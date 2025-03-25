# nyx/core/attentional_controller.py

import logging
import math
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from collections import defaultdict

from agents import Agent, Runner, function_tool, RunContextWrapper

class AttentionalFocus(BaseModel):
    """Schema for current attentional focus"""
    target: str = Field(..., description="Target of attention (modality, concept, task, etc.)")
    strength: float = Field(..., description="Strength of attention (0.0-1.0)", ge=0.0, le=1.0)
    duration_ms: int = Field(..., description="Duration in milliseconds")
    source: str = Field(..., description="Source triggering this attention")
    timestamp: str = Field(..., description="When this focus was established")

class AttentionalControl(BaseModel):
    """Schema for attentional control signal"""
    target: str = Field(..., description="Target to attend to")
    priority: float = Field(..., description="Priority level (0.0-1.0)", ge=0.0, le=1.0)
    duration_ms: int = Field(..., description="Requested duration in milliseconds")
    source: str = Field(..., description="Source requesting attention")
    action: str = Field("focus", description="Action: focus, inhibit, maintain")

class SaliencyConfig(BaseModel):
    """Configuration for saliency calculations"""
    novelty_weight: float = Field(0.3, description="Weight for novelty")
    intensity_weight: float = Field(0.2, description="Weight for intensity")
    emotional_weight: float = Field(0.2, description="Weight for emotional relevance")
    goal_weight: float = Field(0.3, description="Weight for goal relevance")
    
class AttentionalController:
    """
    Controls attention across all system components by determining
    what information to prioritize based on saliency and current goals.
    """
    
    def __init__(self, max_foci: int = 3, emotional_core = None):
        self.max_foci = max_foci  # Maximum number of simultaneous attentional foci
        self.emotional_core = emotional_core  # Reference to emotional system for affective attention
        
        # Current attentional state
        self.current_foci = []  # Current active attentional foci
        self.inhibited_targets = {}  # Targets currently inhibited with expiry times
        self.attentional_history = []  # History of attentional shifts
        self.max_history = 50  # Maximum history size
        
        # Attentional resources and capacity
        self.total_attentional_capacity = 1.0  # Total available attention
        self.attentional_resources = 1.0  # Current available attentional resources
        self.resource_recovery_rate = 0.1  # Rate of resource recovery per second
        self.last_recovery_time = time.time()
        
        # Attentional control requests queue
        self.control_requests = []  # Pending requests
        
        # Salience configuration
        self.saliency_config = SaliencyConfig()
        
        # Attentional bias (can be modified by learning)
        self.attention_biases = defaultdict(float)  # target -> bias
        
        # Performance monitoring
        self.miss_count = 0  # Attention misses
        self.shift_count = 0  # Attention shifts
        
        self.logger = logging.getLogger(__name__)
    
    async def update_attention(self, 
                              salient_items: List[Dict[str, Any]] = None,
                              control_signals: List[AttentionalControl] = None) -> List[AttentionalFocus]:
        """
        Update attentional focus based on salient items and control signals
        
        Args:
            salient_items: Items detected as salient with their properties
            control_signals: Explicit control signals requesting attention
            
        Returns:
            Current attentional foci after update
        """
        # 1. Update attentional resources
        await self._recover_attentional_resources()
        
        # 2. Process any control signals (top-down attention)
        if control_signals:
            for signal in control_signals:
                await self._process_control_signal(signal)
        
        # 3. Add any pending control requests
        for request in self.control_requests:
            if request.action == "focus":
                await self._focus_attention(request.target, request.priority, request.duration_ms, request.source)
            elif request.action == "inhibit":
                await self._inhibit_attention(request.target, request.duration_ms)
            elif request.action == "maintain":
                await self._maintain_attention(request.target, request.duration_ms)
        
        # Clear processed requests
        self.control_requests = []
        
        # 4. Process salient items (bottom-up attention)
        if salient_items:
            # Calculate saliency scores
            scored_items = []
            for item in salient_items:
                score = await self._calculate_saliency(item)
                scored_items.append((item, score))
            
            # Sort by saliency score
            scored_items.sort(key=lambda x: x[1], reverse=True)
            
            # Focus attention on most salient items if resources available
            for item, score in scored_items:
                if score > 0.5 and self.attentional_resources > 0.2:
                    # Get target from item
                    target = item.get("target", item.get("id", "unknown"))
                    
                    # Skip if this target is inhibited
                    if target in self.inhibited_targets:
                        continue
                    
                    # Calculate attention strength based on saliency and available resources
                    strength = min(score, self.attentional_resources)
                    
                    # Calculate duration based on saliency
                    duration_ms = int(2000 * score)  # 0-2 seconds based on saliency
                    
                    # Focus attention
                    await self._focus_attention(target, strength, duration_ms, "saliency")
        
        # 5. Update and expire old foci
        await self._expire_old_foci()
        
        # 6. Update attentional history
        self._update_history()
        
        return self.current_foci
    
    async def _recover_attentional_resources(self):
        """Recover attentional resources over time"""
        current_time = time.time()
        elapsed_seconds = current_time - self.last_recovery_time
        
        if elapsed_seconds > 0:
            # Calculate recovery
            recovery_amount = self.resource_recovery_rate * elapsed_seconds
            
            # Update resources (capped at max capacity)
            self.attentional_resources = min(self.total_attentional_capacity, 
                                           self.attentional_resources + recovery_amount)
            
            # Update last recovery time
            self.last_recovery_time = current_time
    
    async def _process_control_signal(self, signal: AttentionalControl):
        """Process an attentional control signal"""
        # Add to request queue
        self.control_requests.append(signal)
    
    async def _focus_attention(self, 
                             target: str, 
                             strength: float, 
                             duration_ms: int,
                             source: str):
        """Focus attention on a specific target"""
        # Check if already focused
        for focus in self.current_foci:
            if focus.target == target:
                # Update existing focus
                focus.strength = max(focus.strength, strength)
                focus.duration_ms = max(focus.duration_ms, duration_ms)
                focus.source = f"{focus.source}, {source}"
                return
        
        # Check if we have capacity for new focus
        if len(self.current_foci) >= self.max_foci:
            # Remove weakest focus if needed
            self.current_foci.sort(key=lambda x: x.strength)
            if self.current_foci[0].strength < strength:
                self.current_foci.pop(0)
                self.shift_count += 1
            else:
                # Can't focus on this target - attention miss
                self.miss_count += 1
                return
        
        # Create new focus
        new_focus = AttentionalFocus(
            target=target,
            strength=strength,
            duration_ms=duration_ms,
            source=source,
            timestamp=time.time()
        )
        
        # Add to current foci
        self.current_foci.append(new_focus)
        
        # Consume attentional resources
        self.attentional_resources -= strength * 0.2  # Scale resource consumption
        self.attentional_resources = max(0, self.attentional_resources)  # Ensure non-negative
    
    async def _inhibit_attention(self, target: str, duration_ms: int):
        """Inhibit attention to a specific target for a duration"""
        # Remove any current focus on this target
        self.current_foci = [f for f in self.current_foci if f.target != target]
        
        # Add to inhibited targets
        expiry_time = time.time() + (duration_ms / 1000)
        self.inhibited_targets[target] = expiry_time
    
    async def _maintain_attention(self, target: str, duration_ms: int):
        """Maintain attention on a currently focused target"""
        for focus in self.current_foci:
            if focus.target == target:
                # Extend duration
                focus.duration_ms += duration_ms
                return
    
    async def _expire_old_foci(self):
        """Remove expired attentional foci and inhibitions"""
        current_time = time.time()
        
        # Expire foci
        active_foci = []
        for focus in self.current_foci:
            # Check if focus has expired
            focus_end_time = float(focus.timestamp) + (focus.duration_ms / 1000)
            
            if current_time < focus_end_time:
                # Still active
                active_foci.append(focus)
            else:
                # Expired - free up resources
                self.attentional_resources += focus.strength * 0.1  # Partial resource recovery
        
        # Update current foci
        self.current_foci = active_foci
        
        # Expire inhibitions
        to_remove = []
        for target, expiry_time in self.inhibited_targets.items():
            if current_time > expiry_time:
                to_remove.append(target)
                
        # Remove expired inhibitions
        for target in to_remove:
            del self.inhibited_targets[target]
    
    async def _calculate_saliency(self, item: Dict[str, Any]) -> float:
        """Calculate saliency score for an item"""
        # Extract features
        novelty = item.get("novelty", 0.5)
        intensity = item.get("intensity", 0.5)
        emotional_impact = item.get("emotional_impact", 0.5)
        goal_relevance = item.get("goal_relevance", 0.5)
        
        # Get attentional bias for this target
        target = item.get("target", item.get("id", "unknown"))
        bias = self.attention_biases[target]
        
        # Calculate weighted saliency
        config = self.saliency_config
        saliency = (
            novelty * config.novelty_weight +
            intensity * config.intensity_weight +
            emotional_impact * config.emotional_weight +
            goal_relevance * config.goal_weight +
            bias  # Add bias directly
        )
        
        # Check emotional core for additional affective influence
        if self.emotional_core:
            try:
                emotional_state = self.emotional_core.get_emotional_state()
                arousal = self.emotional_core.get_emotional_arousal()
                
                # High arousal amplifies saliency
                if arousal > 0.6:
                    saliency *= 1.2
                elif arousal < 0.3:
                    saliency *= 0.8
                    
                # Check valence influence if strong emotion is present
                strongest_emotion, strength = self.emotional_core.get_dominant_emotion()
                if strength > 0.6:
                    # Check if emotion matches item
                    if "emotion" in item and item["emotion"] == strongest_emotion:
                        saliency *= 1.3  # Boost for emotional congruence
            except Exception as e:
                self.logger.error(f"Error applying emotional influence to saliency: {e}")
        
        # Normalize saliency to 0-1 range
        return max(0.0, min(1.0, saliency))
    
    def _update_history(self):
        """Update attentional history with current focus"""
        # Add current foci to history
        for focus in self.current_foci:
            self.attentional_history.append({
                "target": focus.target,
                "strength": focus.strength,
                "source": focus.source,
                "timestamp": focus.timestamp
            })
            
        # Trim history
        if len(self.attentional_history) > self.max_history:
            self.attentional_history = self.attentional_history[-self.max_history:]
    
    async def request_attention(self, control: AttentionalControl) -> bool:
        """Request attention focus, inhibition, or maintenance"""
        self.control_requests.append(control)
        return True
    
    async def calculate_attention_weight(self, 
                                      item: Any, 
                                      modality: str = None,
                                      context: Any = None) -> float:
        """
        Calculate attention weight for an item based on current attentional focus
        
        Args:
            item: Item to calculate attention for
            modality: Optional modality of the item
            context: Optional context for attention calculation
            
        Returns:
            Attention weight (0.0-1.0)
        """
        # Get target identifier
        if hasattr(item, "id"):
            target = item.id
        elif hasattr(item, "target"):
            target = item.target
        elif modality:
            target = modality
        else:
            target = "unknown"
        
        # Check if target is currently inhibited
        if target in self.inhibited_targets:
            return 0.1  # Minimal attention to inhibited targets
        
        # Check if target is currently in focus
        for focus in self.current_foci:
            if focus.target == target or (modality and focus.target == modality):
                return focus.strength  # Full attention weight
        
        # If not focused but modality is in focus, give partial attention
        if modality:
            for focus in self.current_foci:
                if focus.target == modality:
                    return focus.strength * 0.7  # Partial attention weight
        
        # Default moderate attention if not inhibited and resources available
        if self.attentional_resources > 0.5:
            return 0.5
        else:
            return 0.3  # Reduced attention when resources are low
    
    async def update_attention_bias(self, target: str, adjustment: float):
        """Update attention bias for a target based on learning"""
        current_bias = self.attention_biases[target]
        
        # Apply adjustment with constraints to keep in reasonable range
        new_bias = current_bias + adjustment
        new_bias = max(-0.3, min(0.3, new_bias))  # Limit bias to -0.3 to 0.3
        
        self.attention_biases[target] = new_bias
        
    async def get_attention_statistics(self) -> Dict[str, Any]:
        """Get statistics about attentional system performance"""
        return {
            "current_foci_count": len(self.current_foci),
            "attentional_resources": self.attentional_resources,
            "inhibited_targets_count": len(self.inhibited_targets),
            "attention_shifts": self.shift_count,
            "attention_misses": self.miss_count,
            "most_focused": self._get_most_focused_targets(5),
            "miss_rate": self.miss_count / max(1, self.miss_count + self.shift_count)
        }
    
    def _get_most_focused_targets(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently focused targets"""
        target_counts = defaultdict(int)
        
        for entry in self.attentional_history:
            target_counts[entry["target"]] += 1
            
        # Sort by count
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [{"target": t, "focus_count": c} for t, c in sorted_targets[:limit]]
