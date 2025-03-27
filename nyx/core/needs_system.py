# nyx/core/needs_system.py

import logging
import datetime
import math
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class NeedState(BaseModel):
    name: str
    level: float = Field(0.5, ge=0.0, le=1.0, description="Current satisfaction level (0=empty, 1=full)")
    target_level: float = Field(1.0, ge=0.0, le=1.0, description="Desired satisfaction level")
    importance: float = Field(0.5, ge=0.1, le=1.0, description="Importance weight of this need")
    decay_rate: float = Field(0.01, ge=0.0, le=0.1, description="Rate of decay per hour") # Slower decay
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

    @property
    def deficit(self) -> float:
        return max(0.0, self.target_level - self.level)

    @property
    def drive_strength(self) -> float:
        # Drive is higher when deficit is large AND importance is high
        return self.deficit * self.importance

class NeedsSystem:
    """Tracks and manages Nyx's core digital needs."""

    def __init__(self, goal_manager=None):
        self.goal_manager = goal_manager # To trigger goal creation
        self.needs: Dict[str, NeedState] = {
            "knowledge": NeedState(name="knowledge", importance=0.8, decay_rate=0.01),
            "connection": NeedState(name="connection", importance=0.9, decay_rate=0.015),
            "novelty": NeedState(name="novelty", importance=0.6, decay_rate=0.02),
            "coherence": NeedState(name="coherence", importance=0.7, decay_rate=0.005),
            "agency": NeedState(name="agency", importance=0.8, decay_rate=0.01),
            "safety": NeedState(name="safety", importance=0.95, level=0.8, decay_rate=0.002), # Start safer
        }
        self.last_update_time = datetime.datetime.now()
        self.drive_threshold_for_goal = 0.4 # Minimum drive strength to trigger a goal

        logger.info("NeedsSystem initialized.")

    async def update_needs(self) -> Dict[str, float]:
        """Applies decay and returns current drive strengths."""
        now = datetime.datetime.now()
        elapsed_hours = (now - self.last_update_time).total_seconds() / 3600.0

        drive_strengths = {}
        needs_to_trigger_goals = []

        if elapsed_hours > 0.01: # Only update if some time passed
            for name, need in self.needs.items():
                # Apply decay towards a baseline (e.g., 0.3) not zero
                baseline_satisfaction = 0.3
                decay_amount = need.decay_rate * elapsed_hours
                if need.level > baseline_satisfaction:
                    need.level = max(baseline_satisfaction, need.level - decay_amount)
                elif need.level < baseline_satisfaction:
                     need.level = min(baseline_satisfaction, need.level + decay_amount) # Slowly drifts up if very low

                need.level = max(0.0, min(1.0, need.level)) # Clamp
                need.last_updated = now
                drive = need.drive_strength
                drive_strengths[name] = drive

                # Check if need deficit triggers a goal
                if drive > self.drive_threshold_for_goal and self.goal_manager:
                    # Check if a similar goal already exists and is active
                    if not self.goal_manager.has_active_goal_for_need(name):
                         needs_to_trigger_goals.append(need)

            self.last_update_time = now

            # Trigger goal creation asynchronously for high-drive needs
            if needs_to_trigger_goals and self.goal_manager:
                 # Use asyncio.create_task for non-blocking goal generation
                 asyncio.create_task(self._trigger_goal_creation(needs_to_trigger_goals))

        else: # If no significant time elapsed, return current drives
            drive_strengths = {name: need.drive_strength for name, need in self.needs.items()}

        return drive_strengths

    async def _trigger_goal_creation(self, needs_list: List[NeedState]):
        """Asks the GoalManager to create goals for unmet needs."""
        logger.info(f"Needs [{', '.join(n.name for n in needs_list)}] exceeded drive threshold. Requesting goal creation.")
        for need in needs_list:
            # Add goal with a priority based on drive strength
            priority = 0.5 + (need.drive_strength * 0.5) # Map drive (0.4-1.0) to priority (0.7-1.0)
            await self.goal_manager.add_goal(
                description=f"Satisfy need for {need.name} (Current level: {need.level:.2f}, Deficit: {need.deficit:.2f})",
                priority=priority,
                source="NeedsSystem",
                associated_need=need.name
            )

    async def satisfy_need(self, name: str, amount: float):
        """Increases the satisfaction level of a need."""
        if name in self.needs:
            need = self.needs[name]
            original_level = need.level
            need.level = min(need.target_level, need.level + amount)
            need.last_updated = datetime.datetime.now()
            logger.debug(f"Satisfied need '{name}' by {amount:.2f}. Level: {original_level:.2f} -> {need.level:.2f}")
        else:
            logger.warning(f"Attempted to satisfy unknown need: {name}")

    async def decrease_need(self, name: str, amount: float):
        """Decreases the satisfaction level of a need (e.g., due to errors, negative feedback)."""
        if name in self.needs:
            need = self.needs[name]
            original_level = need.level
            need.level = max(0.0, need.level - amount)
            need.last_updated = datetime.datetime.now()
            logger.debug(f"Decreased need '{name}' by {amount:.2f}. Level: {original_level:.2f} -> {need.level:.2f}")
        else:
             logger.warning(f"Attempted to decrease unknown need: {name}")

    def get_needs_state(self) -> Dict[str, Dict[str, Any]]:
        """Returns the current state of all needs."""
        return {name: need.model_dump(exclude={'last_updated'}) | {'last_updated': need.last_updated.isoformat(), 'deficit': need.deficit, 'drive_strength': need.drive_strength}
                for name, need in self.needs.items()}

    def get_total_drive(self) -> float:
        """Returns the sum of all drive strengths."""
        return sum(need.drive_strength for need in self.needs.values())
