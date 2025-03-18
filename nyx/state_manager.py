import asyncio
import copy
import datetime
from typing import Any, Dict, List, Optional, Callable

class StateManager:
    """Manages state synchronization across autonomous systems"""
    
    def __init__(self):
        self.states = {
            "memory": {},
            "npc": {},
            "lore": {},
            "scene": {}
        }
        self.state_history = []
        self.state_locks = {}
        self.state_subscriptions = {}
        self.state_validations = {}
        self.metrics = {
            "updates": 0,
            "conflicts": 0,
            "resolutions": 0
        }

    async def initialize(self):
        """Initialize state tracking"""
        # Set up state locks
        for system in self.states.keys():
            self.state_locks[system] = asyncio.Lock()
            
        # Set up state validations
        self.state_validations = {
            "memory": self._validate_memory_state,
            "npc": self._validate_npc_state,
            "lore": self._validate_lore_state,
            "scene": self._validate_scene_state
        }
        
        # Initialize history
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "states": copy.deepcopy(self.states),
            "action": "initialization"
        })

    async def update_memory_state(self, update: Dict[str, Any]):
        """Update memory system state"""
        async with self.state_locks["memory"]:
            # Validate update
            if not await self._validate_memory_state(update):
                raise ValueError("Invalid memory state update")
                
            # Check for conflicts
            conflicts = self._detect_state_conflicts("memory", update)
            if conflicts:
                resolution = await self._resolve_state_conflicts("memory", conflicts)
                self.metrics["conflicts"] += 1
                self.metrics["resolutions"] += 1
                
            # Apply update
            self.states["memory"].update(update)
            self.metrics["updates"] += 1
            
            # Record history
            self._record_state_change("memory", update)
            
            # Notify subscribers
            await self._notify_state_change("memory", update)

    async def update_npc_state(self, update: Dict[str, Any]):
        """Update NPC system state"""
        async with self.state_locks["npc"]:
            if not await self._validate_npc_state(update):
                raise ValueError("Invalid NPC state update")
                
            conflicts = self._detect_state_conflicts("npc", update)
            if conflicts:
                resolution = await self._resolve_state_conflicts("npc", conflicts)
                self.metrics["conflicts"] += 1
                self.metrics["resolutions"] += 1
                
            self.states["npc"].update(update)
            self.metrics["updates"] += 1
            
            self._record_state_change("npc", update)
            await self._notify_state_change("npc", update)

    async def update_lore_state(self, update: Dict[str, Any]):
        """Update lore system state"""
        async with self.state_locks["lore"]:
            if not await self._validate_lore_state(update):
                raise ValueError("Invalid lore state update")
                
            conflicts = self._detect_state_conflicts("lore", update)
            if conflicts:
                resolution = await self._resolve_state_conflicts("lore", conflicts)
                self.metrics["conflicts"] += 1
                self.metrics["resolutions"] += 1
                
            self.states["lore"].update(update)
            self.metrics["updates"] += 1
            
            self._record_state_change("lore", update)
            await self._notify_state_change("lore", update)

    async def update_scene_state(self, update: Dict[str, Any]):
        """Update scene system state"""
        async with self.state_locks["scene"]:
            if not await self._validate_scene_state(update):
                raise ValueError("Invalid scene state update")
                
            conflicts = self._detect_state_conflicts("scene", update)
            if conflicts:
                resolution = await self._resolve_state_conflicts("scene", conflicts)
                self.metrics["conflicts"] += 1
                self.metrics["resolutions"] += 1
                
            self.states["scene"].update(update)
            self.metrics["updates"] += 1
            
            self._record_state_change("scene", update)
            await self._notify_state_change("scene", update)

    def _detect_state_conflicts(self, system: str, update: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts between state updates"""
        conflicts = []
        
        # Check for direct conflicts with other systems
        for other_system, state in self.states.items():
            if other_system == system:
                continue
                
            # Check for overlapping keys with different values
            for key, value in update.items():
                if key in state and state[key] != value:
                    conflicts.append({
                        "type": "direct_conflict",
                        "system": other_system,
                        "key": key,
                        "current_value": state[key],
                        "new_value": value
                    })
                    
            # Check for semantic conflicts
            semantic_conflicts = self._check_semantic_conflicts(system, other_system, update)
            conflicts.extend(semantic_conflicts)
            
        return conflicts

    async def _resolve_state_conflicts(self, system: str, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between state updates"""
        resolutions = {}
        
        for conflict in conflicts:
            if conflict["type"] == "direct_conflict":
                # Resolve direct conflicts based on system priority
                resolution = await self._resolve_direct_conflict(system, conflict)
                resolutions[conflict["key"]] = resolution
            elif conflict["type"] == "semantic_conflict":
                # Resolve semantic conflicts through negotiation
                resolution = await self._resolve_semantic_conflict(system, conflict)
                resolutions.update(resolution)
                
        return resolutions

    def _record_state_change(self, system: str, update: Dict[str, Any]):
        """Record state change in history"""
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "system": system,
            "update": update,
            "states": copy.deepcopy(self.states)
        })
        
        # Limit history size
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]

    async def _notify_state_change(self, system: str, update: Dict[str, Any]):
        """Notify subscribers of state changes"""
        if system in self.state_subscriptions:
            for callback in self.state_subscriptions[system]:
                try:
                    await callback(system, update)
                except Exception as e:
                    logger.error(f"Error in state change notification: {e}")

    async def _validate_memory_state(self, state: Dict[str, Any]) -> bool:
        """Validate memory state update"""
        required_keys = {"memories", "beliefs", "emotional_state"}
        if not all(key in state for key in required_keys):
            return False
            
        # Validate memory format
        if "memories" in state:
            for memory in state["memories"]:
                if not self._validate_memory_format(memory):
                    return False
                    
        # Validate belief format
        if "beliefs" in state:
            for belief in state["beliefs"]:
                if not self._validate_belief_format(belief):
                    return False
                    
        # Validate emotional state
        if "emotional_state" in state:
            if not self._validate_emotional_state(state["emotional_state"]):
                return False
                
        return True

    async def _validate_npc_state(self, state: Dict[str, Any]) -> bool:
        """Validate NPC state update"""
        required_keys = {"npcs", "relationships", "behaviors"}
        if not all(key in state for key in required_keys):
            return False
            
        # Validate NPC format
        if "npcs" in state:
            for npc in state["npcs"]:
                if not self._validate_npc_format(npc):
                    return False
                    
        # Validate relationship format
        if "relationships" in state:
            for rel in state["relationships"]:
                if not self._validate_relationship_format(rel):
                    return False
                    
        # Validate behavior format
        if "behaviors" in state:
            for behavior in state["behaviors"]:
                if not self._validate_behavior_format(behavior):
                    return False
                    
        return True

    async def _validate_lore_state(self, state: Dict[str, Any]) -> bool:
        """Validate lore state update"""
        required_keys = {"lore_elements", "narratives", "themes"}
        if not all(key in state for key in required_keys):
            return False
            
        # Validate lore element format
        if "lore_elements" in state:
            for element in state["lore_elements"]:
                if not self._validate_lore_element_format(element):
                    return False
                    
        # Validate narrative format
        if "narratives" in state:
            for narrative in state["narratives"]:
                if not self._validate_narrative_format(narrative):
                    return False
                    
        # Validate theme format
        if "themes" in state:
            for theme in state["themes"]:
                if not self._validate_theme_format(theme):
                    return False
                    
        return True

    async def _validate_scene_state(self, state: Dict[str, Any]) -> bool:
        """Validate scene state update"""
        required_keys = {"scene_elements", "active_npcs", "environment"}
        if not all(key in state for key in required_keys):
            return False
            
        # Validate scene element format
        if "scene_elements" in state:
            for element in state["scene_elements"]:
                if not self._validate_scene_element_format(element):
                    return False
                    
        # Validate active NPC format
        if "active_npcs" in state:
            for npc in state["active_npcs"]:
                if not self._validate_active_npc_format(npc):
                    return False
                    
        # Validate environment format
        if "environment" in state:
            if not self._validate_environment_format(state["environment"]):
                return False
                
        return True

    def _check_semantic_conflicts(self, system1: str, system2: str, update: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for semantic conflicts between systems"""
        conflicts = []
        
        # Check for narrative consistency
        narrative_conflicts = self._check_narrative_consistency(system1, system2, update)
        conflicts.extend(narrative_conflicts)
        
        # Check for character consistency
        character_conflicts = self._check_character_consistency(system1, system2, update)
        conflicts.extend(character_conflicts)
        
        # Check for emotional consistency
        emotional_conflicts = self._check_emotional_consistency(system1, system2, update)
        conflicts.extend(emotional_conflicts)
        
        return conflicts

    async def _resolve_direct_conflict(self, system: str, conflict: Dict[str, Any]) -> Any:
        """Resolve direct conflict between systems"""
        # Get system priorities
        priorities = {
            "memory": 3,
            "npc": 2,
            "lore": 1,
            "scene": 0
        }
        
        # Higher priority system wins
        if priorities[system] >= priorities[conflict["system"]]:
            return conflict["new_value"]
        else:
            return conflict["current_value"]

    async def _resolve_semantic_conflict(self, system: str, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve semantic conflict through negotiation"""
        if conflict["type"] == "narrative_conflict":
            return await self._resolve_narrative_conflict(system, conflict)
        elif conflict["type"] == "character_conflict":
            return await self._resolve_character_conflict(system, conflict)
        elif conflict["type"] == "emotional_conflict":
            return await self._resolve_emotional_conflict(system, conflict)
        else:
            return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get state management metrics"""
        return {
            "updates": self.metrics["updates"],
            "conflicts": self.metrics["conflicts"],
            "resolutions": self.metrics["resolutions"],
            "history_size": len(self.state_history),
            "current_states": {
                system: len(state) for system, state in self.states.items()
            }
        }

    async def subscribe_to_state_changes(self, system: str, callback: Callable):
        """Subscribe to state changes for a system"""
        if system not in self.state_subscriptions:
            self.state_subscriptions[system] = []
        self.state_subscriptions[system].append(callback)

    async def unsubscribe_from_state_changes(self, system: str, callback: Callable):
        """Unsubscribe from state changes"""
        if system in self.state_subscriptions:
            self.state_subscriptions[system].remove(callback)

    def get_state_history(self, system: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get state change history"""
        if system:
            return [
                entry for entry in self.state_history 
                if entry.get("system") == system
            ]
        return self.state_history 