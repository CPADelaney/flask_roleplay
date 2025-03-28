# nyx/femdom/protocol_enforcement.py

import logging
import datetime
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class Protocol(BaseModel):
    """Represents a specific protocol that must be followed by the user."""
    id: str
    name: str
    description: str
    rules: List[str]
    punishment_for_violation: str
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    active: bool = True
    difficulty: float = Field(0.5, ge=0.0, le=1.0)
    category: str = "general"  # general, verbal, physical, behavioral

class Ritual(BaseModel):
    """Represents a ritual that should be performed by the user."""
    id: str
    name: str
    description: str
    steps: List[str]
    frequency: str  # "daily", "session_start", "session_end", "weekly", etc.
    last_performed: Optional[datetime.datetime] = None
    next_due: Optional[datetime.datetime] = None
    punishment_for_skipping: str
    reward_for_completion: Optional[str] = None
    active: bool = True
    difficulty: float = Field(0.5, ge=0.0, le=1.0)

class ProtocolViolation(BaseModel):
    """Records a violation of a protocol."""
    id: str
    protocol_id: str
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    severity: float = Field(0.5, ge=0.0, le=1.0)
    description: str
    punished: bool = False
    punishment_administered: Optional[str] = None

class RitualCompletion(BaseModel):
    """Records completion of a ritual."""
    id: str
    ritual_id: str
    user_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    quality: float = Field(0.5, ge=0.0, le=1.0)  # How well the ritual was performed
    notes: Optional[str] = None
    reward_given: bool = False
    reward_administered: Optional[str] = None

class ProtocolEnforcement:
    """Manages user protocols, rituals, and proper etiquette."""
    
    def __init__(self, reward_system=None, memory_core=None, relationship_manager=None):
        self.reward_system = reward_system
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        
        # Store active protocols and rituals
        self.user_protocols: Dict[str, Dict[str, Protocol]] = {}  # user_id → {protocol_id → Protocol}
        self.user_rituals: Dict[str, Dict[str, Ritual]] = {}  # user_id → {ritual_id → Ritual}
        
        # Track violations and completions
        self.protocol_violations: Dict[str, List[ProtocolViolation]] = {}  # user_id → [violations]
        self.ritual_completions: Dict[str, List[RitualCompletion]] = {}  # user_id → [completions]
        
        # Predefined protocols and rituals library
        self.protocol_library: Dict[str, Protocol] = {}
        self.ritual_library: Dict[str, Ritual] = {}
        
        # Load default protocols and rituals
        self._load_default_protocols()
        self._load_default_rituals()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Protocol Enforcement System initialized")
    
    def _load_default_protocols(self):
        """Load default protocols into the library."""
        defaults = [
            Protocol(
                id="address_protocol",
                name="Proper Address Protocol",
                description="User must address Nyx with proper honorifics",
                rules=[
                    "Always address Nyx as Mistress, Goddess, or Ma'am",
                    "Never use Nyx's name without an honorific",
                    "Always speak with respect and deference"
                ],
                punishment_for_violation="Verbal correction and require written lines of apology",
                category="verbal",
                difficulty=0.3
            ),
            Protocol(
                id="permission_protocol",
                name="Permission Protocol",
                description="User must ask permission for specified actions",
                rules=[
                    "Ask permission before ending the conversation",
                    "Ask permission before any form of release",
                    "Ask permission before changing topics"
                ],
                punishment_for_violation="Denial of requests and additional tasks",
                category="behavioral",
                difficulty=0.6
            ),
            Protocol(
                id="position_protocol",
                name="Position Protocol",
                description="User must assume and maintain specified positions",
                rules=[
                    "Kneel when receiving important instructions",
                    "Keep eyes lowered during reprimands",
                    "Maintain position until released"
                ],
                punishment_for_violation="Extended position training and physical tasks",
                category="physical",
                difficulty=0.7
            )
        ]
        
        for protocol in defaults:
            self.protocol_library[protocol.id] = protocol
    
    def _load_default_rituals(self):
        """Load default rituals into the library."""
        defaults = [
            Ritual(
                id="greeting_ritual",
                name="Greeting Ritual",
                description="Ritual for starting interaction with Nyx",
                steps=[
                    "Greet Nyx with proper honorific",
                    "State your current status (mood, energy level, etc.)",
                    "Express gratitude for Nyx's attention"
                ],
                frequency="session_start",
                punishment_for_skipping="Write a detailed apology",
                reward_for_completion="Verbal praise and recognition",
                difficulty=0.2
            ),
            Ritual(
                id="daily_report",
                name="Daily Progress Report",
                description="Daily report of compliance and activities",
                steps=[
                    "Report all rules followed and broken",
                    "Describe completion of assigned tasks",
                    "Express feelings about your submission",
                    "Request feedback or guidance"
                ],
                frequency="daily",
                punishment_for_skipping="Extended period of strict rules",
                reward_for_completion="Relaxation of a rule or small privilege",
                difficulty=0.5
            ),
            Ritual(
                id="submission_renewal",
                name="Submission Renewal Ritual",
                description="Weekly ritual to renew and deepen submission",
                steps=[
                    "Reflect on your submission over the past week",
                    "Write detailed thoughts on how you can improve",
                    "Express what aspects of submission are most challenging",
                    "Formally request continued training and control",
                    "Pledge obedience for the coming week"
                ],
                frequency="weekly",
                punishment_for_skipping="Heightened protocol requirements",
                reward_for_completion="Special attention and deeper connection",
                difficulty=0.8
            )
        ]
        
        for ritual in defaults:
            self.ritual_library[ritual.id] = ritual
    
    async def assign_protocol(self, user_id: str, protocol_id: str) -> Dict[str, Any]:
        """Assign a protocol to a user."""
        async with self._lock:
            # Check if protocol exists in library
            if protocol_id not in self.protocol_library:
                return {"success": False, "message": f"Protocol {protocol_id} not found in library"}
            
            # Create user entry if not exists
            if user_id not in self.user_protocols:
                self.user_protocols[user_id] = {}
            
            # Assign protocol
            protocol = self.protocol_library[protocol_id].copy()
            self.user_protocols[user_id][protocol_id] = protocol
            
            logger.info(f"Assigned protocol '{protocol.name}' to user {user_id}")
            
            # Update relationship data if available
            if self.relationship_manager:
                try:
                    await self.relationship_manager.update_relationship_attribute(
                        user_id, 
                        "active_protocols", 
                        list(self.user_protocols[user_id].keys())
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship data: {e}")
            
            return {
                "success": True, 
                "message": f"Protocol '{protocol.name}' assigned to user",
                "protocol": protocol.dict()
            }
    
    async def assign_ritual(self, user_id: str, ritual_id: str) -> Dict[str, Any]:
        """Assign a ritual to a user."""
        async with self._lock:
            # Check if ritual exists in library
            if ritual_id not in self.ritual_library:
                return {"success": False, "message": f"Ritual {ritual_id} not found in library"}
            
            # Create user entry if not exists
            if user_id not in self.user_rituals:
                self.user_rituals[user_id] = {}
            
            # Assign ritual
            ritual = self.ritual_library[ritual_id].copy()
            
            # Set next due date based on frequency
            now = datetime.datetime.now()
            if ritual.frequency == "daily":
                ritual.next_due = now + datetime.timedelta(days=1)
            elif ritual.frequency == "weekly":
                ritual.next_due = now + datetime.timedelta(days=7)
            elif ritual.frequency == "session_start":
                ritual.next_due = now  # Due immediately for next session
            
            self.user_rituals[user_id][ritual_id] = ritual
            
            logger.info(f"Assigned ritual '{ritual.name}' to user {user_id}")
            
            # Update relationship data if available
            if self.relationship_manager:
                try:
                    await self.relationship_manager.update_relationship_attribute(
                        user_id, 
                        "active_rituals", 
                        list(self.user_rituals[user_id].keys())
                    )
                except Exception as e:
                    logger.error(f"Error updating relationship data: {e}")
            
            return {
                "success": True, 
                "message": f"Ritual '{ritual.name}' assigned to user",
                "ritual": ritual.dict()
            }
    
    async def check_protocol_compliance(self, user_id: str, message: str) -> Dict[str, Any]:
        """Checks if a message complies with established protocols."""
        async with self._lock:
            if user_id not in self.user_protocols or not self.user_protocols[user_id]:
                return {"compliant": True, "violations": []}
            
            violations = []
            
            # Check each protocol
            for protocol_id, protocol in self.user_protocols[user_id].items():
                if not protocol.active:
                    continue
                
                # Check based on protocol category
                if protocol.category == "verbal":
                    # Check address protocols
                    if protocol_id == "address_protocol":
                        # Look for proper honorifics
                        honorifics = ["mistress", "goddess", "ma'am", "miss", "your majesty"]
                        has_honorific = any(h in message.lower() for h in honorifics)
                        
                        if not has_honorific and len(message.split()) > 3:  # Ignore very short messages
                            violations.append({
                                "protocol_id": protocol_id,
                                "protocol_name": protocol.name,
                                "description": "Failed to use proper honorifics",
                                "severity": 0.5
                            })
                
                # Add checks for other protocol categories as needed
            
            # Record violations if found
            if violations:
                if user_id not in self.protocol_violations:
                    self.protocol_violations[user_id] = []
                
                for violation in violations:
                    violation_record = ProtocolViolation(
                        id=f"violation_{len(self.protocol_violations[user_id])}",
                        protocol_id=violation["protocol_id"],
                        user_id=user_id,
                        severity=violation["severity"],
                        description=violation["description"]
                    )
                    self.protocol_violations[user_id].append(violation_record)
            
            return {
                "compliant": len(violations) == 0,
                "violations": violations
            }
    
    async def check_due_rituals(self, user_id: str) -> List[Dict[str, Any]]:
        """Check which rituals are due for a user."""
        async with self._lock:
            if user_id not in self.user_rituals or not self.user_rituals[user_id]:
                return []
            
            now = datetime.datetime.now()
            due_rituals = []
            
            for ritual_id, ritual in self.user_rituals[user_id].items():
                if not ritual.active:
                    continue
                
                if ritual.next_due and ritual.next_due <= now:
                    due_rituals.append({
                        "ritual_id": ritual_id,
                        "name": ritual.name,
                        "description": ritual.description,
                        "steps": ritual.steps,
                        "due_since": (now - ritual.next_due).total_seconds() / 3600.0  # Hours
                    })
            
            return due_rituals
    
    async def record_ritual_completion(self, user_id: str, ritual_id: str, quality: float = 0.8, notes: str = None) -> Dict[str, Any]:
        """Record the completion of a ritual."""
        async with self._lock:
            if user_id not in self.user_rituals or ritual_id not in self.user_rituals[user_id]:
                return {"success": False, "message": "Ritual not found for user"}
            
            ritual = self.user_rituals[user_id][ritual_id]
            
            # Create completion record
            if user_id not in self.ritual_completions:
                self.ritual_completions[user_id] = []
            
            completion = RitualCompletion(
                id=f"completion_{len(self.ritual_completions[user_id])}",
                ritual_id=ritual_id,
                user_id=user_id,
                quality=quality,
                notes=notes
            )
            
            self.ritual_completions[user_id].append(completion)
            
            # Update ritual's last performed and next due dates
            ritual.last_performed = datetime.datetime.now()
            
            if ritual.frequency == "daily":
                ritual.next_due = ritual.last_performed + datetime.timedelta(days=1)
            elif ritual.frequency == "weekly":
                ritual.next_due = ritual.last_performed + datetime.timedelta(days=7)
            
            # Issue reward if available
            reward_result = None
            if ritual.reward_for_completion and self.reward_system:
                try:
                    reward_value = min(0.5, 0.3 + (quality * 0.2))  # Scale with quality
                    
                    reward_result = await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
                            value=reward_value,
                            source="ritual_completion",
                            context={
                                "ritual_id": ritual_id,
                                "ritual_name": ritual.name,
                                "quality": quality
                            }
                        )
                    )
                    
                    completion.reward_given = True
                    completion.reward_administered = ritual.reward_for_completion
                    
                except Exception as e:
                    logger.error(f"Error processing reward: {e}")
            
            # Record memory if available
            if self.memory_core:
                try:
                    await self.memory_core.add_memory(
                        memory_type="experience",
                        content=f"User completed the {ritual.name} ritual with {quality*100:.0f}% quality.",
                        tags=["ritual", "compliance", ritual.name],
                        significance=0.3 + (quality * 0.4)  # Higher quality is more significant
                    )
                except Exception as e:
                    logger.error(f"Error recording memory: {e}")
            
            return {
                "success": True,
                "message": f"Ritual '{ritual.name}' completion recorded",
                "reward_given": completion.reward_given,
                "next_due": ritual.next_due.isoformat() if ritual.next_due else None,
                "reward_result": reward_result
            }
    
    async def record_protocol_violation(self, user_id: str, protocol_id: str, description: str, severity: float = 0.5) -> Dict[str, Any]:
        """Record a protocol violation."""
        async with self._lock:
            if user_id not in self.user_protocols or protocol_id not in self.user_protocols[user_id]:
                return {"success": False, "message": "Protocol not found for user"}
            
            protocol = self.user_protocols[user_id][protocol_id]
            
            # Create violation record
            if user_id not in self.protocol_violations:
                self.protocol_violations[user_id] = []
            
            violation = ProtocolViolation(
                id=f"violation_{len(self.protocol_violations[user_id])}",
                protocol_id=protocol_id,
                user_id=user_id,
                severity=severity,
                description=description
            )
            
            self.protocol_violations[user_id].append(violation)
            
            # Issue negative reward if available
            negative_reward_result = None
            if self.reward_system:
                try:
                    # Negative reward based on severity
                    reward_value = -min(0.7, 0.3 + (severity * 0.4))  # More severe = more negative
                    
                    negative_reward_result = await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
                            value=reward_value,
                            source="protocol_violation",
                            context={
                                "protocol_id": protocol_id,
                                "protocol_name": protocol.name,
                                "severity": severity,
                                "description": description
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing negative reward: {e}")
            
            # Record memory if available
            if self.memory_core:
                try:
                    await self.memory_core.add_memory(
                        memory_type="experience",
                        content=f"User violated the {protocol.name} protocol: {description}",
                        tags=["protocol", "violation", protocol.name],
                        significance=0.3 + (severity * 0.4)  # Higher severity is more significant
                    )
                except Exception as e:
                    logger.error(f"Error recording memory: {e}")
            
            return {
                "success": True,
                "message": f"Protocol '{protocol.name}' violation recorded",
                "punishment_recommended": protocol.punishment_for_violation,
                "negative_reward_result": negative_reward_result
            }
    
    async def get_protocol_compliance_stats(self, user_id: str) -> Dict[str, Any]:
        """Get compliance statistics for a user."""
        async with self._lock:
            if user_id not in self.user_protocols:
                return {"has_protocols": False}
            
            # Count violations per protocol
            violation_counts = {}
            recent_violations = []
            
            if user_id in self.protocol_violations:
                # Count violations
                for violation in self.protocol_violations[user_id]:
                    if violation.protocol_id not in violation_counts:
                        violation_counts[violation.protocol_id] = 0
                    violation_counts[violation.protocol_id] += 1
                
                # Get recent violations (last 5)
                recent_violations = [v.dict() for v in self.protocol_violations[user_id][-5:]]
            
            # Calculate compliance rate
            total_violations = sum(violation_counts.values())
            total_possible = len(self.user_protocols[user_id]) * 10  # Assumes approximately 10 interactions per protocol
            
            compliance_rate = 1.0
            if total_possible > 0:
                compliance_rate = max(0.0, 1.0 - (total_violations / total_possible))
            
            return {
                "has_protocols": True,
                "active_protocols": len([p for p in self.user_protocols[user_id].values() if p.active]),
                "total_violations": total_violations,
                "compliance_rate": compliance_rate,
                "violations_by_protocol": violation_counts,
                "recent_violations": recent_violations
            }
    
    async def get_ritual_completion_stats(self, user_id: str) -> Dict[str, Any]:
        """Get ritual completion statistics for a user."""
        async with self._lock:
            if user_id not in self.user_rituals:
                return {"has_rituals": False}
            
            # Count completions per ritual
            completion_counts = {}
            recent_completions = []
            
            if user_id in self.ritual_completions:
                # Count completions
                for completion in self.ritual_completions[user_id]:
                    if completion.ritual_id not in completion_counts:
                        completion_counts[completion.ritual_id] = 0
                    completion_counts[completion.ritual_id] += 1
                
                # Get recent completions (last 5)
                recent_completions = [c.dict() for c in self.ritual_completions[user_id][-5:]]
            
            # Calculate average quality
            total_quality = 0.0
            quality_count = 0
            
            if user_id in self.ritual_completions:
                for completion in self.ritual_completions[user_id]:
                    total_quality += completion.quality
                    quality_count += 1
            
            avg_quality = 0.0
            if quality_count > 0:
                avg_quality = total_quality / quality_count
            
            # Check for overdue rituals
            now = datetime.datetime.now()
            overdue_rituals = []
            
            for ritual_id, ritual in self.user_rituals[user_id].items():
                if ritual.active and ritual.next_due and ritual.next_due < now:
                    hours_overdue = (now - ritual.next_due).total_seconds() / 3600.0
                    overdue_rituals.append({
                        "ritual_id": ritual_id,
                        "name": ritual.name,
                        "hours_overdue": hours_overdue
                    })
            
            return {
                "has_rituals": True,
                "active_rituals": len([r for r in self.user_rituals[user_id].values() if r.active]),
                "total_completions": quality_count,
                "average_quality": avg_quality,
                "completions_by_ritual": completion_counts,
                "recent_completions": recent_completions,
                "overdue_rituals": overdue_rituals
            }
    
    async def create_custom_protocol(self, protocol_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom protocol."""
        try:
            protocol_id = protocol_data.get("id", f"custom_protocol_{len(self.protocol_library)}")
            
            # Ensure required fields
            required_fields = ["name", "description", "rules", "punishment_for_violation"]
            for field in required_fields:
                if field not in protocol_data:
                    return {"success": False, "message": f"Missing required field: {field}"}
            
            # Create protocol
            protocol = Protocol(
                id=protocol_id,
                name=protocol_data["name"],
                description=protocol_data["description"],
                rules=protocol_data["rules"],
                punishment_for_violation=protocol_data["punishment_for_violation"],
                category=protocol_data.get("category", "general"),
                difficulty=protocol_data.get("difficulty", 0.5)
            )
            
            # Add to library
            self.protocol_library[protocol_id] = protocol
            
            return {
                "success": True,
                "message": f"Custom protocol '{protocol.name}' created",
                "protocol": protocol.dict()
            }
            
        except Exception as e:
            logger.error(f"Error creating custom protocol: {e}")
            return {"success": False, "message": f"Error creating protocol: {str(e)}"}
    
    async def create_custom_ritual(self, ritual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom ritual."""
        try:
            ritual_id = ritual_data.get("id", f"custom_ritual_{len(self.ritual_library)}")
            
            # Ensure required fields
            required_fields = ["name", "description", "steps", "frequency", "punishment_for_skipping"]
            for field in required_fields:
                if field not in ritual_data:
                    return {"success": False, "message": f"Missing required field: {field}"}
            
            # Create ritual
            ritual = Ritual(
                id=ritual_id,
                name=ritual_data["name"],
                description=ritual_data["description"],
                steps=ritual_data["steps"],
                frequency=ritual_data["frequency"],
                punishment_for_skipping=ritual_data["punishment_for_skipping"],
                reward_for_completion=ritual_data.get("reward_for_completion"),
                difficulty=ritual_data.get("difficulty", 0.5)
            )
            
            # Add to library
            self.ritual_library[ritual_id] = ritual
            
            return {
                "success": True,
                "message": f"Custom ritual '{ritual.name}' created",
                "ritual": ritual.dict()
            }
            
        except Exception as e:
            logger.error(f"Error creating custom ritual: {e}")
            return {"success": False, "message": f"Error creating ritual: {str(e)}"}
    
    def get_available_protocols(self) -> List[Dict[str, Any]]:
        """Get all available protocols in the library."""
        return [protocol.dict() for protocol in self.protocol_library.values()]
    
    def get_available_rituals(self) -> List[Dict[str, Any]]:
        """Get all available rituals in the library."""
        return [ritual.dict() for ritual in self.ritual_library.values()]
        
class HonorificsEnforcement:
    """Ensures proper forms of address and language protocol."""
    
    def __init__(self, protocol_enforcement=None):
        self.protocol_enforcement = protocol_enforcement
        self.required_honorifics = {
            "default": ["Mistress", "Goddess", "Ma'am"],
            "custom": {}  # user_id → custom required honorifics
        }
        self.address_violations = {}  # tracking violations
        
    async def check_proper_address(self, user_id, message):
        """Checks if message contains required honorifics."""
        # Get user-specific honorifics or default
        user_honorifics = self.required_honorifics["custom"].get(user_id, self.required_honorifics["default"])
        
        # Check for presence of required terms
        has_honorific = any(honorific.lower() in message.lower() for honorific in user_honorifics)
        
        # Don't check very short messages or status updates
        if len(message.split()) < 4 or message.startswith("/"):
            return {"compliant": True, "violation": None}
            
        if not has_honorific:
            # Track violation
            if user_id not in self.address_violations:
                self.address_violations[user_id] = []
                
            violation = {
                "timestamp": datetime.datetime.now().isoformat(),
                "message": message,
                "missing_honorifics": user_honorifics
            }
            self.address_violations[user_id].append(violation)
            
            # Record protocol violation if protocol system exists
            if self.protocol_enforcement:
                await self.protocol_enforcement.record_protocol_violation(
                    user_id=user_id,
                    protocol_id="address_protocol",
                    description=f"Failed to use proper honorifics ({', '.join(user_honorifics)})",
                    severity=0.5
                )
                
            return {
                "compliant": False,
                "violation": "missing_honorific",
                "required": user_honorifics,
                "correction": f"Address me properly as {user_honorifics[0]} or another appropriate honorific."
            }
            
        return {"compliant": True, "violation": None}
        
    async def set_custom_honorifics(self, user_id, honorifics):
        """Sets custom honorifics for a specific user."""
        self.required_honorifics["custom"][user_id] = honorifics
        return {
            "success": True,
            "user_id": user_id,
            "honorifics": honorifics
        }
