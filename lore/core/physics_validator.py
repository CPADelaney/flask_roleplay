# lore/core/physics_validator.py
"""
Physics validation for actions and events.
Ensures all actions respect the established physics caps of the world.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PhysicsViolationType(Enum):
    JUMP_HEIGHT = "jump_height"
    THROW_SPEED = "throw_speed"
    FALL_DAMAGE = "fall_damage"
    CARRY_WEIGHT = "carry_weight"
    RUNNING_SPEED = "running_speed"
    TELEPORTATION = "teleportation"
    TIME_TRAVEL = "time_travel"
    EX_NIHILO = "ex_nihilo_creation"
    GRAVITY = "gravity_violation"

@dataclass
class PhysicsViolation:
    """Details of a physics violation"""
    violation_type: PhysicsViolationType
    attempted_value: Any
    maximum_allowed: Any
    description: str
    severity: str  # "minor", "major", "critical"
    consequence: Optional[str] = None

class PhysicsValidator:
    """
    Validates actions against the physics caps of the current world.
    Provides detailed feedback for narrative adjustments.
    """
    
    def __init__(self, physics_caps: Dict[str, Any]):
        self.caps = physics_caps
        self.profile = self._determine_profile(physics_caps)
        
    def _determine_profile(self, caps: Dict[str, Any]) -> str:
        """Determine physics profile from caps"""
        if caps.get('magic_system') == 'extensive':
            return 'hard_magic'
        elif caps.get('magic_system') == 'limited':
            return 'soft_magic'
        elif caps.get('tech_requirement') == 'advanced':
            return 'sci_fi'
        else:
            return 'realistic'
    
    def validate_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[PhysicsViolation]]:
        """
        Validate a specific action against physics caps.
        
        Args:
            action_type: Type of action (jump, throw, fall, etc.)
            parameters: Action parameters with numeric values
            context: Optional context (resources available, buffs, etc.)
            
        Returns:
            (is_valid, violations) tuple
        """
        violations = []
        
        # Route to specific validators
        if action_type == "jump":
            violations.extend(self._validate_jump(parameters, context))
        elif action_type == "throw":
            violations.extend(self._validate_throw(parameters, context))
        elif action_type == "fall":
            violations.extend(self._validate_fall(parameters, context))
        elif action_type == "carry":
            violations.extend(self._validate_carry(parameters, context))
        elif action_type == "run":
            violations.extend(self._validate_run(parameters, context))
        elif action_type == "teleport":
            violations.extend(self._validate_teleport(parameters, context))
        elif action_type == "time_travel":
            violations.extend(self._validate_time_travel(parameters, context))
        elif action_type == "create":
            violations.extend(self._validate_creation(parameters, context))
        elif action_type == "composite":
            # Validate multiple aspects
            for sub_action, sub_params in parameters.items():
                sub_valid, sub_violations = self.validate_action(
                    sub_action, sub_params, context
                )
                violations.extend(sub_violations)
        
        return (len(violations) == 0, violations)
    
    def _validate_jump(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate jump action"""
        violations = []
        height_m = parameters.get('height_m', 0)
        max_height = self.caps['max_jump_height_m']
        
        # Check for buffs/assists
        if context:
            if context.get('has_jump_boost'):
                max_height *= 1.5
            if context.get('low_gravity'):
                max_height *= 2.0
            if context.get('magic_enhancement'):
                if self.caps.get('magic_system'):
                    max_height *= 2.0
        
        if height_m > max_height:
            severity = self._calculate_severity(height_m, max_height)
            violations.append(PhysicsViolation(
                violation_type=PhysicsViolationType.JUMP_HEIGHT,
                attempted_value=height_m,
                maximum_allowed=max_height,
                description=f"Jump of {height_m}m exceeds maximum of {max_height}m",
                severity=severity,
                consequence=self._get_jump_consequence(height_m, max_height)
            ))
        
        return violations
    
    def _validate_throw(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate throw/projectile action"""
        violations = []
        speed_ms = parameters.get('speed_ms', 0)
        max_speed = self.caps['max_throw_speed_ms']
        
        # Check for tools/weapons
        if context:
            if context.get('using_sling'):
                max_speed *= 1.5
            if context.get('using_bow'):
                max_speed *= 2.0
            if context.get('using_gun'):
                # Only in appropriate settings
                if self.profile in ['sci_fi', 'modern']:
                    max_speed = 400  # Bullet speed
        
        if speed_ms > max_speed:
            severity = self._calculate_severity(speed_ms, max_speed)
            violations.append(PhysicsViolation(
                violation_type=PhysicsViolationType.THROW_SPEED,
                attempted_value=speed_ms,
                maximum_allowed=max_speed,
                description=f"Projectile speed of {speed_ms}m/s exceeds maximum of {max_speed}m/s",
                severity=severity,
                consequence=self._get_throw_consequence(speed_ms, max_speed)
            ))
        
        return violations
    
    def _validate_fall(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate fall survival"""
        violations = []
        fall_height = parameters.get('height_m', 0)
        max_safe = self.caps['max_safe_fall_m']
        
        # Check for mitigation
        if context:
            if context.get('has_cushioning'):
                max_safe *= 1.5
            if context.get('superhuman_durability'):
                max_safe *= 3.0
            if context.get('magic_protection'):
                if self.caps.get('magic_system'):
                    max_safe *= 5.0
            if context.get('low_gravity'):
                max_safe *= 2.0
        
        if fall_height > max_safe:
            severity = self._calculate_fall_severity(fall_height, max_safe)
            violations.append(PhysicsViolation(
                violation_type=PhysicsViolationType.FALL_DAMAGE,
                attempted_value=fall_height,
                maximum_allowed=max_safe,
                description=f"Fall from {fall_height}m exceeds safe height of {max_safe}m",
                severity=severity,
                consequence=self._get_fall_consequence(fall_height, max_safe)
            ))
        
        return violations
    
    def _validate_carry(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate carrying capacity"""
        violations = []
        weight_kg = parameters.get('weight_kg', 0)
        max_carry = self.caps['max_carry_weight_kg']
        
        # Check for assists
        if context:
            if context.get('has_exoskeleton'):
                max_carry *= 3.0
            if context.get('using_cart'):
                max_carry *= 5.0
            if context.get('superhuman_strength'):
                max_carry *= 2.0
            if context.get('magic_enhancement'):
                if self.caps.get('magic_system'):
                    max_carry *= 2.0
        
        if weight_kg > max_carry:
            severity = self._calculate_severity(weight_kg, max_carry)
            violations.append(PhysicsViolation(
                violation_type=PhysicsViolationType.CARRY_WEIGHT,
                attempted_value=weight_kg,
                maximum_allowed=max_carry,
                description=f"Carrying {weight_kg}kg exceeds maximum of {max_carry}kg",
                severity=severity,
                consequence="Cannot lift the weight"
            ))
        
        return violations
    
    def _validate_run(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate running speed"""
        violations = []
        speed_ms = parameters.get('speed_ms', 0)
        max_speed = self.caps['max_running_speed_ms']
        
        # Check for enhancements
        if context:
            if context.get('has_speed_boost'):
                max_speed *= 1.5
            if context.get('using_vehicle'):
                # Different cap for vehicles
                return []  # Vehicles have different rules
            if context.get('magic_haste'):
                if self.caps.get('magic_system'):
                    max_speed *= 2.0
        
        if speed_ms > max_speed:
            severity = self._calculate_severity(speed_ms, max_speed)
            violations.append(PhysicsViolation(
                violation_type=PhysicsViolationType.RUNNING_SPEED,
                attempted_value=speed_ms,
                maximum_allowed=max_speed,
                description=f"Running at {speed_ms}m/s exceeds maximum of {max_speed}m/s",
                severity=severity,
                consequence="Cannot achieve that speed"
            ))
        
        return violations
    
    def _validate_teleport(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate teleportation attempt"""
        violations = []
        
        if not self.caps.get('teleportation_allowed', False):
            # Check if there's a valid method
            valid_method = False
            if context:
                if context.get('using_portal') and self.profile in ['hard_magic', 'sci_fi']:
                    valid_method = True
                if context.get('using_teleporter') and self.profile == 'sci_fi':
                    valid_method = True
                if context.get('casting_spell') and self.profile == 'hard_magic':
                    if context.get('has_spell_resource'):
                        valid_method = True
            
            if not valid_method:
                violations.append(PhysicsViolation(
                    violation_type=PhysicsViolationType.TELEPORTATION,
                    attempted_value=True,
                    maximum_allowed=False,
                    description="Teleportation not possible in this reality",
                    severity="critical",
                    consequence="The space between refuses to bend"
                ))
        
        return violations
    
    def _validate_time_travel(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate time travel attempt"""
        violations = []
        
        if not self.caps.get('time_travel_allowed', False):
            violations.append(PhysicsViolation(
                violation_type=PhysicsViolationType.TIME_TRAVEL,
                attempted_value=True,
                maximum_allowed=False,
                description="Time travel not possible in this reality",
                severity="critical",
                consequence="Time flows in one direction only"
            ))
        
        return violations
    
    def _validate_creation(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[PhysicsViolation]:
        """Validate ex nihilo creation"""
        violations = []
        
        creating = parameters.get('creating')
        if not creating:
            return violations
        
        if not self.caps.get('ex_nihilo_creation', False):
            # Check if there's a valid resource
            valid_method = False
            if context:
                if context.get('has_materials'):
                    valid_method = True
                if context.get('using_fabricator') and self.profile == 'sci_fi':
                    valid_method = True
                if context.get('spending_mana') and self.profile in ['soft_magic', 'hard_magic']:
                    if context.get('mana_available', 0) >= parameters.get('mana_cost', 100):
                        valid_method = True
            
            if not valid_method:
                violations.append(PhysicsViolation(
                    violation_type=PhysicsViolationType.EX_NIHILO,
                    attempted_value=creating,
                    maximum_allowed="with_resources_only",
                    description=f"Cannot create {creating} from nothing",
                    severity="major",
                    consequence="Need materials or energy source"
                ))
        
        return violations
    
    def _calculate_severity(self, attempted: float, maximum: float) -> str:
        """Calculate violation severity"""
        ratio = attempted / maximum if maximum > 0 else float('inf')
        
        if ratio <= 1.2:
            return "minor"
        elif ratio <= 2.0:
            return "major"
        else:
            return "critical"
    
    def _calculate_fall_severity(self, fall_height: float, max_safe: float) -> str:
        """Calculate fall damage severity"""
        ratio = fall_height / max_safe if max_safe > 0 else float('inf')
        
        if ratio <= 1.5:
            return "minor"  # Injuries but survivable
        elif ratio <= 3.0:
            return "major"  # Severe injuries
        else:
            return "critical"  # Lethal
    
    def _get_jump_consequence(self, attempted: float, maximum: float) -> str:
        """Get narrative consequence for jump violation"""
        ratio = attempted / maximum
        
        if ratio <= 1.5:
            return "Legs strain but manage partial height"
        elif ratio <= 2.0:
            return "Gravity pulls you back down immediately"
        else:
            return "The attempt doesn't even begin - too impossible"
    
    def _get_throw_consequence(self, attempted: float, maximum: float) -> str:
        """Get narrative consequence for throw violation"""
        ratio = attempted / maximum
        
        if ratio <= 1.5:
            return "The projectile falls short of intended velocity"
        elif ratio <= 2.0:
            return "Your arm cannot generate that force"
        else:
            return "Physics itself refuses the attempt"
    
    def _get_fall_consequence(self, fall_height: float, max_safe: float) -> str:
        """Get narrative consequence for fall"""
        ratio = fall_height / max_safe
        
        if ratio <= 1.5:
            return "Impact causes injuries - broken bones likely"
        elif ratio <= 3.0:
            return "Severe trauma on impact - critical injuries"
        else:
            return "Fatal impact - survival impossible"
    
    def suggest_alternative(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        violations: List[PhysicsViolation]
    ) -> Dict[str, Any]:
        """
        Suggest alternative action that would be valid.
        
        Returns:
            Alternative action parameters that respect physics
        """
        if not violations:
            return parameters
        
        alternative = parameters.copy()
        
        for violation in violations:
            if violation.violation_type == PhysicsViolationType.JUMP_HEIGHT:
                alternative['height_m'] = violation.maximum_allowed * 0.9
                alternative['description'] = f"Jump {alternative['height_m']}m high"
                
            elif violation.violation_type == PhysicsViolationType.THROW_SPEED:
                alternative['speed_ms'] = violation.maximum_allowed * 0.9
                alternative['description'] = f"Throw at {alternative['speed_ms']}m/s"
                
            elif violation.violation_type == PhysicsViolationType.FALL_DAMAGE:
                alternative['height_m'] = violation.maximum_allowed
                alternative['needs_mitigation'] = True
                alternative['description'] = f"Find a way to reduce fall damage"
                
            elif violation.violation_type == PhysicsViolationType.CARRY_WEIGHT:
                alternative['weight_kg'] = violation.maximum_allowed * 0.9
                alternative['description'] = f"Carry {alternative['weight_kg']}kg only"
                
            elif violation.violation_type == PhysicsViolationType.TELEPORTATION:
                alternative['action'] = 'travel_normally'
                alternative['description'] = "Travel by conventional means"
                
            elif violation.violation_type == PhysicsViolationType.EX_NIHILO:
                alternative['requires'] = 'materials'
                alternative['description'] = "Gather materials first"
        
        return alternative
    
    def check_resource_requirement(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check if action requires resources to be valid.
        
        Returns:
            Required resources or None if action is free
        """
        # Check if this violates caps without resources
        valid, violations = self.validate_action(action_type, parameters)
        
        if valid:
            return None  # No resources needed
        
        # Determine what resources would make it valid
        resources = {}
        
        for violation in violations:
            if violation.violation_type == PhysicsViolationType.TELEPORTATION:
                if self.profile == 'hard_magic':
                    resources['mana'] = 50
                    resources['spell'] = 'teleport'
                elif self.profile == 'sci_fi':
                    resources['power'] = 100
                    resources['device'] = 'teleporter'
                    
            elif violation.violation_type == PhysicsViolationType.EX_NIHILO:
                if self.profile in ['soft_magic', 'hard_magic']:
                    resources['mana'] = 30
                    resources['spell'] = 'conjure'
                else:
                    resources['materials'] = self._determine_materials(
                        parameters.get('creating')
                    )
                    
            elif violation.violation_type == PhysicsViolationType.JUMP_HEIGHT:
                if self.profile == 'hard_magic':
                    resources['mana'] = 10
                    resources['spell'] = 'jump'
                elif self.profile == 'sci_fi':
                    resources['device'] = 'jump_pack'
                    resources['power'] = 5
        
        return resources if resources else {'impossible': True}
    
    def _determine_materials(self, creating: str) -> List[str]:
        """Determine required materials for creation"""
        material_map = {
            'weapon': ['metal', 'wood'],
            'armor': ['metal', 'leather'],
            'tool': ['metal', 'wood'],
            'potion': ['herbs', 'water'],
            'food': ['ingredients'],
            'shelter': ['wood', 'stone']
        }
        
        for category, materials in material_map.items():
            if category in creating.lower():
                return materials
        
        return ['raw_materials']


# Helper function for quick validation
def quick_physics_check(
    action: str,
    world_profile: str = 'realistic'
) -> Tuple[bool, str]:
    """
    Quick physics check for common actions.
    
    Args:
        action: Natural language action description
        world_profile: Physics profile of the world
        
    Returns:
        (is_valid, reason) tuple
    """
    # Default caps for profiles
    profile_caps = {
        'realistic': {
            'max_jump_height_m': 1.5,
            'max_throw_speed_ms': 45,
            'max_safe_fall_m': 3,
            'teleportation_allowed': False,
            'ex_nihilo_creation': False
        },
        'soft_magic': {
            'max_jump_height_m': 5,
            'max_throw_speed_ms': 80,
            'max_safe_fall_m': 10,
            'teleportation_allowed': False,
            'ex_nihilo_creation': False
        },
        'hard_magic': {
            'max_jump_height_m': 50,
            'max_throw_speed_ms': 200,
            'max_safe_fall_m': 100,
            'teleportation_allowed': True,
            'ex_nihilo_creation': True
        },
        'sci_fi': {
            'max_jump_height_m': 3,
            'max_throw_speed_ms': 60,
            'max_safe_fall_m': 5,
            'teleportation_allowed': True,
            'ex_nihilo_creation': False
        }
    }
    
    caps = profile_caps.get(world_profile, profile_caps['realistic'])
    validator = PhysicsValidator(caps)
    
    # Parse action for key terms
    action_lower = action.lower()
    
    # Detect action type and parameters
    if 'jump' in action_lower:
        # Try to extract height
        import re
        height_match = re.search(r'(\d+)\s*(?:meter|metre|m)', action_lower)
        if height_match:
            height = float(height_match.group(1))
            valid, violations = validator.validate_action(
                'jump',
                {'height_m': height}
            )
            if not valid:
                return (False, violations[0].description)
    
    elif 'teleport' in action_lower:
        valid, violations = validator.validate_action('teleport', {})
        if not valid:
            return (False, violations[0].description)
    
    elif 'create' in action_lower or 'conjure' in action_lower or 'summon' in action_lower:
        valid, violations = validator.validate_action(
            'create',
            {'creating': action}
        )
        if not valid:
            return (False, violations[0].description)
    
    elif 'fall' in action_lower:
        # Try to extract height
        import re
        height_match = re.search(r'(\d+)\s*(?:meter|metre|m)', action_lower)
        if height_match:
            height = float(height_match.group(1))
            valid, violations = validator.validate_action(
                'fall',
                {'height_m': height}
            )
            if not valid:
                return (False, violations[0].consequence)
    
    return (True, "Action appears valid")
