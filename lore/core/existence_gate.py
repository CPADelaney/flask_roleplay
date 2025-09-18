# lore/core/existence_gate.py
"""
Existence Gate - Reality enforcement for all canonical entity creation.
Ensures physics caps, technology constraints, and setting consistency.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class GateDecision(Enum):
    ALLOW = "allow"
    ANALOG = "analog"  # Substitute with setting-appropriate version
    DENY = "deny"
    DEFER = "defer"    # Return as lead/rumor, not canon

class ExistenceGate:
    """
    Central gatekeeper for all entity creation in the world.
    Enforces hard physics caps, technology constraints, and setting rules.
    """
    
    def __init__(self, ctx):
        self.ctx = ctx
        self.user_id = ctx.user_id
        self.conversation_id = ctx.conversation_id
        self._caps_cache = None
        self._infra_cache = None
    
    async def assess_entity(
        self,
        entity_type: str,
        entity_data: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """
        Main assessment function - determines if an entity can exist.
        
        Returns:
            (decision, details) where details contains reasoning and alternatives
        """
        # Load physics caps and infrastructure
        caps = await self._load_physics_caps()
        infra = await self._load_infrastructure()
        
        # Route to appropriate checker
        if entity_type == "location":
            return await self._assess_location_existence(entity_data, caps, infra, scene_context)
        elif entity_type == "npc":
            return await self._assess_npc_existence(entity_data, caps, infra, scene_context)
        elif entity_type == "item":
            return await self._assess_item_existence(entity_data, caps, infra, scene_context)
        elif entity_type == "event":
            return await self._assess_event_existence(entity_data, caps, infra, scene_context)
        elif entity_type == "magic":
            return await self._assess_magic_existence(entity_data, caps, infra, scene_context)
        elif entity_type == "creature":
            return await self._assess_creature_existence(entity_data, caps, infra, scene_context)
        else:
            # Default permissive for unknown types
            return (GateDecision.ALLOW, {"reason": "Unknown entity type, defaulting to allow"})
    
    async def _load_physics_caps(self) -> Dict[str, Any]:
        """Load physics caps from world configuration"""
        if self._caps_cache:
            return self._caps_cache
        
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Get physics profile
            profile_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='PhysicsProfile'
            """, self.user_id, self.conversation_id)
            
            profile = profile_row['value'] if profile_row else 'realistic'
            
            # Define caps based on profile
            caps = self._get_caps_for_profile(profile)
            
            # Check for custom caps
            custom_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='PhysicsCaps'
            """, self.user_id, self.conversation_id)
            
            if custom_row:
                try:
                    custom = json.loads(custom_row['value'])
                    caps.update(custom)
                except:
                    pass
            
            self._caps_cache = caps
            return caps
    
    def _get_caps_for_profile(self, profile: str) -> Dict[str, Any]:
        """Get default caps for a physics profile"""
        profiles = {
            'realistic': {
                'max_jump_height_m': 1.5,
                'max_throw_speed_ms': 45,
                'max_safe_fall_m': 3,
                'max_carry_weight_kg': 100,
                'max_running_speed_ms': 12,
                'teleportation_allowed': False,
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,
                'gravity_constant': 9.8
            },
            'soft_magic': {
                'max_jump_height_m': 5,
                'max_throw_speed_ms': 80,
                'max_safe_fall_m': 10,
                'max_carry_weight_kg': 200,
                'max_running_speed_ms': 20,
                'teleportation_allowed': False,
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,  # Must use resources
                'gravity_constant': 9.8,
                'magic_system': 'limited'
            },
            'hard_magic': {
                'max_jump_height_m': 50,
                'max_throw_speed_ms': 200,
                'max_safe_fall_m': 100,
                'max_carry_weight_kg': 1000,
                'max_running_speed_ms': 50,
                'teleportation_allowed': True,
                'time_travel_allowed': False,
                'ex_nihilo_creation': True,  # With spell cost
                'gravity_constant': 9.8,
                'magic_system': 'extensive',
                'spell_resource': 'mana'
            },
            'sci_fi': {
                'max_jump_height_m': 3,  # With assist
                'max_throw_speed_ms': 60,
                'max_safe_fall_m': 5,
                'max_carry_weight_kg': 300,  # With exoskeleton
                'max_running_speed_ms': 15,
                'teleportation_allowed': True,  # Technology-based
                'time_travel_allowed': False,
                'ex_nihilo_creation': False,  # Must use fabricators
                'gravity_constant': 'variable',
                'tech_requirement': 'advanced'
            }
        }
        
        return profiles.get(profile, profiles['realistic'])
    
    async def _load_infrastructure(self) -> Dict[str, Any]:
        """Load infrastructure flags from world configuration"""
        if self._infra_cache:
            return self._infra_cache
        
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Get technology level
            tech_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='TechnologyLevel'
            """, self.user_id, self.conversation_id)
            
            tech_level = tech_row['value'] if tech_row else 'modern'
            
            # Get era
            era_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay  
                WHERE user_id=$1 AND conversation_id=$2 AND key='SettingEra'
            """, self.user_id, self.conversation_id)
            
            era = era_row['value'] if era_row else 'contemporary'
            
            infra = self._get_infra_for_tech(tech_level, era)
            
            # Check for custom infrastructure
            custom_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='InfrastructureFlags'
            """, self.user_id, self.conversation_id)
            
            if custom_row:
                try:
                    custom = json.loads(custom_row['value'])
                    infra.update(custom)
                except:
                    pass
            
            self._infra_cache = infra
            return infra
    
    def _get_infra_for_tech(self, tech_level: str, era: str) -> Dict[str, Any]:
        """Get infrastructure flags for technology level"""
        tech_infra = {
            'primitive': {
                'electricity': False,
                'refrigeration': False,
                'mass_packaging': False,
                'night_economy': False,
                'plumbing': False,
                'printing': False,
                'global_trade': False,
                'instant_communication': False,
                'metalworking': True  # Bronze/iron age
            },
            'medieval': {
                'electricity': False,
                'refrigeration': False,
                'mass_packaging': False,
                'night_economy': False,
                'plumbing': False,  # Basic only
                'printing': False,
                'global_trade': False,
                'instant_communication': False,
                'metalworking': True
            },
            'industrial': {
                'electricity': True,
                'refrigeration': False,  # Ice houses only
                'mass_packaging': False,
                'night_economy': False,  # Limited
                'plumbing': True,
                'printing': True,
                'global_trade': True,  # Ships
                'instant_communication': False,  # Telegraph only
                'telegraph': True,
                'steam_power': True,
                'industrial_machinery': True,
                'metalworking': True
            },
            'modern': {
                'electricity': True,
                'refrigeration': True,
                'mass_packaging': True,
                'night_economy': True,
                'plumbing': True,
                'printing': True,
                'global_trade': True,
                'instant_communication': True,
                'aviation': True,
                'microprocessors': True,
                'transistors': True,
                'metalworking': True
            },
            'advanced': {
                'electricity': True,
                'refrigeration': True,
                'mass_packaging': True,
                'night_economy': True,
                'plumbing': True,
                'printing': True,  # 3D printing
                'global_trade': True,
                'instant_communication': True,
                'quantum_computing': True,
                'ai_systems': True,
                'nano_fabrication': False,
                'advanced_optics': True,
                'augmented_reality': True,
                'metalworking': True
            },
            'futuristic': {
                'electricity': True,
                'refrigeration': True,
                'mass_packaging': True,
                'night_economy': True,  # 24/7 economy
                'plumbing': True,
                'printing': True,  # Molecular assembly
                'global_trade': True,  # Interplanetary
                'instant_communication': True,  # FTL comms
                'quantum_computing': True,
                'ai_systems': True,
                'nano_fabrication': True,
                'matter_replication': True,
                'interstellar_travel': True,
                'holographic_projection': True,
                'power_cells': True,
                'metalworking': True
            }
        }
        
        return tech_infra.get(tech_level, tech_infra['modern'])
    
    def _has_infra(self, infra: Dict[str, Any], req: str) -> bool:
        """Check infrastructure with synonym mapping."""
        synonyms = {
            'modern_aviation': 'aviation',
            'industrial_communication': 'telegraph',
            'futuristic_tech': 'interstellar_travel',
            'space_travel': 'interstellar_travel',
            'holographic_projection': 'holographic_projection',
            'industrial_refrigeration': 'refrigeration',
            'advanced_optics': 'advanced_optics',
            'computers': 'microprocessors',
            'microprocessors': 'microprocessors',
            'transistors': 'transistors',
            'social_media': 'instant_communication',
            'metalworking': 'metalworking',
            'writing': 'printing',
            'power_cells': 'power_cells',
        }
        key = synonyms.get(req, req)
        return bool(infra.get(key, False))
    
    def _is_brandlike(self, name: str) -> bool:
        """Detect brandlike names that should be analogized."""
        import re
        BRANDLIKE = re.compile(
            r"(?:[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:®|™|Inc\.|LLC))|"
            r"(?:Starbucks|McDonald'?s|Coca[- ]Cola|Walmart|Amazon|Google|Microsoft|Apple|Nike|Adidas)",
            re.I
        )
        return bool(BRANDLIKE.search(name))
    
    async def _assess_location_existence(
        self,
        location_data: Dict[str, Any],
        caps: Dict[str, Any],
        infra: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]]
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Assess if a location can exist"""
        location_name = location_data.get('location_name', '')
        location_type = location_data.get('location_type', 'generic')
        district_type = location_data.get('district_type')
        
        # Check for brandlike names first
        if self._is_brandlike(location_name):
            analog = self._suggest_location_analog(location_type, infra) or 'local_shop'
            return (GateDecision.ANALOG, {
                "reason": "Brandlike name detected",
                "analog": analog,
                "analog_data": self._transform_to_analog(location_data, analog)
            })
        
        # Check against infrastructure requirements
        infra_violations = []
        
        # Map location types to required infrastructure
        requirements = {
            '24h_convenience_store': ['electricity', 'refrigeration', 'night_economy', 'mass_packaging'],
            'internet_cafe': ['electricity', 'instant_communication'],
            'airport': ['electricity', 'aviation'],
            'space_port': ['space_travel'],
            'quantum_lab': ['quantum_computing'],
            'telegraph_office': ['telegraph'],
            'ice_house': ['industrial_refrigeration']
        }
        
        required_infra = requirements.get(location_type, [])
        for req in required_infra:
            if not self._has_infra(infra, req):
                infra_violations.append(req)
        
        # Check district appropriateness
        if district_type:
            district_violations = self._check_district_spawn(location_type, district_type)
            if district_violations:
                # Suggest analog
                analog = self._suggest_location_analog(location_type, infra)
                return (GateDecision.ANALOG, {
                    "reason": f"Wrong district for {location_type}",
                    "violations": district_violations,
                    "analog": analog,
                    "lead": f"You might find something like that in the {self._suggest_district(location_type)} district"
                })
        
        if infra_violations:
            # Try to find an era-appropriate analog
            analog = self._suggest_location_analog(location_type, infra)
            if analog:
                return (GateDecision.ANALOG, {
                    "reason": f"Infrastructure missing: {', '.join(infra_violations)}",
                    "analog": analog,
                    "analog_data": self._transform_to_analog(location_data, analog)
                })
            else:
                return (GateDecision.DEFER, {
                    "reason": f"Cannot exist here - missing: {', '.join(infra_violations)}",
                    "lead": f"Perhaps in a more developed area..."
                })
        
        # Check topology (rural vs urban)
        if scene_context:
            density = scene_context.get('population_density', 'medium')
            if self._violates_topology(location_type, density):
                return (GateDecision.DEFER, {
                    "reason": f"{location_type} cannot exist in {density} density area",
                    "lead": f"Try the nearest city for that"
                })
        
        return (GateDecision.ALLOW, {"reason": "Location meets all requirements"})
    
    def _suggest_location_analog(self, modern_type: str, infra: Dict[str, Any]) -> Optional[str]:
        """Suggest era-appropriate analog for modern location"""
        analogs = {
            '24h_convenience_store': {
                'primitive': 'village_market',
                'medieval': 'market_square',
                'industrial': 'general_store',
                'modern': 'corner_shop'
            },
            'internet_cafe': {
                'primitive': 'storyteller_circle',
                'medieval': 'scribe_house',
                'industrial': 'telegraph_office',
                'modern': 'library_terminals'
            },
            'gym': {
                'primitive': 'training_grounds',
                'medieval': 'practice_yard',
                'industrial': 'boxing_club',
                'modern': 'fitness_center'
            },
            'hospital': {
                'primitive': 'healer_hut',
                'medieval': 'monastery_infirmary',
                'industrial': 'medical_college',
                'modern': 'medical_center'
            }
        }
        
        if modern_type not in analogs:
            return None
        
        # Determine era from infrastructure
        era = self._determine_era_from_infra(infra)
        return analogs[modern_type].get(era)
    
    def _transform_to_analog(self, original_data: Dict[str, Any], analog_type: str) -> Dict[str, Any]:
        """Transform modern location data to era-appropriate analog"""
        analog_data = original_data.copy()
        analog_data['location_type'] = analog_type
        
        # Transform features
        feature_transforms = {
            'neon_signs': 'painted_signs',
            'electric_lights': 'oil_lamps',
            'air_conditioning': 'fans',
            'elevators': 'stairs',
            'automatic_doors': 'regular_doors',
            'digital_displays': 'bulletin_boards',
            'security_cameras': 'guards'
        }
        
        if 'features' in analog_data:
            new_features = []
            for feature in analog_data['features']:
                new_features.append(feature_transforms.get(feature, feature))
            analog_data['features'] = new_features
        
        return analog_data
    
    def _check_district_spawn(self, location_type: str, district_type: str) -> List[str]:
        """Check if location type is appropriate for district"""
        allowed_categories = {
            'residential': ['residential', 'park', 'civic-small', 'shop'],
            'commercial': ['shop', 'restaurant', 'office', 'bank', 'entertainment'],
            'industrial': ['industrial', 'warehouse', 'power', 'factory'],
            'entertainment': ['entertainment', 'restaurant'],
            'civic': ['civic', 'bank', 'office']
        }
        
        category = self._get_location_category(location_type)
        allowed = allowed_categories.get(district_type, [])
        
        if category not in allowed:
            return [f"{location_type} (category: {category}) not appropriate for {district_type} district"]
        
        return []
    
    def _get_location_category(self, location_type: str) -> str:
        """Get general category for a location type"""
        categories = {
            'shop': ['store', 'market', 'boutique', 'shop', 'corner_shop', 'general_store', 
                    'mall', '24h_convenience_store'],
            'restaurant': ['restaurant', 'cafe', 'diner', 'eatery', 'tavern', 'inn', 'bar'],
            'entertainment': ['club', 'theater', 'casino', 'arcade', 'arena', 'stadium'],
            'residential': ['house', 'apartment', 'condo', 'dwelling', 'home'],
            'industrial': ['factory', 'warehouse', 'workshop', 'power_plant', 'mill', 
                          'refinery', 'plant'],
            'civic': ['courthouse', 'station', 'hall', 'library', 'school', 
                     'university', 'hospital', 'city_hall', 'police_station'],
            'bank': ['bank'],  # Banks can be in commercial or civic districts
            'office': ['office', 'headquarters'],
            'park': ['park', 'garden', 'square']
        }
        
        location_lower = location_type.lower()
        for category, keywords in categories.items():
            if any(keyword in location_lower for keyword in keywords):
                return category
        
        return 'generic'
    
    def _suggest_district(self, location_type: str) -> str:
        """Suggest appropriate district for location type"""
        category = self._get_location_category(location_type)
        district_map = {
            'shop': 'commercial',
            'restaurant': 'commercial',
            'entertainment': 'entertainment',
            'residential': 'residential',
            'civic': 'civic',
            'bank': 'commercial',  # Added bank mapping
            'generic': 'mixed-use'
        }
        return district_map.get(category, 'commercial')
    
    def _violates_topology(self, location_type: str, density: str) -> bool:
        """Check if location violates population density constraints"""
        # Locations that require urban density
        urban_only = [
            'subway_station', 'skyscraper', 'mall', 'arena',
            'airport', 'university', 'hospital', 'theater_district'
        ]
        
        # Locations that require rural setting
        rural_only = [
            'farm', 'ranch', 'wilderness_camp', 'hunting_lodge',
            'mine', 'quarry', 'lumber_mill'
        ]
        
        if location_type in urban_only and density in ['rural', 'wilderness']:
            return True
        
        if location_type in rural_only and density in ['urban', 'metropolitan']:
            return True
        
        return False
    
    async def _assess_npc_existence(
        self,
        npc_data: Dict[str, Any],
        caps: Dict[str, Any],
        infra: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]]
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Assess if an NPC role can exist"""
        role = npc_data.get('role', 'citizen')
        
        # Check role feasibility
        role_requirements = {
            'hacker': ['instant_communication'],
            'pilot': ['aviation'],
            'astronaut': ['space_travel'],
            'internet_influencer': ['instant_communication', 'social_media'],
            'influencer': ['instant_communication'],  # Standardized key
            'AI_researcher': ['ai_systems'],
            'quantum_physicist': ['quantum_computing'],
            'telegraph_operator': ['telegraph'],
            'blacksmith': ['metalworking'],
            'scribe': ['writing', '!printing'],  # ! means "not required"
            'programmer': ['computers']
        }
        
        required = role_requirements.get(role, [])
        violations = []
        
        for req in required:
            if req.startswith('!'):
                # Negative requirement - must NOT have this
                if self._has_infra(infra, req[1:]):
                    violations.append(f"Role obsolete due to {req[1:]}")
            else:
                if not self._has_infra(infra, req):
                    violations.append(f"Missing requirement: {req}")
        
        if violations:
            # Suggest era-appropriate analog role
            analog = self._suggest_role_analog(role, infra)
            if analog:
                return (GateDecision.ANALOG, {
                    "reason": f"Role '{role}' cannot exist: {', '.join(violations)}",
                    "analog": analog,
                    "analog_data": {**npc_data, 'role': analog}
                })
            else:
                return (GateDecision.DENY, {
                    "reason": f"Role '{role}' impossible in this setting",
                    "violations": violations
                })
        
        # Check if institution exists for role
        institution = self._get_required_institution(role)
        if institution and scene_context:
            if not scene_context.get(f'has_{institution}', True):
                return (GateDecision.DEFER, {
                    "reason": f"No {institution} exists here to support {role}",
                    "lead": f"You might find a {role} where there's a {institution}"
                })
        
        return (GateDecision.ALLOW, {"reason": "NPC role is feasible"})
    
    def _suggest_role_analog(self, modern_role: str, infra: Dict[str, Any]) -> Optional[str]:
        """Suggest era-appropriate analog for modern role"""
        role_analogs = {
            'programmer': {
                'primitive': 'pattern_weaver',
                'medieval': 'scribe',
                'industrial': 'calculator',
                'modern': 'computer_operator'
            },
            'hacker': {
                'primitive': 'trickster',
                'medieval': 'spy',
                'industrial': 'telegraph_interceptor',
                'modern': 'phone_phreaker'
            },
            'influencer': {
                'primitive': 'storyteller',
                'medieval': 'herald',
                'industrial': 'newspaper_columnist',
                'modern': 'radio_personality'
            },
            'scientist': {
                'primitive': 'shaman',
                'medieval': 'alchemist',
                'industrial': 'natural_philosopher',
                'modern': 'researcher'
            }
        }
        
        if modern_role not in role_analogs:
            return None
        
        era = self._determine_era_from_infra(infra)
        return role_analogs[modern_role].get(era)
    
    def _get_required_institution(self, role: str) -> Optional[str]:
        """Get institution required for a role to exist"""
        institution_map = {
            'consul': 'republic',
            'senator': 'senate',
            'customs_officer': 'border_control',
            'professor': 'university',
            'priest': 'temple',
            'judge': 'courthouse',
            'banker': 'bank'
        }
        return institution_map.get(role)
    
    async def _assess_item_existence(
        self,
        item_data: Dict[str, Any],
        caps: Dict[str, Any],
        infra: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]]
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Assess if an item can exist"""
        item_name = item_data.get('item_name', '')
        item_type = item_data.get('item_type', 'generic')
        tech_band = item_data.get('tech_band')
        
        # Check for brandlike names
        if self._is_brandlike(item_name):
            analog = self._suggest_item_analog(item_type, infra) or 'generic_item'
            return (GateDecision.ANALOG, {
                "reason": "Brandlike item name",
                "analog": analog,
                "analog_data": {**item_data, 'item_name': analog}
            })
        
        # Check tech band before specific requirements
        if tech_band:
            world_band = infra.get('tech_requirement', 'modern')
            allowed = ['primitive', 'medieval', 'industrial', 'modern', 'advanced', 'futuristic']
            if allowed.index(tech_band) > allowed.index(world_band):
                analog = self._suggest_item_analog(item_type, infra) or 'tool'
                return (GateDecision.ANALOG, {
                    "reason": f"Tech band '{tech_band}' exceeds world '{world_band}'",
                    "analog": analog,
                    "analog_data": {**item_data, 'item_type': analog}
                })
        
        # Check technology requirements
        tech_requirements = {
            'smartphone': ['electricity', 'instant_communication', 'microprocessors'],
            'laser_gun': ['advanced_optics', 'power_cells'],
            'steam_engine': ['industrial_machinery'],
            'printing_press': ['printing'],
            'computer': ['electricity', 'transistors'],
            'hologram': ['holographic_projection']
        }
        
        required = tech_requirements.get(item_type, [])
        violations = []
        
        for req in required:
            if not self._has_infra(infra, req):
                violations.append(f"Missing tech: {req}")
        
        # Check materials/supply chain
        materials = item_data.get('required_materials', [])
        if materials:
            supply_chain_check = await self._check_supply_chain(materials, scene_context)
            if not supply_chain_check['available']:
                return (GateDecision.DEFER, {
                    "reason": "Required materials not available",
                    "missing": supply_chain_check['missing'],
                    "lead": f"Need access to: {', '.join(supply_chain_check['missing'])}"
                })
        
        if violations:
            # Try analog
            analog = self._suggest_item_analog(item_type, infra)
            if analog:
                return (GateDecision.ANALOG, {
                    "reason": f"Item requires: {', '.join(violations)}",
                    "analog": analog,
                    "analog_data": {**item_data, 'item_name': analog, 'item_type': analog}
                })
            else:
                return (GateDecision.DENY, {
                    "reason": f"Item impossible in this setting",
                    "violations": violations
                })
        
        # Check resource cost if this is creation/crafting
        if item_data.get('is_crafting'):
            cost_check = await self._check_resource_cost(item_data)
            if not cost_check['affordable']:
                return (GateDecision.DENY, {
                    "reason": "Insufficient resources",
                    "required": cost_check['required'],
                    "available": cost_check['available']
                })
        
        return (GateDecision.ALLOW, {"reason": "Item can exist"})
    
    async def _check_supply_chain(
        self,
        materials: List[str],
        scene_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check if required materials are available"""
        if not scene_context:
            return {"available": True, "missing": []}
        
        available_materials = scene_context.get('available_materials', [])
        missing = [m for m in materials if m not in available_materials]
        
        return {
            "available": len(missing) == 0,
            "missing": missing
        }
    
    async def _check_resource_cost(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if player has resources to create item"""
        from db.connection import get_db_connection_context
        
        player = item_data.get('player_name', 'Player')
        cost = item_data.get('resource_cost', {'money': 0, 'supplies': 0})
        
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT money, supplies FROM PlayerResources
                WHERE user_id=$1 AND conversation_id=$2 AND player_name=$3
            """, self.user_id, self.conversation_id, player)
        
        available = dict(row or {'money': 0, 'supplies': 0})
        ok = (available.get('money', 0) >= cost.get('money', 0) and 
              available.get('supplies', 0) >= cost.get('supplies', 0))
        
        return {
            'affordable': ok,
            'required': cost,
            'available': available
        }
    
    def _suggest_item_analog(self, modern_item: str, infra: Dict[str, Any]) -> Optional[str]:
        """Suggest era-appropriate analog for modern item"""
        item_analogs = {
            'flashlight': {
                'primitive': 'torch',
                'medieval': 'lantern',
                'industrial': 'oil_lamp',
                'modern': 'electric_torch'
            },
            'gun': {
                'primitive': 'bow',
                'medieval': 'crossbow',
                'industrial': 'musket',
                'modern': 'pistol'
            },
            'phone': {
                'primitive': 'messenger',
                'medieval': 'letter',
                'industrial': 'telegraph',
                'modern': 'telephone'
            },
            'car': {
                'primitive': 'legs',
                'medieval': 'horse',
                'industrial': 'carriage',
                'modern': 'automobile'
            }
        }
        
        if modern_item not in item_analogs:
            return None
        
        era = self._determine_era_from_infra(infra)
        return item_analogs[modern_item].get(era)
    
    async def _assess_event_existence(
        self,
        event_data: Dict[str, Any],
        caps: Dict[str, Any],
        infra: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]]
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Assess if an event can occur"""
        event_type = event_data.get('event_type')
        
        # Check timeline consistency
        if 'timestamp' in event_data:
            timeline_check = await self._check_timeline_consistency(event_data)
            if not timeline_check['consistent']:
                return (GateDecision.DENY, {
                    "reason": "Violates timeline",
                    "violation": timeline_check['violation'],
                    "conflict": timeline_check['conflict']
                })
        
        # Check causal requirements
        if 'requires' in event_data:
            for req in event_data['requires']:
                if not await self._check_event_requirement(req):
                    return (GateDecision.DEFER, {
                        "reason": f"Prerequisite not met: {req}",
                        "lead": f"First need: {req}"
                    })
        
        # Check against physics caps for action events
        if event_type == 'action':
            physics_check = self._check_physics_violation(event_data, caps)
            if physics_check['violates']:
                return (GateDecision.DENY, {
                    "reason": "Violates physics",
                    "violations": physics_check['violations'],
                    "caps": physics_check['exceeded_caps']
                })
        
        return (GateDecision.ALLOW, {"reason": "Event can occur"})
    
    async def _assess_magic_existence(
        self,
        entity_data: Dict[str, Any],
        caps: Dict[str, Any],
        infra: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]]
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Assess if magic can exist/occur"""
        if not caps.get('magic_system'):
            return (GateDecision.DENY, {"reason": "Magic not part of this reality"})
        
        # Check spell level vs magic system
        spell_level = entity_data.get('spell_level', 'cantrip')
        magic_level = caps.get('magic_system')
        
        if magic_level == 'limited' and spell_level in ['legendary', 'epic']:
            return (GateDecision.DENY, {
                "reason": f"Spell level '{spell_level}' exceeds limited magic system"
            })
        
        # Check mana/resource cost
        if 'resource_cost' in entity_data:
            cost_check = await self._check_resource_cost(entity_data)
            if not cost_check['affordable']:
                return (GateDecision.DENY, {
                    "reason": "Insufficient magical resources",
                    "required": cost_check['required'],
                    "available": cost_check['available']
                })
        
        return (GateDecision.ALLOW, {"reason": "Magic permitted by profile"})
    
    async def _assess_creature_existence(
        self,
        entity_data: Dict[str, Any],
        caps: Dict[str, Any],
        infra: Dict[str, Any],
        scene_context: Optional[Dict[str, Any]]
    ) -> Tuple[GateDecision, Dict[str, Any]]:
        """Assess if creature can exist in setting"""
        creature_type = entity_data.get('creature_type', 'animal')
        habitat = entity_data.get('habitat', 'any')
        
        # Check if magical creature in non-magical world
        if creature_type in ['dragon', 'unicorn', 'phoenix', 'griffin']:
            if not caps.get('magic_system'):
                return (GateDecision.DENY, {
                    "reason": f"Magical creature '{creature_type}' cannot exist without magic"
                })
        
        # Check habitat compatibility
        if scene_context and habitat != 'any':
            current_biome = scene_context.get('biome', 'temperate')
            if not self._habitat_compatible(habitat, current_biome):
                return (GateDecision.DEFER, {
                    "reason": f"Creature requires {habitat} habitat, not {current_biome}",
                    "lead": f"Look for this creature in {habitat} regions"
                })
        
        return (GateDecision.ALLOW, {"reason": "Creature plausible for biome/era"})
    
    def _habitat_compatible(self, required: str, current: str) -> bool:
        """Check if habitats are compatible"""
        compatible = {
            'arctic': ['arctic', 'tundra'],
            'desert': ['desert', 'arid'],
            'forest': ['forest', 'jungle', 'temperate'],
            'ocean': ['ocean', 'coastal'],
            'mountain': ['mountain', 'alpine'],
            'swamp': ['swamp', 'wetland']
        }
        
        return current in compatible.get(required, [required])
    
    async def _check_timeline_consistency(
        self,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if event violates established timeline"""
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Check for conflicting events - now uses structured data
            conflicts = await conn.fetch("""
                SELECT event_id, event_text, timestamp, affects_entities
                FROM CanonicalEvents
                WHERE user_id=$1 AND conversation_id=$2
                AND timestamp > $3
                AND significance >= 5
                ORDER BY timestamp DESC
                LIMIT 5
            """, self.user_id, self.conversation_id, event_data['timestamp'])
            
            for conflict in conflicts:
                # Check using structured data if available
                if conflict.get('affects_entities'):
                    affected = json.loads(conflict['affects_entities'])
                    if self._would_invalidate_structured(event_data, affected):
                        return {
                            "consistent": False,
                            "violation": "Would invalidate future event",
                            "conflict": conflict['event_text']
                        }
                else:
                    # Fallback to text-based checking
                    if self._would_invalidate(event_data, conflict):
                        return {
                            "consistent": False,
                            "violation": "Would invalidate future event",
                            "conflict": conflict['event_text']
                        }
        
        return {"consistent": True}
    
    def _would_invalidate(self, new_event: Dict[str, Any], existing_event: Dict[str, Any]) -> bool:
        """Check if new event would invalidate an existing one (text-based)"""
        # Check for death/destruction that would prevent future events
        if 'destroys' in new_event:
            if new_event['destroys'] in existing_event['event_text']:
                return True
        
        if 'kills' in new_event:
            if new_event['kills'] in existing_event['event_text']:
                return True
        
        return False
    
    def _would_invalidate_structured(
        self,
        new_event: Dict[str, Any],
        affected_entities: Dict[str, Any]
    ) -> bool:
        """Check invalidation using structured data"""
        # Check if destroying something needed
        if 'destroys' in new_event:
            destroyed_id = new_event.get('destroys_id')
            if destroyed_id in affected_entities.get('requires_locations', []):
                return True
        
        # Check if killing someone needed
        if 'kills' in new_event:
            killed_id = new_event.get('kills_id')
            if killed_id in affected_entities.get('requires_npcs', []):
                return True
        
        return False
    
    async def _check_event_requirement(self, requirement: str) -> bool:
        """Check if an event requirement is met"""
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Check if requirement exists in canon
            exists = await conn.fetchval("""
                SELECT COUNT(*) FROM CanonicalEvents
                WHERE user_id=$1 AND conversation_id=$2
                AND event_text ILIKE $3
            """, self.user_id, self.conversation_id, f'%{requirement}%')
            
            return exists > 0
    
    def _check_physics_violation(
        self,
        event_data: Dict[str, Any],
        caps: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if event violates physics caps"""
        violations = []
        exceeded = {}
        
        # Check jump height
        if 'jump_height_m' in event_data:
            if event_data['jump_height_m'] > caps['max_jump_height_m']:
                violations.append('jump_height')
                exceeded['jump_height'] = {
                    'attempted': event_data['jump_height_m'],
                    'max': caps['max_jump_height_m']
                }
        
        # Check throw speed
        if 'projectile_speed_ms' in event_data:
            if event_data['projectile_speed_ms'] > caps['max_throw_speed_ms']:
                violations.append('throw_speed')
                exceeded['throw_speed'] = {
                    'attempted': event_data['projectile_speed_ms'],
                    'max': caps['max_throw_speed_ms']
                }
        
        # Check fall survival
        if 'fall_height_m' in event_data:
            if event_data['fall_height_m'] > caps['max_safe_fall_m']:
                violations.append('fall_survival')
                exceeded['fall_survival'] = {
                    'attempted': event_data['fall_height_m'],
                    'max': caps['max_safe_fall_m']
                }
        
        return {
            "violates": len(violations) > 0,
            "violations": violations,
            "exceeded_caps": exceeded
        }
    
    def _determine_era_from_infra(self, infra: Dict[str, Any]) -> str:
        """Determine era from infrastructure flags"""
        if infra.get('quantum_computing'):
            return 'futuristic'
        elif infra.get('instant_communication'):
            return 'modern'
        elif infra.get('electricity'):
            return 'industrial'
        elif infra.get('printing'):
            return 'medieval'
        else:
            return 'primitive'


# Convenience function for integration
async def assess_location_existence(
    ctx,
    name: str,
    desc: str,
    location_type: str = "generic"  # Added location_type parameter with default
) -> Tuple[GateDecision, Dict[str, Any]]:
    """Quick assessment for location existence"""
    gate = ExistenceGate(ctx)
    return await gate.assess_entity('location', {
        'location_name': name,
        'description': desc,
        'location_type': location_type
    })
