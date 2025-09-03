# lore/core/lore_system.py
"""
The unified LoreSystem - Primary implementation.
This is the core module that contains all lore-related logic and operations.
Complete refactor including ALL functionality from the original.
Enhanced with multi-relationship support for NPCs.
"""
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

from agents import RunContextWrapper
from lore.core.registry import ManagerRegistry
from lore.core import canon
from db.connection import get_db_connection_context
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority, NyxUnifiedGovernor
from nyx.governance_helpers import with_governance
from datetime import timedelta

# Import all managers
from lore.managers.education import EducationalSystemManager
from lore.managers.religion import ReligionManager
from lore.managers.local_lore import LocalLoreManager
from lore.managers.geopolitical import GeopoliticalSystemManager
from lore.systems.regional_culture import RegionalCultureSystem
from lore.managers.base_manager import BaseLoreManager
from lore.managers.politics import WorldPoliticsManager
from lore.systems.dynamics import LoreDynamicsSystem

# Import data access and integration components
from lore.data_access import NPCDataAccess, LocationDataAccess, FactionDataAccess, LoreKnowledgeAccess
from lore.integration import NPCLoreIntegration, ConflictIntegration, ContextEnhancer
from lore.lore_generator import DynamicLoreGenerator

logger = logging.getLogger(__name__)

class LoreSystem:
    """
    The core unified LoreSystem that handles all lore-related functionality.
    This is the primary implementation containing all logic.
    Enhanced with multi-relationship support for NPCs.
    """
    _instances: Dict[str, "LoreSystem"] = {}

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.registry = ManagerRegistry(user_id, conversation_id)
        self.initialized = False
        self._initializing = False
        self.governor = None
        
        # Initialize all data access components
        self.npc_data = NPCDataAccess(user_id, conversation_id)
        self.location_data = LocationDataAccess(user_id, conversation_id)
        self.faction_data = FactionDataAccess(user_id, conversation_id)
        self.lore_knowledge = LoreKnowledgeAccess(user_id, conversation_id)
        
        # Initialize integration components
        self.npc_integration = NPCLoreIntegration(user_id, conversation_id)
        self.conflict_integration = ConflictIntegration(user_id, conversation_id)
        self.context_enhancer = ContextEnhancer(user_id, conversation_id)
        
        # Initialize specialized managers
        self.education_manager = EducationalSystemManager(user_id, conversation_id)
        self.religion_manager = ReligionManager(user_id, conversation_id)
        self.local_lore_manager = LocalLoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.regional_culture_system = RegionalCultureSystem(user_id, conversation_id)
        self.world_politics_manager = WorldPoliticsManager(user_id, conversation_id)
        self.lore_dynamics_system = LoreDynamicsSystem(user_id, conversation_id)
        
        # Initialize generator (will be fully set up during initialization)
        self.generator = None
        
        # Store prohibited actions from directives
        self.prohibited_actions = []
        # Store modifications from directives
        self.action_modifications = {}
        
        # Track which managers need governance registration
        self._managers_needing_registration = []


    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> "LoreSystem":
        """Singleton instance retrieval for a given user/conversation context."""
        key = f"{user_id}:{conversation_id}"
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            # Don't await initialization here - let ensure_initialized handle it
            cls._instances[key] = instance
        return cls._instances[key]

    async def ensure_initialized(self):
        """Ensure the LoreSystem is initialized."""
        if self.initialized:
            return
        
        # Actually initialize the system if not already done
        if not self._initializing:
            await self.initialize()
        else:
            # Wait for initialization to complete if it's in progress
            while self._initializing:
                await asyncio.sleep(0.1)
        
        logger.info(f"LoreSystem initialized for user {self.user_id}, conversation {self.conversation_id}.")

    async def _maybe_acquire_governor(self) -> None:
        """
        Try to obtain a central governor if one wasn't provided.
        Uses a lazy import to avoid module-level import cycles.
        """
        if getattr(self, "governor", None):
            return
    
        try:
            # Import here to avoid import cycles on module load
            from nyx.integrate import get_central_governance
    
            gov = await get_central_governance(self.user_id, self.conversation_id)
            if gov:
                self.governor = gov
                logger.info("[LoreSystem] Governor acquired via central governance")
        except Exception as e:
            logger.debug(f"[LoreSystem] Could not acquire governor automatically: {e}")

    def set_governor(self, governor: Any) -> None:
        """Set the governor instance (can be called before or after initialize)."""
        self.governor = governor
        logger.info(f"[LoreSystem] Governor set for user {self.user_id}, conversation {self.conversation_id}")
        # If we’re already initialized, register components now (fire-and-forget)
        if self.initialized:
            try:
                asyncio.create_task(self._register_all_with_governance())
            except Exception:
        logger.debug("[LoreSystem] Deferred governance registration scheduling failed", exc_info=True)

    async def initialize(self, governor=None) -> bool:
        """
        Initialize the LoreSystem. Optionally accept a governor instance.
        Attempts to auto-acquire a governor if not provided.
        """
        # Guard against re-entrancy
        if getattr(self, "initialized", False) or getattr(self, "_initializing", False):
            logger.debug("[LoreSystem] Already initialized or initializing, skipping")
            return True
    
        self._initializing = True
        try:
            if governor is not None:
                self.governor = governor
                logger.info("[LoreSystem] Using provided governor instance")
    
            # NEW: attempt to acquire governor if not provided
            if not getattr(self, "governor", None):
                await self._maybe_acquire_governor()
    
            if getattr(self, "governor", None):
                logger.info("[LoreSystem] Governor set, proceeding with initialization")
            else:
                logger.warning("[LoreSystem] No governor set yet, some features may be limited")
    
            await self._initialize_components()
    
            if getattr(self, "governor", None):
                await self._register_all_with_governance()
    
            self.initialized = True
            logger.info("[LoreSystem] Initialization successful.")
            return True
    
        except Exception as e:
            logger.exception(f"[LoreSystem] ERROR during initialization: {e}")
            self.initialized = False
            return False
        finally:
            self._initializing = False

    async def _initialize_components(self):
        """Initialize all components WITHOUT governance registration."""
        # Initialize the data access components
        logger.info("[LoreSystem] Initializing: NPCDataAccess")
        await self.npc_data.initialize()

        logger.info("[LoreSystem] Initializing: LocationDataAccess")
        await self.location_data.initialize()

        logger.info("[LoreSystem] Initializing: FactionDataAccess")
        await self.faction_data.initialize()

        logger.info("[LoreSystem] Initializing: LoreKnowledgeAccess")
        await self.lore_knowledge.initialize()

        logger.info("[LoreSystem] Initializing: NPCLoreIntegration")
        # Set the governor on NPCLoreIntegration before initializing
        if hasattr(self.npc_integration, 'set_governor'):
            self.npc_integration.set_governor(self.governor)
        await self.npc_integration.initialize()
        self._managers_needing_registration.append(('npc_integration', self.npc_integration))

        logger.info("[LoreSystem] Initializing: ConflictIntegration")
        # Set governor if the component supports it
        if hasattr(self.conflict_integration, 'set_governor'):
            self.conflict_integration.set_governor(self.governor)
        await self.conflict_integration.initialize()
        self._managers_needing_registration.append(('conflict_integration', self.conflict_integration))

        logger.info("[LoreSystem] Initializing: ContextEnhancer")
        # Set governor if the component supports it
        if hasattr(self.context_enhancer, 'set_governor'):
            self.context_enhancer.set_governor(self.governor)
        await self.context_enhancer.initialize()
        self._managers_needing_registration.append(('context_enhancer', self.context_enhancer))

        logger.info("[LoreSystem] Initializing: DynamicLoreGenerator")
        # Get the instance with the governor to avoid circular dependency
        self.generator = DynamicLoreGenerator.get_instance(self.user_id, self.conversation_id, self.governor)
        await self.generator.initialize()
        self._managers_needing_registration.append(('generator', self.generator))

        # Initialize additional integrated managers
        await self._initialize_specialized_managers()

    async def _initialize_specialized_managers(self):
        """Initialize specialized managers with timeout protection."""
        managers = [
            ("EducationalSystemManager", self.education_manager),
            ("ReligionManager", self.religion_manager),
            ("LocalLoreManager", self.local_lore_manager),
            ("GeopoliticalSystemManager", self.geopolitical_manager),
            ("RegionalCultureSystem", self.regional_culture_system),
            ("WorldPoliticsManager", self.world_politics_manager),
            ("LoreDynamicsSystem", self.lore_dynamics_system)
        ]
        
        for manager_name, manager in managers:
            logger.info(f"[LoreSystem] Ensuring: {manager_name}")
            
            # Pass governor if the manager supports it
            if hasattr(manager, 'set_governor'):
                manager.set_governor(self.governor)
            
            try:
                # Add timeout to catch hanging initializations
                await asyncio.wait_for(
                    manager.ensure_initialized(),
                    timeout=15.0
                )
                self._managers_needing_registration.append((manager_name.lower(), manager))
                logger.info(f"[LoreSystem] {manager_name} initialized successfully")
                
            except asyncio.TimeoutError:
                logger.error(f"[LoreSystem] {manager_name} init timed-out - dumping tasks...")
                for task in asyncio.all_tasks():
                    logger.error(f"↳ {task} - {task.get_stack(limit=5)}")
                raise RuntimeError(f"{manager_name} initialization timed out")
            except Exception as e:
                logger.error(f"[LoreSystem] Error initializing {manager_name}: {e}")
                raise

    async def _register_all_with_governance(self):
        """Register all components with governance after they're initialized."""
        logger.info("[LoreSystem] Registering all components with Nyx governance")
        
        # First register the LoreSystem itself
        await self.register_with_governance()
        
        # Then register all managers that need it
        for manager_id, manager in self._managers_needing_registration:
            try:
                if hasattr(manager, 'register_with_governance_deferred'):
                    await manager.register_with_governance_deferred()
                elif hasattr(manager, '_get_agent_type') and hasattr(manager, '_get_agent_id'):
                    # Use the base pattern
                    await self.governor.register_agent(
                        agent_type=manager._get_agent_type(),
                        agent_id=manager._get_agent_id(),
                        agent_instance=manager
                    )
                    logger.info(f"[LoreSystem] Registered {manager_id} with governance")
            except Exception as e:
                logger.warning(f"[LoreSystem] Failed to register {manager_id} with governance: {e}")
                # Continue with other registrations even if one fails

    async def register_with_governance(self):
        """Register the lore system with Nyx governance."""
        if not self.governor:
            logger.warning("[LoreSystem] Cannot register with governance - no governor set")
            return

        # Register this system with governance
        await self.governor.register_agent(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_system",
            agent_instance=self
        )

        # Issue standard directives
        await self._issue_standard_directives()

    async def _issue_standard_directives(self):
        """Issue standard directives for the lore system."""
        if not self.governor:
            return

        # Example directive for lore generation
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="lore_generator",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Maintain world lore consistency and generate new lore as needed.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24 * 60  # 24 hours
        )

        # Example directive for NPC lore integration
        await self.governor.issue_directive(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="npc_lore_integration",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Ensure NPCs have appropriate lore knowledge based on their backgrounds.",
                "scope": "narrative"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24 * 60
        )

    def is_initializing(self) -> bool:
        """Check if the LoreSystem is currently initializing."""
        return self._initializing
    
    def is_initialized(self) -> bool:
        """Check if the LoreSystem has been initialized."""
        return self.initialized

    # ---------------------------------------------------------------------
    # NPC Lore Methods
    # ---------------------------------------------------------------------

    async def get_npc_lore_knowledge(self, npc_id: int) -> List[Dict[str, Any]]:
        """Get all lore known by an NPC."""
        return await self.lore_knowledge.get_entity_knowledge("npc", npc_id)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="initialize_npc_lore_knowledge",
        action_description="Initializing lore knowledge for NPC {npc_id}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def initialize_npc_lore_knowledge(self, ctx, npc_id: int,
                                            cultural_background: str,
                                            faction_affiliations: List[str]) -> Dict[str, Any]:
        """Initialize an NPC's lore knowledge based on background."""
        return await self.npc_integration.initialize_npc_lore_knowledge(
            ctx, npc_id, cultural_background, faction_affiliations
        )

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="process_npc_lore_interaction",
        action_description="Processing lore interaction for NPC {npc_id}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def process_npc_lore_interaction(self, ctx, npc_id: int, player_input: str) -> Dict[str, Any]:
        """Handle a lore interaction between the player and an NPC."""
        return await self.npc_integration.process_npc_lore_interaction(ctx, npc_id, player_input)

    # ---------------------------------------------------------------------
    # Location Lore Methods
    # ---------------------------------------------------------------------

    async def get_location_lore(self, location_id: int) -> Dict[str, Any]:
        """Get lore specific to a location."""
        return await self.location_data.get_location_with_lore(location_id)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_comprehensive_location_context",
        action_description="Getting comprehensive lore for location: {location_name}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def get_comprehensive_location_context(self, ctx, location_name: str) -> Dict[str, Any]:
        """Get a full lore context for a location, including culture/politics environment."""
        location = await self.location_data.get_location_by_name(location_name)
        if not location:
            return {}

        location_id = location.get("id")
        location_lore = await self.location_data.get_location_with_lore(location_id)
        cultural_info = await self.location_data.get_cultural_context_for_location(location_id)
        political_info = await self.location_data.get_political_context_for_location(location_id)
        environment_info = await self.location_data.get_environmental_conditions(location_id)

        return {
            "location": location,
            "lore": location_lore,
            "cultural_context": cultural_info,
            "political_context": political_info,
            "environment": environment_info
        }

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_scene_description",
        action_description="Generating scene description for location: {location}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_scene_description_with_lore(self, ctx, location: str) -> Dict[str, Any]:
        """Generate a scene description enhanced with relevant lore."""
        return await self.context_enhancer.generate_scene_description(location)

    # ---------------------------------------------------------------------
    # Education / Religion / Local Lore Accessors
    # ---------------------------------------------------------------------

    async def generate_educational_systems(self) -> List[Dict[str, Any]]:
        """
        High-level call to produce educational systems (or retrieve them)
        by delegating to the EducationalSystemManager.
        """
        # For example:
        return await self.education_manager.generate_educational_systems(None)

    async def generate_knowledge_traditions(self) -> List[Dict[str, Any]]:
        """
        Similarly, a high-level call to produce knowledge traditions
        from the EducationalSystemManager.
        """
        return await self.education_manager.generate_knowledge_traditions(None)

    async def generate_religion(self) -> Dict[str, Any]:
        """
        High-level call to produce a complete faith system
        from the ReligionManager.
        """
        return await self.religion_manager.generate_complete_faith_system(None)

    async def distribute_religions_across_nations(self) -> List[Dict[str, Any]]:
        """
        Use ReligionManager to distribute religions across world nations.
        """
        return await self.religion_manager.distribute_religions(None)

    async def generate_local_lore_for_location(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        High-level call to produce local myths/histories/landmarks
        from the LocalLoreManager.
        """
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        return await self.local_lore_manager._generate_location_lore_impl(ctx, location_data)

    # ---------------------------------------------------------------------
    # Lore Generation & Evolution
    # ---------------------------------------------------------------------

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_lore",
        action_description="Generating complete lore for environment: {environment_desc}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def generate_complete_lore(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """Generate a complete set of lore for a game world via the generator."""
        # Ensure the system is initialized
        await self.ensure_initialized()
        
        # Check if generator exists
        if not hasattr(self, 'generator') or self.generator is None:
            logger.error("LoreSystem generator not available after initialization")
            return {"error": "Lore generator not properly initialized"}
        
        return await self.generator.generate_complete_lore(environment_desc)

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_lore_with_event",
        action_description="Evolving lore based on event: {event_description}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def evolve_lore_with_event(self, ctx, event_description: str) -> Dict[str, Any]:
        """Update world lore based on a significant narrative event."""
        # Ensure the system is initialized
        await self.ensure_initialized()
        
        # Check if generator exists
        if not hasattr(self, 'generator') or self.generator is None:
            logger.error("LoreSystem generator not available after initialization")
            return {"error": "Lore generator not properly initialized"}
        
        return await self.generator.evolve_lore_with_event(event_description)

    async def mature_world_over_time(self, days_passed: int = 7) -> Dict[str, Any]:
        await self.ensure_initialized()
        ctx = RunContextWrapper(context={
            "user_id": self.user_id,
            "conversation_id": self.conversation_id
        })
        try:
            return await self.lore_dynamics_system.mature_lore_over_time(ctx, days_passed)
        except TypeError:
            return await self.lore_dynamics_system.mature_lore_over_time(days_passed)
    
    # ---------------------------------------------------------------------
    # Enhanced Relationship Methods - Multi-Relationship Support
    # ---------------------------------------------------------------------

    def _merge_compatible_relationships(self, current_relationships, new_relationships) -> Tuple[bool, List[Dict]]:
        """
        Merge relationship lists, allowing multiple compatible relationship types per entity.
        Handles both JSON strings and Python lists/dicts.
        
        Args:
            current_relationships: Can be JSON string, list of dicts, or None
            new_relationships: Can be JSON string, list of dicts, or None
        
        Returns:
            (is_compatible, merged_relationships)
        """
        import json  # Import here if not available at module level
        
        # Parse JSON strings if needed
        def parse_relationships(rels):
            """Parse relationships from various formats into a list of dicts."""
            if rels is None:
                return []
            if isinstance(rels, str):
                try:
                    parsed = json.loads(rels) if rels else []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse relationships JSON: {rels}")
                    return []
            else:
                parsed = rels
            
            # Ensure it's a list
            if not isinstance(parsed, list):
                if isinstance(parsed, dict):
                    return [parsed]  # Wrap single dict in list
                else:
                    return []
            
            return parsed
        
        # Parse both inputs
        current_relationships = parse_relationships(current_relationships)
        new_relationships = parse_relationships(new_relationships)
        
        # If either is empty, just return the other
        if not current_relationships:
            return True, new_relationships
        if not new_relationships:
            return True, current_relationships
        
        # Group current relationships by entity
        current_by_entity = {}
        for rel in current_relationships:
            if not isinstance(rel, dict):
                logger.warning(f"Skipping non-dict relationship: {rel}")
                continue
            key = (rel.get('entity_id'), rel.get('entity_type'))
            if key not in current_by_entity:
                current_by_entity[key] = []
            current_by_entity[key].append(rel)
        
        # Group new relationships by entity  
        new_by_entity = {}
        for rel in new_relationships:
            if not isinstance(rel, dict):
                logger.warning(f"Skipping non-dict relationship: {rel}")
                continue
            key = (rel.get('entity_id'), rel.get('entity_type'))
            if key not in new_by_entity:
                new_by_entity[key] = []
            new_by_entity[key].append(rel)
        
        # Only truly mutually exclusive relationships (can't be both at once)
        mutually_exclusive_groups = [
            # Can't be multiple family members at once
            {'mother', 'stepmother', 'aunt', 'older sister', 'stepsister', 'sister', 'cousin'},
            # Can't have multiple positions in SAME hierarchy
            {'boss', 'supervisor', 'boss/supervisor', 'employee', 'underling'},
            {'teacher', 'principal', 'teacher/principal'},
            # Remove these - they often coexist:
            # {'best friend', 'friend'} - best friend IS a friend
            # {'childhood friend', 'friend'} - childhood friend IS a friend
        ]
        
        # Relationships that supersede others (stronger includes weaker)
        superseding_relationships = {
            'best friend': {'friend', 'acquaintance'},
            'childhood friend': {'friend', 'acquaintance'},
            'friend': {'acquaintance'},
            'enemy': {'rival'},  # enemy is stronger than rival
            'nemesis': {'enemy', 'rival'},  # nemesis is ultimate enemy
            'lover': {'friend'},  # lovers are usually friends too
            'ex-wife': {'ex-partner'},  # more specific
            'ex-girlfriend': {'ex-partner'},  # more specific
        }
        
        # Very few truly impossible combinations
        truly_impossible_combinations = [
            # Can't be both superior and subordinate in same hierarchy
            ({'boss', 'supervisor', 'boss/supervisor'}, {'employee', 'underling'}),
        ]
        
        # Complicated but common combinations for narrative purposes
        complicated_combinations = [
            # Romantic complications
            ({'lover'}, {'ex-partner', 'ex-girlfriend', 'ex-wife'}),  # on-again, off-again
            ({'lover'}, {'the one who got away'}),  # finally together
            ({'lover'}, {'roommate'}),  # living together
            ({'lover'}, {'colleague', 'boss', 'supervisor', 'employee'}),  # workplace romance
            
            # Family complications  
            ({'mother', 'stepmother', 'aunt', 'older sister', 'stepsister', 'sister', 'cousin'}, 
             {'lover', 'ex-partner', 'ex-girlfriend', 'ex-wife'}),  # taboo
            ({'mother', 'stepmother'}, {'enemy', 'nemesis', 'rival'}),  # family conflict
            
            # Social complications
            ({'best friend', 'friend', 'childhood friend'}, {'enemy', 'nemesis', 'rival'}),  # frenemies
            ({'best friend', 'friend'}, {'ex-partner', 'ex-girlfriend', 'ex-wife'}),  # dated your friend
            ({'roommate'}, {'enemy', 'nemesis', 'rival'}),  # living with someone you hate
            
            # Professional complications
            ({'therapist'}, {'lover', 'ex-partner', 'friend', 'neighbor'}),  # boundary issues
            ({'landlord'}, {'lover', 'ex-partner', 'friend'}),  # mixing business and personal
            ({'boss', 'supervisor'}, {'friend', 'best friend'}),  # friendship with power dynamics
            
            # Authority complications
            ({'babysitter'}, {'lover', 'ex-partner'}),  # if they babysit your kids
            ({'teacher', 'principal'}, {'neighbor', 'friend'}),  # personal connection with authority
        ]
        
        # Natural relationship combinations that make sense
        natural_combinations = [
            ({'colleague'}, {'friend', 'best friend', 'rival', 'enemy'}),  # work relationships
            ({'neighbor'}, {'friend', 'best friend', 'landlord', 'enemy'}),  # proximity breeds all
            ({'roommate'}, {'friend', 'best friend', 'lover', 'ex-partner'}),  # close quarters
            ({'classmate'}, {'friend', 'best friend', 'rival', 'lover'}),  # school relationships
            ({'online friend'}, {'best friend', 'lover'}),  # online to real connection
        ]
        
        # Relationship strength (for resolving conflicts)
        relationship_strength = {
            # Family ties (strongest)
            'mother': 100,
            'stepmother': 95,
            'aunt': 90,
            'older sister': 85,
            'sister': 85,
            'stepsister': 84,
            'cousin': 80,
            
            # Deep emotional bonds
            'nemesis': 78,  # obsessive enemy
            'lover': 75,
            'best friend': 70,
            
            # Authority relationships
            'boss': 70,
            'supervisor': 69,
            'boss/supervisor': 70,
            'principal': 66,
            'teacher': 65,
            'therapist': 60,
            'landlord': 55,
            'head of household': 58,
            'domestic authority': 57,
            'babysitter': 50,
            
            # Strong connections
            'enemy': 60,
            'childhood friend': 55,
            'rival': 50,
            'friend': 45,
            
            # Professional/social
            'colleague': 40,
            'roommate': 38,
            'teammate': 35,
            'employee': 35,
            'underling': 34,
            'classmate': 30,
            'online friend': 28,
            'neighbor': 25,
            
            # Historical relationships  
            'ex-wife': 30,
            'ex-partner': 28,
            'ex-girlfriend': 27,
            'the one who got away': 35,  # higher because of emotional weight
            
            # Weakest
            'acquaintance': 10,
        }
        
        # Check for incompatibilities and merge
        merged = {}
        relationship_notes = []
        
        for entity_key in set(current_by_entity.keys()) | set(new_by_entity.keys()):
            merged[entity_key] = []
            
            # Collect all relationships for this entity
            all_rels = []
            rel_labels = set()
            
            for rel in current_by_entity.get(entity_key, []):
                all_rels.append(('current', rel))
                rel_labels.add(rel.get('relationship_label'))
                
            for rel in new_by_entity.get(entity_key, []):
                all_rels.append(('new', rel))
                rel_labels.add(rel.get('relationship_label'))
            
            # Check truly impossible combinations
            for group1, group2 in truly_impossible_combinations:
                if (rel_labels & group1) and (rel_labels & group2):
                    return False, []
            
            # Handle superseding relationships
            labels_to_remove = set()
            for stronger, weaker_set in superseding_relationships.items():
                if stronger in rel_labels:
                    labels_to_remove.update(weaker_set & rel_labels)
            
            # Remove superseded relationships
            all_rels = [(source, rel) for source, rel in all_rels 
                        if rel.get('relationship_label') not in labels_to_remove]
            
            # Check mutually exclusive groups
            for exclusive_group in mutually_exclusive_groups:
                group_matches = rel_labels & exclusive_group
                if len(group_matches) > 1:
                    strongest_label = max(group_matches, 
                                        key=lambda x: relationship_strength.get(x, 0))
                    for source, rel in all_rels[:]:
                        if (rel.get('relationship_label') in group_matches and 
                            rel.get('relationship_label') != strongest_label):
                            all_rels.remove((source, rel))
            
            # Update rel_labels after removals
            rel_labels = {rel.get('relationship_label') for _, rel in all_rels}
            
            # Check for natural combinations
            for group1, group2 in natural_combinations:
                if (rel_labels & group1) and (rel_labels & group2):
                    relationship_notes.append({
                        'entity': entity_key,
                        'type': 'natural',
                        'combo': f"{list(rel_labels & group1)[0]} + {list(rel_labels & group2)[0]}"
                    })
            
            # Check for complicated combinations
            for group1, group2 in complicated_combinations:
                if (rel_labels & group1) and (rel_labels & group2):
                    relationship_notes.append({
                        'entity': entity_key,
                        'type': 'complicated',
                        'combo': f"{list(rel_labels & group1)[0]} + {list(rel_labels & group2)[0]}",
                        'drama_potential': 'high'
                    })
            
            # Build final merged list
            seen_labels = set()
            for source, rel in all_rels:
                label = rel.get('relationship_label')
                if label not in seen_labels:
                    # Add metadata about relationship dynamics
                    if any(note['entity'] == entity_key and note['type'] == 'complicated' 
                          for note in relationship_notes):
                        rel['is_complicated'] = True
                        
                    # Track if this is a multi-faceted relationship
                    if len([r for _, r in all_rels if r.get('entity_id') == rel.get('entity_id')]) > 1:
                        rel['is_multifaceted'] = True
                        
                    merged[entity_key].append(rel)
                    seen_labels.add(label)
                else:
                    # Update existing with new data
                    for existing_rel in merged[entity_key]:
                        if existing_rel.get('relationship_label') == label:
                            for key, value in rel.items():
                                if key != 'relationship_label' and value is not None:
                                    existing_rel[key] = value
                            break
        
        # Convert back to flat list
        merged_list = []
        for relationships in merged.values():
            merged_list.extend(relationships)
        
        # Log interesting relationship dynamics
        if relationship_notes:
            natural = [n for n in relationship_notes if n['type'] == 'natural']
            complicated = [n for n in relationship_notes if n['type'] == 'complicated']
            if natural:
                logger.debug(f"Natural relationship combinations: {natural}")
            if complicated:
                logger.info(f"Complicated relationships detected (narrative gold!): {complicated}")
        
        return True, merged_list
    
    # ---------------------------------------------------------------------
    # Canon Methods - This is critical for the system to work
    # ---------------------------------------------------------------------
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="handle_narrative_event",
        action_description="Handling narrative event: {event_description}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def handle_narrative_event(
        self, 
        ctx,
        event_description: str,
        affected_lore_ids: List[str] = None,
        resolution_type: str = "standard",
        impact_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        Handle impacts of a narrative event on the world lore.
        
        Args:
            ctx: Governance context
            event_description: Description of the event that occurred
            affected_lore_ids: Optional list of specifically affected lore IDs
            resolution_type: Type of resolution (standard, conflict_generation, etc.)
            impact_level: Impact level (low, medium, high)
            
        Returns:
            Dictionary with update results
        """
        await self.ensure_initialized()
        
        try:
            # Log the event
            logger.info(f"Handling narrative event: {event_description[:100]}...")
            
            # Use the generator to evolve lore based on the event
            if hasattr(self, 'generator') and self.generator:
                evolution_result = await self.generator.evolve_lore_with_event(event_description)
            else:
                evolution_result = {
                    "status": "no_evolution",
                    "message": "Lore generator not available"
                }
            
            # Create canonical event log
            async with get_db_connection_context() as conn:
                await canon.log_canonical_event(
                    ctx, conn,
                    event_description,
                    tags=["narrative_event", resolution_type, impact_level],
                    significance=7 if impact_level == "high" else 5 if impact_level == "medium" else 3
                )
            
            # Return results
            return {
                "success": True,
                "event": event_description,
                "evolution_result": evolution_result,
                "affected_lore_ids": affected_lore_ids or [],
                "resolution_type": resolution_type,
                "impact_level": impact_level
            }
            
        except Exception as e:
            logger.error(f"Error handling narrative event: {e}")
            return {
                "success": False,
                "error": str(e),
                "event": event_description
            }
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="propose_and_enact_change",
        action_description="Proposing change to {entity_type} based on: {reason}",
        id_from_context=lambda ctx: "lore_system"
    )
    async def propose_and_enact_change(
        self,
        ctx,
        entity_type: str,
        entity_identifier: Dict[str, Any],
        updates: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """
        Canonical state-change helper with proper list comparison and enhanced relationship support.
        """
        from lore.core import canon
        from db.connection import get_db_connection_context
    
        # Helper function to check if a list is a subset of another list (for dicts)
        def _list_is_subset(smaller_list, larger_list):
            """Check if all items in smaller_list exist in larger_list (for lists of dicts)."""
            if not isinstance(smaller_list, list) or not isinstance(larger_list, list):
                return False
            
            # For each item in smaller list, check if it exists in larger list
            for item in smaller_list:
                if isinstance(item, dict):
                    # For dicts, we need to find an exact match
                    found = False
                    for larger_item in larger_list:
                        if isinstance(larger_item, dict) and item == larger_item:
                            found = True
                            break
                    if not found:
                        return False
                else:
                    # For non-dict items, use simple containment
                    if item not in larger_list:
                        return False
            return True
    
        logger.info(f"[propose_and_enact_change] Starting for entity_type={entity_type}, "
                    f"identifier={entity_identifier}, reason='{reason[:100]}...'")
        logger.debug(f"[propose_and_enact_change] Full updates: {json.dumps(updates, indent=2)}")

        # ------------------------------------------------------------------ #
        # Helpers                                                             #
        # ------------------------------------------------------------------ #
        def _is_effectively_empty(val) -> bool:
            """
            Returns True when *val* should be considered an "empty placeholder".

            Handles:
              • NULL/None
              • empty string / whitespace
              • literal string placeholders: 'None', 'null', '{}', '[]'
              • empty list / empty dict
            """
            if val is None:
                return True
            if isinstance(val, str):
                stripped = val.strip()
                return stripped in {"", "null", "None", "{}", "[]"}
            if isinstance(val, (list, tuple, set, dict)):
                return len(val) == 0
            return False
        
        # Comprehensive mapping of entity types to their JSON/JSONB fields and array fields
        json_fields = {
            # Core tables
            "StateUpdates": ["update_payload"],
            "Settings": ["enhanced_features", "stat_modifiers", "activity_examples"],
            "Locations": ["open_hours"],  # Arrays handled separately
            
            # NPC-related tables
            "NPCStats": ["relationships", "personality_traits", "hobbies", "likes", 
                         "dislikes", "affiliations", "schedule", "archetypes", 
                         "physical_description", "memory", "personality_patterns",
                         "trauma_triggers", "flashback_triggers", "revelation_plan"],
            "NPCGroups": ["group_data"],
            "NPCEvolution": ["mask_slippage_events", "evolution_events"],
            "NPCAgentState": ["current_state", "last_decision"],
            "NPCVisualAttributes": ["outfit_variations", "accessories", "expressions", "poses"],
            "NPCVisualEvolution": ["previous_state", "current_state"],
            "NPCMemories": ["associated_entities"],
            "NPCMasks": ["mask_data"],
            
            # Player-related tables
            "PlayerInventory": ["item_properties"],
            "PlayerPerks": ["perk_properties"],
            "PlayerSpecialRewards": ["reward_properties"],
            "PlayerJournal": ["entry_metadata", "tags"],
            "PlayerManipulationAttempts": ["goal", "leverage_used"],
            "PlayerConflictInvolvement": ["actions_taken", "manipulated_by"],
            
            # Memory system tables
            "unified_memories": ["metadata"],
            "MemoryMaintenanceSchedule": ["maintenance_schedule"],
            "memory_telemetry": ["metadata"],
            "SemanticNetworks": ["network_data"],
            
            # Game mechanics tables
            "Archetypes": ["baseline_stats", "progression_rules", "setting_examples", "unique_traits"],
            "Activities": ["purpose", "stat_integration", "intensity_tiers", "setting_variants"],
            "ActivityEffects": ["effects", "flags"],
            "IntensityTiers": ["key_features", "activity_examples", "permanent_effects"],
            "Interactions": ["detailed_rules", "task_examples", "agency_overrides"],
            "PlotTriggers": ["key_features", "stat_dynamics", "examples", "triggers"],
            
            # Faction/political tables
            "Factions": ["leadership_structure"],
            
            # World lore tables
            "NationalConflicts": ["public_opinion"],
            "JointMemories": ["tags", "metadata"],
            "PoliticalEntities": ["relations", "power_centers"],
            "ConflictSimulations": ["primary_actors", "timeline", "diplomatic_events", 
                                    "military_events", "civilian_impact", "resolution_scenarios", 
                                    "most_likely_outcome"],
            "BorderDisputes": ["resolution_attempts"],
            
            # Social/relationship tables
            "SocialLinks": ["link_history", "dynamics", "experienced_crossroads", 
                            "experienced_rituals"],
            "RelationshipEvolution": ["evolution_history"],
            "interaction_history": ["emotional_impact", "relationship_changes"],
            
            # Conflict system tables
            "ConflictStakeholders": ["alliances", "rivalries"],
            "ResolutionPaths": ["requirements", "stakeholders_involved", "key_challenges"],
            "PathStoryBeats": ["involved_npcs"],
            
            # Nyx system tables
            "nyx_brain_checkpoints": ["serialized_state"],
            "nyx_brain_events": ["event_payload"],
            "NyxAgentState": ["current_goals", "predicted_futures"],
            "nyx_dm_messages": ["message"],
            "NyxNPCDirectives": ["directive"],
            "JointMemoryGraph": ["tags", "metadata"],
            "NyxAgentDirectives": ["directive"],
            "NyxActionTracking": ["action_data", "result_data"],
            "NyxAgentCommunication": ["message_content", "response_content"],
            "NyxJointMemoryGraph": ["tags", "metadata"],
            "NyxNarrativeGovernance": ["governance_policy", "theme_directives", 
                                       "pacing_directives", "character_directives"],
            "nyx1_strategy_injections": ["payload"],
            "nyx1_strategy_logs": ["kink_profile", "decision_meta"],
            "NyxAgentRegistry": ["capabilities"],
            
            # Other system tables
            "scenario_states": ["state_data"],
            "performance_metrics": ["metrics", "error_log"],
            "learning_metrics": ["metrics", "learned_patterns"],
            "ImageFeedback": ["npc_names"],
            "UserKinkProfile": ["trigger_context"],
            "ContextEvolution": ["context_data", "changes"],
        }
        
        # Array fields (TEXT[], INTEGER[], etc) - these need special handling
        array_fields = {
            "Locations": ["notable_features", "hidden_aspects", "access_restrictions", "local_customs"],
            "Factions": ["values", "goals", "membership_requirements", "rivals", "allies", 
                         "secret_activities", "recruitment_methods"],
            "Nations": ["major_resources", "major_cities", "cultural_traits", "neighboring_nations"],
            "NationalConflicts": ["involved_nations", "recent_developments"],
            "CulturalElements": ["practiced_by"],
            "CulinaryTraditions": ["ingredients", "adopted_by"],
            "SocialCustoms": ["adopted_by"],
            "GeographicRegions": ["natural_resources", "notable_features", "major_settlements", 
                                  "cultural_traits", "dangers", "terrain_features"],
            "PoliticalEntities": ["internal_conflicts"],
            "BorderDisputes": ["female_leaders_involved"],
            "UrbanMyths": ["regions_known", "themes", "matriarchal_elements"],
            "LocalHistories": ["notable_figures", "connected_myths", "related_landmarks"],
            "Landmarks": ["legends", "connected_histories"],
            "HistoricalEvents": ["involved_entities", "consequences", "disputed_facts", 
                                 "commemorations", "primary_sources"],
            "NotableFigures": ["faction_affiliations", "achievements", "failures", 
                               "personality_traits", "hidden_aspects", "influence_areas", 
                               "controversial_actions", "relationships"],
            "CanonicalEvents": ["tags"],
            "nyx_brain_checkpoints": ["merged_from"],
            "nyx1_scene_templates": ["tags"],
            "unified_memories": ["tags"],
        }
        
        # Convert Python objects to JSON strings for comparison and storage
        processed_updates = {}
        entity_json_fields  = json_fields.get(entity_type, [])
        entity_array_fields = array_fields.get(entity_type, [])

        for field, value in updates.items():
            if field in entity_json_fields and value is not None and not isinstance(value, str):
                processed_updates[field] = json.dumps(value)
            elif field in entity_array_fields and value is not None:
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = [value]
                if not isinstance(value, list):
                    value = [value]
                processed_updates[field] = value
            else:
                processed_updates[field] = value

        logger.info(f"Proposing change to {entity_type} ({entity_identifier}) because: {reason}")

        try:
            async with get_db_connection_context() as conn:
                async with conn.transaction():
                    # 1. Fetch current row
                    logger.debug(f"[propose_and_enact_change] Fetching existing entity...")
                    where_sql = " AND ".join(f"{k} = ${i+1}"
                                             for i, k in enumerate(entity_identifier))
                    existing = await conn.fetchrow(
                        f"SELECT * FROM {entity_type} WHERE {where_sql}",
                        *entity_identifier.values()
                    )
                    if not existing:
                        logger.warning(f"[propose_and_enact_change] Entity not found: {entity_type} {entity_identifier}")
                        return {"status": "error",
                                "message": f"{entity_type} not found with {entity_identifier}"}
    
                    logger.debug(f"[propose_and_enact_change] Found existing entity: {existing.get('name', existing.get('npc_name', 'unknown'))}")
    
                    # 2. Conflict detection
                    conflicts = []
    
                    for field, new_val in processed_updates.items():
                        if field not in existing:
                            logger.debug(f"[propose_and_enact_change] Field '{field}' not in existing entity, skipping")
                            continue
    
                        current_val = existing[field]
                        logger.debug(f"[propose_and_enact_change] Checking field '{field}': "
                                   f"current={current_val!r}, new={new_val!r}")
    
                        # SHORT-CIRCUIT: if the current value is "blank", allow overwrite
                        if _is_effectively_empty(current_val):
                            logger.debug(f"[propose_and_enact_change] Field '{field}' is effectively empty, allowing overwrite")
                            continue
    
                        # SPECIAL CASE: Enhanced relationships handling
                        if field == "relationships" and field in entity_json_fields:
                            # Special handling for relationships to allow multiple compatible types
                            is_compatible, merged_relationships = self._merge_compatible_relationships(current_val, new_val)
                            
                            if is_compatible:
                                logger.debug(f"[propose_and_enact_change] Relationships are compatible, merging")
                                # Update the processed_updates to use merged relationships
                                processed_updates[field] = json.dumps(merged_relationships)
                                continue
                            else:
                                # True conflict - incompatible relationship states
                                conflict_msg = (
                                    f"Incompatible relationships for field '{field}': "
                                    f"Cannot combine {json.dumps(current_val)} with {json.dumps(new_val)}"
                                )
                                conflicts.append(conflict_msg)
                                logger.warning(f"[propose_and_enact_change] {conflict_msg}")
                            
                        elif field in entity_json_fields:
                            # Original JSON field handling for non-relationship fields
                            try:
                                cur_obj = (json.loads(current_val)
                                           if isinstance(current_val, str) else current_val) or {}
                                new_obj = (json.loads(new_val)
                                           if isinstance(new_val, str) else new_val) or {}
    
                                logger.debug(f"[propose_and_enact_change] JSON field '{field}' parsed successfully")
                                logger.debug(f"  Current: {json.dumps(cur_obj, indent=2)}")
                                logger.debug(f"  New: {json.dumps(new_obj, indent=2)}")
    
                                # Early exit: allow adding to empty dict/list
                                if _is_effectively_empty(cur_obj):
                                    logger.debug(f"[propose_and_enact_change] Current JSON is empty, allowing")
                                    continue
                                
                                # Handle list comparison specially
                                if isinstance(cur_obj, list) and isinstance(new_obj, list):
                                    # Check if it's a list of dicts (like relationships)
                                    if cur_obj and isinstance(cur_obj[0], dict):
                                        # Use custom subset check for lists of dicts
                                        if _list_is_subset(cur_obj, new_obj):
                                            logger.debug(f"[propose_and_enact_change] Current list is subset of new (dict comparison), allowing")
                                            continue
                                    else:
                                        # For lists of simple types, convert to set
                                        try:
                                            if set(cur_obj).issubset(set(new_obj)):
                                                logger.debug(f"[propose_and_enact_change] Current list is subset of new, allowing")
                                                continue
                                        except TypeError:
                                            # If items aren't hashable, fall back to equality check
                                            logger.debug(f"[propose_and_enact_change] List items not hashable, using equality check")
                                
                                # Handle dict comparison
                                elif isinstance(cur_obj, dict) and isinstance(new_obj, dict):
                                    if cur_obj.items() <= new_obj.items():
                                        logger.debug(f"[propose_and_enact_change] Current dict is subset of new, allowing")
                                        continue
    
                                # If not handled above, check for equality
                                if cur_obj != new_obj:
                                    conflict_msg = (
                                        f"Conflict on JSON field '{field}': "
                                        f"{json.dumps(cur_obj)} → {json.dumps(new_obj)}"
                                    )
                                    conflicts.append(conflict_msg)
                                    logger.warning(f"[propose_and_enact_change] {conflict_msg}")
                                    
                            except TypeError as e:
                                # Handle unhashable type error specifically
                                logger.error(f"[propose_and_enact_change] Type error on field '{field}': {e}")
                                conflict_msg = (
                                    f"Conflict on field '{field}' (type error): "
                                    f"{current_val!r} → {new_val!r}"
                                )
                                conflicts.append(conflict_msg)
                            except Exception as e:
                                conflict_msg = (
                                    f"Conflict on field '{field}' (parse error): "
                                    f"{current_val!r} → {new_val!r}"
                                )
                                conflicts.append(conflict_msg)
                                logger.error(f"[propose_and_enact_change] JSON parse error on field '{field}': {e}")
                                logger.debug(f"[propose_and_enact_change] Current value type: {type(current_val)}")
                                logger.debug(f"[propose_and_enact_change] New value type: {type(new_val)}")


                        # -------------------------------------------------- #
                        # ARRAY fields                                       #
                        # -------------------------------------------------- #
                        elif field in entity_array_fields:
                            cur_set = set(current_val or [])
                            new_set = set(new_val or [])

                            if _is_effectively_empty(cur_set):
                                continue
                            if cur_set.issubset(new_set):
                                continue
                            if cur_set != new_set:
                                conflicts.append(
                                    f"Conflict on array field '{field}': "
                                    f"{list(cur_set)} → {list(new_set)}"
                                )

                        # -------------------------------------------------- #
                        # Plain scalar fields                                #
                        # -------------------------------------------------- #
                        else:
                            if str(current_val) != str(new_val):
                                conflicts.append(
                                    f"Conflict on field '{field}': "
                                    f"{current_val!r} → {new_val!r}"
                                )

                    # 3. Handle conflicts or perform UPDATE
                    if conflicts:
                        logger.warning(f"[propose_and_enact_change] {len(conflicts)} conflicts detected")
                        for i, conflict in enumerate(conflicts):
                            logger.warning(f"  Conflict {i+1}: {conflict}")
                        
                        dynamics = await self.registry.get_lore_dynamics()
                        evt = self._create_conflict_event_description(
                            entity_type, entity_identifier, existing, updates, conflicts, reason
                        )
                        logger.info(f"[propose_and_enact_change] Generated conflict event: '{evt[:200]}...'")
                        
                        try:
                            await dynamics.evolve_lore_with_event(ctx, evt)
                            logger.info(f"[propose_and_enact_change] Lore evolution completed for conflict")
                        except Exception as e:
                            logger.error(f"[propose_and_enact_change] Lore evolution failed: {e}")
                            logger.error(f"[propose_and_enact_change] Event description was: {evt}")
                            raise
                        
                        return {"status": "conflict_generated", "details": conflicts}
    
                    # Step 3b: No conflict, commit the change
                    logger.info(f"[propose_and_enact_change] No conflicts, applying {len(processed_updates)} updates")
                    # Add type conversion for known boolean columns
                    boolean_columns = ['introduced', 'is_active', 'is_consolidated', 'is_archived',
                                       'fantasy_flag', 'consolidated', 'has_triggered_consequence',
                                       'is_revealed', 'is_public', 'is_completed', 'public_knowledge',
                                       'success', 'willing_to_betray_faction', 'equipped', 'used',
                                       'marked_for_review', 'dismissed', 'active']
                    
                    # Build SET clauses with proper casting for boolean and array columns
                    set_parts, vals = [], []
                    for i, (col, val) in enumerate(processed_updates.items()):
                        if col in boolean_columns:
                            set_parts.append(f"{col} = ${i+1}::boolean")
                        elif col in entity_array_fields:
                            set_parts.append(f"{col} = ${i+1}::{self._get_array_type(entity_type, col)}")
                        else:
                            set_parts.append(f"{col} = ${i+1}")
                        vals.append(val)

                    where_parts = [f"{k} = ${len(vals) + j + 1}"
                                   for j, k in enumerate(entity_identifier)]
                    vals.extend(entity_identifier.values())
    
                    await conn.execute(
                        f"UPDATE {entity_type} SET {', '.join(set_parts)} "
                        f"WHERE {' AND '.join(where_parts)}",
                        *vals
                    )
                    logger.info(f"[propose_and_enact_change] Database update successful")
    
                    # 5. Canonical event log
                    await canon.log_canonical_event(
                        ctx, conn,
                        f"{entity_type} {entity_identifier} updated. Reason: {reason}.",
                        tags=[entity_type.lower(), "state_change"],
                        significance=8
                    )

            # 6. Post-commit lore evolution (if significant)
            evt_desc = self._create_detailed_event_description(
                entity_type, entity_identifier, updates, reason
            )
            if evt_desc:
                logger.info(f"[propose_and_enact_change] Triggering post-commit lore evolution: '{evt_desc[:200]}...'")
                dynamics = await self.registry.get_lore_dynamics()
                await dynamics.evolve_lore_with_event(ctx, evt_desc)
            else:
                logger.debug(f"[propose_and_enact_change] No significant changes for lore evolution")
    
            logger.info(f"[propose_and_enact_change] Successfully committed changes to {entity_type} {entity_identifier}")
            return {"status": "committed",
                    "entity_type": entity_type,
                    "identifier": entity_identifier,
                    "changes": updates}
    
        except Exception as e:
            logger.exception(f"[propose_and_enact_change] Failed to enact change for {entity_type} {entity_identifier}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_array_type(self, entity_type: str, field: str) -> str:
        """Determine the PostgreSQL array type for a given field."""
        # Map of fields to their array types
        text_array_fields = {
            "values", "goals", "membership_requirements", "secret_activities", 
            "recruitment_methods", "major_resources", "major_cities", "cultural_traits",
            "neighboring_nations", "recent_developments", "practiced_by", "ingredients",
            "adopted_by", "resources", "notable_features", "major_settlements", 
            "dangers", "terrain_features", "internal_conflicts", "female_leaders_involved",
            "regions_known", "themes", "matriarchal_elements", "notable_figures",
            "connected_myths", "related_landmarks", "legends", "connected_histories",
            "involved_entities", "consequences", "disputed_facts", "commemorations",
            "primary_sources", "faction_affiliations", "achievements", "failures",
            "personality_traits", "hidden_aspects", "influence_areas", "controversial_actions",
            "relationships", "tags", "merged_from", "hidden_aspects", "access_restrictions",
            "local_customs", "natural_resources"
        }
        
        integer_array_fields = {
            "rivals", "allies", "involved_nations", "adopted_by"
        }
        
        if field in text_array_fields:
            return "TEXT[]"
        elif field in integer_array_fields:
            return "INTEGER[]"
        else:
            # Default to TEXT[] if unknown
            return "TEXT[]"

    def _create_detailed_event_description(
        self, 
        entity_type: str, 
        entity_identifier: Dict[str, Any], 
        updates: Dict[str, Any], 
        reason: str
    ) -> Optional[str]:
        """
        Create a detailed event description that will pass validation.
        Returns None if the change is too minor to warrant lore evolution.
        """
        # Determine if this change is significant enough
        significant_fields = {
            'NPCStats': ['power_level', 'status', 'loyalty'],
            'Factions': ['leader_npc_id', 'power_level', 'territory', 'allies', 'rivals'],
            'Nations': ['leader_npc_id', 'government_type', 'stability'],
            'Locations': ['controlling_faction', 'strategic_value'],
            'NPCs': ['status', 'faction_affiliation', 'current_location']
        }
        
        # Check if any significant fields were updated
        if entity_type in significant_fields:
            updated_significant_fields = [
                field for field in significant_fields[entity_type] 
                if field in updates
            ]
            if not updated_significant_fields:
                # Not significant enough for lore evolution
                return None
        
        # Build a detailed event description based on the entity type and changes
        event_parts = []
        
        if entity_type == "NPCStats" and "power_level" in updates:
            event_parts.append(f"A significant shift in power has occurred - someone's influence has changed dramatically")
        elif entity_type == "Factions" and "leader_npc_id" in updates:
            event_parts.append(f"Leadership has changed within a major faction")
        elif entity_type == "Nations" and "leader_npc_id" in updates:
            event_parts.append(f"A nation has experienced a change in leadership")
        elif entity_type == "Locations" and "controlling_faction" in updates:
            event_parts.append(f"Control of a strategic location has shifted to new hands")
        else:
            # Generic but still specific
            event_parts.append(f"The {entity_type.lower()} structure has undergone important changes")
        
        # Add the reason
        event_parts.append(f"This occurred because: {reason}")
        
        # Add specific change details for context
        change_details = []
        for field, value in updates.items():
            if field in ['power_level', 'status', 'loyalty', 'stability']:
                change_details.append(f"{field.replace('_', ' ')} has shifted")
            elif field in ['leader_npc_id', 'controlling_faction']:
                change_details.append(f"new leadership has been established")
            elif field in ['territory', 'allies', 'rivals']:
                change_details.append(f"{field} relationships have changed")
        
        if change_details:
            event_parts.append(f"Specifically: {', '.join(change_details)}")
        
        # Add context about the broader implications
        if entity_type in ["Nations", "Factions"]:
            event_parts.append("This shift in power will ripple through the political landscape, affecting alliances and rivalries")
        elif entity_type == "Locations":
            event_parts.append("This territorial change may spark new conflicts or opportunities in the region")
        elif entity_type == "NPCStats":
            event_parts.append("This personal transformation will influence their relationships and standing in society")
        
        return ". ".join(event_parts)
    
    def _create_specific_event_description(self, entity_type: str, entity_identifier: Dict[str, Any], 
                                          existing_entity: Any, updates: Dict[str, Any], reason: str) -> str:
        """Create a specific event description for lore evolution based on entity type."""
        
        # Try to get the entity name from various common fields
        entity_name = None
        for name_field in ['name', 'location_name', 'quest_name', 'title']:
            if name_field in existing_entity:
                entity_name = existing_entity[name_field]
                break
        
        if not entity_name:
            entity_name = f"{entity_type} #{entity_identifier.get('id', 'unknown')}"
        
        # Create type-specific descriptions
        if entity_type == "Nations":
            change_details = []
            if 'government_type' in updates:
                change_details.append(f"its government shifted from {existing_entity.get('government_type', 'unknown')} to {updates['government_type']}")
            if 'leader_npc_id' in updates:
                change_details.append("new leadership took power")
            if 'capital_id' in updates:
                change_details.append("the capital was relocated")
            
            changes_text = ", ".join(change_details) if change_details else "significant internal changes occurred"
            return f"The nation of {entity_name} underwent a major transformation: {reason}. As a result, {changes_text}. This political shift sends ripples through neighboring regions and allied factions."
        
        elif entity_type == "Factions":
            change_details = []
            if 'type' in updates:
                change_details.append(f"transformed from a {existing_entity.get('type', 'unknown')} faction to a {updates['type']} faction")
            if 'headquarters' in updates:
                change_details.append(f"relocated their headquarters to {updates['headquarters']}")
            if 'allies' in updates or 'rivals' in updates:
                change_details.append("shifted their diplomatic alignments")
            
            changes_text = ", ".join(change_details) if change_details else "underwent significant restructuring"
            return f"The {entity_name} faction experienced a pivotal moment: {reason}. The faction {changes_text}, marking a new chapter in their influence and operations. Their members and affiliates must now adapt to this new reality."
        
        elif entity_type == "CulturalElements":
            element_type = existing_entity.get('type', 'tradition')
            return f"The cultural {element_type} known as '{entity_name}' evolved significantly: {reason}. This transformation affects how it is practiced and perceived, potentially influencing related customs and beliefs throughout the regions where it holds sway."
        
        elif entity_type == "HistoricalEvents":
            return f"New revelations about the historical event '{entity_name}' have come to light: {reason}. These discoveries force scholars and lorekeepers to reconsider its true impact and meaning, potentially rewriting portions of accepted history."
        
        elif entity_type == "GeographicRegions":
            change_details = []
            if 'governing_faction' in updates:
                change_details.append(f"control shifted to {updates['governing_faction']}")
            if 'climate' in updates:
                change_details.append("experienced dramatic environmental changes")
            
            changes_text = ", ".join(change_details) if change_details else "underwent significant changes"
            return f"The region of {entity_name} experienced momentous change: {reason}. The region {changes_text}, affecting all who dwell within its borders and potentially altering trade routes and strategic considerations."
        
        elif entity_type == "Locations":
            return f"The location known as {entity_name} was dramatically altered: {reason}. These changes affect its significance, accessibility, or nature, potentially impacting nearby settlements and those who depend on or fear this place."
        
        elif entity_type == "QuestHooks":
            return f"The quest '{entity_name}' has evolved unexpectedly: {reason}. New complications, opportunities, or revelations have emerged, changing the stakes for any who might pursue this adventure."
        
        else:
            # Generic but still more detailed than before
            field_changes = []
            for field, new_value in updates.items():
                if field in existing_entity:
                    old_value = existing_entity[field]
                    if old_value != new_value:
                        field_changes.append(f"{field} changed from '{old_value}' to '{new_value}'")
            
            changes_summary = "; ".join(field_changes[:3])  # Limit to first 3 changes
            if len(field_changes) > 3:
                changes_summary += f"; and {len(field_changes) - 3} other changes"
            
            return f"A significant transformation occurred in the world's {entity_type}: {reason}. Specifically, {changes_summary}. These changes ripple through connected systems and relationships."
    
    def _create_conflict_event_description(self, entity_type: str, entity_identifier: Dict[str, Any],
                                         existing_entity: Any, updates: Dict[str, Any], 
                                         conflicts: List[str], reason: str) -> str:
        """Create an event description for when a conflict is detected during updates."""
        logger.info(f"[_create_conflict_event_description] Creating description for {entity_type} conflict")
        logger.debug(f"[_create_conflict_event_description] Entity identifier: {entity_identifier}")
        logger.debug(f"[_create_conflict_event_description] Conflicts: {conflicts}")
        logger.debug(f"[_create_conflict_event_description] Reason: {reason}")
        
        # Try to get the entity name - expanded list to handle more entity types
        entity_name = None
        name_fields = [
            'name', 'npc_name', 'location_name', 'quest_name', 'title', 
            'event_name',  # For LocalHistories
            'conflict_name',  # For Conflicts
            'group_name',  # For NPCGroups
            'faction_name',  # For various faction-related tables
            'currency_name',  # For CurrencySystem
            'trigger_name',  # For PlotTriggers
            'interaction_name',  # For Interactions
            'region_name',  # For regions
            'landmark_name'  # For landmarks
        ]
        
        for name_field in name_fields:
            if name_field in existing_entity:
                entity_name = existing_entity[name_field]
                logger.debug(f"[_create_conflict_event_description] Found entity name: {entity_name}")
                break
        
        if not entity_name:
            # Try to construct a meaningful identifier from available fields
            if 'id' in entity_identifier:
                entity_name = f"{entity_type} #{entity_identifier['id']}"
            elif 'npc_id' in entity_identifier:
                entity_name = f"{entity_type} #NPC{entity_identifier['npc_id']}"
            elif 'user_id' in entity_identifier and 'conversation_id' in entity_identifier:
                entity_name = f"{entity_type} (User:{entity_identifier['user_id']}, Conv:{entity_identifier['conversation_id']})"
            else:
                entity_name = f"{entity_type} #{entity_identifier.get('id', entity_identifier.get('npc_id', 'unknown'))}"
            logger.debug(f"[_create_conflict_event_description] Using fallback entity name: {entity_name}")
        
        # Don't pass technical parsing errors to lore evolution
        technical_conflict = any('parse error' in c or 'JSON' in c or 'Type error' in c for c in conflicts)
        logger.info(f"[_create_conflict_event_description] Technical conflict detected: {technical_conflict}")
        
        if technical_conflict:
            # Log the technical details before converting to narrative
            logger.warning(f"[_create_conflict_event_description] Converting technical conflict to narrative")
            logger.debug(f"[_create_conflict_event_description] Original conflicts: {json.dumps(conflicts, indent=2)}")
            
            # Create narrative descriptions based on the specific technical error
            if entity_type == "NPCStats" and "relationships" in str(conflicts[0]):
                description = (f"Complex social dynamics prevented {entity_name} from forming new relationships as intended. "
                              f"The existing web of connections resisted change, suggesting deeper loyalties or "
                              f"obligations that must be addressed before new bonds can form. Original intention: {reason}")
            elif entity_type == "LocalHistories" and any("name" in c for c in conflicts):
                description = (f"Historical records for {entity_name} proved resistant to revision. "
                              f"The chronicles seem to have their own inertia, as if the past itself refuses "
                              f"to be rewritten. Original intention: {reason}")
            elif "JSON" in str(conflicts[0]) or "parse error" in str(conflicts[0]):
                description = (f"The intricate nature of {entity_name} resisted simplification or change. "
                              f"Its complex essence proved too intertwined with the world's fabric to alter "
                              f"in the intended way. Original intention: {reason}")
            else:
                description = (f"Mysterious forces prevented changes to {entity_name}. The very fabric of reality "
                              f"seemed to resist the transformation, as if protecting some essential truth about "
                              f"their nature. Original intention: {reason}")
        else:
            # For non-technical conflicts, create a proper narrative
            conflict_summary = conflicts[0] if conflicts else "Multiple conflicting changes were attempted"
            
            # Extract field names from conflict messages for better narratives
            conflicting_fields = []
            for conflict in conflicts:
                if "Conflict on field" in conflict:
                    field_match = conflict.split("'")[1] if "'" in conflict else None
                    if field_match:
                        conflicting_fields.append(field_match)
            
            if conflicting_fields:
                # Create specific narratives based on the conflicting fields
                if "power_level" in conflicting_fields or "influence" in conflicting_fields:
                    description = (f"A power struggle emerged around {entity_name}: {reason}. "
                                  f"Competing forces vie for influence, each attempting to shape their destiny "
                                  f"in different ways. The balance of power remains contested.")
                elif "territory" in conflicting_fields or "location" in conflicting_fields:
                    description = (f"Territorial disputes arose concerning {entity_name}: {reason}. "
                                  f"Multiple claims to the same domain created tension, as different factions "
                                  f"assert their rights to control or influence this valuable asset.")
                elif "allies" in conflicting_fields or "rivals" in conflicting_fields:
                    description = (f"Diplomatic tensions surrounded {entity_name}: {reason}. "
                                  f"Conflicting allegiances and rivalries created a complex web of relationships "
                                  f"that resisted simple resolution, reflecting deeper political currents.")
                else:
                    field_text = ", ".join(conflicting_fields[:3])
                    description = (f"Conflicting changes were attempted on {entity_name} regarding {field_text}: {reason}. "
                                  f"These competing alterations revealed underlying tensions in the world's structure, "
                                  f"suggesting that some aspects of reality resist easy modification.")
            else:
                # Generic conflict narrative
                if "already has value" in conflict_summary:
                    description = (f"A conflict arose when attempting to change {entity_name}: {reason}. "
                                  f"The attempted changes were blocked because the entity already possesses "
                                  f"the qualities being imposed, creating a paradox of transformation.")
                else:
                    description = (f"A conflict arose when attempting to change {entity_name}: {reason}. "
                                  f"The attempted changes were blocked by competing forces within the world, "
                                  f"creating tension between what was and what might be. This resistance "
                                  f"suggests deeper currents at work in the world's fabric.")
        
        logger.info(f"[_create_conflict_event_description] Generated narrative description: '{description[:100]}...'")
        return description

    # --- Convenience Wrappers (Optional but Recommended) ---
    # These methods provide a clean, high-level API but all use the generic method internally.

    async def execute_coup(self, ctx, nation_id: int, new_leader_id: int, reason: str):
        """A high-level wrapper for a coup event."""
        # Ensure proper context
        if isinstance(ctx, str) or not hasattr(ctx, 'context'):
            ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
        
        return await self.propose_and_enact_change(
            ctx=ctx,
            entity_type="Nations",
            entity_identifier={"id": nation_id},
            updates={"leader_npc_id": new_leader_id},
            reason=reason
        )

    async def assign_faction_territory(self, ctx, faction_id: int, location_name: str, reason: str):
        """A high-level wrapper for a faction taking over territory."""
        # Ensure proper context
        if isinstance(ctx, str) or not hasattr(ctx, 'context'):
            ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
        
        location_id = None
        # First, ensure the location exists canonically
        async with get_db_connection_context() as conn:
            location_id = await canon.find_or_create_location(ctx, conn, location_name)

        return await self.propose_and_enact_change(
            ctx=ctx,
            entity_type="Factions",
            entity_identifier={"id": faction_id},
            updates={"territory": location_name}, # Assuming territory is a text field
            reason=reason
        )

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    async def cleanup(self):
        """Clean up resources used by the LoreSystem (and managers)."""
        try:
            # Collect all cleanup tasks in parallel
            cleanup_tasks = [
                self.npc_data.cleanup(),
                self.location_data.cleanup(),
                self.faction_data.cleanup(),
                self.lore_knowledge.cleanup(),
                self.npc_integration.cleanup(),
                self.conflict_integration.cleanup(),
                self.context_enhancer.cleanup(),
                self.generator.cleanup()
            ]
    
            # Example approach: if each manager implements a 'cleanup' method, call it
            for manager in [
                self.education_manager,
                self.religion_manager,
                self.local_lore_manager,
                self.geopolitical_manager,
                self.regional_culture_system,
                self.world_politics_manager,
                self.lore_dynamics_system
            ]:
                if hasattr(manager, "cleanup") and callable(manager.cleanup):
                    cleanup_tasks.append(manager.cleanup())
    
            # Execute all cleanup tasks concurrently
            await asyncio.gather(*cleanup_tasks)
    
            # Remove this instance from the global cache
            key = f"{self.user_id}:{self.conversation_id}"
            if key in self._instances:
                del self._instances[key]
    
        except Exception as e:
            logger.error(f"Error during LoreSystem cleanup: {e}")
