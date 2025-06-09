# logic/conflict_system/integration.py
"""
Main integration point for the conflict system
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

from agents import RunContextWrapper
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType

from .core import ConflictCore
from .generation import ConflictGenerator
from .evolution import ConflictEvolution
from .resolution import ConflictResolver
from .stakeholders import StakeholderManager

logger = logging.getLogger(__name__)

class ConflictSystemIntegration:
    """Main integration class for the conflict system"""
    
    _instances: Dict[str, 'ConflictSystemIntegration'] = {}
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.is_initialized = False
        
        # Core components
        self.core = ConflictCore(user_id, conversation_id)
        self.generator = ConflictGenerator(user_id, conversation_id)
        self.evolution = ConflictEvolution(user_id, conversation_id)
        self.resolver = ConflictResolver(user_id, conversation_id)
        self.stakeholders = StakeholderManager(user_id, conversation_id)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_task = None
    
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> 'ConflictSystemIntegration':
        """Get or create instance"""
        key = f"{user_id}:{conversation_id}"
        if key not in cls._instances:
            instance = cls(user_id, conversation_id)
            await instance.initialize()
            cls._instances[key] = instance
        return cls._instances[key]
    
    async def initialize(self):
        """Initialize the system"""
        if self.is_initialized:
            return
        
        logger.info(f"Initializing conflict system for user {self.user_id}")
        
        # Register with governance
        await self._register_with_governance()
        
        # Start monitoring if enabled
        if await self._should_auto_monitor():
            await self.start_monitoring()
        
        self.is_initialized = True
    
    async def _register_with_governance(self):
        """Register with central governance"""
        try:
            governance = await get_central_governance(self.user_id, self.conversation_id)
            
            await governance.register_agent(
                "conflict_system",
                AgentType.CONFLICT_ANALYST,
                self
            )
            
            logger.info("Conflict system registered with governance")
        except Exception as e:
            logger.error(f"Error registering with governance: {e}")
    
    async def _should_auto_monitor(self) -> bool:
        """Check if automatic monitoring should be enabled"""
        # Could check settings, but default to True
        return True
    
    async def start_monitoring(self):
        """Start automatic conflict monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started conflict monitoring")
    
    async def stop_monitoring(self):
        """Stop automatic conflict monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped conflict monitoring")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for new conflicts
                await self.check_and_generate_conflicts()
                
                # Process natural evolution
                await self.evolve_active_conflicts()
                
                # Process stakeholder turns
                await self.process_all_stakeholder_turns()
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def check_and_generate_conflicts(self) -> Optional[Dict[str, Any]]:
        """Check world state and generate conflict if appropriate"""
        try:
            # Analyze pressure
            pressure = await self.generator.analyzer.analyze_world_state()
            
            # Check active conflicts
            active = await self.core.get_active_conflicts()
            
            # Generate if appropriate
            if pressure['total_pressure'] > 100 and len(active) < 3:
                return await self.generator.generate_organic_conflict()
                
            return None
            
        except Exception as e:
            logger.error(f"Error checking conflicts: {e}")
            return None
    
    async def evolve_active_conflicts(self):
        """Process natural evolution for all active conflicts"""
        try:
            conflicts = await self.core.get_active_conflicts()
            
            for conflict in conflicts:
                await self.evolution.process_natural_evolution(
                    conflict['conflict_id']
                )
                
        except Exception as e:
            logger.error(f"Error evolving conflicts: {e}")
    
    async def process_all_stakeholder_turns(self):
        """Process stakeholder turns for all conflicts"""
        try:
            conflicts = await self.core.get_active_conflicts()
            
            for conflict in conflicts:
                await self.stakeholders.process_stakeholder_turns(
                    conflict['conflict_id']
                )
                
        except Exception as e:
            logger.error(f"Error processing stakeholder turns: {e}")
    
    # Public API methods
    
    async def generate_conflict(self, conflict_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate a new conflict"""
        return await self.generator.generate_organic_conflict(
            force_archetype=conflict_type
        )
    
    async def get_active_conflicts(self) -> List[Dict[str, Any]]:
        """Get all active conflicts"""
        return await self.core.get_active_conflicts()
    
    async def get_conflict(self, conflict_id: int) -> Optional[Dict[str, Any]]:
        """Get specific conflict details"""
        return await self.core.get_conflict_details(conflict_id)
    
    async def advance_conflict(self, conflict_id: int,
                             event_description: str,
                             involved_npcs: Optional[List[int]] = None) -> Dict[str, Any]:
        """Advance a conflict based on an event"""
        return await self.evolution.evolve_conflict(
            conflict_id,
            "player_action",
            {
                "description": event_description,
                "involved_npcs": involved_npcs or []
            }
        )
    
    async def resolve_conflict(self, conflict_id: int,
                             resolution_method: str = "negotiated") -> Dict[str, Any]:
        """Resolve a conflict"""
        return await self.resolver.resolve_conflict(
            conflict_id,
            resolution_method,
            {}
        )
    
    async def handle_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Handle directive from governance"""
        directive_type = directive.get("type")
        directive_data = directive.get("data", {})
        
        if directive_type == DirectiveType.ACTION:
            action = directive_data.get("action")
            if action == "generate_conflict":
                result = await self.generate_conflict()
                return {"success": True, "result": result}
            elif action == "resolve_conflict":
                conflict_id = directive_data.get("conflict_id")
                result = await self.resolve_conflict(conflict_id)
                return {"success": True, "result": result}
        
        return {"success": False, "error": "Unknown directive"}
