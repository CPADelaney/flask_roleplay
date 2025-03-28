# nyx/core/conditioning_maintenance.py

import asyncio
import logging
import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ConditioningMaintenanceSystem:
    """
    Handles periodic maintenance tasks for the conditioning system.
    """
    
    def __init__(self, conditioning_system, reward_system=None):
        self.conditioning_system = conditioning_system
        self.reward_system = reward_system
        
        # Maintenance configuration
        self.maintenance_interval_hours = 24  # Run daily
        self.extinction_threshold = 0.05  # Remove associations below this strength
        self.reinforcement_threshold = 0.3  # Reinforce associations above this strength
        self.consolidation_interval_days = 7  # Consolidate weekly
        
        # Maintenance stats
        self.last_maintenance_time = None
        self.maintenance_history = []
        self.max_history_entries = 30
        
        # Background task
        self.maintenance_task = None
        
        logger.info("Conditioning maintenance system initialized")
    
    async def start_maintenance_scheduler(self):
        """Start the periodic maintenance scheduler"""
        if self.maintenance_task is not None:
            logger.warning("Maintenance scheduler already running")
            return
        
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        logger.info("Maintenance scheduler started")
    
    async def stop_maintenance_scheduler(self):
        """Stop the periodic maintenance scheduler"""
        if self.maintenance_task is None:
            logger.warning("Maintenance scheduler not running")
            return
        
        self.maintenance_task.cancel()
        try:
            await self.maintenance_task
        except asyncio.CancelledError:
            pass
        self.maintenance_task = None
        logger.info("Maintenance scheduler stopped")
    
    async def _maintenance_loop(self):
        """Internal maintenance loop"""
        try:
            while True:
                # Run maintenance
                try:
                    await self.run_maintenance()
                except Exception as e:
                    logger.error(f"Error in maintenance run: {e}")
                
                # Sleep until next maintenance
                sleep_seconds = self.maintenance_interval_hours * 3600
                logger.info(f"Next maintenance scheduled in {self.maintenance_interval_hours} hours")
                await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            logger.info("Maintenance loop cancelled")
            raise
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance on the conditioning system
        
        Returns:
            Maintenance results
        """
        logger.info("Starting conditioning system maintenance")
        start_time = datetime.datetime.now()
        
        # 1. Run extinction on associations
        extinction_results = await self.conditioning_system.run_maintenance()
        
        # 2. Check personality balance
        personality_balance = await self._check_personality_balance()
        
        # 3. Reinforce core traits if needed
        reinforcement_results = await self._reinforce_core_traits(personality_balance)
        
        # 4. Consolidate similar associations if due
        consolidation_due = False
        if self.last_maintenance_time:
            days_since_consolidation = (datetime.datetime.now() - self.last_maintenance_time).days
            consolidation_due = days_since_consolidation >= self.consolidation_interval_days
        else:
            consolidation_due = True
        
        consolidation_results = None
        if consolidation_due:
            consolidation_results = await self._consolidate_associations()
        
        # 5. Record maintenance
        duration = (datetime.datetime.now() - start_time).total_seconds()
        
        maintenance_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "duration_seconds": duration,
            "extinction_results": extinction_results,
            "personality_balance": personality_balance,
            "reinforcement_results": reinforcement_results,
            "consolidation_results": consolidation_results
        }
        
        # Update maintenance history
        self.maintenance_history.append(maintenance_record)
        if len(self.maintenance_history) > self.max_history_entries:
            self.maintenance_history = self.maintenance_history[-self.max_history_entries:]
        
        self.last_maintenance_time = datetime.datetime.now()
        
        logger.info(f"Conditioning system maintenance completed in {duration:.2f} seconds")
        return maintenance_record
    
    async def _check_personality_balance(self) -> Dict[str, Any]:
        """Check if personality traits are balanced"""
        # Get all classical associations
        classical_associations = self.conditioning_system.classical_associations
        
        # Count associations by stimulus category
        stimulus_counts = {}
        
        for key, association in classical_associations.items():
            stimulus = association.stimulus
            
            # Skip weak associations
            if association.association_strength < self.reinforcement_threshold:
                continue
            
            if stimulus not in stimulus_counts:
                stimulus_counts[stimulus] = 0
            
            stimulus_counts[stimulus] += 1
        
        # Get all operant associations
        operant_associations = self.conditioning_system.operant_associations
        
        # Count behaviors by category
        behavior_counts = {}
        
        for key, association in operant_associations.items():
            behavior = association.stimulus  # The behavior is stored as stimulus
            
            # Skip weak associations
            if association.association_strength < self.reinforcement_threshold:
                continue
            
            if behavior not in behavior_counts:
                behavior_counts[behavior] = 0
            
            behavior_counts[behavior] += 1
        
        # Analyze balance
        imbalances = []
        
        # Check if any behavior is overrepresented
        total_behaviors = sum(behavior_counts.values())
        for behavior, count in behavior_counts.items():
            if total_behaviors > 0 and count / total_behaviors > 0.3:
                imbalances.append({
                    "type": "behavior_overrepresented",
                    "behavior": behavior,
                    "count": count,
                    "percentage": count / total_behaviors
                })
        
        # Check if any stimulus is overrepresented
        total_stimuli = sum(stimulus_counts.values())
        for stimulus, count in stimulus_counts.items():
            if total_stimuli > 0 and count / total_stimuli > 0.3:
                imbalances.append({
                    "type": "stimulus_overrepresented",
                    "stimulus": stimulus,
                    "count": count,
                    "percentage": count / total_stimuli
                })
        
        return {
            "behavior_counts": behavior_counts,
            "stimulus_counts": stimulus_counts,
            "imbalances": imbalances,
            "balanced": len(imbalances) == 0
        }
    
    async def _reinforce_core_traits(self, personality_balance: Dict[str, Any]) -> Dict[str, Any]:
        """Reinforce core personality traits if needed"""
        # Define core traits and their ideal values
        core_traits = {
            "dominance": 0.8,
            "playfulness": 0.7,
            "strictness": 0.6,
            "creativity": 0.7
        }
        
        reinforcement_results = []
        
        # Check for imbalances and reinforce relevant traits
        for imbalance in personality_balance.get("imbalances", []):
            # Handle behavior overrepresentation by weakening that behavior
            if imbalance["type"] == "behavior_overrepresented":
                behavior = imbalance["behavior"]
                
                # Apply a mild negative reinforcement to reduce dominance
                result = await self.conditioning_system.process_operant_conditioning(
                    behavior=behavior,
                    consequence_type="negative_punishment",  # Remove positive reinforcement
                    intensity=0.3,  # Mild effect
                    context={"source": "maintenance_balancing"}
                )
                
                reinforcement_results.append({
                    "type": "balance_correction",
                    "behavior": behavior,
                    "action": "reduce_dominance",
                    "result": result
                })
        
        # Reinforce core traits to maintain personality
        for trait, value in core_traits.items():
            # Only reinforce traits that should be strong (value >= 0.6)
            if value >= 0.6:
                # Use a mild reinforcement to maintain the trait
                result = await self.conditioning_system.condition_personality_trait(
                    trait=trait,
                    value=value * 0.3,  # Scale down to avoid overreinforcement
                    context={"source": "maintenance_reinforcement"}
                )
                
                reinforcement_results.append({
                    "type": "trait_reinforcement",
                    "trait": trait,
                    "target_value": value,
                    "reinforcement_value": value * 0.3,
                    "result": result
                })
        
        return {
            "reinforcements": reinforcement_results,
            "core_traits": core_traits
        }
    
    async def _consolidate_associations(self) -> Dict[str, Any]:
        """Consolidate similar associations"""
        # Get classical associations
        classical_associations = self.conditioning_system.classical_associations
        
        # Group by stimulus
        stimulus_groups = {}
        
        for key, association in classical_associations.items():
            stimulus = association.stimulus
            
            if stimulus not in stimulus_groups:
                stimulus_groups[stimulus] = []
            
            stimulus_groups[stimulus].append((key, association))
        
        # Find groups with multiple associations for the same stimulus
        consolidation_candidates = {}
        
        for stimulus, associations in stimulus_groups.items():
            if len(associations) > 1:
                consolidation_candidates[stimulus] = associations
        
        # Consolidate associations within each group
        consolidations = []
        
        for stimulus, associations in consolidation_candidates.items():
            # Group by response to find duplicates
            response_groups = {}
            
            for key, association in associations:
                response = association.response
                
                if response not in response_groups:
                    response_groups[response] = []
                
                response_groups[response].append((key, association))
            
            # Consolidate associations with the same response
            for response, response_associations in response_groups.items():
                if len(response_associations) > 1:
                    # Get the strongest association
                    strongest = max(response_associations, key=lambda x: x[1].association_strength)
                    strongest_key, strongest_association = strongest
                    
                    # Consolidate other associations into the strongest
                    for key, association in response_associations:
                        if key != strongest_key:
                            # Strengthen the strongest association slightly
                            new_strength = min(1.0, strongest_association.association_strength + (association.association_strength * 0.2))
                            strongest_association.association_strength = new_strength
                            
                            # Combine context keys
                            for context_key in association.context_keys:
                                if context_key not in strongest_association.context_keys:
                                    strongest_association.context_keys.append(context_key)
                            
                            # Remove the weaker association
                            del classical_associations[key]
                            
                            consolidations.append({
                                "type": "classical",
                                "stimulus": stimulus,
                                "response": response,
                                "consolidated_key": key,
                                "into_key": strongest_key,
                                "new_strength": new_strength
                            })
        
        # Do similar consolidation for operant associations
        operant_associations = self.conditioning_system.operant_associations
        
        # Group by behavior
        behavior_groups = {}
        
        for key, association in operant_associations.items():
            behavior = association.stimulus  # The behavior is stored as stimulus
            
            if behavior not in behavior_groups:
                behavior_groups[behavior] = []
            
            behavior_groups[behavior].append((key, association))
        
        # Find groups with multiple associations for the same behavior
        for behavior, associations in behavior_groups.items():
            if len(associations) > 1:
                # Group by consequence type
                consequence_groups = {}
                
                for key, association in associations:
                    consequence = association.response  # The consequence is stored as response
                    
                    if consequence not in consequence_groups:
                        consequence_groups[consequence] = []
                    
                    consequence_groups[consequence].append((key, association))
                
                # Consolidate associations with the same consequence
                for consequence, consequence_associations in consequence_groups.items():
                    if len(consequence_associations) > 1:
                        # Get the strongest association
                        strongest = max(consequence_associations, key=lambda x: x[1].association_strength)
                        strongest_key, strongest_association = strongest
                        
                        # Consolidate other associations into the strongest
                        for key, association in consequence_associations:
                            if key != strongest_key:
                                # Strengthen the strongest association slightly
                                new_strength = min(1.0, strongest_association.association_strength + (association.association_strength * 0.2))
                                strongest_association.association_strength = new_strength
                                
                                # Combine context keys
                                for context_key in association.context_keys:
                                    if context_key not in strongest_association.context_keys:
                                        strongest_association.context_keys.append(context_key)
                                
                                # Remove the weaker association
                                del operant_associations[key]
                                
                                consolidations.append({
                                    "type": "operant",
                                    "behavior": behavior,
                                    "consequence": consequence,
                                    "consolidated_key": key,
                                    "into_key": strongest_key,
                                    "new_strength": new_strength
                                })
        
        return {
            "consolidations": consolidations,
            "total_count": len(consolidations)
        }
    
    async def get_maintenance_stats(self) -> Dict[str, Any]:
        """Get maintenance statistics"""
        return {
            "last_maintenance_time": self.last_maintenance_time.isoformat() if self.last_maintenance_time else None,
            "maintenance_count": len(self.maintenance_history),
            "maintenance_interval_hours": self.maintenance_interval_hours,
            "extinction_threshold": self.extinction_threshold,
            "reinforcement_threshold": self.reinforcement_threshold,
            "consolidation_interval_days": self.consolidation_interval_days,
            "task_running": self.maintenance_task is not None,
            "recent_history": self.maintenance_history[-5:] if self.maintenance_history else []
        }
