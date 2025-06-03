# nyx/core/conditioning_maintenance.py

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional

from agents import Agent, Runner, ModelSettings, trace, handoff
from nyx.core.conditioning_models import *
from nyx.core.conditioning_tools import *

logger = logging.getLogger(__name__)

# Maintenance-specific tools

@function_tool
async def analyze_trait_distribution(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Analyze the distribution of personality traits"""
    trait_counts = {}
    trait_strengths = {}
    
    # Count trait references in associations
    all_associations = list(ctx.context.conditioning_system.classical_associations.values()) + \
                      list(ctx.context.conditioning_system.operant_associations.values())
    
    trait_names = ["dominance", "playfulness", "strictness", "creativity", "intensity", "patience"]
    
    for assoc in all_associations:
        for trait in trait_names:
            if trait in assoc.stimulus.lower() or trait in assoc.response.lower():
                if trait not in trait_counts:
                    trait_counts[trait] = 0
                    trait_strengths[trait] = []
                trait_counts[trait] += 1
                trait_strengths[trait].append(assoc.association_strength)
    
    # Calculate averages
    average_strengths = {}
    for trait, strengths in trait_strengths.items():
        average_strengths[trait] = sum(strengths) / len(strengths) if strengths else 0.0
    
    return {
        "trait_counts": trait_counts,
        "average_strengths": average_strengths,
        "total_traits": len(trait_counts)
    }

@function_tool
async def identify_extinction_candidates(ctx: RunContextWrapper) -> List[Dict[str, Any]]:
    """Identify associations that are candidates for extinction"""
    candidates = []
    threshold = ctx.context.extinction_threshold
    
    # Check both association types
    for assoc_type, associations in [
        ("classical", ctx.context.conditioning_system.classical_associations),
        ("operant", ctx.context.conditioning_system.operant_associations)
    ]:
        for key, assoc in associations.items():
            if assoc.association_strength < threshold:
                candidates.append({
                    "association_key": key,
                    "association_type": assoc_type,
                    "strength": assoc.association_strength,
                    "reason": "strength_below_threshold"
                })
    
    return candidates

@function_tool
async def apply_extinction_to_association(ctx: RunContextWrapper, association_key: str, association_type: str) -> Dict[str, Any]:
    """Apply extinction to a specific association"""
    try:
        result = await ctx.context.conditioning_system.apply_extinction(association_key, association_type)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

@function_tool
async def reinforce_core_trait(ctx: RunContextWrapper, trait: str, adjustment: float) -> Dict[str, Any]:
    """Reinforce a core personality trait"""
    try:
        result = await ctx.context.conditioning_system.condition_personality_trait(
            trait=trait,
            target_value=adjustment,
            context={"source": "maintenance_reinforcement"}
        )
        return {
            "success": True,
            "trait": trait,
            "adjustment": adjustment,
            "result": result
        }
    except Exception as e:
        return {"success": False, "trait": trait, "error": str(e)}

@function_tool
async def get_maintenance_status(ctx: RunContextWrapper) -> Dict[str, Any]:
    """Get the current status of maintenance"""
    now = datetime.datetime.now()
    
    if ctx.context.last_maintenance_time:
        delta = now - ctx.context.last_maintenance_time
        hours_since_last = delta.total_seconds() / 3600
        days_since_last = hours_since_last / 24
        next_maintenance = ctx.context.last_maintenance_time + \
                          datetime.timedelta(hours=ctx.context.maintenance_interval_hours)
        hours_until_next = max(0, (next_maintenance - now).total_seconds() / 3600)
    else:
        hours_since_last = None
        days_since_last = None
        hours_until_next = 0
    
    maintenance_due = hours_until_next <= 0 or ctx.context.last_maintenance_time is None
    consolidation_due = (days_since_last is not None and 
                        days_since_last >= ctx.context.consolidation_interval_days) or \
                       ctx.context.last_maintenance_time is None
    
    return {
        "last_maintenance_time": ctx.context.last_maintenance_time.isoformat() if ctx.context.last_maintenance_time else None,
        "hours_since_last_maintenance": hours_since_last,
        "maintenance_due": maintenance_due,
        "consolidation_due": consolidation_due,
        "maintenance_interval_hours": ctx.context.maintenance_interval_hours
    }

@function_tool
async def record_maintenance_history(ctx: RunContextWrapper, maintenance_record: Dict[str, Any]) -> Dict[str, Any]:
    """Record maintenance history"""
    try:
        if "timestamp" not in maintenance_record:
            maintenance_record["timestamp"] = datetime.datetime.now().isoformat()
        
        ctx.context.maintenance_history.append(maintenance_record)
        
        if len(ctx.context.maintenance_history) > ctx.context.max_history_entries:
            ctx.context.maintenance_history = ctx.context.maintenance_history[-ctx.context.max_history_entries:]
        
        return {
            "success": True,
            "history_count": len(ctx.context.maintenance_history)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

class ConditioningMaintenanceSystem:
    """Handles periodic maintenance tasks for the conditioning system"""
    
    def __init__(self, conditioning_system, reward_system=None):
        self.context = MaintenanceContext(conditioning_system, reward_system)
        self._create_agents()
        logger.info("Conditioning maintenance system initialized")
    
    def _create_agents(self):
        """Create maintenance agents"""
        
        # Balance Analysis Agent
        self.balance_analysis_agent = Agent(
            name="Balance_Analyzer",
            instructions="""
            Analyze personality trait and behavior balance.
            Identify imbalances and recommend adjustments.
            Output a BalanceAnalysisOutput with your findings.
            """,
            tools=[
                analyze_trait_distribution,
                check_trait_balance
            ],
            output_type=BalanceAnalysisOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.2)
        )
        
        # Association Maintenance Agent
        self.association_maintenance_agent = Agent(
            name="Association_Maintainer",
            instructions="""
            Maintain conditioning associations.
            Apply extinction to weak associations.
            When using apply_extinction_to_association:
            - Provide association_key (string)
            - Provide association_type ('classical' or 'operant')
            """,
            tools=[
                identify_extinction_candidates,
                apply_extinction_to_association
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
        
        # Trait Maintenance Agent
        self.trait_maintenance_agent = Agent(
            name="Trait_Maintainer",
            instructions="""
            Maintain and reinforce personality traits.
            Apply appropriate adjustments to core traits.
            """,
            tools=[
                reinforce_core_trait
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
        
        # Maintenance Orchestrator
        self.maintenance_orchestrator = Agent(
            name="Maintenance_Orchestrator",
            instructions="""
            Orchestrate maintenance processes.
            Check status, run maintenance tasks via handoffs,
            and produce a MaintenanceSummaryOutput.
            
            You must populate all fields in the output:
            - tasks_performed
            - time_taken_seconds
            - associations_modified
            - traits_adjusted
            - extinction_count
            - improvements
            - next_maintenance_recommendation
            """,
            handoffs=[
                handoff(self.balance_analysis_agent,
                       tool_name_override="analyze_balance"),
                handoff(self.association_maintenance_agent,
                       tool_name_override="maintain_associations"),
                handoff(self.trait_maintenance_agent,
                       tool_name_override="maintain_traits")
            ],
            tools=[
                get_maintenance_status,
                record_maintenance_history
            ],
            output_type=MaintenanceSummaryOutput,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.3)
        )
    
    async def start_maintenance_scheduler(self, run_immediately=False):
        """Start the periodic maintenance scheduler"""
        if self.context.maintenance_task is not None:
            logger.warning("Maintenance scheduler already running")
            return
        
        self.context.maintenance_task = asyncio.create_task(
            self._maintenance_loop(run_immediately)
        )
        logger.info(f"Maintenance scheduler started")
    
    async def stop_maintenance_scheduler(self):
        """Stop the periodic maintenance scheduler"""
        if self.context.maintenance_task is None:
            return
        
        self.context.maintenance_task.cancel()
        try:
            await self.context.maintenance_task
        except asyncio.CancelledError:
            pass
        self.context.maintenance_task = None
        logger.info("Maintenance scheduler stopped")
    
    async def _maintenance_loop(self, run_immediately=False):
        """Internal maintenance loop"""
        try:
            if run_immediately:
                await self.run_maintenance()
            
            while True:
                await asyncio.sleep(self.context.maintenance_interval_hours * 3600)
                await self.run_maintenance()
        except asyncio.CancelledError:
            raise
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """Run maintenance on the conditioning system"""
        start_time = datetime.datetime.now()
        
        with trace(workflow_name="conditioning_maintenance", group_id=self.context.trace_group_id):
            try:
                # Run orchestrator
                result = await Runner.run(
                    self.maintenance_orchestrator,
                    json.dumps({"action": "perform_maintenance"}),
                    context=self.context
                )
                
                summary = result.final_output
                duration = (datetime.datetime.now() - start_time).total_seconds()
                
                maintenance_record = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "duration_seconds": duration,
                    "status": "completed",
                    "tasks_performed": summary.tasks_performed,
                    "associations_modified": summary.associations_modified,
                    "traits_adjusted": summary.traits_adjusted,
                    "extinction_count": summary.extinction_count,
                    "improvements": summary.improvements,
                    "next_recommendation": summary.next_maintenance_recommendation
                }
                
                self.context.last_maintenance_time = datetime.datetime.now()
                
                # Record history
                await record_maintenance_history(
                    RunContextWrapper(context=self.context),
                    maintenance_record
                )
                
                return maintenance_record
                
            except Exception as e:
                logger.error(f"Error in maintenance: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "duration_seconds": (datetime.datetime.now() - start_time).total_seconds()
                }
    
    async def get_maintenance_stats(self) -> Dict[str, Any]:
        """Get maintenance statistics"""
        ctx_wrapper = RunContextWrapper(context=self.context)
        status = await get_maintenance_status(ctx_wrapper)
        
        return {
            "last_maintenance_time": status.get("last_maintenance_time"),
            "maintenance_count": len(self.context.maintenance_history),
            "task_running": self.context.maintenance_task is not None,
            "recent_history": self.context.maintenance_history[-5:],
            "maintenance_due": status.get("maintenance_due", False)
        }
