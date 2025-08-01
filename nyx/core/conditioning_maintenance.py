# nyx/core/conditioning_maintenance.py

import asyncio
import datetime
import logging
from typing import Dict, List, Any, Optional, Literal

from agents import Agent, Runner, ModelSettings, trace, handoff
from nyx.core.conditioning_models import *
from nyx.core.conditioning_tools import *

logger = logging.getLogger(__name__)

# Maintenance-specific tools

class RecordHistoryParams(BaseModel):
    """JSON string with a full maintenance-record object"""
    maintenance_record_json: str

class TraitStat(BaseModel):
    name: str
    count: int
    average_strength: float

class TraitDistributionResult(BaseModel):
    traits: List[TraitStat]
    total_traits: int

class ExtinctionCandidate(BaseModel):
    association_key: str
    association_type: Literal["classical", "operant"]
    strength: float
    reason: str

class ExtinctionCandidatesResult(BaseModel):
    candidates: List[ExtinctionCandidate]
    total: int

class ApplyExtinctionResult(BaseModel):
    success: bool
    association_key: str
    association_type: Literal["classical", "operant"]
    old_strength: Optional[float] = None
    new_strength: Optional[float] = None
    error: Optional[str] = None
    message: Optional[str] = None

class ReinforceTraitResult(BaseModel):
    success: bool
    trait: str
    adjustment: float
    new_value: Optional[float] = None
    error: Optional[str] = None

class MaintenanceStatusResult(BaseModel):
    last_maintenance_time: Optional[str] = None
    hours_since_last_maintenance: Optional[float] = None
    hours_until_next: Optional[float] = None
    maintenance_due: bool
    consolidation_due: bool
    maintenance_interval_hours: float

class RecordHistoryResult(BaseModel):
    success: bool
    history_count: int
    error: Optional[str] = None

# Models for maintenance history records
class MaintenanceRecord(BaseModel):
    """Explicit model for maintenance history records"""
    timestamp: str
    duration_seconds: float
    status: str
    tasks_performed: List[MaintenanceTask]
    associations_modified: int
    traits_adjusted: int
    extinction_count: int
    improvements: List[str]
    next_recommendation: str


# ──────────────────────────────────────────────────────────────────────────────
#  Helper – convert dynamic dicts → strict lists
# ──────────────────────────────────────────────────────────────────────────────
def _stats_from_maps(counts: Dict[str, int],
                     avgs: Dict[str, float]) -> List[TraitStat]:
    """Combine the two loose maps into a list of TraitStat models."""
    out: list[TraitStat] = []
    for name, cnt in counts.items():
        out.append(
            TraitStat(
                name=name,
                count=cnt,
                average_strength=avgs.get(name, 0.0),
            )
        )
    return out



@function_tool
async def analyze_trait_distribution(
    ctx: RunContextWrapper,
) -> TraitDistributionResult:
    """Strict version – analyse trait distribution."""
    trait_counts: Dict[str, int] = {}
    trait_strengths: Dict[str, list[float]] = {}

    all_associations = (
        list(ctx.context.conditioning_system.classical_associations.values())
        + list(ctx.context.conditioning_system.operant_associations.values())
    )

    trait_names = [
        "dominance",
        "playfulness",
        "strictness",
        "creativity",
        "intensity",
        "patience",
    ]
    for assoc in all_associations:
        for tr in trait_names:
            if tr in assoc.stimulus.lower() or tr in assoc.response.lower():
                trait_counts.setdefault(tr, 0)
                trait_strengths.setdefault(tr, [])
                trait_counts[tr] += 1
                trait_strengths[tr].append(assoc.association_strength)

    average_strengths = {
        tr: (sum(vals) / len(vals) if vals else 0.0)
        for tr, vals in trait_strengths.items()
    }

    return TraitDistributionResult(
        traits=_stats_from_maps(trait_counts, average_strengths),
        total_traits=len(trait_counts),
    )

@function_tool
async def identify_extinction_candidates(
    ctx: RunContextWrapper,
) -> ExtinctionCandidatesResult:
    """Strict version – list extinction candidates."""
    threshold = ctx.context.extinction_threshold
    cand: list[ExtinctionCandidate] = []

    for assoc_type, mapping in [
        ("classical", ctx.context.conditioning_system.classical_associations),
        ("operant", ctx.context.conditioning_system.operant_associations),
    ]:
        for key, assoc in mapping.items():
            if assoc.association_strength < threshold:
                cand.append(
                    ExtinctionCandidate(
                        association_key=key,
                        association_type=assoc_type,  # type: ignore[arg-type]
                        strength=assoc.association_strength,
                        reason="strength_below_threshold",
                    )
                )

    return ExtinctionCandidatesResult(candidates=cand, total=len(cand))

@function_tool
async def apply_extinction_to_association(                # noqa: N802
    ctx: RunContextWrapper,
    association_key: str,
    association_type: Literal["classical", "operant"],
) -> ApplyExtinctionResult:
    """Strict wrapper around conditioning_system.apply_extinction()."""
    try:
        res = await ctx.context.conditioning_system.apply_extinction(
            association_key, association_type
        )
        return ApplyExtinctionResult(
            success=res.get("success", False),
            association_key=association_key,
            association_type=association_type,
            old_strength=res.get("old_strength"),
            new_strength=res.get("new_strength"),
            message=res.get("message"),
        )
    except Exception as exc:
        return ApplyExtinctionResult(
            success=False,
            association_key=association_key,
            association_type=association_type,
            error=str(exc),
        )


@function_tool
async def reinforce_core_trait(
    ctx: RunContextWrapper,
    trait: str,
    adjustment: float,
) -> ReinforceTraitResult:
    """Strict wrapper for trait reinforcement."""
    try:
        res = await ctx.context.conditioning_system.condition_personality_trait(
            trait=trait,
            target_value=adjustment,
            context={"source": "maintenance_reinforcement"},
        )
        return ReinforceTraitResult(
            success=True,
            trait=trait,
            adjustment=adjustment,
            new_value=res.get("new_value") if isinstance(res, dict) else None,
        )
    except Exception as exc:
        return ReinforceTraitResult(
            success=False,
            trait=trait,
            adjustment=adjustment,
            error=str(exc),
        )

@function_tool
async def get_maintenance_status(
    ctx: RunContextWrapper,
) -> MaintenanceStatusResult:
    """Strict maintenance status."""
    now = datetime.datetime.now()

    if ctx.context.last_maintenance_time:
        delta = now - ctx.context.last_maintenance_time
        hours_since_last = delta.total_seconds() / 3600
        next_maintenance = ctx.context.last_maintenance_time + datetime.timedelta(
            hours=ctx.context.maintenance_interval_hours
        )
        hours_until_next = max(
            0.0, (next_maintenance - now).total_seconds() / 3600
        )
    else:
        hours_since_last = None
        hours_until_next = 0.0

    maintenance_due = hours_until_next <= 0 or ctx.context.last_maintenance_time is None
    consolidation_due = (
        hours_since_last is not None
        and hours_since_last / 24 >= ctx.context.consolidation_interval_days
    ) or ctx.context.last_maintenance_time is None

    return MaintenanceStatusResult(
        last_maintenance_time=(
            ctx.context.last_maintenance_time.isoformat()
            if ctx.context.last_maintenance_time
            else None
        ),
        hours_since_last_maintenance=hours_since_last,
        hours_until_next=hours_until_next,
        maintenance_due=maintenance_due,
        consolidation_due=consolidation_due,
        maintenance_interval_hours=ctx.context.maintenance_interval_hours,
    )

@function_tool
async def record_maintenance_history(                    # noqa: N802
    ctx: RunContextWrapper,
    params: RecordHistoryParams,                         # ← strict wrapper
) -> RecordHistoryResult:
    """Strict recorder – accepts JSON string."""
    import json, datetime

    # ① decode
    try:
        rec = json.loads(params.maintenance_record_json or "{}")
        if not isinstance(rec, dict):
            raise TypeError("maintenance_record_json must decode to an object")
        
        # Validate that it contains the expected structure
        # Convert to MaintenanceRecord and back to ensure compatibility
        # This validates the structure without storing the model
        _ = MaintenanceRecord(**rec)
        
    except Exception as exc:
        return RecordHistoryResult(
            success=False,
            history_count=len(ctx.context.maintenance_history),
            error=f"Bad JSON or invalid structure: {exc}",
        )

    # ② augment & store
    rec.setdefault("timestamp", datetime.datetime.now().isoformat())

    ctx.context.maintenance_history.append(rec)
    if len(ctx.context.maintenance_history) > ctx.context.max_history_entries:
        ctx.context.maintenance_history = ctx.context.maintenance_history[
            -ctx.context.max_history_entries :
        ]

    # ③ done
    return RecordHistoryResult(
        success=True, history_count=len(ctx.context.maintenance_history)
    )
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
                
                # Create structured maintenance record
                maintenance_record = MaintenanceRecord(
                    timestamp=datetime.datetime.now().isoformat(),
                    duration_seconds=duration,
                    status="completed",
                    tasks_performed=summary.tasks_performed,
                    associations_modified=summary.associations_modified,
                    traits_adjusted=summary.traits_adjusted,
                    extinction_count=summary.extinction_count,
                    improvements=summary.improvements,
                    next_recommendation=summary.next_maintenance_recommendation
                )
                
                self.context.last_maintenance_time = datetime.datetime.now()
                
                # Record history using JSON string
                await record_maintenance_history(
                    RunContextWrapper(context=self.context),
                    RecordHistoryParams(
                        maintenance_record_json=maintenance_record.model_dump_json()
                    )
                )
                
                return maintenance_record.model_dump()
                
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
            "last_maintenance_time": status.last_maintenance_time,
            "maintenance_count": len(self.context.maintenance_history),
            "task_running": self.context.maintenance_task is not None,
            "recent_history": self.context.maintenance_history[-5:],
            "maintenance_due": status.maintenance_due
        }
