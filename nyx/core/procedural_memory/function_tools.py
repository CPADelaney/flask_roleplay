# nyx/core/procedural_memory/function_tools.py

import asyncio
import datetime
import logging
import random
import json
import uuid
from typing import List, Any, Optional, Tuple, Union

# OpenAI Agents SDK imports
from agents import Agent, Runner, trace, function_tool, RunContextWrapper, ModelSettings, handoff, custom_span, FunctionTool
from agents.exceptions import UserError

from .models import Procedure, ProcedureStats, TransferStats, ProcedureTransferRecord

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)

# ===== PYDANTIC MODELS FOR ALL INPUTS AND OUTPUTS =====

class ProcedureStep(BaseModel):
    """Model for a single procedure step"""
    id: str
    function: str
    parameters: dict = Field(default_factory=dict)  # Using lowercase dict is fine in Pydantic
    description: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class ExecutionContext(BaseModel):
    """Model for procedure execution context"""
    model_config = ConfigDict(extra="allow")

class StepParameters(BaseModel):
    """Model for step parameters in refinement"""
    model_config = ConfigDict(extra="allow")

# Return type models
class AddProcedureResponse(BaseModel):
    """Response model for add_procedure"""
    success: bool
    procedure_id: Optional[str] = None
    name: Optional[str] = None
    domain: Optional[str] = None
    steps_count: Optional[int] = None
    error: Optional[str] = None

class ExecuteProcedureResponse(BaseModel):
    """Response model for execute_procedure"""
    success: bool
    execution_time: float
    error: Optional[str] = None
    strategy: Optional[str] = None
    results: List[Any] = Field(default_factory=list)

class TransferProcedureResponse(BaseModel):
    """Response model for transfer_procedure"""
    success: bool
    source_name: str
    target_name: str
    source_domain: str
    target_domain: str
    steps_count: int
    procedure_id: str
    error: Optional[str] = None

class ProcedureSummary(BaseModel):
    """Model for procedure summary in list_procedures"""
    name: str
    id: str
    description: str
    domain: str
    proficiency: float
    proficiency_level: str
    steps_count: int
    execution_count: int
    is_chunked: bool
    created_at: str
    last_updated: str
    last_execution: Optional[str] = None

class ChunkingOpportunityResponse(BaseModel):
    """Response model for identify_chunking_opportunities"""
    can_chunk: bool
    procedure_name: str
    potential_chunks: Optional[List[List[str]]] = None
    chunk_count: Optional[int] = None
    reason: Optional[str] = None

class ApplyChunkingResponse(BaseModel):
    """Response model for apply_chunking"""
    success: bool
    chunks_applied: Optional[int] = None
    chunk_ids: Optional[List[str]] = None
    procedure_name: str
    error: Optional[str] = None

class GeneralizeChunkResponse(BaseModel):
    """Response model for generalize_chunk_from_steps"""
    success: bool
    template_id: Optional[str] = None
    name: Optional[str] = None
    domain: Optional[str] = None
    actions_count: Optional[int] = None
    can_transfer: Optional[bool] = None
    error: Optional[str] = None

class ChunkMatch(BaseModel):
    """Model for chunk match in find_matching_chunks"""
    template_id: str
    similarity: float
    domain: str
    name: str

class FindMatchingChunksResponse(BaseModel):
    """Response model for find_matching_chunks"""
    success: bool
    chunks_found: bool
    matches: Optional[List[ChunkMatch]] = None
    count: Optional[int] = None
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

class TransferChunkResponse(BaseModel):
    """Response model for transfer_chunk"""
    success: bool
    steps: Optional[List[dict]] = None
    steps_count: Optional[int] = None
    target_domain: Optional[str] = None
    error: Optional[str] = None

class ChunkTransferDetail(BaseModel):
    """Model for chunk transfer details"""
    chunk_id: str
    template_id: str
    steps_count: int
    newly_created: Optional[bool] = None

class TransferWithChunkingResponse(BaseModel):
    """Response model for transfer_with_chunking"""
    success: bool
    source_name: str
    target_name: str
    source_domain: str
    target_domain: str
    steps_count: int
    chunks_transferred: Optional[int] = None
    procedure_id: str
    chunk_transfer_details: Optional[List[ChunkTransferDetail]] = None
    error: Optional[str] = None

class SimilarProcedure(BaseModel):
    """Model for similar procedure in find_similar_procedures"""
    name: str
    id: str
    domain: str
    similarity: float
    steps_count: int
    proficiency: float

class RefineStepResponse(BaseModel):
    """Response model for refine_step"""
    success: bool
    procedure_name: Optional[str] = None
    step_id: Optional[str] = None
    function_updated: Optional[bool] = None
    parameters_updated: Optional[bool] = None
    description_updated: Optional[bool] = None
    chunking_reset: Optional[bool] = None
    error: Optional[str] = None

# ===== FUNCTION TOOLS WITH UPDATED SIGNATURES =====

@function_tool(strict_mode=False)  # Disable strict schema for this function
async def add_procedure(
    ctx: RunContextWrapper[Any],
    name: str,
    steps_json: Union[str, List[ProcedureStep]],  # Already using Pydantic model
    description: Optional[str] = None,
    domain: Optional[str] = None,
) -> AddProcedureResponse:  # Changed return type
    # Handle both string and list inputs
    if isinstance(steps_json, list):
        steps = [step.model_dump() if isinstance(step, BaseModel) else dict(step) for step in steps_json]
    elif isinstance(steps_json, str):
        logger.debug(f"ADD_PROCEDURE_DEBUG: Received steps_json of type: <class 'str'>")
        logger.debug(f"ADD_PROCEDURE_DEBUG: Length of steps_json string: {len(steps_json)}")
        logger.debug(f"ADD_PROCEDURE_DEBUG: repr(steps_json): {repr(steps_json)}")
        
        try:
            # Attempt to parse the string as JSON
            # Generalize the fix for "Extra data" errors
            try:
                steps = json.loads(steps_json)
            except json.JSONDecodeError as e_inner:
                if "Extra data" in str(e_inner) and e_inner.pos < len(steps_json):
                    # This means valid JSON was parsed up to e_inner.pos,
                    # and there was garbage data afterwards.
                    logger.warning(
                        f"ADD_PROCEDURE_WARN: JSONDecodeError with extra data. "
                        f"Original len: {len(steps_json)}, error char index: {e_inner.pos}. "
                        f"Attempting to parse substring up to error index."
                    )
                    valid_json_part = steps_json[:e_inner.pos]
                    steps = json.loads(valid_json_part) # Try parsing just the valid part
                    logger.info(
                        f"ADD_PROCEDURE_INFO: Successfully parsed substring of length {len(valid_json_part)}."
                    )
                else:
                    # Not an "Extra data" error, or e.inner.pos is not useful, re-raise.
                    raise e_inner 

            if not isinstance(steps, list):
                # This check is important if parsing was successful but didn't yield a list
                logger.error(f"ADD_PROCEDURE_ERROR: Decoded JSON is not a list of steps. Type: {type(steps)}")
                raise ValueError("Decoded JSON is not a list of steps.")

        except Exception as e: # Catch parsing errors (original, from substring, or ValueError)
            logger.error(f"ADD_PROCEDURE_ERROR: Failed to parse steps_json string. Error: {e}")
            logger.error(f"ADD_PROCEDURE_ERROR: Problematic steps_json (len {len(steps_json)}): {repr(steps_json)}")
            return AddProcedureResponse(error=f"Invalid steps_json string: {e}", success=False)
    else:
        logger.error(f"ADD_PROCEDURE_ERROR: steps_json is of unexpected type: {type(steps_json)}")
        return AddProcedureResponse(error=f"steps_json is of unexpected type: {type(steps_json)}", success=False)

    # ---------- 2️⃣  Validate / enrich each step ----------
    # (This part of your code seems fine)
    for i, step_data in enumerate(steps): # Renamed 'step' to 'step_data' to avoid confusion
        if not isinstance(step_data, dict):
            logger.error(f"ADD_PROCEDURE_ERROR: Step {i+1} is not an object (dictionary). Found: {type(step_data)}")
            return AddProcedureResponse(error=f"Step {i+1} is not an object (dictionary).", success=False)

        step_data.setdefault("id", f"step_{i+1}")

        if "function" not in step_data or not step_data["function"]:
            logger.error(f"ADD_PROCEDURE_ERROR: Step {step_data['id']} is missing the required 'function' field.")
            return AddProcedureResponse(
                error=f"Step {step_data['id']} is missing the required 'function' field.",
                success=False,
            )
        step_data.setdefault("parameters", {})

    # ---------- 3️⃣  House‑keeping ----------
    # (This part of your code seems fine)
    domain = domain or "general"
    manager = ctx.context.manager # Assuming AgentContext has a 'manager' attribute
    procedure_id = f"proc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"

    # ---------- 4️⃣  Register any *callable* step.functions ---------
    # (This part of your code seems fine)
    for step_data in steps:
        fn = step_data["function"]
        if callable(fn):
            fn_name = fn.__name__
            if hasattr(ctx.context, 'register_function') and callable(ctx.context.register_function):
                 ctx.context.register_function(fn_name, fn)
            elif hasattr(manager, 'register_function') and callable(manager.register_function): # Fallback if register_function is on manager
                 manager.register_function(fn_name, fn)
            else:
                logger.warning(f"Could not register callable function {fn_name} as register_function not found on context or manager.")
            step_data["function"] = fn_name

    # ---------- 5️⃣  Persist ----------
    # (This part of your code seems fine)
    procedure = Procedure(
        id=procedure_id,
        name=name,
        description=description or f"Procedure for {name}",
        domain=domain,
        steps=steps, # Use the processed 'steps' list
        created_at=datetime.datetime.now().isoformat(),
        last_updated=datetime.datetime.now().isoformat(),
    )

    manager.procedures[name] = procedure
    logger.info(f"Added procedure '{name}' ({len(steps)} steps) in domain '{domain}') with ID {procedure_id}")

    return AddProcedureResponse(
        success=True,
        procedure_id=procedure_id,
        name=name,
        domain=domain,
        steps_count=len(steps),
    )


@function_tool(use_docstring_info=True, strict_mode=False)  # Disable strict schema
async def execute_procedure(
    ctx: RunContextWrapper[Any],
    name: str,
    context: Optional[ExecutionContext] = None,  # Already using Pydantic model
    force_conscious: bool = False
) -> ExecuteProcedureResponse:  # Changed return type
    """
    Execute a procedure by name.
    
    Args:
        name: Name of the procedure to execute
        context: Context data for execution
        force_conscious: Force deliberate execution even if proficient
        
    Returns:
        Execution results including success status, execution time, and any output data
    """
    manager = ctx.context.manager
    
    if name not in manager.procedures:
        return ExecuteProcedureResponse(error=f"Procedure '{name}' not found", success=False, execution_time=0.0)
    
    procedure = manager.procedures[name]
    execution_context = context.model_dump() if context and isinstance(context, BaseModel) else (context or {})
    
    # Create trace span for execution
    with custom_span(
        name="execute_procedure", 
        data={
            "procedure_name": name,
            "procedure_id": procedure.id,
            "domain": procedure.domain,
            "steps_count": len(procedure.steps),
            "force_conscious": force_conscious
        }
    ):
        # Check for execution strategy selection
        strategy = "deliberate" if force_conscious else "automatic"
        if hasattr(manager, "strategy_selector") and manager.strategy_selector:
            if procedure.proficiency > 0.8 and not force_conscious:
                strategy = "automatic"
            elif procedure.proficiency < 0.5 or force_conscious:
                strategy = "deliberate"
            else:
                strategy = "adaptive"
        
        # Log the execution attempt
        logger.info(f"Executing procedure '{name}' using {strategy} strategy")
        
        # Execute the procedure
        start_time = datetime.datetime.now()
        try:
            result = await manager.execute_procedure_steps(
                procedure, 
                execution_context, 
                force_conscious=force_conscious
            )
            
            # Calculate execution time if not provided
            if "execution_time" not in result:
                execution_time = (datetime.datetime.now() - start_time).total_seconds()
                result["execution_time"] = execution_time
            
            # Record execution context and update stats
            if hasattr(procedure, "record_execution"):
                procedure.record_execution(
                    success=result.get("success", False),
                    execution_time=result["execution_time"],
                    context=execution_context
                )
            
            return ExecuteProcedureResponse(**result)
            
        except Exception as e:
            logger.error(f"Error executing procedure '{name}': {str(e)}")
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            return ExecuteProcedureResponse(
                success=False,
                execution_time=execution_time,
                error=str(e),
                strategy=strategy,
                results=[]
            )

@function_tool(use_docstring_info=True)
async def transfer_procedure(
    ctx: RunContextWrapper[Any],
    source_name: str,
    target_name: str,
    target_domain: str
) -> TransferProcedureResponse:  # Changed return type
    """
    Transfer a procedure from one domain to another.
    
    Args:
        source_name: Name of the source procedure
        target_name: Name for the new procedure
        target_domain: Domain for the new procedure
        
    Returns:
        Transfer results with adapted procedure details
    """
    manager = ctx.context.manager
    
    if source_name not in manager.procedures:
        return TransferProcedureResponse(
            success=False,
            error=f"Source procedure '{source_name}' not found",
            source_name=source_name,
            target_name=target_name,
            source_domain="unknown",
            target_domain=target_domain,
            steps_count=0,
            procedure_id=""
        )
    
    source = manager.procedures[source_name]
    
    # Create trace span for the transfer operation
    with custom_span(
        name="transfer_procedure", 
        data={
            "source_name": source_name,
            "source_domain": source.domain,
            "target_name": target_name,
            "target_domain": target_domain
        }
    ):
        # Use the chunk library to map steps to the new domain
        mapped_steps = []
        
        for step in source.steps:
            # Try to map step to new domain
            mapped_step = await manager.map_step_to_domain(
                step=step,
                source_domain=source.domain,
                target_domain=target_domain
            )
            
            if mapped_step:
                mapped_steps.append(mapped_step)
        
        if not mapped_steps:
            return TransferProcedureResponse(
                success=False,
                error="Could not map any steps to the target domain",
                source_name=source_name,
                target_name=target_name,
                source_domain=source.domain,
                target_domain=target_domain,
                steps_count=0,
                procedure_id=""
            )
        
        # Create new procedure
        import json
        new_procedure = await add_procedure(
            ctx,
            name=target_name,
            steps_json=json.dumps(mapped_steps),
            description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
            domain=target_domain
        )
        
        if not new_procedure.success:
            return TransferProcedureResponse(
                success=False,
                error=new_procedure.error or "Failed to create new procedure",
                source_name=source_name,
                target_name=target_name,
                source_domain=source.domain,
                target_domain=target_domain,
                steps_count=len(mapped_steps),
                procedure_id=""
            )
        
        # Record transfer
        transfer_record = ProcedureTransferRecord(
            source_procedure_id=source.id,
            source_domain=source.domain,
            target_procedure_id=new_procedure.procedure_id,
            target_domain=target_domain,
            transfer_date=datetime.datetime.now().isoformat(),
            success_level=0.8,  # Initial estimate
            practice_needed=5  # Initial estimate
        )
        
        if hasattr(manager, "chunk_library") and manager.chunk_library:
            manager.chunk_library.record_transfer(transfer_record)
        
        # Update transfer stats
        if hasattr(manager, "transfer_stats"):
            manager.transfer_stats["total_transfers"] += 1
            manager.transfer_stats["successful_transfers"] += 1
        
        return TransferProcedureResponse(
            success=True,
            source_name=source_name,
            target_name=target_name,
            source_domain=source.domain,
            target_domain=target_domain,
            steps_count=len(mapped_steps),
            procedure_id=new_procedure.procedure_id
        )

@function_tool(use_docstring_info=True)
async def get_procedure_proficiency(
    ctx: RunContextWrapper[Any], 
    name: str
) -> ProcedureStats:  # Already returns a Pydantic model
    """
    Get the current proficiency level for a procedure.
    
    Args:
        name: Name of the procedure
        
    Returns:
        Detailed statistics about the procedure including proficiency level and execution history
    """
    manager = ctx.context.manager
    
    if name not in manager.procedures:
        raise UserError(f"Procedure '{name}' not found")
    
    procedure = manager.procedures[name]
    
    # Create trace span for the operation
    with custom_span(
        name="get_procedure_proficiency", 
        data={
            "procedure_name": name,
            "procedure_id": procedure.id,
            "domain": procedure.domain
        }
    ):
        # Determine proficiency level
        proficiency_level = "novice"
        if procedure.proficiency >= 0.95:
            proficiency_level = "automatic"
        elif procedure.proficiency >= 0.8:
            proficiency_level = "expert"
        elif procedure.proficiency >= 0.5:
            proficiency_level = "competent"
        
        # Calculate success rate
        success_rate = procedure.successful_executions / max(1, procedure.execution_count)
        
        return ProcedureStats(
            procedure_name=name,
            procedure_id=procedure.id,
            proficiency=procedure.proficiency,
            level=proficiency_level,
            execution_count=procedure.execution_count,
            success_rate=success_rate,
            average_execution_time=procedure.average_execution_time,
            is_chunked=procedure.is_chunked,
            chunks_count=len(procedure.chunked_steps) if procedure.is_chunked else 0,
            steps_count=len(procedure.steps),
            last_execution=procedure.last_execution,
            domain=procedure.domain,
            generalized_chunks=len(procedure.generalized_chunks) if hasattr(procedure, "generalized_chunks") else 0,
            refinement_opportunities=len(procedure.refinement_opportunities) if hasattr(procedure, "refinement_opportunities") else 0
        )

@function_tool(use_docstring_info=True)
async def list_procedures(
    ctx: RunContextWrapper[Any],
    domain: Optional[str] = None
) -> List[ProcedureSummary]:  # Changed return type
    """
    List all procedures, optionally filtered by domain.
    
    Args:
        domain: Optional domain to filter by
        
    Returns:
        List of procedure summaries with key information
    """
    manager = ctx.context.manager
    
    # Create trace span for the operation
    with custom_span(
        name="list_procedures", 
        data={
            "domain_filter": domain,
            "procedures_count": len(manager.procedures)
        }
    ):
        procedure_list = []
        
        for name, procedure in manager.procedures.items():
            # Filter by domain if specified
            if domain and procedure.domain != domain:
                continue
                
            # Get proficiency level
            proficiency_level = "novice"
            if procedure.proficiency >= 0.95:
                proficiency_level = "automatic"
            elif procedure.proficiency >= 0.8:
                proficiency_level = "expert"
            elif procedure.proficiency >= 0.5:
                proficiency_level = "competent"
                
            # Create summary
            procedure_list.append(ProcedureSummary(
                name=name,
                id=procedure.id,
                description=procedure.description,
                domain=procedure.domain,
                proficiency=procedure.proficiency,
                proficiency_level=proficiency_level,
                steps_count=len(procedure.steps),
                execution_count=procedure.execution_count,
                is_chunked=procedure.is_chunked,
                created_at=procedure.created_at,
                last_updated=procedure.last_updated,
                last_execution=procedure.last_execution
            ))
            
        # Sort by domain and then name
        procedure_list.sort(key=lambda x: (x.domain, x.name))
        
        return procedure_list

@function_tool(use_docstring_info=True)
async def get_transfer_statistics(
    ctx: RunContextWrapper[Any]
) -> TransferStats:  # Already returns a Pydantic model
    """
    Get statistics about procedure transfers across domains.
    
    Returns:
        Comprehensive transfer statistics including success rates and domain coverage
    """
    manager = ctx.context.manager
    
    # Create trace span for the operation
    with custom_span(
        name="get_transfer_statistics", 
        data={"timestamp": datetime.datetime.now().isoformat()}
    ):
        # Get chunks by domain
        chunks_by_domain = {}
        if hasattr(manager, "chunk_library") and manager.chunk_library:
            for domain, chunk_ids in manager.chunk_library.domain_chunks.items():
                chunks_by_domain[domain] = len(chunk_ids)
        
        # Get recent transfers
        recent_transfers = []
        if hasattr(manager, "chunk_library") and manager.chunk_library:
            for record in manager.chunk_library.transfer_records[-5:]:
                recent_transfers.append({
                    "source_domain": record.source_domain,
                    "target_domain": record.target_domain,
                    "transfer_date": record.transfer_date,
                    "success_level": record.success_level
                })
        
        # Get transfer stats
        transfer_stats = getattr(manager, "transfer_stats", {
            "total_transfers": 0,
            "successful_transfers": 0,
            "avg_success_level": 0.0,
            "avg_practice_needed": 0
        })
        
        return TransferStats(
            total_transfers=transfer_stats.get("total_transfers", 0),
            successful_transfers=transfer_stats.get("successful_transfers", 0),
            avg_success_level=transfer_stats.get("avg_success_level", 0.0),
            avg_practice_needed=transfer_stats.get("avg_practice_needed", 0),
            chunks_by_domain=chunks_by_domain,
            recent_transfers=recent_transfers,
            templates_count=len(manager.chunk_library.chunk_templates) if hasattr(manager, "chunk_library") else 0,
            actions_count=len(manager.chunk_library.action_templates) if hasattr(manager, "chunk_library") else 0
        )

@function_tool(use_docstring_info=True)
async def identify_chunking_opportunities(
    ctx: RunContextWrapper[Any], 
    procedure_name: str
) -> ChunkingOpportunityResponse:  # Changed return type
    """
    Identify opportunities to chunk steps in a procedure.
    
    Args:
        procedure_name: Name of the procedure to analyze
        
    Returns:
        Analysis of potential chunks including which steps should be grouped together
    """
    manager = ctx.context.manager
    
    if procedure_name not in manager.procedures:
        raise UserError(f"Procedure '{procedure_name}' not found")
    
    procedure = manager.procedures[procedure_name]
    
    # Create trace span for the operation
    with custom_span(
        name="identify_chunking_opportunities", 
        data={
            "procedure_name": procedure_name,
            "procedure_id": procedure.id,
            "steps_count": len(procedure.steps),
            "execution_count": procedure.execution_count
        }
    ):
        # Need at least 3 steps and some executions to consider chunking
        if len(procedure.steps) < 3 or procedure.execution_count < 5:
            return ChunkingOpportunityResponse(
                can_chunk=False,
                procedure_name=procedure_name,
                reason=f"Need at least 3 steps and 5 executions (has {len(procedure.steps)} steps and {procedure.execution_count} executions)"
            )
        
        # Find sequences of steps that could be chunked
        chunks = []
        current_chunk = []
        
        for i in range(len(procedure.steps) - 1):
            # Start a new potential chunk
            if not current_chunk:
                current_chunk = [procedure.steps[i]["id"]]
            
            # Check if next step is consistently executed after this one
            co_occurrence = manager.calculate_step_co_occurrence(
                procedure,
                procedure.steps[i]["id"], 
                procedure.steps[i+1]["id"]
            )
            
            if co_occurrence > 0.8:  # High co-occurrence threshold
                # Add to current chunk
                current_chunk.append(procedure.steps[i+1]["id"])
            else:
                # End current chunk if it has multiple steps
                if len(current_chunk) > 1:
                    chunks.append(current_chunk)
                current_chunk = []
        
        # Add the last chunk if it exists
        if len(current_chunk) > 1:
            chunks.append(current_chunk)
        
        return ChunkingOpportunityResponse(
            can_chunk=len(chunks) > 0,
            potential_chunks=chunks,
            chunk_count=len(chunks),
            procedure_name=procedure_name
        )

@function_tool(use_docstring_info=True)
async def apply_chunking(
    ctx: RunContextWrapper[Any], 
    procedure_name: str
) -> ApplyChunkingResponse:  # Changed return type
    """
    Apply chunking to a procedure based on execution patterns.
    
    Args:
        procedure_name: Name of the procedure to chunk
        
    Returns:
        Results of the chunking process including new chunks created
    """
    manager = ctx.context.manager
    
    if procedure_name not in manager.procedures:
        raise UserError(f"Procedure '{procedure_name}' not found")
    
    procedure = manager.procedures[procedure_name]
    
    # Create trace span for the operation
    with custom_span(
        name="apply_chunking", 
        data={
            "procedure_name": procedure_name,
            "procedure_id": procedure.id,
            "steps_count": len(procedure.steps)
        }
    ):
        # Find chunking opportunities
        chunking_result = await identify_chunking_opportunities(ctx, procedure_name)
        
        if not chunking_result.can_chunk:
            return ApplyChunkingResponse(
                success=False,
                error=chunking_result.reason,
                procedure_name=procedure_name
            )
        
        # Apply chunks
        chunks = chunking_result.potential_chunks
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{i+1}"
            procedure.chunked_steps[chunk_id] = chunk
            
            # Look for context patterns if chunk selector is available
            if hasattr(manager, "chunk_selector") and manager.chunk_selector:
                context_pattern = manager.chunk_selector.create_context_pattern_from_history(
                    chunk_id=chunk_id,
                    domain=procedure.domain
                )
                
                if context_pattern:
                    # Store reference to context pattern
                    procedure.chunk_contexts[chunk_id] = context_pattern.id
        
        # Mark as chunked
        procedure.is_chunked = True
        procedure.last_updated = datetime.datetime.now().isoformat()
        
        return ApplyChunkingResponse(
            success=True,
            chunks_applied=len(chunks),
            chunk_ids=list(procedure.chunked_steps.keys()),
            procedure_name=procedure_name
        )

@function_tool(use_docstring_info=True)
async def generalize_chunk_from_steps(
    ctx: RunContextWrapper[Any],
    chunk_name: str,
    procedure_name: str,
    step_ids: List[str],
    domain: Optional[str] = None
) -> GeneralizeChunkResponse:  # Changed return type
    """
    Create a generalizable chunk template from specific procedure steps.
    
    Args:
        chunk_name: Name for the new chunk template
        procedure_name: Name of the procedure containing the steps
        step_ids: IDs of the steps to include in the chunk
        domain: Optional domain override (defaults to procedure's domain)
        
    Returns:
        Information about the created chunk template
    """
    manager = ctx.context.manager
    
    if procedure_name not in manager.procedures:
        return GeneralizeChunkResponse(error=f"Procedure '{procedure_name}' not found", success=False)
    
    procedure = manager.procedures[procedure_name]
    
    # Create trace span for the operation
    with custom_span(
        name="generalize_chunk", 
        data={
            "chunk_name": chunk_name,
            "procedure_name": procedure_name,
            "step_count": len(step_ids)
        }
    ):
        # Get the steps by ID
        steps = []
        for step_id in step_ids:
            step = next((s for s in procedure.steps if s["id"] == step_id), None)
            if step:
                steps.append(step)
        
        if not steps:
            return GeneralizeChunkResponse(error="No valid steps found", success=False)
        
        # Use the procedure's domain if not specified
        chunk_domain = domain or procedure.domain
        
        # Generate a unique template ID
        template_id = f"template_{chunk_name}_{int(datetime.datetime.now().timestamp())}"
        
        # Create the chunk template
        if not hasattr(manager, "chunk_library") or not manager.chunk_library:
            return GeneralizeChunkResponse(error="Chunk library not available", success=False)
            
        template = manager.chunk_library.create_chunk_template_from_steps(
            chunk_id=template_id,
            name=chunk_name,
            steps=steps,
            domain=chunk_domain
        )
        
        if not template:
            return GeneralizeChunkResponse(error="Failed to create chunk template", success=False)
        
        # If these steps were already part of a chunk, store the template reference
        for chunk_id, step_list in procedure.chunked_steps.items():
            if all(step_id in step_list for step_id in step_ids):
                # This existing chunk contains all our steps
                procedure.generalized_chunks[chunk_id] = template.id
                break
        
        return GeneralizeChunkResponse(
            success=True,
            template_id=template.id,
            name=template.name,
            domain=chunk_domain,
            actions_count=len(template.actions),
            can_transfer=True
        )

@function_tool(use_docstring_info=True)
async def find_matching_chunks(
    ctx: RunContextWrapper[Any],
    procedure_name: str,
    target_domain: str
) -> FindMatchingChunksResponse:  # Changed return type
    """
    Find library chunks that match a procedure's steps for transfer.
    
    Args:
        procedure_name: Name of the procedure to find chunks for
        target_domain: Target domain to transfer to
        
    Returns:
        List of matching chunks with similarity scores
    """
    manager = ctx.context.manager
    
    if procedure_name not in manager.procedures:
        return FindMatchingChunksResponse(error=f"Procedure '{procedure_name}' not found", success=False, chunks_found=False)
    
    procedure = manager.procedures[procedure_name]
    
    # Create trace span for the operation
    with custom_span(
        name="find_matching_chunks", 
        data={
            "procedure_name": procedure_name,
            "source_domain": procedure.domain,
            "target_domain": target_domain
        }
    ):
        # Ensure we have a chunk library
        if not hasattr(manager, "chunk_library") or not manager.chunk_library:
            return FindMatchingChunksResponse(
                chunks_found=False,
                message="Chunk library not available",
                success=False
            )
        
        # Find matching chunks
        matches = manager.chunk_library.find_matching_chunks(
            steps=procedure.steps,
            source_domain=procedure.domain,
            target_domain=target_domain
        )
        
        if not matches:
            return FindMatchingChunksResponse(
                chunks_found=False,
                message="No matching chunks found for transfer",
                success=True
            )
        
        # Convert matches to ChunkMatch models
        chunk_matches = [
            ChunkMatch(
                template_id=match["template_id"],
                similarity=match["similarity"],
                domain=match.get("domain", target_domain),
                name=match.get("name", "")
            ) for match in matches
        ]
        
        return FindMatchingChunksResponse(
            chunks_found=True,
            matches=chunk_matches,
            count=len(chunk_matches),
            source_domain=procedure.domain,
            target_domain=target_domain,
            success=True
        )

@function_tool(use_docstring_info=True)
async def transfer_chunk(
    ctx: RunContextWrapper[Any],
    template_id: str,
    target_domain: str
) -> TransferChunkResponse:  # Changed return type
    """
    Transfer a chunk template to a new domain.
    
    Args:
        template_id: ID of the chunk template to transfer
        target_domain: Domain to transfer to
        
    Returns:
        Mapped steps for the new domain
    """
    manager = ctx.context.manager
    
    # Create trace span for the operation
    with custom_span(
        name="transfer_chunk", 
        data={
            "template_id": template_id,
            "target_domain": target_domain
        }
    ):
        # Ensure we have a chunk library
        if not hasattr(manager, "chunk_library") or not manager.chunk_library:
            return TransferChunkResponse(
                success=False,
                error="Chunk library not available"
            )
        
        # Map the chunk to the new domain
        mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
            template_id=template_id,
            target_domain=target_domain
        )
        
        if not mapped_steps:
            return TransferChunkResponse(
                success=False,
                error="Failed to map chunk to new domain"
            )
        
        return TransferChunkResponse(
            success=True,
            steps=mapped_steps,
            steps_count=len(mapped_steps),
            target_domain=target_domain
        )

@function_tool(use_docstring_info=True)
async def transfer_with_chunking(
    ctx: RunContextWrapper[Any],
    source_name: str,
    target_name: str,
    target_domain: str
) -> TransferWithChunkingResponse:  # Changed return type
    """
    Transfer a procedure from one domain to another using chunk-level transfer.
    
    Args:
        source_name: Name of the source procedure
        target_name: Name for the new procedure
        target_domain: Domain for the new procedure
        
    Returns:
        Transfer results with chunking details
    """
    manager = ctx.context.manager
    
    if source_name not in manager.procedures:
        return TransferWithChunkingResponse(
            success=False,
            error=f"Source procedure '{source_name}' not found",
            source_name=source_name,
            target_name=target_name,
            source_domain="unknown",
            target_domain=target_domain,
            steps_count=0,
            procedure_id=""
        )
    
    source = manager.procedures[source_name]
    
    # Create trace span for the operation
    with custom_span(
        name="transfer_with_chunking", 
        data={
            "source_name": source_name,
            "source_domain": source.domain,
            "target_name": target_name,
            "target_domain": target_domain,
            "is_chunked": source.is_chunked
        }
    ):
        # Ensure we have a chunk library
        if not hasattr(manager, "chunk_library") or not manager.chunk_library:
            return TransferWithChunkingResponse(
                success=False,
                error="Chunk library not available",
                source_name=source_name,
                target_name=target_name,
                source_domain=source.domain,
                target_domain=target_domain,
                steps_count=0,
                procedure_id=""
            )
        
        # Get chunks from source if chunked
        steps_from_chunks = set()
        transferred_chunks = []
        all_steps = []
        
        if source.is_chunked:
            # Try to transfer each chunk
            for chunk_id, step_ids in source.chunked_steps.items():
                # Get chunk steps
                chunk_steps = [s for s in source.steps if s["id"] in step_ids]
                
                # Check if this chunk has a template
                if hasattr(source, "generalized_chunks") and chunk_id in source.generalized_chunks:
                    template_id = source.generalized_chunks[chunk_id]
                    
                    # Map template to new domain
                    mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
                        template_id=template_id,
                        target_domain=target_domain
                    )
                    
                    if mapped_steps:
                        # Add steps from this chunk
                        all_steps.extend(mapped_steps)
                        transferred_chunks.append(ChunkTransferDetail(
                            chunk_id=chunk_id,
                            template_id=template_id,
                            steps_count=len(mapped_steps)
                        ))
                        
                        # Track which source steps were covered
                        steps_from_chunks.update(step_ids)
                        continue
                
                # Find matching templates
                matches = manager.chunk_library.find_matching_chunks(
                    steps=chunk_steps,
                    source_domain=source.domain,
                    target_domain=target_domain
                )
                
                if matches:
                    # Use best match
                    best_match = matches[0]
                    template_id = best_match["template_id"]
                    
                    # Map template to new domain
                    mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
                        template_id=template_id,
                        target_domain=target_domain
                    )
                    
                    if mapped_steps:
                        # Add steps from this chunk
                        all_steps.extend(mapped_steps)
                        transferred_chunks.append(ChunkTransferDetail(
                            chunk_id=chunk_id,
                            template_id=template_id,
                            steps_count=len(mapped_steps)
                        ))
                        
                        # Track which source steps were covered
                        steps_from_chunks.update(step_ids)
                
                else:
                    # No matching templates, try to create one
                    template = manager.chunk_library.create_chunk_template_from_steps(
                        chunk_id=f"template_{chunk_id}_{int(datetime.datetime.now().timestamp())}",
                        name=f"{source_name}_{chunk_id}",
                        steps=chunk_steps,
                        domain=source.domain
                    )
                    
                    if template:
                        # Map to new domain
                        mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
                            template_id=template.id,
                            target_domain=target_domain
                        )
                        
                        if mapped_steps:
                            all_steps.extend(mapped_steps)
                            transferred_chunks.append(ChunkTransferDetail(
                                chunk_id=chunk_id,
                                template_id=template.id,
                                steps_count=len(mapped_steps),
                                newly_created=True
                            ))
                            
                            # Track which source steps were covered
                            steps_from_chunks.update(step_ids)
        
        # Get remaining steps not covered by chunks
        remaining_steps = [s for s in source.steps if s["id"] not in steps_from_chunks]
        
        # Map remaining steps individually
        for step in remaining_steps:
            mapped_step = await manager.map_step_to_domain(
                step=step,
                source_domain=source.domain,
                target_domain=target_domain
            )
            
            if mapped_step:
                all_steps.append(mapped_step)
        
        if not all_steps:
            return TransferWithChunkingResponse(
                success=False,
                error="Could not map any steps or chunks to the target domain",
                source_name=source_name,
                target_name=target_name,
                source_domain=source.domain,
                target_domain=target_domain,
                steps_count=0,
                procedure_id=""
            )
        
        # Create new procedure
        import json
        new_procedure = await add_procedure(
            ctx,
            name=target_name,
            steps_json=json.dumps(all_steps),
            description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
            domain=target_domain
        )
        
        if not new_procedure.success:
            return TransferWithChunkingResponse(
                success=False,
                error=new_procedure.error or "Failed to create new procedure",
                source_name=source_name,
                target_name=target_name,
                source_domain=source.domain,
                target_domain=target_domain,
                steps_count=len(all_steps),
                procedure_id=""
            )
        
        # Record transfer
        transfer_record = ProcedureTransferRecord(
            source_procedure_id=source.id,
            source_domain=source.domain,
            target_procedure_id=new_procedure.procedure_id,
            target_domain=target_domain,
            transfer_date=datetime.datetime.now().isoformat(),
            adaptation_steps=[{
                "chunk_id": chunk.chunk_id,
                "template_id": chunk.template_id,
                "steps_count": chunk.steps_count
            } for chunk in transferred_chunks],
            success_level=0.8,  # Initial estimate
            practice_needed=5  # Initial estimate
        )
        
        manager.chunk_library.record_transfer(transfer_record)
        
        # Update transfer stats
        if hasattr(manager, "transfer_stats"):
            manager.transfer_stats["total_transfers"] += 1
            manager.transfer_stats["successful_transfers"] += 1
        
        return TransferWithChunkingResponse(
            success=True,
            source_name=source_name,
            target_name=target_name,
            source_domain=source.domain,
            target_domain=target_domain,
            steps_count=len(all_steps),
            chunks_transferred=len(transferred_chunks),
            procedure_id=new_procedure.procedure_id,
            chunk_transfer_details=transferred_chunks
        )

@function_tool(use_docstring_info=True)
async def find_similar_procedures(
    ctx: RunContextWrapper[Any],
    name: str, 
    target_domain: Optional[str] = None
) -> List[SimilarProcedure]:  # Changed return type
    """
    Find procedures similar to the specified one.
    
    Args:
        name: Name of the procedure to compare
        target_domain: Optional domain to filter by
        
    Returns:
        List of similar procedures with similarity scores
    """
    manager = ctx.context.manager
    
    if name not in manager.procedures:
        return []
    
    source = manager.procedures[name]
    
    # Create trace span for the operation
    with custom_span(
        name="find_similar_procedures", 
        data={
            "source_name": name,
            "source_domain": source.domain,
            "target_domain": target_domain
        }
    ):
        similar_procedures = []
        for proc_name, procedure in manager.procedures.items():
            # Skip self
            if proc_name == name:
                continue
                
            # Filter by domain if specified
            if target_domain and procedure.domain != target_domain:
                continue
                
            # Calculate similarity
            similarity = await manager.calculate_procedure_similarity(source, procedure)
            
            if similarity > 0.3:  # Minimum similarity threshold
                similar_procedures.append(SimilarProcedure(
                    name=proc_name,
                    id=procedure.id,
                    domain=procedure.domain,
                    similarity=similarity,
                    steps_count=len(procedure.steps),
                    proficiency=procedure.proficiency
                ))
        
        # Sort by similarity
        similar_procedures.sort(key=lambda x: x.similarity, reverse=True)
        
        return similar_procedures

@function_tool(use_docstring_info=True, strict_mode=False)  # Disable strict schema
async def refine_step(
    ctx: RunContextWrapper[Any],
    procedure_name: str,
    step_id: str,
    new_function: Optional[str] = None,
    new_parameters: Optional[StepParameters] = None,  # Already using Pydantic model
    new_description: Optional[str] = None
) -> RefineStepResponse:  # Changed return type
    """
    Refine a specific step in a procedure.
    
    Args:
        procedure_name: Name of the procedure
        step_id: ID of the step to refine
        new_function: Optional new function name
        new_parameters: Optional new parameters
        new_description: Optional new description
        
    Returns:
        Result of the refinement with details about what was changed
    """
    manager = ctx.context.manager  # ✅ This should be first
    
    if procedure_name not in manager.procedures:
        return RefineStepResponse(error=f"Procedure '{procedure_name}' not found", success=False)
    
    procedure = manager.procedures[procedure_name]
    
    # Create trace span for the operation
    with custom_span(
        name="refine_step", 
        data={
            "procedure_name": procedure_name,
            "step_id": step_id,
            "function_updated": new_function is not None,
            "parameters_updated": new_parameters is not None,
            "description_updated": new_description is not None
        }
    ):
        # Find the step
        step = None
        for s in procedure.steps:
            if s["id"] == step_id:
                step = s
                break
                
        if not step:
            return RefineStepResponse(
                error=f"Step '{step_id}' not found in procedure '{procedure_name}'",
                success=False
            )
            
        # Update function if provided
        if new_function:
            if callable(new_function):
                func_name = new_function.__name__
                ctx.context.register_function(func_name, new_function)
                step["function"] = func_name
            else:
                step["function"] = new_function
            
        # Update parameters if provided - ✅ MOVED TO CORRECT LOCATION
        if new_parameters:
            # Convert Pydantic model to dict
            step["parameters"] = new_parameters.model_dump() if isinstance(new_parameters, BaseModel) else new_parameters
            
        # Update description if provided
        if new_description:
            step["description"] = new_description
            
        # Update procedure
        procedure.last_updated = datetime.datetime.now().isoformat()
        
        # If this step is part of a chunk, unchunk
        affected_chunks = []
        if procedure.is_chunked:
            # Find chunks containing this step
            for chunk_id, step_ids in procedure.chunked_steps.items():
                if step_id in step_ids:
                    affected_chunks.append(chunk_id)
                    
            # If any chunks are affected, reset chunking
            if affected_chunks:
                procedure.is_chunked = False
                procedure.chunked_steps = {}
                procedure.chunk_contexts = {}
                procedure.generalized_chunks = {}
        
        # Remove this step from refinement opportunities if it was there
        if hasattr(procedure, "refinement_opportunities"):
            procedure.refinement_opportunities = [
                r for r in procedure.refinement_opportunities if r.get("step_id") != step_id
            ]
        
        return RefineStepResponse(
            success=True,
            procedure_name=procedure_name,
            step_id=step_id,
            function_updated=new_function is not None,
            parameters_updated=new_parameters is not None,
            description_updated=new_description is not None,
            chunking_reset=len(affected_chunks) > 0
        )
