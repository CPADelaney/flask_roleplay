# nyx/core/procedural_memory/function_tools.py

import asyncio
import datetime
import logging
import random
from typing import Dict, List, Any, Optional

# OpenAI Agents SDK imports
from agents import function_tool, trace
from agents.exceptions import UserError

from .models import Procedure, ProcedureStats, TransferStats

logger = logging.getLogger(__name__)

@function_tool
async def add_procedure(
    ctx,
    name: str, 
    steps: List[Dict[str, Any]], 
    description: str = None,
    domain: str = "general"
) -> Dict[str, Any]:
    """
    Add a new procedure to the procedural memory system.
    
    Args:
        name: Name of the procedure
        steps: List of step definitions with function, description and parameters
        description: Optional description of what the procedure accomplishes
        domain: Domain/context where this procedure applies
        
    Returns:
        Information about the created procedure
    """
    manager = ctx.context
    
    # Generate a unique ID for the procedure
    procedure_id = f"proc_{int(datetime.datetime.now().timestamp())}_{random.randint(1000, 9999)}"
    
    # Create procedure object
    procedure = Procedure(
        id=procedure_id,
        name=name,
        description=description or f"Procedure for {name}",
        domain=domain,
        steps=steps,
        created_at=datetime.datetime.now().isoformat(),
        last_updated=datetime.datetime.now().isoformat()
    )
    
    # Register function names if needed
    for step in steps:
        function_name = step.get("function")
        if function_name and callable(function_name):
            # It's a callable, register it by name
            func_name = function_name.__name__
            manager.register_function(func_name, function_name)
            step["function"] = func_name
    
    # Store the procedure
    manager.procedures[name] = procedure
    
    # Create a trace for analytics
    with trace(workflow_name="add_procedure"):
        logger.info(f"Added new procedure '{name}' with {len(steps)} steps in {domain} domain")
    
    return {
        "procedure_id": procedure_id,
        "name": name,
        "domain": domain,
        "steps_count": len(steps)
    }

@function_tool
async def execute_procedure(
    ctx,
    name: str,
    context: Dict[str, Any] = None,
    force_conscious: bool = False
) -> Dict[str, Any]:
    """
    Execute a procedure by name
    
    Args:
        name: Name of the procedure to execute
        context: Context data for execution
        force_conscious: Force deliberate execution even if proficient
        
    Returns:
        Execution results including success and execution time
    """
    manager = ctx.context
    
    if name not in manager.procedures:
        return {"error": f"Procedure '{name}' not found"}
    
    procedure = manager.procedures[name]
    
    # Create execution trace
    with trace(workflow_name="execute_procedure"):
        # Execute the procedure
        result = await manager.execute_procedure_steps(procedure, context or {}, force_conscious)
        
        # Record execution context
        if hasattr(procedure, "context_history"):
            execution_context = (context or {}).copy()
            execution_context["timestamp"] = datetime.datetime.now().isoformat()
            execution_context["result"] = result["success"]
            execution_context["execution_time"] = result["execution_time"]
            
            if len(procedure.context_history) >= procedure.max_history:
                procedure.context_history = procedure.context_history[-(procedure.max_history-1):]
            procedure.context_history.append(execution_context)
    
    return result

@function_tool
async def transfer_procedure(
    ctx,
    source_name: str,
    target_name: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Transfer a procedure from one domain to another
    
    Args:
        source_name: Name of the source procedure
        target_name: Name for the new procedure
        target_domain: Domain for the new procedure
        
    Returns:
        Transfer results with adapted procedure details
    """
    manager = ctx.context
    
    if source_name not in manager.procedures:
        return {"error": f"Source procedure '{source_name}' not found"}
    
    source = manager.procedures[source_name]
    
    # Create a trace for the transfer operation
    with trace(workflow_name="transfer_procedure"):
        # Use the chunk library to map steps to the new domain
        mapped_steps = []
        
        for step in source.steps:
            # Try to map step to new domain
            mapped_step = manager.map_step_to_domain(
                step=step,
                source_domain=source.domain,
                target_domain=target_domain
            )
            
            if mapped_step:
                mapped_steps.append(mapped_step)
        
        if not mapped_steps:
            return {
                "success": False,
                "error": "Could not map any steps to the target domain"
            }
        
        # Create new procedure
        new_procedure = await add_procedure(
            ctx,
            name=target_name,
            steps=mapped_steps,
            description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
            domain=target_domain
        )
        
        # Record transfer
        transfer_record = ProcedureTransferRecord(
            source_procedure_id=source.id,
            source_domain=source.domain,
            target_procedure_id=new_procedure["procedure_id"],
            target_domain=target_domain,
            transfer_date=datetime.datetime.now().isoformat(),
            success_level=0.8,  # Initial estimate
            practice_needed=5  # Initial estimate
        )
        
        manager.chunk_library.record_transfer(transfer_record)
        
        # Update transfer stats
        manager.transfer_stats["total_transfers"] += 1
        manager.transfer_stats["successful_transfers"] += 1
    
    return {
        "success": True,
        "source_name": source_name,
        "target_name": target_name,
        "source_domain": source.domain,
        "target_domain": target_domain,
        "steps_count": len(mapped_steps),
        "procedure_id": new_procedure["procedure_id"]
    }

@function_tool
async def get_procedure_proficiency(ctx, name: str) -> ProcedureStats:
    """
    Get the current proficiency level for a procedure
    
    Args:
        name: Name of the procedure
        
    Returns:
        Proficiency information including level and execution statistics
    """
    manager = ctx.context
    
    if name not in manager.procedures:
        raise UserError(f"Procedure '{name}' not found")
    
    procedure = manager.procedures[name]
    
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

@function_tool
async def list_procedures(ctx, domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all procedures, optionally filtered by domain
    
    Args:
        domain: Optional domain to filter by
        
    Returns:
        List of procedure summaries
    """
    manager = ctx.context
    
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
        procedure_list.append({
            "name": name,
            "id": procedure.id,
            "description": procedure.description,
            "domain": procedure.domain,
            "proficiency": procedure.proficiency,
            "proficiency_level": proficiency_level,
            "steps_count": len(procedure.steps),
            "execution_count": procedure.execution_count,
            "is_chunked": procedure.is_chunked,
            "created_at": procedure.created_at,
            "last_updated": procedure.last_updated,
            "last_execution": procedure.last_execution
        })
        
    # Sort by domain and then name
    procedure_list.sort(key=lambda x: (x["domain"], x["name"]))
    
    return procedure_list

@function_tool
async def get_transfer_statistics(ctx) -> TransferStats:
    """
    Get statistics about procedure transfers
    
    Returns:
        Transfer statistics including success rates and domain coverage
    """
    manager = ctx.context
    
    # Get chunks by domain
    chunks_by_domain = {}
    for domain, chunk_ids in manager.chunk_library.domain_chunks.items():
        chunks_by_domain[domain] = len(chunk_ids)
    
    # Get recent transfers
    recent_transfers = []
    for record in manager.chunk_library.transfer_records[-5:]:
        recent_transfers.append({
            "source_domain": record.source_domain,
            "target_domain": record.target_domain,
            "transfer_date": record.transfer_date,
            "success_level": record.success_level
        })
    
    return TransferStats(
        total_transfers=manager.transfer_stats["total_transfers"],
        successful_transfers=manager.transfer_stats["successful_transfers"],
        avg_success_level=manager.transfer_stats.get("avg_success_level", 0.0),
        avg_practice_needed=manager.transfer_stats.get("avg_practice_needed", 0),
        chunks_by_domain=chunks_by_domain,
        recent_transfers=recent_transfers,
        templates_count=len(manager.chunk_library.chunk_templates),
        actions_count=len(manager.chunk_library.action_templates)
    )

@function_tool
async def identify_chunking_opportunities(ctx, procedure_name: str) -> Dict[str, Any]:
    """
    Identify opportunities to chunk steps in a procedure
    
    Args:
        procedure_name: Name of the procedure to analyze
        
    Returns:
        Potential chunks that could be formed
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        raise UserError(f"Procedure '{procedure_name}' not found")
    
    procedure = manager.procedures[procedure_name]
    
    # Need at least 3 steps and some executions to consider chunking
    if len(procedure.steps) < 3 or procedure.execution_count < 5:
        return {
            "can_chunk": False,
            "reason": f"Need at least 3 steps and 5 executions (has {len(procedure.steps)} steps and {procedure.execution_count} executions)"
        }
    
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
    
    return {
        "can_chunk": len(chunks) > 0,
        "potential_chunks": chunks,
        "chunk_count": len(chunks),
        "procedure_name": procedure_name
    }

@function_tool
async def apply_chunking(ctx, procedure_name: str) -> Dict[str, Any]:
    """
    Apply chunking to a procedure based on execution patterns
    
    Args:
        procedure_name: Name of the procedure to chunk
        
    Returns:
        Results of the chunking process
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        raise UserError(f"Procedure '{procedure_name}' not found")
    
    procedure = manager.procedures[procedure_name]
    
    # Find chunking opportunities
    chunking_result = await identify_chunking_opportunities(ctx, procedure_name)
    
    if not chunking_result["can_chunk"]:
        return chunking_result
    
    # Apply chunks
    chunks = chunking_result["potential_chunks"]
    
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
    
    return {
        "success": True,
        "chunks_applied": len(chunks),
        "chunk_ids": list(procedure.chunked_steps.keys()),
        "procedure_name": procedure_name
    }

@function_tool
async def generalize_chunk_from_steps(
    ctx,
    chunk_name: str,
    procedure_name: str,
    step_ids: List[str],
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a generalizable chunk template from specific procedure steps
    
    Args:
        chunk_name: Name for the new chunk template
        procedure_name: Name of the procedure containing the steps
        step_ids: IDs of the steps to include in the chunk
        domain: Optional domain override (defaults to procedure's domain)
        
    Returns:
        Information about the created chunk template
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        return {"error": f"Procedure '{procedure_name}' not found"}
    
    procedure = manager.procedures[procedure_name]
    
    # Get the steps by ID
    steps = []
    for step_id in step_ids:
        step = next((s for s in procedure.steps if s["id"] == step_id), None)
        if step:
            steps.append(step)
    
    if not steps:
        return {"error": "No valid steps found"}
    
    # Use the procedure's domain if not specified
    chunk_domain = domain or procedure.domain
    
    # Generate a unique template ID
    template_id = f"template_{chunk_name}_{int(datetime.datetime.now().timestamp())}"
    
    # Create the chunk template
    template = manager.chunk_library.create_chunk_template_from_steps(
        chunk_id=template_id,
        name=chunk_name,
        steps=steps,
        domain=chunk_domain
    )
    
    if not template:
        return {"error": "Failed to create chunk template"}
    
    # If these steps were already part of a chunk, store the template reference
    for chunk_id, step_list in procedure.chunked_steps.items():
        if all(step_id in step_list for step_id in step_ids):
            # This existing chunk contains all our steps
            procedure.generalized_chunks[chunk_id] = template.id
            break
    
    return {
        "template_id": template.id,
        "name": template.name,
        "domain": chunk_domain,
        "actions_count": len(template.actions),
        "can_transfer": True
    }

@function_tool
async def find_matching_chunks(
    ctx,
    procedure_name: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Find library chunks that match a procedure's steps for transfer
    
    Args:
        procedure_name: Name of the procedure to find chunks for
        target_domain: Target domain to transfer to
        
    Returns:
        List of matching chunks with similarity scores
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        return {"error": f"Procedure '{procedure_name}' not found"}
    
    procedure = manager.procedures[procedure_name]
    
    # Find matching chunks
    matches = manager.chunk_library.find_matching_chunks(
        steps=procedure.steps,
        source_domain=procedure.domain,
        target_domain=target_domain
    )
    
    if not matches:
        return {
            "chunks_found": False,
            "message": "No matching chunks found for transfer"
        }
    
    return {
        "chunks_found": True,
        "matches": matches,
        "count": len(matches),
        "source_domain": procedure.domain,
        "target_domain": target_domain
    }

@function_tool
async def transfer_chunk(
    ctx,
    template_id: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Transfer a chunk template to a new domain
    
    Args:
        template_id: ID of the chunk template to transfer
        target_domain: Domain to transfer to
        
    Returns:
        Mapped steps for the new domain
    """
    manager = ctx.context
    
    # Map the chunk to the new domain
    mapped_steps = manager.chunk_library.map_chunk_to_new_domain(
        template_id=template_id,
        target_domain=target_domain
    )
    
    if not mapped_steps:
        return {
            "success": False,
            "error": "Failed to map chunk to new domain"
        }
    
    return {
        "success": True,
        "steps": mapped_steps,
        "steps_count": len(mapped_steps),
        "target_domain": target_domain
    }

@function_tool
async def transfer_with_chunking(
    ctx,
    source_name: str,
    target_name: str,
    target_domain: str
) -> Dict[str, Any]:
    """
    Transfer a procedure from one domain to another using chunk-level transfer
    
    Args:
        source_name: Name of the source procedure
        target_name: Name for the new procedure
        target_domain: Domain for the new procedure
        
    Returns:
        Transfer results
    """
    manager = ctx.context
    
    if source_name not in manager.procedures:
        return {"error": f"Source procedure '{source_name}' not found"}
    
    source = manager.procedures[source_name]
    
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
                    transferred_chunks.append({
                        "chunk_id": chunk_id,
                        "template_id": template_id,
                        "steps_count": len(mapped_steps)
                    })
                    
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
                    transferred_chunks.append({
                        "chunk_id": chunk_id,
                        "template_id": template_id,
                        "steps_count": len(mapped_steps)
                    })
                    
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
                        transferred_chunks.append({
                            "chunk_id": chunk_id,
                            "template_id": template.id,
                            "steps_count": len(mapped_steps),
                            "newly_created": True
                        })
                        
                        # Track which source steps were covered
                        steps_from_chunks.update(step_ids)
    
    # Get remaining steps not covered by chunks
    remaining_steps = [s for s in source.steps if s["id"] not in steps_from_chunks]
    
    # Map remaining steps individually
    for step in remaining_steps:
        mapped_step = manager.map_step_to_domain(
            step=step,
            source_domain=source.domain,
            target_domain=target_domain
        )
        
        if mapped_step:
            all_steps.append(mapped_step)
    
    if not all_steps:
        return {
            "success": False,
            "error": "Could not map any steps or chunks to the target domain"
        }
    
    # Create new procedure
    new_procedure = await add_procedure(
        ctx,
        name=target_name,
        steps=all_steps,
        description=f"Transferred from {source_name} ({source.domain} to {target_domain})",
        domain=target_domain
    )
    
    # Record transfer
    transfer_record = ProcedureTransferRecord(
        source_procedure_id=source.id,
        source_domain=source.domain,
        target_procedure_id=new_procedure["procedure_id"],
        target_domain=target_domain,
        transfer_date=datetime.datetime.now().isoformat(),
        adaptation_steps=[{
            "chunk_id": info["chunk_id"],
            "template_id": info["template_id"],
            "steps_count": info["steps_count"]
        } for info in transferred_chunks],
        success_level=0.8,  # Initial estimate
        practice_needed=5  # Initial estimate
    )
    
    manager.chunk_library.record_transfer(transfer_record)
    
    # Update transfer stats
    manager.transfer_stats["total_transfers"] += 1
    manager.transfer_stats["successful_transfers"] += 1
    
    return {
        "success": True,
        "source_name": source_name,
        "target_name": target_name,
        "source_domain": source.domain,
        "target_domain": target_domain,
        "steps_count": len(all_steps),
        "chunks_transferred": len(transferred_chunks),
        "procedure_id": new_procedure["procedure_id"],
        "chunk_transfer_details": transferred_chunks
    }

@function_tool
async def find_similar_procedures(
    ctx,
    name: str, 
    target_domain: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Find procedures similar to the specified one
    
    Args:
        name: Name of the procedure to compare
        target_domain: Optional domain to filter by
        
    Returns:
        List of similar procedures with similarity scores
    """
    manager = ctx.context
    
    if name not in manager.procedures:
        return []
    
    source = manager.procedures[name]
    
    similar_procedures = []
    for proc_name, procedure in manager.procedures.items():
        # Skip self
        if proc_name == name:
            continue
            
        # Filter by domain if specified
        if target_domain and procedure.domain != target_domain:
            continue
            
        # Calculate similarity
        similarity = manager.calculate_procedure_similarity(source, procedure)
        
        if similarity > 0.3:  # Minimum similarity threshold
            similar_procedures.append({
                "name": proc_name,
                "id": procedure.id,
                "domain": procedure.domain,
                "similarity": similarity,
                "steps_count": len(procedure.steps),
                "proficiency": procedure.proficiency
            })
    
    # Sort by similarity
    similar_procedures.sort(key=lambda x: x["similarity"], reverse=True)
    
    return similar_procedures

@function_tool
async def refine_step(
    ctx,
    procedure_name: str,
    step_id: str,
    new_function: Optional[str] = None,
    new_parameters: Optional[Dict[str, Any]] = None,
    new_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Refine a specific step in a procedure
    
    Args:
        procedure_name: Name of the procedure
        step_id: ID of the step to refine
        new_function: Optional new function name
        new_parameters: Optional new parameters
        new_description: Optional new description
        
    Returns:
        Result of the refinement
    """
    manager = ctx.context
    
    if procedure_name not in manager.procedures:
        return {"error": f"Procedure '{procedure_name}' not found"}
    
    procedure = manager.procedures[procedure_name]
    
    # Find the step
    step = None
    for s in procedure.steps:
        if s["id"] == step_id:
            step = s
            break
            
    if not step:
        return {"error": f"Step '{step_id}' not found in procedure '{procedure_name}'"}
        
    # Update function if provided
    if new_function:
        if callable(new_function):
            func_name = new_function.__name__
            manager.register_function(func_name, new_function)
            step["function"] = func_name
        else:
            step["function"] = new_function
        
    # Update parameters if provided
    if new_parameters:
        step["parameters"] = new_parameters
        
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
    
    return {
        "success": True,
        "procedure_name": procedure_name,
        "step_id": step_id,
        "function_updated": new_function is not None,
        "parameters_updated": new_parameters is not None,
        "description_updated": new_description is not None,
        "chunking_reset": len(affected_chunks) > 0
    }
