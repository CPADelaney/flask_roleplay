# nyx/core/integration/knowledge_memory_reasoning_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class KnowledgeMemoryReasoningBridge:
    """
    Integrates knowledge core with memory and reasoning systems.
    
    This bridge enables:
    1. Knowledge retrieval enhanced by memory context
    2. Reasoning over knowledge to derive new facts
    3. Memory-guided knowledge exploration
    4. Bi-directional updates between knowledge and memory
    """
    
    def __init__(self, 
                knowledge_core=None,
                memory_orchestrator=None,
                reasoning_core=None,
                reflection_engine=None):
        """Initialize the knowledge-memory-reasoning bridge."""
        self.knowledge_core = knowledge_core
        self.memory_orchestrator = memory_orchestrator
        self.reasoning_core = reasoning_core
        self.reflection_engine = reflection_engine
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.memory_knowledge_threshold = 0.7  # Min memory significance for knowledge
        self.knowledge_confidence_threshold = 0.6  # Min confidence for memory creation
        self.reasoning_confidence_threshold = 0.7  # Min confidence for reasoning-based knowledge
        
        # Tracking variables
        self.memory_to_knowledge_mappings = {}  # memory_id -> knowledge_node_id
        self.knowledge_to_memory_mappings = {}  # knowledge_node_id -> list of memory_ids
        self.last_integration_run = datetime.datetime.now()
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("KnowledgeMemoryReasoningBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("memory_added", self._handle_memory_added)
                self.event_bus.subscribe("knowledge_updated", self._handle_knowledge_updated)
                self.event_bus.subscribe("reflection_created", self._handle_reflection_created)
                self._subscribed = True
            
            logger.info("KnowledgeMemoryReasoningBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing KnowledgeMemoryReasoningBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeMemoryReasoning")
    async def memory_guided_knowledge_query(self, 
                                         query: str, 
                                         context: Optional[Dict[str, Any]] = None,
                                         memory_enhancement: bool = True) -> Dict[str, Any]:
        """
        Query knowledge with memory enhancement.
        
        Args:
            query: Knowledge query string
            context: Optional context information
            memory_enhancement: Whether to enhance results with memory
            
        Returns:
            Query results with knowledge and relevant memories
        """
        if not self.knowledge_core or not self.memory_orchestrator:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Parse query into knowledge query format
            knowledge_query = await self._parse_query_to_knowledge_format(query, context)
            
            # Execute knowledge query
            knowledge_results = await self.knowledge_core.query_knowledge(knowledge_query)
            
            # If no memory enhancement requested, return knowledge results directly
            if not memory_enhancement:
                return {
                    "status": "success",
                    "knowledge_results": knowledge_results,
                    "query": query,
                    "memory_enhanced": False
                }
            
            # Enhance with memory if requested
            memory_query = {"text": query}
            if context:
                memory_query.update(context)
                
            # Retrieve relevant memories
            memories = await self.memory_orchestrator.retrieve_memories(
                query=query, 
                limit=5, 
                context=context
            )
            
            # Enhance knowledge results with memory context
            enhanced_results = await self._enhance_with_memory(knowledge_results, memories)
            
            # Update mappings for any new connections discovered
            for k_node in enhanced_results:
                if "memory_connections" in k_node:
                    k_id = k_node["id"]
                    if k_id not in self.knowledge_to_memory_mappings:
                        self.knowledge_to_memory_mappings[k_id] = []
                        
                    for m_id in k_node["memory_connections"]:
                        if m_id not in self.knowledge_to_memory_mappings[k_id]:
                            self.knowledge_to_memory_mappings[k_id].append(m_id)
            
            return {
                "status": "success",
                "knowledge_results": enhanced_results,
                "memory_results": memories,
                "query": query,
                "memory_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Error in memory guided knowledge query: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeMemoryReasoning")
    async def derive_knowledge_from_memory(self, 
                                        memory_id: str,
                                        confidence_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Derive knowledge from a memory.
        
        Args:
            memory_id: ID of the memory to analyze
            confidence_override: Optional override for confidence
            
        Returns:
            Results of the knowledge derivation
        """
        if not self.knowledge_core or not self.memory_orchestrator:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Retrieve the memory
            memory = await self.memory_orchestrator.get_memory(memory_id)
            if not memory:
                return {"status": "error", "message": f"Memory {memory_id} not found"}
                
            # Check if memory is significant enough
            significance = memory.get("significance", 5)
            normalized_significance = significance / 10.0 if significance > 1 else significance
            
            if normalized_significance < self.memory_knowledge_threshold and confidence_override is None:
                return {
                    "status": "skipped", 
                    "reason": f"Memory significance {normalized_significance} below threshold",
                    "memory_id": memory_id
                }
            
            # Extract facts from memory
            knowledge_facts = await self._extract_knowledge_facts(memory)
            
            if not knowledge_facts:
                return {
                    "status": "no_facts",
                    "memory_id": memory_id,
                    "memory_text": memory.get("memory_text", "")
                }
            
            # Determine confidence (from significance or override)
            confidence = confidence_override if confidence_override is not None else normalized_significance
            
            # Add knowledge nodes
            added_nodes = []
            for fact in knowledge_facts:
                # Create knowledge content
                content = {
                    "fact": fact["statement"],
                    "context": fact.get("context", ""),
                    "derived_from_memory": True,
                    "memory_id": memory_id
                }
                
                # Add domain information if available
                if "domain" in fact:
                    content["domain"] = fact["domain"]
                if "topic" in fact:
                    content["topic"] = fact["topic"]
                
                # Add to knowledge graph
                node_id = await self.knowledge_core.add_knowledge(
                    type="fact",
                    content=content,
                    source="memory_derivation",
                    confidence=confidence
                )
                
                if node_id:
                    added_nodes.append({
                        "node_id": node_id,
                        "fact": fact["statement"],
                        "confidence": confidence
                    })
                    
                    # Update mappings
                    self.memory_to_knowledge_mappings[memory_id] = node_id
                    
                    if node_id not in self.knowledge_to_memory_mappings:
                        self.knowledge_to_memory_mappings[node_id] = []
                        
                    if memory_id not in self.knowledge_to_memory_mappings[node_id]:
                        self.knowledge_to_memory_mappings[node_id].append(memory_id)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "derived_knowledge": added_nodes,
                "node_count": len(added_nodes)
            }
            
        except Exception as e:
            logger.error(f"Error deriving knowledge from memory: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeMemoryReasoning")
    async def create_memory_from_knowledge(self, 
                                        node_id: str,
                                        memory_type: str = "derived",
                                        tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a memory from a knowledge node.
        
        Args:
            node_id: ID of the knowledge node
            memory_type: Type of memory to create
            tags: Optional tags for the memory
            
        Returns:
            Results of the memory creation
        """
        if not self.knowledge_core or not self.memory_orchestrator:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Retrieve the knowledge node
            node_data = None
            query_result = await self.knowledge_core.query_knowledge({
                "content_filter": {"id": node_id},
                "limit": 1
            })
            
            if query_result and len(query_result) > 0:
                node_data = query_result[0]
            
            if not node_data:
                return {"status": "error", "message": f"Knowledge node {node_id} not found"}
                
            # Check confidence threshold
            confidence = node_data.get("confidence", 0.5)
            if confidence < self.knowledge_confidence_threshold:
                return {
                    "status": "skipped", 
                    "reason": f"Knowledge confidence {confidence} below threshold",
                    "node_id": node_id
                }
            
            # Create memory text
            memory_text = await self._format_knowledge_as_memory(node_data)
            
            # Set up memory tags
            memory_tags = tags or ["knowledge_derived"]
            memory_tags.append(node_data.get("type", "knowledge"))
            
            # Calculate significance (0-10 scale)
            significance = int(min(10, confidence * 10))
            
            # Add to memory
            memory_id = await self.memory_orchestrator.add_memory(
                memory_text=memory_text,
                memory_type=memory_type,
                significance=significance,
                tags=memory_tags,
                metadata={
                    "derived_from_knowledge": True,
                    "knowledge_node_id": node_id,
                    "knowledge_type": node_data.get("type"),
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
            
            if memory_id:
                # Update mappings
                self.memory_to_knowledge_mappings[memory_id] = node_id
                
                if node_id not in self.knowledge_to_memory_mappings:
                    self.knowledge_to_memory_mappings[node_id] = []
                    
                if memory_id not in self.knowledge_to_memory_mappings[node_id]:
                    self.knowledge_to_memory_mappings[node_id].append(memory_id)
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "node_id": node_id,
                "memory_text": memory_text,
                "significance": significance
            }
            
        except Exception as e:
            logger.error(f"Error creating memory from knowledge: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeMemoryReasoning")
    async def reason_over_knowledge(self, 
                                 query: str, 
                                 reasoning_type: str = "inductive",
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply reasoning over knowledge to derive new insights.
        
        Args:
            query: Question or query to reason about
            reasoning_type: Type of reasoning (deductive, inductive, abductive)
            context: Optional additional context
            
        Returns:
            Results of the reasoning process
        """
        if not self.knowledge_core or not self.reasoning_core:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # First, query relevant knowledge
            knowledge_query = await self._parse_query_to_knowledge_format(query, context)
            knowledge_results = await self.knowledge_core.query_knowledge(knowledge_query)
            
            if not knowledge_results:
                return {
                    "status": "insufficient_knowledge",
                    "query": query,
                    "reasoning_type": reasoning_type
                }
            
            # Format knowledge for reasoning
            knowledge_context = await self._format_knowledge_for_reasoning(knowledge_results)
            
            # Set up reasoning parameters
            reasoning_params = {
                "query": query,
                "reasoning_type": reasoning_type,
                "knowledge_context": knowledge_context
            }
            
            if context:
                reasoning_params.update(context)
            
            # Execute reasoning
            if reasoning_type == "deductive":
                reasoning_result = await self.reasoning_core.reason_deductively(reasoning_params)
            elif reasoning_type == "abductive":
                reasoning_result = await self.reasoning_core.reason_abductively(reasoning_params)
            else:  # default to inductive
                reasoning_result = await self.reasoning_core.reason_inductively(reasoning_params)
            
            # Extract confidence and conclusion
            conclusion = reasoning_result.get("conclusion", "")
            confidence = reasoning_result.get("confidence", 0.5)
            
            # Create new knowledge if confidence is high enough
            new_knowledge_id = None
            if conclusion and confidence >= self.reasoning_confidence_threshold:
                knowledge_content = {
                    "derived_conclusion": conclusion,
                    "reasoning_type": reasoning_type,
                    "supporting_evidence": reasoning_result.get("evidence", []),
                    "query": query
                }
                
                new_knowledge_id = await self.knowledge_core.add_knowledge(
                    type="derived_fact",
                    content=knowledge_content,
                    source="reasoning_engine",
                    confidence=confidence
                )
                
                # Add relations to supporting knowledge
                if new_knowledge_id:
                    for k_result in knowledge_results:
                        await self.knowledge_core.add_relation(
                            source_id=new_knowledge_id,
                            target_id=k_result["id"],
                            type="derived_from",
                            weight=0.8
                        )
            
            return {
                "status": "success",
                "reasoning_type": reasoning_type,
                "conclusion": conclusion,
                "confidence": confidence,
                "evidence": reasoning_result.get("evidence", []),
                "new_knowledge_id": new_knowledge_id,
                "knowledge_sources": [k["id"] for k in knowledge_results]
            }
            
        except Exception as e:
            logger.error(f"Error reasoning over knowledge: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeMemoryReasoning")
    async def enhance_reflection_with_knowledge(self, 
                                            reflection_id: str) -> Dict[str, Any]:
        """
        Enhance a reflection with knowledge.
        
        Args:
            reflection_id: ID of the reflection to enhance
            
        Returns:
            Results of the knowledge enhancement
        """
        if not self.knowledge_core or not self.memory_orchestrator or not self.reflection_engine:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Retrieve reflection
            reflection = await self.memory_orchestrator.get_memory(reflection_id)
            if not reflection:
                return {"status": "error", "message": f"Reflection {reflection_id} not found"}
                
            # Extract key topics and query knowledge
            topics = await self._extract_topics_from_reflection(reflection)
            
            enhanced_content = {}
            for topic in topics:
                # Query knowledge about this topic
                knowledge_query = {
                    "content_filter": {"topic": topic},
                    "limit": 3
                }
                
                knowledge_results = await self.knowledge_core.query_knowledge(knowledge_query)
                
                if knowledge_results:
                    enhanced_content[topic] = knowledge_results
            
            if not enhanced_content:
                return {
                    "status": "no_enhancements",
                    "reflection_id": reflection_id
                }
            
            # Generate enhanced reflection
            enhanced_text = await self.reflection_engine.enhance_reflection(
                reflection_text=reflection.get("memory_text", ""),
                knowledge_enhancements=enhanced_content
            )
            
            # Update the reflection
            await self.memory_orchestrator.update_memory(
                memory_id=reflection_id,
                updates={
                    "memory_text": enhanced_text,
                    "metadata": {
                        **reflection.get("metadata", {}),
                        "knowledge_enhanced": True,
                        "enhancement_timestamp": datetime.datetime.now().isoformat(),
                        "knowledge_topics": list(enhanced_content.keys())
                    }
                }
            )
            
            return {
                "status": "success",
                "reflection_id": reflection_id,
                "enhanced_topics": list(enhanced_content.keys()),
                "knowledge_nodes_used": sum(len(nodes) for nodes in enhanced_content.values())
            }
            
        except Exception as e:
            logger.error(f"Error enhancing reflection with knowledge: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeMemoryReasoning")
    async def run_integration_cycle(self) -> Dict[str, Any]:
        """
        Run a cycle to integrate memory and knowledge.
        
        Returns:
            Results of the integration cycle
        """
        self.last_integration_run = datetime.datetime.now()
        
        results = {
            "memories_analyzed": 0,
            "knowledge_nodes_analyzed": 0,
            "memories_to_knowledge": 0,
            "knowledge_to_memories": 0,
            "start_time": self.last_integration_run.isoformat()
        }
        
        try:
            # 1. Find significant memories without knowledge mappings
            recent_memories = await self.memory_orchestrator.get_recent_memories(limit=20)
            
            # Filter for significant memories without knowledge mapping
            significant_unmapped = []
            for memory in recent_memories:
                memory_id = memory.get("id")
                significance = memory.get("significance", 5)
                normalized_significance = significance / 10.0 if significance > 1 else significance
                
                if (normalized_significance >= self.memory_knowledge_threshold and 
                    memory_id not in self.memory_to_knowledge_mappings):
                    significant_unmapped.append(memory)
            
            # Process unmapped memories to create knowledge
            for memory in significant_unmapped:
                result = await self.derive_knowledge_from_memory(memory.get("id"))
                if result.get("status") == "success":
                    results["memories_to_knowledge"] += 1
                    
            results["memories_analyzed"] = len(recent_memories)
            
            # 2. Find high-confidence knowledge without memory mappings
            knowledge_query = {
                "content_filter": {"confidence": 0.8},  # High confidence
                "limit": 20
            }
            
            knowledge_results = await self.knowledge_core.query_knowledge(knowledge_query)
            
            # Filter for unmapped knowledge
            high_conf_unmapped = []
            for node in knowledge_results:
                node_id = node.get("id")
                if node_id not in self.knowledge_to_memory_mappings:
                    high_conf_unmapped.append(node)
            
            # Create memories from unmapped knowledge
            for node in high_conf_unmapped:
                result = await self.create_memory_from_knowledge(node.get("id"))
                if result.get("status") == "success":
                    results["knowledge_to_memories"] += 1
                    
            results["knowledge_nodes_analyzed"] = len(knowledge_results)
            
            # 3. Update stats
            results["end_time"] = datetime.datetime.now().isoformat()
            results["duration_seconds"] = (
                datetime.datetime.now() - self.last_integration_run
            ).total_seconds()
            
            logger.info(f"Integration cycle complete: {results['memories_to_knowledge']} memories to knowledge, "
                       f"{results['knowledge_to_memories']} knowledge to memories")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in integration cycle: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    async def _handle_memory_added(self, event: Event) -> None:
        """
        Handle memory added events.
        
        Args:
            event: Memory added event
        """
        try:
            # Extract event data
            memory_id = event.data.get("memory_id")
            significance = event.data.get("significance", 5)
            
            if not memory_id:
                return
                
            # Normalize significance if needed
            normalized_significance = significance / 10.0 if significance > 1 else significance
            
            # Check if significant enough to convert to knowledge
            if normalized_significance >= self.memory_knowledge_threshold:
                # Derive knowledge with slight delay to ensure memory is fully available
                await asyncio.sleep(0.5)
                asyncio.create_task(self.derive_knowledge_from_memory(memory_id))
                
        except Exception as e:
            logger.error(f"Error handling memory added event: {e}")
    
    async def _handle_knowledge_updated(self, event: Event) -> None:
        """
        Handle knowledge updated events.
        
        Args:
            event: Knowledge updated event
        """
        try:
            # Extract event data
            node_id = event.data.get("node_id")
            confidence = event.data.get("confidence", 0.5)
            
            if not node_id:
                return
                
            # Check if confidence is high enough for memory creation
            if confidence >= self.knowledge_confidence_threshold:
                # Check if already has a memory mapping
                if node_id in self.knowledge_to_memory_mappings and self.knowledge_to_memory_mappings[node_id]:
                    # Already has memory mapping
                    return
                    
                # Create memory with slight delay to ensure knowledge is fully updated
                await asyncio.sleep(0.5)
                asyncio.create_task(self.create_memory_from_knowledge(node_id))
                
        except Exception as e:
            logger.error(f"Error handling knowledge updated event: {e}")
    
    async def _handle_reflection_created(self, event: Event) -> None:
        """
        Handle reflection created events.
        
        Args:
            event: Reflection created event
        """
        try:
            # Extract event data
            reflection_id = event.data.get("reflection_id")
            
            if not reflection_id:
                return
                
            # Enhance reflection with knowledge
            asyncio.create_task(self.enhance_reflection_with_knowledge(reflection_id))
                
        except Exception as e:
            logger.error(f"Error handling reflection created event: {e}")
    
    async def _parse_query_to_knowledge_format(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Parse a natural language query into knowledge query format."""
        # Extract query parameters
        knowledge_query = {}
        
        # Add type filter if specified
        if context and "type" in context:
            knowledge_query["type"] = context["type"]
            
        # Add content filters
        content_filter = {}
        
        # Try to extract domain/topic from query or context
        if context and "domain" in context:
            content_filter["domain"] = context["domain"]
        if context and "topic" in context:
            content_filter["topic"] = context["topic"]
        
        # Add full text for semantic search if available
        content_filter["text_search"] = query
        
        # Add limit if specified
        knowledge_query["content_filter"] = content_filter
        knowledge_query["limit"] = context.get("limit", 10) if context else 10
        
        return knowledge_query
    
    async def _enhance_with_memory(self, knowledge_results: List[Dict[str, Any]], memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance knowledge results with memory context."""
        if not knowledge_results or not memories:
            return knowledge_results
            
        # Copy to avoid modifying originals
        enhanced_results = []
        for k_node in knowledge_results:
            enhanced_node = k_node.copy()
            
            # Initialize memory connections
            memory_connections = []
            memory_context = []
            
            # Check for existing mappings first
            node_id = k_node.get("id")
            if node_id in self.knowledge_to_memory_mappings:
                for m_id in self.knowledge_to_memory_mappings[node_id]:
                    memory_connections.append(m_id)
            
            # Check for connections in memory content
            for memory in memories:
                memory_id = memory.get("id")
                
                # Skip if already in connections
                if memory_id in memory_connections:
                    continue
                    
                # Check if memory references this knowledge
                metadata = memory.get("metadata", {})
                if metadata.get("knowledge_node_id") == node_id:
                    memory_connections.append(memory_id)
                    continue
                    
                # Check for content similarity
                if self.reasoning_core:
                    k_content = json.dumps(k_node.get("content", {}))
                    m_text = memory.get("memory_text", "")
                    
                    similarity = await self.reasoning_core.calculate_text_similarity(k_content, m_text)
                    if similarity > 0.7:  # High similarity threshold
                        memory_connections.append(memory_id)
                        memory_context.append({
                            "memory_id": memory_id,
                            "text": m_text[:100] + "..." if len(m_text) > 100 else m_text,
                            "similarity": similarity
                        })
                        
                        # Update mappings
                        if node_id not in self.knowledge_to_memory_mappings:
                            self.knowledge_to_memory_mappings[node_id] = []
                            
                        if memory_id not in self.knowledge_to_memory_mappings[node_id]:
                            self.knowledge_to_memory_mappings[node_id].append(memory_id)
                            
                        self.memory_to_knowledge_mappings[memory_id] = node_id
            
            # Add memory information to the enhanced node
            if memory_connections:
                enhanced_node["memory_connections"] = memory_connections
            if memory_context:
                enhanced_node["memory_context"] = memory_context
                
            enhanced_results.append(enhanced_node)
            
        return enhanced_results
    
    async def _extract_knowledge_facts(self, memory: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract factual knowledge from a memory."""
        memory_text = memory.get("memory_text", "")
        metadata = memory.get("metadata", {})
        
        # If reasoning core is available, use it for extraction
        if self.reasoning_core and hasattr(self.reasoning_core, "extract_facts"):
            return await self.reasoning_core.extract_facts(memory_text, metadata)
        
        # Simple fallback extraction
        # This is a simplistic approach - a real implementation would be more sophisticated
        facts = []
        
        # Create a simple fact from the memory
        facts.append({
            "statement": f"Nyx experienced: {memory_text}",
            "confidence": 0.7,
            "context": "Derived from personal memory",
            "domain": "personal_experience",
            "topic": memory.get("tags", ["experience"])[0] if memory.get("tags") else "experience"
        })
        
        return facts
    
    async def _format_knowledge_as_memory(self, node_data: Dict[str, Any]) -> str:
        """Format a knowledge node as memory text."""
        # Extract content
        content = node_data.get("content", {})
        node_type = node_data.get("type", "knowledge")
        
        # Different formatting based on node type
        if node_type == "fact":
            # Extract fact statement
            fact = content.get("fact", "")
            context = content.get("context", "")
            
            if context:
                return f"Knowledge fact: {fact} (Context: {context})"
            else:
                return f"Knowledge fact: {fact}"
                
        elif node_type == "concept":
            # Format concept
            concept = content.get("name", "")
            definition = content.get("definition", "")
            
            return f"Concept: {concept} - {definition}"
            
        elif node_type == "rule":
            # Format rule
            rule = content.get("rule", "")
            conditions = content.get("conditions", "")
            
            if conditions:
                return f"Rule: {rule} when {conditions}"
            else:
                return f"Rule: {rule}"
        
        # Default format for other types
        return f"Knowledge ({node_type}): {json.dumps(content)}"
    
    async def _format_knowledge_for_reasoning(self, knowledge_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format knowledge nodes for reasoning."""
        formatted = []
        
        for node in knowledge_results:
            # Extract key information
            node_id = node.get("id")
            node_type = node.get("type", "knowledge")
            content = node.get("content", {})
            confidence = node.get("confidence", 0.5)
            
            formatted_node = {
                "id": node_id,
                "type": node_type,
                "confidence": confidence
            }
            
            # Format content based on type
            if node_type == "fact":
                formatted_node["statement"] = content.get("fact", "")
                if "context" in content:
                    formatted_node["context"] = content["context"]
                    
            elif node_type == "concept":
                formatted_node["concept"] = content.get("name", "")
                formatted_node["definition"] = content.get("definition", "")
                
            elif node_type == "rule":
                formatted_node["rule"] = content.get("rule", "")
                if "conditions" in content:
                    formatted_node["conditions"] = content["conditions"]
            
            # For other types, include full content
            else:
                formatted_node["content"] = content
                
            formatted.append(formatted_node)
            
        return formatted
    
    async def _extract_topics_from_reflection(self, reflection: Dict[str, Any]) -> List[str]:
        """Extract key topics from a reflection for knowledge enhancement."""
        reflection_text = reflection.get("memory_text", "")
        
        # Use reasoning core if available
        if self.reasoning_core and hasattr(self.reasoning_core, "extract_topics"):
            return await self.reasoning_core.extract_topics(reflection_text)
        
        # Fallback to simple topic extraction
        # Get topics from tags
        topics = reflection.get("tags", [])
        
        # Filter out generic tags
        generic_tags = ["reflection", "memory", "experience", "personal"]
        topics = [t for t in topics if t not in generic_tags]
        
        # If no topics found, add a default one
        if not topics:
            topics.append("personal_reflection")
            
        return topics

# Function to create the bridge
def create_knowledge_memory_reasoning_bridge(nyx_brain):
    """Create a knowledge-memory-reasoning bridge for the given brain."""
    return KnowledgeMemoryReasoningBridge(
        knowledge_core=nyx_brain.knowledge_core if hasattr(nyx_brain, "knowledge_core") else None,
        memory_orchestrator=nyx_brain.memory_orchestrator if hasattr(nyx_brain, "memory_orchestrator") else None,
        reasoning_core=nyx_brain.reasoning_core if hasattr(nyx_brain, "reasoning_core") else None,
        reflection_engine=nyx_brain.reflection_engine if hasattr(nyx_brain, "reflection_engine") else None
    )
