# memory/memory_retriever.py

"""
Memory Retrieval Agent (OpenAI-native)

Removes LangChain/HuggingFace; uses OpenAI Embeddings + Responses.
Retrieves memories via MemoryEmbeddingService and synthesizes analysis
with OpenAI Responses API (JSON output).
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Import our memory service
from config.pipeline_config import PipelineConfig
from memory.memory_service import MemoryEmbeddingService
from rag import ask as rag_ask

# Configure logging
logger = logging.getLogger(__name__)

# Define Pydantic models for structured output
class MemoryInsight(BaseModel):
    """A structured insight from a set of memories."""
    insight: str = Field(description="The insight derived from the memories")
    relevance: float = Field(description="Relevance score from 0 to 1")
    supporting_memory_ids: List[str] = Field(description="IDs of memories supporting this insight")

class MemoryAnalysis(BaseModel):
    """Analysis of a set of memories."""
    primary_theme: str = Field(description="The primary theme of the memories")
    insights: List[MemoryInsight] = Field(description="List of insights derived from the memories")
    suggested_response: str = Field(description="Suggested response incorporating the memories")

class MemoryRetrieverAgent:
    """Agent for retrieving and analyzing memories relevant to a query using native OpenAI APIs."""
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        llm_type: str = "openai",
        memory_service: Optional[MemoryEmbeddingService] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the memory retriever agent.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            llm_type: Type of LLM to use (only "openai" is supported)
            memory_service: Optional memory service instance
            config: Optional configuration
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.llm_type = llm_type
        self.config = config or {}
        
        # Use provided memory service or create a new one
        self.memory_service = memory_service
        
        # Initialize variables to be set later
        self._openai: Optional[OpenAI] = None
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the memory retriever agent."""
        if self.initialized:
            return
        
        try:
            # 1. Set up memory service if not provided
            if not self.memory_service:
                self.memory_service = MemoryEmbeddingService(
                    user_id=self.user_id,
                    conversation_id=self.conversation_id,
                    vector_store_type=self.config.get("vector_store_type", "chroma"),
                    embedding_model=self.config.get("embedding_model", "local"),
                    config=self.config
                )
                await self.memory_service.initialize()

            if self.llm_type not in ("openai", None):
                raise ValueError("Only 'openai' llm_type is supported after LangChain removal")

            # 2. Create OpenAI client
            self._openai = OpenAI()

            self.initialized = True
            logger.info("Memory retriever agent initialized (OpenAI-native)")
            
        except Exception as e:
            logger.error(f"Error initializing memory retriever agent: {e}")
            raise
    
    async def _analyze_with_openai(self, query: str, formatted_memories: str) -> Dict[str, Any]:
        """Call the OpenAI Responses API to get structured JSON analysis."""
        if not self._openai:
            self._openai = OpenAI()

        system_prompt = (
            "You analyze retrieved memories and respond with JSON containing the keys: "
            "primary_theme (string), insights (array of objects with insight, relevance, supporting_memory_ids), "
            "and suggested_response (string). Return JSON only."
        )
        user_prompt = (
            f"Query:\n{query}\n\nRetrieved Memories:\n{formatted_memories}\n\n"
            "Respond strictly with valid JSON matching the schema."
        )

        def _invoke() -> str:
            response = self._openai.responses.create(
                model=self.config.get("openai_model_name", "gpt-5-nano"),
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.output_text

        raw_text = await asyncio.to_thread(_invoke)
        try:
            payload = json.loads(raw_text)
            try:
                validated = MemoryAnalysis.model_validate(payload)
                payload = validated.model_dump()
            except ValidationError as exc:
                logger.warning("OpenAI analysis payload failed validation: %s", exc)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response from OpenAI; returning fallback payload")
            payload = {
                "primary_theme": "analysis",
                "insights": [],
                "suggested_response": raw_text.strip(),
            }
        return payload
    
    async def retrieve_memories(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to a query.
        
        Args:
            query: Search query
            entity_types: List of entity types to search
            top_k: Number of results to return
            threshold: Relevance threshold (0-1)
            
        Returns:
            List of relevant memories
        """
        if not self.initialized:
            await self.initialize()
        
        # Default to searching all entity types if none specified
        if not entity_types:
            entity_types = ["memory", "npc", "location", "narrative"]
        
        async def _legacy_retrieval() -> List[Dict[str, Any]]:
            collected: List[Dict[str, Any]] = []

            for entity_type in entity_types:
                try:
                    memories = await self.memory_service.search_memories(
                        query_text=query,
                        entity_type=entity_type,
                        top_k=top_k,
                        fetch_content=True
                    )
                except Exception as exc:
                    logger.error(f"Error retrieving {entity_type} memories: {exc}")
                    continue

                for memory in memories:
                    relevance = float(memory.get("relevance", 0.0))
                    if relevance >= threshold:
                        collected.append(memory)

            collected.sort(key=lambda item: item.get("relevance", 0.0), reverse=True)
            return collected[:top_k]

        response = await rag_ask(
            query,
            mode="retrieval",
            limit=top_k,
            metadata={
                "component": "memory_retriever",
                "operation": "retrieve_memories",
                "entity_types": entity_types,
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "threshold": threshold,
            },
            legacy_fallback=_legacy_retrieval,
            backend=PipelineConfig.get_rag_backend().value,
        )

        documents = response.get("documents", [])
        return self._normalise_documents(documents, threshold, top_k)

    async def analyze_memories(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze memories to extract insights and suggested responses (OpenAI-native)."""
        if not self.initialized:
            await self.initialize()

        formatted_memories = ""
        for idx, memory in enumerate(memories, start=1):
            metadata = memory.get("metadata", {})
            memory_text = memory.get("memory_text", metadata.get("content", ""))
            entity_type = metadata.get("entity_type", "unknown")
            created_at = metadata.get("timestamp", "unknown time")
            relevance = float(memory.get("relevance", 0.0))

            formatted_memories += (
                f"Memory {idx} [ID: {memory.get('id')}, Type: {entity_type}, "
                f"Relevance: {relevance:.2f}, Timestamp: {created_at}]:\n{memory_text}\n\n"
            )

        return await self._analyze_with_openai(query, formatted_memories)
    
    async def retrieve_and_analyze(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Retrieve and analyze memories relevant to a query.
        
        Args:
            query: Search query
            entity_types: List of entity types to search
            top_k: Number of results to return
            threshold: Relevance threshold (0-1)
            
        Returns:
            Dictionary with retrieved memories and analysis
        """
        if not self.initialized:
            await self.initialize()
        
        # 1. Retrieve relevant memories
        memories = await self.retrieve_memories(
            query=query,
            entity_types=entity_types,
            top_k=top_k,
            threshold=threshold
        )
        
        # 2. If no relevant memories found, return early
        if not memories:
            return {
                "found_memories": False,
                "memories": [],
                "analysis": None
            }
        
        # 3. Analyze memories
        analysis = await self.analyze_memories(query, memories)
        
        # 4. Return combined result
        return {
            "found_memories": True,
            "memories": memories,
            "analysis": analysis
        }

    async def close(self) -> None:
        """Close the memory retriever agent."""
        if self.memory_service:
            await self.memory_service.close()

        self.initialized = False

    @staticmethod
    def _normalise_documents(
        documents: Any,
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        normalised: List[Dict[str, Any]] = []

        if not isinstance(documents, list):
            documents = [documents]

        for raw in documents:
            if not isinstance(raw, dict):
                continue

            entry = dict(raw)
            metadata = entry.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            entry["metadata"] = metadata

            if "entity_type" not in metadata and entry.get("entity_type"):
                metadata["entity_type"] = entry["entity_type"]

            relevance = entry.get("relevance")
            if relevance is None:
                relevance = entry.get("score") or metadata.get("relevance") or metadata.get("score")
            try:
                relevance_value = float(relevance) if relevance is not None else 0.0
            except (TypeError, ValueError):
                relevance_value = 0.0

            if relevance_value < threshold:
                continue

            entry["relevance"] = relevance_value

            text = entry.get("memory_text")
            if text is None:
                text = entry.get("text") or entry.get("content") or entry.get("value")
            if text is not None:
                entry["memory_text"] = str(text)

            normalised.append(entry)

        normalised.sort(key=lambda item: item.get("relevance", 0.0), reverse=True)
        return normalised[:top_k]
