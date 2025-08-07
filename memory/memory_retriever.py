# memory/memory_retriever.py

"""
Memory Retrieval Agent using LangChain

This module provides a memory retrieval agent that uses LangChain to
retrieve and synthesize relevant memory snippets for an AI system.
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Import our memory service
from memory.memory_service import MemoryEmbeddingService

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
    """
    Agent for retrieving and analyzing memories relevant to a query.
    Uses LangChain components and LLM to synthesize information.
    """
    
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
            llm_type: Type of LLM to use ("openai" or "huggingface")
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
        self.llm = None
        self.memory_analysis_chain = None
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
            
            # 2. Set up LLM
            await self._setup_llm()
            
            # 3. Set up LangChain components
            await self._setup_chains()
            
            self.initialized = True
            logger.info(f"Memory retriever agent initialized with {self.llm_type} LLM")
            
        except Exception as e:
            logger.error(f"Error initializing memory retriever agent: {e}")
            raise
    
    async def _setup_llm(self) -> None:
        """Set up the LLM component."""
        # Use asyncio.to_thread for potentially blocking operations
        loop = asyncio.get_event_loop()
        
        if self.llm_type == "openai":
            # OpenAI API requires an API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI")
            
            model_name = self.config.get("openai_model_name", "gpt-5-nano")
            temperature = self.config.get("temperature", 0.0)  # Low temperature for factual responses
            
            self.llm = await loop.run_in_executor(
                None,
                lambda: ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=openai_api_key
                )
            )
            
        elif self.llm_type == "huggingface":
            # HuggingFace Hub requires an API token
            hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
            if not hf_token:
                raise ValueError("HUGGINGFACE_API_TOKEN environment variable is required for HuggingFace")
            
            # You can use a remote model on the Hub
            model_name = self.config.get("hf_model_name", "mistralai/Mistral-7B-Instruct-v0.2")
            
            self.llm = await loop.run_in_executor(
                None,
                lambda: HuggingFaceHub(
                    repo_id=model_name,
                    huggingfacehub_api_token=hf_token,
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
            )
            
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
    
    async def _setup_chains(self) -> None:
        """Set up LangChain components."""
        # Define prompt template for memory analysis
        memory_analysis_template = """
        You are analyzing a set of memories retrieved for an AI assistant.
        Based on the query and retrieved memories, identify the main theme, 
        key insights, and suggest a response that incorporates the relevant information.
        
        Query: {query}
        
        Retrieved Memories:
        {memories}
        
        Analyze these memories and respond with a structured summary including:
        1. The primary theme connecting these memories
        2. Key insights derived from the memories with their relevance scores
        3. A suggested response for the AI assistant that subtly incorporates the relevant information
        
        {format_instructions}
        """
        
        # Output parser for structured output
        parser = PydanticOutputParser(pydantic_object=MemoryAnalysis)
        
        # Create prompt with template and output parser
        prompt = PromptTemplate(
            template=memory_analysis_template,
            input_variables=["query", "memories"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create the chain
        self.memory_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=parser
        )
    
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
        
        # Collect memories from all entity types
        all_memories = []
        
        for entity_type in entity_types:
            try:
                # Search for memories of this entity type
                memories = await self.memory_service.search_memories(
                    query_text=query,
                    entity_type=entity_type,
                    top_k=top_k,
                    fetch_content=True
                )
                
                # Filter by threshold
                memories = [m for m in memories if m["relevance"] >= threshold]
                
                all_memories.extend(memories)
                
            except Exception as e:
                logger.error(f"Error retrieving {entity_type} memories: {e}")
        
        # Sort by relevance
        all_memories.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Limit to top_k overall
        return all_memories[:top_k]
    
    async def analyze_memories(
        self,
        query: str,
        memories: List[Dict[str, Any]]
    ) -> MemoryAnalysis:
        """
        Analyze memories to extract insights and suggested responses.
        
        Args:
            query: Original query
            memories: List of retrieved memories
            
        Returns:
            MemoryAnalysis object with insights and suggested response
        """
        if not self.initialized:
            await self.initialize()
        
        # Format memories for the prompt
        formatted_memories = ""
        for i, memory in enumerate(memories):
            memory_text = memory.get("memory_text", memory.get("metadata", {}).get("content", ""))
            entity_type = memory.get("metadata", {}).get("entity_type", "unknown")
            created_at = memory.get("metadata", {}).get("timestamp", "unknown time")
            relevance = memory.get("relevance", 0.0)
            
            formatted_memories += f"Memory {i+1} [ID: {memory['id']}, Type: {entity_type}, Relevance: {relevance:.2f}]:\n"
            formatted_memories += f"{memory_text}\n\n"
        
        # Use the memory analysis chain to analyze memories
        loop = asyncio.get_event_loop()
        
        # Run the LLM chain (potentially blocking) in an executor
        result = await loop.run_in_executor(
            None,
            lambda: self.memory_analysis_chain.run(
                query=query,
                memories=formatted_memories
            )
        )
        
        return result
    
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
