# nyx/llm_integration.py

import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
import openai
import numpy as np


from nyx.core.orchestrator import prepare_context

logger = logging.getLogger(__name__)

# Configure API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Temperature settings for different tasks
TEMPERATURE_SETTINGS = {
    "decision": 0.7,       # More creative for responses
    "reflection": 0.5,     # Balanced for reflections
    "abstraction": 0.4,    # More deterministic for abstractions
    "introspection": 0.6,  # Slightly creative for introspection
    "memory": 0.3          # Very deterministic for memory operations
}

async def generate_text_completion(
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int = 1000,
    stop_sequences: List[str] | None = None,
    task_type: str = "decision",
) -> str:
    """
    Single-shot completion via Responses API with back-off.
    """
    temperature = temperature if temperature is not None else \
                  TEMPERATURE_SETTINGS.get(task_type, 0.7)

    system_prompt = await prepare_context(system_prompt, user_prompt)
    client = get_openai_client()

    for attempt in range(3):
        try:
            resp = await client.responses.create(
                model="gpt-4.1-nano",
                instructions=system_prompt,     # “system”
                input=user_prompt,              # “user”
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )
            return resp.output_text.strip()
        except client.error.RateLimitError:
            wait = 2 ** attempt
            logger.warning("Rate limit – retrying in %ss", wait)
            await asyncio.sleep(wait)

    logger.error("Exceeded retries for text completion.")
    return "I'm having trouble processing your request right now."


async def create_semantic_abstraction(memory_text: str) -> str:
    """
    Create a semantic abstraction from a specific memory.
    
    Args:
        memory_text: The memory text to abstract
        
    Returns:
        Abstract version of the memory
    """
    prompt = f"""
    Convert this specific observation into a general insight or pattern:
    
    Observation: {memory_text}
    
    Create a concise semantic memory that:
    1. Extracts the general principle or pattern from this specific event
    2. Forms a higher-level abstraction that could apply to similar situations
    3. Phrases it as a generalized insight rather than a specific event
    4. Keeps it under 50 words
    
    Example transformation:
    Observation: "Chase hesitated when Monica asked him about his past, changing the subject quickly."
    Semantic abstraction: "Chase appears uncomfortable discussing his past and employs deflection when questioned about it."
    """
    
    try:
        return await generate_text_completion(
            system_prompt="You are an AI that extracts semantic meaning from specific observations.",
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=100,
            task_type="abstraction"
        )
    except Exception as e:
        logger.error(f"Error creating semantic abstraction: {e}")
        # Create a simple fallback abstraction
        words = memory_text.split()
        if len(words) > 15:
            # Just take first portion and add "..." for simple fallback
            return " ".join(words[:15]) + "... [Pattern detected]"
        return memory_text + " [Pattern detected]"

async def generate_reflection(
    memory_texts: List[str],
    topic: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a reflection based on memories and optional topic.
    
    Args:
        memory_texts: List of memory texts to reflect on
        topic: Optional topic to focus the reflection
        context: Optional additional context
        
    Returns:
        Generated reflection
    """
    # Format memories for the prompt
    memories_formatted = "\n".join([f"- {text}" for text in memory_texts])
    
    topic_str = f' about "{topic}"' if topic else ""
    
    prompt = f"""
    As Nyx, create a thoughtful reflection{topic_str} based on these memories:
    
    {memories_formatted}
    
    Your reflection should:
    1. Identify patterns, themes, or insights
    2. Express an appropriate level of confidence based on the memories
    3. Use first-person perspective ("I")
    4. Be concise but insightful (100-200 words)
    5. Maintain your confident, dominant personality
    """
    
    # Add context information if provided
    if context:
        context_str = "\n\nAdditional context:\n"
        for key, value in context.items():
            context_str += f"{key}: {value}\n"
        prompt += context_str
    
    try:
        return await generate_text_completion(
            system_prompt="You are Nyx, reflecting on your memories and observations.",
            user_prompt=prompt,
            temperature=0.5,
            max_tokens=300,
            task_type="reflection"
        )
    except Exception as e:
        logger.error(f"Error generating reflection: {e}")
        # Return a simple reflection as fallback
        if memory_texts:
            return f"Based on what I've observed, {memory_texts[0]} This seems to be a pattern worth noting."
        return "I don't have enough memories to form a meaningful reflection at this time."

async def analyze_preferences(text: str) -> Dict[str, Any]:
    """
    Analyze text for user preferences and interests.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of detected preferences
    """
    prompt = f"""
    Analyze the following text for potential user preferences, interests, or behaviors:
    
    "{text}"
    
    Extract:
    1. Explicit preferences/interests (directly stated)
    2. Implicit preferences/interests (implied)
    3. Behavioral patterns or tendencies
    4. Emotional responses or triggers
    
    Format your response as a JSON object with these categories.
    """
    
    try:
        response = await generate_text_completion(
            system_prompt="You are an AI that specializes in analyzing preferences and behavior patterns from text.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=400,
            task_type="abstraction"
        )
        
        # Try to parse the response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If not valid JSON, extract sections manually
            result = {
                "explicit_preferences": [],
                "implicit_preferences": [],
                "behavioral_patterns": [],
                "emotional_responses": []
            }
            
            current_section = None
            for line in response.split("\n"):
                line = line.strip()
                
                if "explicit preferences" in line.lower():
                    current_section = "explicit_preferences"
                elif "implicit preferences" in line.lower():
                    current_section = "implicit_preferences"
                elif "behavioral patterns" in line.lower():
                    current_section = "behavioral_patterns"
                elif "emotional responses" in line.lower():
                    current_section = "emotional_responses"
                elif current_section and line.startswith("-"):
                    item = line[1:].strip()
                    if item and current_section in result:
                        result[current_section].append(item)
            
            return result
            
    except Exception as e:
        logger.error(f"Error analyzing preferences: {e}")
        return {
            "explicit_preferences": [],
            "implicit_preferences": [],
            "behavioral_patterns": [],
            "emotional_responses": [],
            "error": str(e)
        }

async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for text using OpenAI's API.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Embedding vector as list of floats
    """
    try:
        response = await openai.Embedding.acreate(
            model="text-embedding-ada-002",
            input=text
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        # Return a zero vector of the correct dimension as fallback
        return [0.0] * 1536  # ada-002 embeddings have 1536 dimensions

# Add these functions to the existing llm_integration.py file

async def get_text_embedding(text: str, model: str = "text-embedding-3-small", dimensions: Optional[int] = None) -> List[float]:
    """
    Get embedding vector for text using OpenAI's latest embedding models.
    
    Args:
        text: Text to get embedding for
        model: Embedding model to use (default: text-embedding-3-small)
        dimensions: Optional dimensions to reduce embedding size (must be less than model's default)
        
    Returns:
        List of embedding vector values
    """
    try:
        # Validate model choice
        valid_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        if model not in valid_models:
            logger.warning(f"Invalid model {model}, using text-embedding-3-small")
            model = "text-embedding-3-small"
        
        # Check dimensions parameter
        max_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if dimensions:
            if dimensions > max_dimensions[model]:
                logger.warning(f"Requested dimensions {dimensions} exceeds max {max_dimensions[model]} for {model}")
                dimensions = None
            elif dimensions < 1:
                logger.warning(f"Invalid dimensions {dimensions}, must be positive")
                dimensions = None
        
        # Clean and validate text
        text = text.replace("\n", " ").strip()
        if not text:
            logger.warning("Empty text provided for embedding, returning zero vector")
            return [0.0] * (dimensions or max_dimensions[model])
        
        # Check token count (8192 limit for all models)
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(encoding.encode(text))
            
            if num_tokens > 8192:
                logger.warning(f"Text has {num_tokens} tokens, exceeding limit of 8192. Truncating...")
                # Truncate to roughly 8000 tokens to leave some buffer
                tokens = encoding.encode(text)[:8000]
                text = encoding.decode(tokens)
        except ImportError:
            # If tiktoken not available, use character-based approximation
            # ~4 chars per token is a rough estimate
            max_chars = 32000  # ~8000 tokens
            if len(text) > max_chars:
                logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
                text = text[:max_chars] + "..."
        
        # Get client
        if not openai.api_key:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
        
        # Create embedding with retry logic
        for attempt in range(3):
            try:
                # Build request parameters
                params = {
                    "model": model,
                    "input": text,
                    "encoding_format": "float"  # Ensure we get floats, not base64
                }
                
                # Add dimensions parameter if specified
                if dimensions:
                    params["dimensions"] = dimensions
                
                # Make async request
                # Note: The latest OpenAI SDK might use a different async pattern
                if hasattr(openai, 'AsyncOpenAI'):
                    # New client style
                    client = openai.AsyncOpenAI(api_key=openai.api_key)
                    response = await client.embeddings.create(**params)
                else:
                    # Old style with asyncio.to_thread for sync client
                    response = await asyncio.to_thread(
                        openai.embeddings.create,
                        **params
                    )
                
                # Extract embedding
                embedding = response.data[0].embedding
                
                # Validate embedding
                expected_dims = dimensions or max_dimensions[model]
                if len(embedding) != expected_dims:
                    logger.warning(
                        f"Unexpected embedding dimension: {len(embedding)}, expected {expected_dims}"
                    )
                
                # Ensure all values are floats
                return list(map(float, embedding))
                
            except openai.RateLimitError as e:
                if attempt < 2:
                    wait_time = 2 ** (attempt + 1)  # 2, 4 seconds
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Rate limit exceeded after retries: {e}")
                    raise
                    
            except openai.BadRequestError as e:
                # Handle token limit errors specifically
                if "maximum context length" in str(e) or "tokens" in str(e).lower():
                    # Retry with more aggressive truncation
                    text = text[:len(text)//2] + "..."
                    logger.warning(f"Token limit error, retrying with truncated text ({len(text)} chars)")
                    if attempt < 2:
                        continue
                logger.error(f"Bad request error: {e}")
                raise
                
            except Exception as e:
                if attempt < 2:
                    wait_time = 2 ** attempt
                    logger.warning(f"Error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to get embedding after retries: {e}")
                    raise
        
    except Exception as e:
        logger.error(f"Error getting text embedding: {e}")
        
        # Return zero vector with appropriate dimensions
        default_dims = 1536  # Default to small model dimensions
        if model == "text-embedding-3-large" and not dimensions:
            default_dims = 3072
        elif dimensions:
            default_dims = dimensions
            
        return [0.0] * default_dims


# Add a helper function for cosine similarity (mentioned in the docs)
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity between -1 and 1
    """
    import numpy as np
    
    a = np.array(a)
    b = np.array(b)
    
    # Handle zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Compute cosine similarity
    return np.dot(a, b) / (norm_a * norm_b)


# Update the existing generate_embedding function to use the new models
async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for text using OpenAI's API.
    Legacy wrapper that calls get_text_embedding.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        Embedding vector as list of floats
    """
    return await get_text_embedding(text, model="text-embedding-3-small")
