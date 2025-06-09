# nyx/llm_integration.py

import os
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
import openai

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
    temperature: float = None,
    max_tokens: int = 1000,
    stop_sequences: List[str] = None,
    task_type: str = "decision"
) -> str:
    """
    Generate a text completion using GPT model.
    
    Args:
        system_prompt: System prompt to guide the model's behavior
        user_prompt: User prompt / query
        temperature: Temperature setting (0.0-1.0)
        max_tokens: Maximum tokens to generate
        stop_sequences: Optional sequences to stop generation
        task_type: Type of task for default temperature
        
    Returns:
        Generated text response
    """
    # Use task-specific temperature if not specified
    if temperature is None:
        temperature = TEMPERATURE_SETTINGS.get(task_type, 0.7)

    # Inject relevant memories into the system prompt
    system_prompt = await prepare_context(system_prompt, user_prompt)
    
    # Create messages array
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Call OpenAI API with simple exponential backoff on rate limits
    response = None
    for attempt in range(3):
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4.1-nano",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
            )
            break
        except openai.error.RateLimitError:
            wait = 2 ** attempt
            logger.warning(
                f"Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/3)"
            )
            await asyncio.sleep(wait)
    else:
        logger.error("Exceeded retry attempts due to rate limiting")
        return "I'm having trouble processing your request at the moment."

    try:
        # Extract and return the text
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating text completion: {e}")
        return (
            "I'm having trouble processing your request at the moment. "
            + str(e)[:50]
        )

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

async def get_text_embedding(text: str) -> np.ndarray:
    """
    Get embedding vector for text.
    
    Args:
        text: Text to get embedding for
        
    Returns:
        numpy array with embedding vector
    """
    try:
        # In production, this would call an actual LLM API
        # For example, with OpenAI:
        # response = await openai.Embedding.acreate(
        #     input=text,
        #     model="text-embedding-ada-002"
        # )
        # return np.array(response["data"][0]["embedding"])
        
        # For this implementation, we'll use a simpler approach
        # This is a placeholder - not for production use
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create a simple embedding based on TF-IDF
        vectorizer = TfidfVectorizer(max_features=1536)
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Convert to dense array and pad/truncate to 1536 dimensions
        dense_vector = tfidf_matrix.toarray()[0]
        padded_vector = np.zeros(1536)
        padded_vector[:min(len(dense_vector), 1536)] = dense_vector[:1536]
        
        return padded_vector
        
    except Exception as e:
        logger.error(f"Error getting text embedding: {e}")
        # Return a zero vector as fallback
        return np.zeros(1536)
