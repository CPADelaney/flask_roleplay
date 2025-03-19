# nyx/eternal/openai_api.py

from typing import Dict, Any, Optional, Callable, Awaitable
import asyncio
import logging
import time
import os

from logic.chatgpt_integration import get_chatgpt_response
from .openai_facade import NyxAgentsFacade

logger = logging.getLogger(__name__)

class OpenAIAgentsAPI:
    """API for OpenAI Agents SDK integration that leverages existing ChatGPT integration"""
    
    def __init__(self, 
               api_key: Optional[str] = None,
               original_processor: Optional[Callable] = None):
        """
        Initialize the API.
        
        Args:
            api_key: OpenAI API key
            original_processor: Function that processes with original Nyx system
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.original_processor = original_processor
        self.facades = {}
    
    def get_facade(self, user_id: int, conversation_id: int) -> NyxAgentsFacade:
        """Get or create facade instance"""
        key = f"{user_id}:{conversation_id}"
        
        if key not in self.facades:
            self.facades[key] = NyxAgentsFacade(user_id, conversation_id, self.api_key)
        
        return self.facades[key]
    
    async def process_with_enhancement(self,
                                 user_id: int,
                                 conversation_id: int,
                                 user_input: str,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input with original Nyx, enhanced by OpenAI components.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: User's input message
            context: Context information
            
        Returns:
            Processing result
        """
        if not self.original_processor:
            return {
                "success": False,
                "error": "Original processor not provided"
            }
        
        # Get facade
        facade = self.get_facade(user_id, conversation_id)
        await facade.initialize()
        
        # Apply enhancements
        start_time = time.time()
        enhancement = await facade.enhance_processing(user_input, context or {})
        
        # Create enhanced context
        enhanced_context = context.copy() if context else {}
        if enhancement.get("selected_strategy"):
            enhanced_context["strategy"] = enhancement["selected_strategy"]
        if enhancement.get("feature_importance"):
            enhanced_context["feature_importance"] = enhancement["feature_importance"]
        
        # Call original processor with the enhanced context
        result = await self.original_processor(user_id, conversation_id, user_input, enhanced_context)
        
        # Evaluate response
        if result.get("success", False) and "response" in result:
            evaluation = await facade.evaluate_response(
                result["response"],
                {
                    **enhanced_context,
                    "response_time": time.time() - start_time
                }
            )
            result["evaluation"] = evaluation
        
        # Return enhanced result
        return {
            **result,
            "enhancement": enhancement
        }
    
    async def process_standalone(self,
                           user_id: int,
                           conversation_id: int,
                           user_input: str,
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process user input with standalone implementation using existing ChatGPT integration.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: User's input message
            context: Context information
            
        Returns:
            Processing result
        """
        try:
            # Get aggregated context from existing logic
            from logic.aggregator import get_aggregated_roleplay_context
            aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
            
            # Check if context contains universal update
            if context and context.get("universal_update"):
                # Apply universal update if provided
                from logic.universal_updater import apply_universal_updates
                apply_universal_updates(
                    user_id,
                    conversation_id,
                    context["universal_update"]
                )
                # Refresh aggregator data post-update
                aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
            
            # Get reflection setting from context
            reflection_enabled = context.get("reflection_enabled", False)
            
            # Use existing ChatGPT integration
            response = get_chatgpt_response(
                conversation_id, 
                aggregator_data, 
                user_input, 
                reflection_enabled
            )
            
            # Process response
            if response["type"] == "function_call":
                return {
                    "success": True,
                    "type": "function_call",
                    "function_name": response["function_name"],
                    "function_args": response["function_args"],
                    "message": response["function_args"].get("narrative", ""),
                    "tokens_used": response["tokens_used"]
                }
            else:
                return {
                    "success": True,
                    "type": "text",
                    "response": response["response"],
                    "message": response["response"],
                    "tokens_used": response["tokens_used"]
                }
                
        except Exception as e:
            logger.error(f"Error in standalone processing: {e}")
            return {
                "success": False,
                "error": str(e)
            }
