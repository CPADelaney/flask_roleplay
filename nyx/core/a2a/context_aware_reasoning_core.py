# nyx/core/a2a/context_aware_reasoning_core.py

import numpy as np
import torch
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime
import logging
from collections import defaultdict
import networkx as nx
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import asyncio
from concurrent.futures import ThreadPoolExecutor

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - semantic similarity will use fallback methods")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - syntactic analysis will be disabled")

try:
    import nltk
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - using basic stopword list")

# Complete __init__ method for ContextAwareReasoningCore

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import logging
import asyncio

# Import at the top of the file
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available - semantic similarity will use fallback methods")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - syntactic analysis will be disabled")

try:
    import nltk
    # Download required NLTK data
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - using basic stopword list")

class ContextAwareReasoningCore(ContextAwareModule):
    """
    Context-aware wrapper for ReasoningCore with full A2A integration
    """
    
    def __init__(self, original_reasoning_core, lazy_load: bool = False, 
                 max_workers: int = 4, model_cache_dir: str = None):
        """
        Initialize the context-aware reasoning core.
        
        Args:
            original_reasoning_core: The original ReasoningCore instance
            lazy_load: If True, defer loading heavy models until first use
            max_workers: Maximum worker threads for async operations
            model_cache_dir: Directory to cache downloaded models
        """
        super().__init__("reasoning_core")
        self.original_core = original_reasoning_core
        self.integration_layer = None             
        
        # ========================================================================
        # A2A INTEGRATION SETUP
        # ========================================================================
        
        self.context_subscriptions = [
            "emotional_state_update", "memory_retrieval_complete", 
            "goal_context_available", "knowledge_update",
            "perception_input", "multimodal_integration",
            "causal_discovery_request", "conceptual_blend_request",
            "intervention_request", "counterfactual_query"
        ]
        
        # ========================================================================
        # ACTIVE REASONING STATE
        # ========================================================================
        
        # Track active reasoning processes
        self.active_models = set()
        self.active_spaces = set()
        self.active_interventions = set()
        self.reasoning_context = {}
        
        # ========================================================================
        # NLP AND SEMANTIC MODELS
        # ========================================================================
        
        # Configuration
        self.lazy_load = lazy_load
        self.model_cache_dir = model_cache_dir
        self._models_loaded = False
        
        # Initialize model placeholders
        self.semantic_model = None
        self.nlp = None
        
        # Load models immediately if not lazy loading
        if not lazy_load:
            self._initialize_nlp_models()
        
        # ========================================================================
        # THREAD POOL FOR ASYNC OPERATIONS
        # ========================================================================
        
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="reasoning_worker"
        )
        
        # ========================================================================
        # MODULE DISCOVERY AND PERSISTENCE
        # ========================================================================
        
        # Module discovery registry
        self.discovered_modules = {}
        
        # Context persistence tracking
        self._last_user_input = ""
        self._last_session_id = ""
        self._last_emotional_state = {}
        self._last_goal_context = {}
        self._last_memory_context = {}
        
        # ========================================================================
        # PERFORMANCE METRICS AND MONITORING
        # ========================================================================
        
        # Performance metrics
        self._contexts_processed = 0
        self._updates_received = 0
        self._updates_sent = 0
        self._avg_processing_time = 0.0
        self._processing_times = []  # For calculating average
        
        # Detailed metrics tracking
        self._processing_metrics = defaultdict(list)  # Track per-operation metrics
        self._handler_success_count = defaultdict(int)  # Track handler success rates
        self._model_inference_times = defaultdict(list)  # Track model inference times
        self._cache_hit_rates = defaultdict(lambda: {"hits": 0, "misses": 0})
        
        # ========================================================================
        # CACHING AND OPTIMIZATION
        # ========================================================================
        
        # Semantic similarity cache (LRU-style)
        self._similarity_cache = {}
        self._max_cache_size = 10000
        
        # Community detection cache
        self._community_cache = {}
        self._community_cache_ttl = 300  # 5 minutes TTL
        
        # ========================================================================
        # ERROR TRACKING
        # ========================================================================
        
        self._error_counts = defaultdict(int)
        self._last_errors = defaultdict(list)
        self._max_error_history = 10
        
        # ========================================================================
        # AUTO-SAVE AND BACKGROUND TASKS
        # ========================================================================
        
        # Auto-save flag
        self._auto_save_enabled = False
        self._background_tasks = []
        self._discovery_completed = False
        
        # ========================================================================
        # STARTUP TASKS
        # ========================================================================
        
        # Schedule initial module discovery
        discovery_task = asyncio.create_task(self.discover_modules())
        self._background_tasks.append(discovery_task)
        
        # Log initialization
        logger.info(f"ContextAwareReasoningCore initialized with {max_workers} workers")
        logger.info(f"Lazy loading: {lazy_load}, Model cache: {model_cache_dir}")
    
    def _initialize_nlp_models(self):
        """
        Initialize NLP models with proper error handling and fallbacks.
        This is called either during __init__ or on first use if lazy loading.
        """
        if self._models_loaded:
            return
        
        logger.info("Initializing NLP models...")
        
        # Initialize sentence transformer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use cache directory if specified
                cache_folder = self.model_cache_dir if self.model_cache_dir else None
                
                # Try to load the model
                self.semantic_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    cache_folder=cache_folder
                )
                
                # Warm up the model with a test encoding
                _ = self.semantic_model.encode(["test"], show_progress_bar=False)
                
                logger.info("Sentence transformer model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load sentence transformer: {e}")
                logger.warning("Semantic similarity will use fallback methods")
                self.semantic_model = None
        
        # Initialize spaCy
        if SPACY_AVAILABLE:
            try:
                # Try to load the English model
                self.nlp = spacy.load('en_core_web_sm')
                
                # Disable unnecessary components for speed
                disabled_components = ['ner', 'textcat']  # Keep tagger and parser
                for component in disabled_components:
                    if component in self.nlp.pipe_names:
                        self.nlp.disable_pipe(component)
                
                # Set max length for processing
                self.nlp.max_length = 1000000  # 1M characters
                
                # Warm up the model
                _ = self.nlp("test")
                
                logger.info("spaCy model loaded successfully")
                
            except OSError:
                logger.error("spaCy model 'en_core_web_sm' not found")
                logger.info("Run: python -m spacy download en_core_web_sm")
                self.nlp = None
            except Exception as e:
                logger.error(f"Failed to load spaCy: {e}")
                self.nlp = None
        
        self._models_loaded = True

    def set_integration_layer(self, integration_layer):
        """Set the integration layer reference and trigger discovery"""
        self.integration_layer = integration_layer
        
        # Now trigger discovery after integration layer is set
        if not self._discovery_completed:
            discovery_task = asyncio.create_task(self.discover_modules())
            self._background_tasks.append(discovery_task)
            self._discovery_completed = True
        
    def _ensure_models_loaded(self):
        """Ensure models are loaded (for lazy loading)"""
        if not self._models_loaded:
            self._initialize_nlp_models()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        # Shutdown thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        # Cancel background tasks
        if hasattr(self, '_background_tasks'):
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
    
    # ========================================================================
    # PERFORMANCE MONITORING METHODS
    # ========================================================================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        stats = {
            "contexts_processed": self._contexts_processed,
            "updates_received": self._updates_received,
            "updates_sent": self._updates_sent,
            "avg_processing_time": self._avg_processing_time,
            "handler_success_rates": {},
            "model_performance": {},
            "cache_performance": {},
            "error_summary": {}
        }
        
        # Calculate handler success rates
        for handler, count in self._handler_success_count.items():
            total_calls = count + self._error_counts.get(handler, 0)
            if total_calls > 0:
                stats["handler_success_rates"][handler] = count / total_calls
        
        # Model inference statistics
        for model_name, times in self._model_inference_times.items():
            if times:
                stats["model_performance"][model_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "calls": len(times)
                }
        
        # Cache hit rates
        for cache_name, cache_stats in self._cache_hit_rates.items():
            total = cache_stats["hits"] + cache_stats["misses"]
            if total > 0:
                stats["cache_performance"][cache_name] = {
                    "hit_rate": cache_stats["hits"] / total,
                    "total_requests": total
                }
        
        # Error summary
        for error_type, count in self._error_counts.items():
            stats["error_summary"][error_type] = {
                "count": count,
                "recent_errors": self._last_errors.get(error_type, [])[-3:]  # Last 3 errors
            }
        
        return stats
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        self._similarity_cache.clear()
        self._community_cache.clear()
        logger.info("Caches cleared")

    async def send_context_update(self, update_type: str, data: Dict[str, Any], 
                                  priority: ContextPriority = ContextPriority.NORMAL,
                                  target_modules: List[str] = None,
                                  scope: ContextScope = ContextScope.GLOBAL) -> bool:
        """
        Send context update to other modules through the integration layer.
        
        Args:
            update_type: Type of update (e.g., "reasoning_context_available")
            data: Update data to send
            priority: Priority level for the update
            target_modules: Specific modules to target (None = broadcast to all)
            scope: Scope of the update (LOCAL, GLOBAL, or CROSS_SESSION)
            
        Returns:
            bool: True if update was sent successfully
        """
        try:
            # Track for metrics
            self._updates_sent += 1
            
            # Create context update object
            update = ContextUpdate(
                source_module=self.module_id,
                update_type=update_type,
                data=data,
                priority=priority,
                scope=scope,
                timestamp=datetime.now(),
                session_id=getattr(self, '_last_session_id', None)
            )
            
            # Add metadata
            update.metadata = {
                "reasoning_context": {
                    "active_models": list(self.active_models),
                    "active_spaces": list(self.active_spaces),
                    "confidence": data.get("confidence", 0.0)
                }
            }
            
            # Log the update
            logger.debug(f"ReasoningCore sending {update_type} update with priority {priority.name}")
            
            # Send through integration layer
            if target_modules:
                # Send to specific modules
                success = True
                for target_module in target_modules:
                    # Check if module is available
                    if not await self.is_module_available(target_module):
                        logger.warning(f"Target module {target_module} not available")
                        continue
                        
                    try:
                        await self.integration_layer.send_update_to_module(
                            target_module=target_module,
                            update=update
                        )
                    except Exception as e:
                        logger.error(f"Failed to send update to {target_module}: {e}")
                        success = False
                
                return success
            else:
                # Broadcast to all subscribed modules
                await self.integration_layer.broadcast_update(update)
                return True
                
        except Exception as e:
            logger.error(f"Error sending context update: {e}")
            return False
    
    async def send_priority_update(self, update_type: str, data: Dict[str, Any],
                                   target_modules: List[str] = None) -> bool:
        """
        Send high-priority context update that requires immediate attention.
        
        Args:
            update_type: Type of update
            data: Update data
            target_modules: Specific modules to target
            
        Returns:
            bool: True if update was sent successfully
        """
        return await self.send_context_update(
            update_type=update_type,
            data=data,
            priority=ContextPriority.CRITICAL,
            target_modules=target_modules,
            scope=ContextScope.GLOBAL
        )
    
    async def request_context_from_module(self, module_name: str, request_type: str,
                                          request_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Request specific context information from another module.
        
        Args:
            module_name: Name of the module to request from
            request_type: Type of request
            request_data: Additional request parameters
            
        Returns:
            Response data from the module, or None if request failed
        """
        try:
            # Check if module is available
            if not await self.is_module_available(module_name):
                logger.warning(f"Module {module_name} not available for request")
                return None
            
            # Create request update
            request_id = f"{self.module_id}_request_{datetime.now().timestamp()}"
            
            await self.send_context_update(
                update_type=f"{module_name}_request",
                data={
                    "request_id": request_id,
                    "request_type": request_type,
                    "request_data": request_data or {},
                    "response_needed": True
                },
                priority=ContextPriority.HIGH,
                target_modules=[module_name]
            )
            
            # Wait for response (with timeout)
            response = await self._wait_for_response(request_id, timeout=5.0)
            
            return response
            
        except Exception as e:
            logger.error(f"Error requesting context from {module_name}: {e}")
            return None
    
    async def _wait_for_response(self, request_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Wait for a response to a specific request.
        
        Args:
            request_id: ID of the request to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Response data or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if response has been received
            messages = await self.get_cross_module_messages()
            
            for source_module, module_messages in messages.items():
                for message in module_messages:
                    if message.get("response_to") == request_id:
                        return message.get("data", {})
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
        
        logger.warning(f"Timeout waiting for response to request {request_id}")
        return None
    
    async def broadcast_discovery(self, discovery_type: str, discovery_data: Dict[str, Any]):
        """
        Broadcast a discovery or insight to all interested modules.
        
        Args:
            discovery_type: Type of discovery (e.g., "causal_pattern", "novel_concept")
            discovery_data: Details of the discovery
        """
        await self.send_context_update(
            update_type=f"reasoning_discovery_{discovery_type}",
            data={
                "discovery_type": discovery_type,
                "discovery_data": discovery_data,
                "timestamp": datetime.now().isoformat(),
                "confidence": discovery_data.get("confidence", 0.7),
                "models_involved": list(self.active_models),
                "spaces_involved": list(self.active_spaces)
            },
            priority=ContextPriority.HIGH,
            scope=ContextScope.GLOBAL
        )
        
        logger.info(f"Broadcasted {discovery_type} discovery to all modules")
    
    # Helper method to get cross-module messages (if not already defined)
    async def get_cross_module_messages(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get messages from other modules since last check.
        
        Returns:
            Dictionary mapping module names to their messages
        """
        if hasattr(self, 'integration_layer'):
            return await self.integration_layer.get_messages_for_module(self.module_id)
        else:
            # Fallback if integration layer not available
            return {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize reasoning processing for this context"""
        # Track context for persistence
        self._last_user_input = context.user_input
        self._last_session_id = context.session_context.get("session_id", "")
        self._last_emotional_state = context.emotional_state or {}
        self._last_goal_context = context.goal_context or {}
        self._last_memory_context = context.memory_context or {}
        self._contexts_processed += 1
        
        start_time = time.time()
        logger.debug(f"ReasoningCore received context for user: {context.user_id}")
        
        # Analyze input for reasoning-related content
        reasoning_implications = await self._analyze_input_for_reasoning(context.user_input)
        
        # Get relevant models and spaces for context
        relevant_models = await self._get_contextually_relevant_models(context)
        relevant_spaces = await self._get_contextually_relevant_spaces(context)
        
        # Send initial reasoning context to other modules
        await self.send_context_update(
            update_type="reasoning_context_available",
            data={
                "reasoning_implications": reasoning_implications,
                "available_models": relevant_models,
                "available_spaces": relevant_spaces,
                "active_reasoning_type": reasoning_implications.get("reasoning_type", "none"),
                "confidence": reasoning_implications.get("confidence", 0.0)
            },
            priority=ContextPriority.HIGH
        )
        # Track processing time
        processing_time = time.time() - start_time
        self._processing_times.append(processing_time)
        if len(self._processing_times) > 100:  # Keep last 100
            self._processing_times.pop(0)
        self._avg_processing_time = sum(self._processing_times) / len(self._processing_times)
    
    async def on_context_update(self, update: ContextUpdate):
        """
        Handle updates from other modules that affect reasoning.
        Production version with comprehensive error handling and validation.
        """
        start_time = time.time()
        
        try:
            # Track metrics
            self._updates_received += 1
            
            # Validate update
            if not self._validate_update(update):
                logger.warning(f"Invalid update received: {update.update_type}")
                return
            
            # Log update receipt
            logger.debug(f"ReasoningCore received {update.update_type} from {update.source_module}")
            
            # Route to appropriate handler
            handlers = {
                "perception_input": self._handle_perception_update,
                "causal_discovery_request": self._handle_causal_discovery_request,
                "conceptual_blend_request": self._handle_conceptual_blend_request,
                "emotional_state_update": self._handle_emotional_update,
                "goal_context_available": self._handle_goal_update,
                "memory_retrieval_complete": self._handle_memory_update,
                "intervention_request": self._handle_intervention_request,
                "counterfactual_query": self._handle_counterfactual_query,
                "knowledge_update": self._handle_knowledge_update,
                "multimodal_integration": self._handle_multimodal_update
            }
            
            # Check for handler
            handler = handlers.get(update.update_type)
            
            if handler:
                # Execute handler asynchronously
                try:
                    await handler(update)
                    
                    # Track success
                    if not hasattr(self, '_handler_success_count'):
                        self._handler_success_count = defaultdict(int)
                    self._handler_success_count[update.update_type] += 1
                    
                except Exception as e:
                    logger.error(f"Error in handler for {update.update_type}: {e}", exc_info=True)
                    
                    # Send error notification if critical
                    if update.priority == ContextPriority.CRITICAL:
                        await self.send_context_update(
                            update_type="reasoning_error",
                            data={
                                "error_type": "handler_failure",
                                "original_update": update.update_type,
                                "error_message": str(e)
                            },
                            priority=ContextPriority.HIGH
                        )
            else:
                # Unknown update type - log and check if it's a module-specific request
                if "_request" in update.update_type or "_query" in update.update_type:
                    await self._handle_generic_request(update)
                else:
                    logger.info(f"No handler for update type: {update.update_type}")
            
            # Track processing time
            processing_time = time.time() - start_time
            self._track_processing_time(update.update_type, processing_time)
            
        except Exception as e:
            logger.error(f"Critical error in on_context_update: {e}", exc_info=True)
    
    def _validate_update(self, update: ContextUpdate) -> bool:
        """Validate incoming update"""
        if not update:
            return False
        
        if not hasattr(update, 'update_type') or not update.update_type:
            return False
        
        if not hasattr(update, 'data') or update.data is None:
            return False
        
        return True
    
    async def _handle_perception_update(self, update: ContextUpdate):
        """Handle perception input updates"""
        percept = update.data.get("percept")
        
        if not percept:
            logger.warning("Perception update missing percept data")
            return
        
        # Update causal models with new perceptual data
        if hasattr(self.original_core, 'update_with_perception'):
            # Run in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.original_core.update_with_perception,
                percept
            )
        
        # Extract modality information safely
        modality = "unknown"
        if hasattr(percept, 'modality'):
            modality = str(percept.modality)
        elif isinstance(percept, dict) and 'modality' in percept:
            modality = percept['modality']
        
        # Notify about model updates
        await self.send_context_update(
            update_type="causal_models_updated",
            data={
                "updated_models": list(self.active_models),
                "perception_modality": modality,
                "update_count": len(self.active_models),
                "timestamp": datetime.now().isoformat()
            },
            priority=ContextPriority.MEDIUM
        )
    
    async def _handle_causal_discovery_request(self, update: ContextUpdate):
        """Handle causal discovery requests"""
        model_id = update.data.get("model_id")
        
        if not model_id:
            logger.warning("Causal discovery request missing model_id")
            return
        
        # Validate model exists
        if model_id not in self.original_core.causal_models:
            await self.send_context_update(
                update_type="causal_discovery_error",
                data={
                    "model_id": model_id,
                    "error": "Model not found",
                    "available_models": list(self.original_core.causal_models.keys())
                },
                priority=ContextPriority.HIGH
            )
            return
        
        try:
            # Perform discovery
            discovery_result = await self.original_core.discover_causal_relations(model_id)
            
            # Prepare detailed response
            response_data = {
                "model_id": model_id,
                "discovery_result": discovery_result,
                "new_relations": discovery_result.get("accepted_relations", 0),
                "potential_relations": discovery_result.get("potential_relations", 0),
                "validation_score": discovery_result.get("validation", {}).get("score", 0.0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add discovery insights if available
            if discovery_result.get("accepted_relations", 0) > 0:
                response_data["insights"] = self._generate_discovery_insights(
                    model_id, discovery_result
                )
            
            await self.send_context_update(
                update_type="causal_discovery_complete",
                data=response_data,
                priority=ContextPriority.HIGH
            )
            
        except Exception as e:
            logger.error(f"Error in causal discovery: {e}")
            await self.send_context_update(
                update_type="causal_discovery_error",
                data={
                    "model_id": model_id,
                    "error": str(e)
                },
                priority=ContextPriority.HIGH
            )
    
    async def _handle_conceptual_blend_request(self, update: ContextUpdate):
        """Handle conceptual blending requests"""
        space_ids = update.data.get("space_ids", [])
        blend_type = update.data.get("blend_type", "composition")
        constraints = update.data.get("constraints", {})
        
        # Validate inputs
        if len(space_ids) < 2:
            await self.send_context_update(
                update_type="conceptual_blend_error",
                data={
                    "error": "At least 2 concept spaces required",
                    "provided_spaces": len(space_ids)
                },
                priority=ContextPriority.HIGH
            )
            return
        
        # Validate spaces exist
        missing_spaces = [sid for sid in space_ids 
                         if sid not in self.original_core.concept_spaces]
        
        if missing_spaces:
            await self.send_context_update(
                update_type="conceptual_blend_error",
                data={
                    "error": "Space(s) not found",
                    "missing_spaces": missing_spaces,
                    "available_spaces": list(self.original_core.concept_spaces.keys())
                },
                priority=ContextPriority.HIGH
            )
            return
        
        try:
            # Perform blending
            blend_result = await self._perform_contextual_blending(
                space_ids, blend_type, constraints
            )
            
            # Send result
            await self.send_context_update(
                update_type="conceptual_blend_complete",
                data={
                    "blend_result": blend_result,
                    "input_spaces": space_ids,
                    "blend_type": blend_type,
                    "success": blend_result.get("success", False),
                    "timestamp": datetime.now().isoformat()
                },
                priority=ContextPriority.HIGH
            )
            
        except Exception as e:
            logger.error(f"Error in conceptual blending: {e}")
            await self.send_context_update(
                update_type="conceptual_blend_error",
                data={
                    "error": str(e),
                    "space_ids": space_ids,
                    "blend_type": blend_type
                },
                priority=ContextPriority.HIGH
            )
    
    async def _handle_emotional_update(self, update: ContextUpdate):
        """Handle emotional state updates"""
        emotional_data = update.data
        
        # Validate emotional data
        if not isinstance(emotional_data, dict):
            logger.warning("Invalid emotional data format")
            return
        
        # Update reasoning based on emotion
        await self._adjust_reasoning_from_emotion(emotional_data)
        
        # Log emotional influence
        dominant_emotion = emotional_data.get("dominant_emotion")
        if dominant_emotion:
            logger.info(f"Reasoning adjusted for emotion: {dominant_emotion}")
    
    async def _handle_goal_update(self, update: ContextUpdate):
        """Handle goal context updates"""
        goal_data = update.data
        
        # Validate goal data
        if not isinstance(goal_data, dict):
            logger.warning("Invalid goal data format")
            return
        
        # Align reasoning with goals
        await self._align_reasoning_with_goals(goal_data)
        
        # Check if any goals require specific reasoning
        active_goals = goal_data.get("active_goals", [])
        for goal in active_goals:
            if goal.get("requires_reasoning", False):
                await self._initiate_goal_reasoning(goal)
    
    async def _handle_memory_update(self, update: ContextUpdate):
        """Handle memory retrieval updates"""
        memory_data = update.data
        
        # Validate memory data
        if not isinstance(memory_data, dict):
            logger.warning("Invalid memory data format")
            return
        
        # Inform reasoning from memory
        await self._inform_reasoning_from_memory(memory_data)
        
        # Check if memories suggest model updates
        if memory_data.get("suggests_model_update", False):
            await self._update_models_from_memory(memory_data)
    
    async def _handle_generic_request(self, update: ContextUpdate):
        """Handle generic reasoning requests"""
        request_type = update.data.get("request_type", "unknown")
        request_data = update.data.get("request_data", {})
        
        logger.info(f"Handling generic request: {request_type}")
        
        # Prepare response
        response = {
            "request_id": update.data.get("request_id"),
            "request_type": request_type,
            "status": "processing"
        }
        
        # Send acknowledgment
        await self.send_context_update(
            update_type=f"{update.source_module}_response",
            data=response,
            priority=update.priority
        )
    
    def _track_processing_time(self, update_type: str, processing_time: float):
        """Track processing time metrics"""
        if not hasattr(self, '_processing_metrics'):
            self._processing_metrics = defaultdict(list)
        
        self._processing_metrics[update_type].append(processing_time)
        
        # Keep only last 100 entries per type
        if len(self._processing_metrics[update_type]) > 100:
            self._processing_metrics[update_type].pop(0)
        
        # Log if processing time is unusually high
        if processing_time > 1.0:  # More than 1 second
            logger.warning(f"High processing time for {update_type}: {processing_time:.2f}s")
    
    def _generate_discovery_insights(self, model_id: str, 
                                   discovery_result: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from discovery results"""
        insights = []
        
        new_relations = discovery_result.get("accepted_relations", 0)
        if new_relations > 0:
            insights.append(f"Discovered {new_relations} new causal relationships")
        
        validation_score = discovery_result.get("validation", {}).get("score", 0.0)
        if validation_score > 0.8:
            insights.append("High confidence in discovered relationships")
        elif validation_score < 0.5:
            insights.append("Low confidence - more data may be needed")
        
        return insights
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with full context awareness for reasoning"""
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Analyze input for reasoning needs
        reasoning_analysis = await self._analyze_input_for_reasoning(context.user_input)
        
        # Determine reasoning approach based on context
        reasoning_approach = await self._determine_contextual_reasoning_approach(
            context, reasoning_analysis, messages
        )
        
        # Execute reasoning based on approach
        reasoning_results = {}
        
        if reasoning_approach["type"] == "causal":
            reasoning_results = await self._execute_causal_reasoning(
                context, reasoning_approach
            )
        elif reasoning_approach["type"] == "conceptual":
            reasoning_results = await self._execute_conceptual_reasoning(
                context, reasoning_approach
            )
        elif reasoning_approach["type"] == "integrated":
            reasoning_results = await self._execute_integrated_reasoning(
                context, reasoning_approach
            )
        
        # Update context with reasoning results
        await self.send_context_update(
            update_type="reasoning_process_complete",
            data={
                "reasoning_type": reasoning_approach["type"],
                "reasoning_results": reasoning_results,
                "models_used": list(self.active_models),
                "spaces_used": list(self.active_spaces),
                "cross_module_integration": len(messages) > 0
            }
        )
        
        return {
            "reasoning_analysis": reasoning_analysis,
            "reasoning_approach": reasoning_approach,
            "reasoning_results": reasoning_results,
            "context_aware_reasoning": True
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze reasoning possibilities in current context"""
        # Get all relevant context
        messages = await self.get_cross_module_messages()
        
        # Analyze available reasoning resources
        available_models = list(self.original_core.causal_models.keys())
        available_spaces = list(self.original_core.concept_spaces.keys())
        available_blends = list(self.original_core.blends.keys())
        
        # Analyze reasoning potential based on context
        reasoning_potential = await self._analyze_reasoning_potential(
            context, messages, available_models, available_spaces
        )
        
        # Identify reasoning opportunities
        reasoning_opportunities = await self._identify_reasoning_opportunities(
            context, messages
        )
        
        # Assess reasoning coherence with other modules
        coherence_analysis = await self._analyze_reasoning_coherence(
            context, messages
        )
        
        return {
            "available_models": available_models,
            "available_spaces": available_spaces,
            "available_blends": available_blends,
            "reasoning_potential": reasoning_potential,
            "reasoning_opportunities": reasoning_opportunities,
            "coherence_analysis": coherence_analysis,
            "analysis_complete": True
        }

    async def discover_modules(self):
        """Dynamically discover available modules and update subscriptions"""
        try:
            # Get available modules from the integration layer
            available_modules = await self.integration_layer.get_available_modules()
            
            # Build dynamic subscription list based on module capabilities
            dynamic_subscriptions = []
            
            for module_info in available_modules:
                module_name = module_info.get("name", "")
                capabilities = module_info.get("capabilities", [])
                
                # Map module capabilities to subscription types
                capability_mappings = {
                    "emotion": ["emotional_state_update", "emotion_analysis_complete"],
                    "memory": ["memory_retrieval_complete", "memory_storage_update"],
                    "goal": ["goal_context_available", "goal_update", "goal_completion"],
                    "knowledge": ["knowledge_update", "knowledge_query_result"],
                    "perception": ["perception_input", "sensory_update"],
                    "multimodal": ["multimodal_integration", "modality_fusion_complete"],
                    "planning": ["plan_generated", "plan_execution_update"],
                    "language": ["language_understanding_complete", "generation_ready"],
                    "attention": ["attention_shift", "focus_update"],
                    "motor": ["action_execution_update", "motor_feedback"]
                }
                
                # Add subscriptions based on capabilities
                for capability in capabilities:
                    if capability in capability_mappings:
                        dynamic_subscriptions.extend(capability_mappings[capability])
                
                # Also subscribe to module-specific reasoning requests
                dynamic_subscriptions.extend([
                    f"{module_name}_reasoning_request",
                    f"{module_name}_inference_request",
                    f"{module_name}_analysis_request"
                ])
            
            # Add core reasoning subscriptions that should always be present
            core_subscriptions = [
                "causal_discovery_request",
                "conceptual_blend_request", 
                "intervention_request",
                "counterfactual_query",
                "reasoning_request"  # Generic reasoning requests
            ]
            
            # Combine and deduplicate
            self.context_subscriptions = list(set(dynamic_subscriptions + core_subscriptions))
            
            # Update the integration layer with new subscriptions
            await self.integration_layer.update_subscriptions(self.module_id, self.context_subscriptions)
            
            # Log discovered modules
            logger.info(f"ReasoningCore discovered {len(available_modules)} modules")
            logger.debug(f"Updated subscriptions: {self.context_subscriptions}")
            
            # Store module registry for future reference
            self.discovered_modules = {
                module["name"]: module for module in available_modules
            }
            
            return {
                "modules_discovered": len(available_modules),
                "subscriptions_updated": len(self.context_subscriptions),
                "module_names": [m["name"] for m in available_modules]
            }
            
        except Exception as e:
            logger.error(f"Error during module discovery: {e}")
            # Fall back to default subscriptions
            self.context_subscriptions = [
                "emotional_state_update", "memory_retrieval_complete", 
                "goal_context_available", "knowledge_update",
                "perception_input", "multimodal_integration",
                "causal_discovery_request", "conceptual_blend_request",
                "intervention_request", "counterfactual_query"
            ]
            return {"error": str(e), "fallback": True}
    
    async def register_module_capabilities(self, module_name: str, capabilities: List[str]):
        """Register a new module's capabilities and update subscriptions"""
        if module_name not in self.discovered_modules:
            self.discovered_modules[module_name] = {
                "name": module_name,
                "capabilities": capabilities,
                "registered_at": datetime.now().isoformat()
            }
            
            # Trigger re-discovery to update subscriptions
            await self.discover_modules()
            
            logger.info(f"Registered new module: {module_name} with capabilities: {capabilities}")
    
    async def is_module_available(self, module_name: str) -> bool:
        """Check if a specific module is available"""
        return module_name in self.discovered_modules
    
    # ========================================================================================
    # CONTEXT PERSISTENCE METHODS
    # ========================================================================================
    
    async def save_context_state(self, filepath: str = None) -> Dict[str, Any]:
        """Save current context state for persistence across sessions"""
        if filepath is None:
            filepath = f"reasoning_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            context_state = {
                "timestamp": datetime.now().isoformat(),
                "module_id": self.module_id,
                "version": "1.0",
                
                # Active reasoning state
                "active_models": list(self.active_models),
                "active_spaces": list(self.active_spaces),
                "active_interventions": list(self.active_interventions),
                "reasoning_context": self.reasoning_context,
                
                # Discovered modules and subscriptions
                "discovered_modules": self.discovered_modules,
                "context_subscriptions": self.context_subscriptions,
                
                # Current context if available
                "last_context": {
                    "user_input": getattr(self, '_last_user_input', ""),
                    "session_id": getattr(self, '_last_session_id', ""),
                    "emotional_state": getattr(self, '_last_emotional_state', {}),
                    "goal_context": getattr(self, '_last_goal_context', {}),
                    "memory_context": getattr(self, '_last_memory_context', {})
                },
                
                # Cross-module message history (last N messages)
                "recent_messages": self._get_recent_message_history(limit=50),
                
                # Performance metrics
                "performance_metrics": {
                    "total_contexts_processed": getattr(self, '_contexts_processed', 0),
                    "total_updates_received": getattr(self, '_updates_received', 0),
                    "total_updates_sent": getattr(self, '_updates_sent', 0),
                    "average_processing_time": getattr(self, '_avg_processing_time', 0.0)
                }
            }
            
            # Save to file
            import json
            with open(filepath, 'w') as f:
                json.dump(context_state, f, indent=2)
            
            logger.info(f"Saved reasoning context state to {filepath}")
            
            return {
                "success": True,
                "filepath": filepath,
                "state_size": len(json.dumps(context_state)),
                "timestamp": context_state["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error saving context state: {e}")
            return {"success": False, "error": str(e)}
    
    async def load_context_state(self, filepath: str) -> Dict[str, Any]:
        """Load context state from a previous session"""
        try:
            import json
            with open(filepath, 'r') as f:
                context_state = json.load(f)
            
            # Validate version compatibility
            if context_state.get("version") != "1.0":
                logger.warning(f"Context state version mismatch: {context_state.get('version')}")
            
            # Restore active reasoning state
            self.active_models = set(context_state.get("active_models", []))
            self.active_spaces = set(context_state.get("active_spaces", []))
            self.active_interventions = set(context_state.get("active_interventions", []))
            self.reasoning_context = context_state.get("reasoning_context", {})
            
            # Restore discovered modules and subscriptions
            self.discovered_modules = context_state.get("discovered_modules", {})
            self.context_subscriptions = context_state.get("context_subscriptions", self.context_subscriptions)
            
            # Update integration layer with restored subscriptions
            await self.integration_layer.update_subscriptions(self.module_id, self.context_subscriptions)
            
            # Restore last context information
            last_context = context_state.get("last_context", {})
            self._last_user_input = last_context.get("user_input", "")
            self._last_session_id = last_context.get("session_id", "")
            self._last_emotional_state = last_context.get("emotional_state", {})
            self._last_goal_context = last_context.get("goal_context", {})
            self._last_memory_context = last_context.get("memory_context", {})
            
            # Restore performance metrics
            metrics = context_state.get("performance_metrics", {})
            self._contexts_processed = metrics.get("total_contexts_processed", 0)
            self._updates_received = metrics.get("total_updates_received", 0)
            self._updates_sent = metrics.get("total_updates_sent", 0)
            self._avg_processing_time = metrics.get("average_processing_time", 0.0)
            
            # Validate restored state
            validation_result = await self._validate_restored_state()
            
            logger.info(f"Loaded reasoning context state from {filepath}")
            
            return {
                "success": True,
                "filepath": filepath,
                "state_timestamp": context_state.get("timestamp"),
                "models_restored": len(self.active_models),
                "spaces_restored": len(self.active_spaces),
                "modules_discovered": len(self.discovered_modules),
                "validation": validation_result
            }
            
        except Exception as e:
            logger.error(f"Error loading context state: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_restored_state(self) -> Dict[str, Any]:
        """Validate restored state and check if referenced models/spaces still exist"""
        validation = {
            "valid_models": 0,
            "invalid_models": [],
            "valid_spaces": 0,
            "invalid_spaces": [],
            "warnings": []
        }
        
        # Check if active models still exist
        for model_id in list(self.active_models):
            if model_id in self.original_core.causal_models:
                validation["valid_models"] += 1
            else:
                validation["invalid_models"].append(model_id)
                self.active_models.discard(model_id)
                validation["warnings"].append(f"Removed non-existent model: {model_id}")
        
        # Check if active spaces still exist
        for space_id in list(self.active_spaces):
            if space_id in self.original_core.concept_spaces:
                validation["valid_spaces"] += 1
            else:
                validation["invalid_spaces"].append(space_id)
                self.active_spaces.discard(space_id)
                validation["warnings"].append(f"Removed non-existent space: {space_id}")
        
        # Re-discover modules to ensure current state
        discovery_result = await self.discover_modules()
        if discovery_result.get("error"):
            validation["warnings"].append("Module re-discovery failed, using restored subscriptions")
        
        validation["is_valid"] = len(validation["invalid_models"]) == 0 and len(validation["invalid_spaces"]) == 0
        
        return validation
    
    def _get_recent_message_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent cross-module message history for persistence"""
        # This would need to be implemented based on how messages are stored
        # For now, return empty list as placeholder
        return []
    
    async def auto_save_context(self, interval_minutes: int = 30):
        """Automatically save context at regular intervals"""
        import asyncio
        
        self._auto_save_enabled = True
        save_dir = "reasoning_context_autosave"
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        while self._auto_save_enabled:
            try:
                # Wait for the specified interval
                await asyncio.sleep(interval_minutes * 60)
                
                # Save context
                filepath = os.path.join(save_dir, f"autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                await self.save_context_state(filepath)
                
                # Clean up old autosaves (keep only last 10)
                autosaves = sorted([f for f in os.listdir(save_dir) if f.startswith("autosave_")])
                if len(autosaves) > 10:
                    for old_file in autosaves[:-10]:
                        os.remove(os.path.join(save_dir, old_file))
                        
            except Exception as e:
                logger.error(f"Error during auto-save: {e}")
    
    def stop_auto_save(self):
        """Stop automatic context saving"""
        self._auto_save_enabled = False

    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize reasoning insights for response generation"""
        messages = await self.get_cross_module_messages()
        
        # Generate reasoning-based insights
        reasoning_synthesis = {
            "causal_insights": await self._synthesize_causal_insights(context, messages),
            "conceptual_insights": await self._synthesize_conceptual_insights(context, messages),
            "counterfactual_insights": await self._synthesize_counterfactual_insights(context, messages),
            "intervention_suggestions": await self._synthesize_intervention_suggestions(context, messages),
            "reasoning_narrative": await self._generate_reasoning_narrative(context, messages),
            "confidence_assessment": await self._assess_reasoning_confidence(context)
        }
        
        # Check if we should announce any discoveries
        if reasoning_synthesis["causal_insights"].get("new_discoveries"):
            await self.send_context_update(
                update_type="causal_discovery_announcement",
                data={
                    "discoveries": reasoning_synthesis["causal_insights"]["new_discoveries"],
                    "impact": "high"
                },
                priority=ContextPriority.HIGH
            )
        
        return {
            "reasoning_synthesis": reasoning_synthesis,
            "synthesis_complete": True,
            "integrated_with_modules": list(messages.keys())
        }
    
    # ========================================================================================
    # REASONING ANALYSIS METHODS
    # ========================================================================================
    
    async def _analyze_input_for_reasoning(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input for reasoning-related implications"""
        input_lower = user_input.lower()
        
        analysis = {
            "suggests_causal_reasoning": any(kw in input_lower for kw in 
                ["cause", "effect", "why", "because", "leads to", "results in", "if then"]),
            "suggests_conceptual_reasoning": any(kw in input_lower for kw in 
                ["concept", "idea", "blend", "combine", "creative", "imagine"]),
            "suggests_counterfactual": any(kw in input_lower for kw in 
                ["what if", "would have", "could have", "alternative", "instead"]),
            "suggests_intervention": any(kw in input_lower for kw in 
                ["change", "intervene", "modify", "alter", "influence"]),
            "domain_keywords": self._extract_domain_keywords(input_lower),
            "confidence": 0.0
        }
        
        # Calculate confidence based on keyword matches
        confidence = 0.0
        if analysis["suggests_causal_reasoning"]:
            confidence += 0.3
        if analysis["suggests_conceptual_reasoning"]:
            confidence += 0.2
        if analysis["suggests_counterfactual"]:
            confidence += 0.3
        if analysis["suggests_intervention"]:
            confidence += 0.2
        
        analysis["confidence"] = min(1.0, confidence)
        
        # Determine primary reasoning type
        if analysis["suggests_counterfactual"]:
            analysis["reasoning_type"] = "counterfactual"
        elif analysis["suggests_causal_reasoning"] and analysis["suggests_conceptual_reasoning"]:
            analysis["reasoning_type"] = "integrated"
        elif analysis["suggests_causal_reasoning"]:
            analysis["reasoning_type"] = "causal"
        elif analysis["suggests_conceptual_reasoning"]:
            analysis["reasoning_type"] = "conceptual"
        else:
            analysis["reasoning_type"] = "none"
        
        return analysis
    
    async def _get_contextually_relevant_models(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get causal models relevant to current context"""
        relevant_models = []
        
        # Extract domain from context
        domain_keywords = self._extract_domain_keywords(context.user_input.lower())
        
        for model_id, model in self.original_core.causal_models.items():
            relevance_score = 0.0
            
            # Check domain match
            if model.domain:
                for keyword in domain_keywords:
                    if keyword in model.domain.lower():
                        relevance_score += 0.3
            
            # Check if model relates to current goals
            if context.goal_context:
                active_goals = context.goal_context.get("active_goals", [])
                for goal in active_goals:
                    if goal.get("associated_need") in model.domain.lower():
                        relevance_score += 0.2
            
            # Check if model relates to emotional context
            if context.emotional_state:
                dominant_emotion = context.emotional_state.get("dominant_emotion")
                if dominant_emotion and dominant_emotion[0].lower() in model.metadata.get("emotional_relevance", []):
                    relevance_score += 0.1
            
            if relevance_score > 0.1:
                relevant_models.append({
                    "model_id": model_id,
                    "name": model.name,
                    "domain": model.domain,
                    "relevance_score": relevance_score,
                    "node_count": len(model.nodes),
                    "relation_count": len(model.relations)
                })
        
        # Sort by relevance
        relevant_models.sort(key=lambda m: m["relevance_score"], reverse=True)
        return relevant_models[:5]  # Top 5 models
    
    async def _get_contextually_relevant_spaces(self, context: SharedContext) -> List[Dict[str, Any]]:
        """Get concept spaces relevant to current context"""
        relevant_spaces = []
        
        # Extract domain from context
        domain_keywords = self._extract_domain_keywords(context.user_input.lower())
        
        for space_id, space in self.original_core.concept_spaces.items():
            relevance_score = 0.0
            
            # Check domain match
            if space.domain:
                for keyword in domain_keywords:
                    if keyword in space.domain.lower():
                        relevance_score += 0.3
            
            # Check if space relates to memory context
            if context.memory_context:
                memory_types = context.memory_context.get("memory_types", [])
                if "experience" in memory_types and "experiential" in space.name.lower():
                    relevance_score += 0.2
            
            if relevance_score > 0.1:
                relevant_spaces.append({
                    "space_id": space_id,
                    "name": space.name,
                    "domain": space.domain,
                    "relevance_score": relevance_score,
                    "concept_count": len(space.concepts),
                    "relation_count": len(space.relations)
                })
        
        # Sort by relevance
        relevant_spaces.sort(key=lambda s: s["relevance_score"], reverse=True)
        return relevant_spaces[:5]  # Top 5 spaces
    
    async def _determine_contextual_reasoning_approach(self, context: SharedContext, 
                                                     reasoning_analysis: Dict[str, Any],
                                                     messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Determine the best reasoning approach based on full context"""
        approach = {
            "type": reasoning_analysis.get("reasoning_type", "none"),
            "confidence": reasoning_analysis.get("confidence", 0.0),
            "strategy": "default",
            "resources": []
        }
        
        # Adjust based on emotional context
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                emotion_name, strength = dominant_emotion
                
                if emotion_name == "Curiosity" and strength > 0.5:
                    # Curiosity favors exploratory reasoning
                    approach["strategy"] = "exploratory"
                    if approach["type"] == "none":
                        approach["type"] = "conceptual"
                elif emotion_name == "Anxiety" and strength > 0.6:
                    # Anxiety favors understanding causes
                    approach["strategy"] = "explanatory"
                    if approach["type"] == "none":
                        approach["type"] = "causal"
        
        # Adjust based on goal context
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            for goal in active_goals:
                if "understand" in goal.get("description", "").lower():
                    approach["strategy"] = "analytical"
                elif "solve" in goal.get("description", "").lower():
                    approach["strategy"] = "problem_solving"
                    approach["type"] = "integrated"
        
        # Check messages from other modules
        if "knowledge_core" in messages:
            # Knowledge updates might require model updates
            approach["resources"].append("knowledge_integration")
        
        if "multimodal_integrator" in messages:
            # Multimodal input might provide evidence for causal discovery
            approach["resources"].append("perceptual_evidence")
        
        return approach
    
    # ========================================================================================
    # REASONING EXECUTION METHODS
    # ========================================================================================
    
    async def _execute_causal_reasoning(self, context: SharedContext, 
                                      approach: Dict[str, Any]) -> Dict[str, Any]:
        """Execute causal reasoning with context awareness"""
        results = {
            "models_analyzed": [],
            "causal_insights": [],
            "new_relations_discovered": 0
        }
        
        # Get relevant models
        relevant_models = await self._get_contextually_relevant_models(context)
        
        if not relevant_models:
            # Create a new model if needed
            domain = self._infer_domain_from_context(context)
            model_id = await self.original_core.create_causal_model(
                name=f"Context-driven model for {domain}",
                domain=domain,
                metadata={"created_from_context": True, "session_id": context.session_context.get("session_id")}
            )
            self.active_models.add(model_id)
            
            results["models_analyzed"].append({
                "model_id": model_id,
                "status": "newly_created"
            })
        else:
            # Analyze existing models
            for model_info in relevant_models[:2]:  # Analyze top 2 models
                model_id = model_info["model_id"]
                self.active_models.add(model_id)
                
                # Check if we should run causal discovery
                if approach["strategy"] == "exploratory":
                    discovery_result = await self.original_core.discover_causal_relations(model_id)
                    
                    results["models_analyzed"].append({
                        "model_id": model_id,
                        "discovery_result": discovery_result
                    })
                    
                    if discovery_result.get("accepted_relations", 0) > 0:
                        results["new_relations_discovered"] += discovery_result["accepted_relations"]
                else:
                    # Just analyze the model
                    model = self.original_core.causal_models[model_id]
                    
                    # Find relevant nodes based on context
                    relevant_nodes = []
                    for node_id, node in model.nodes.items():
                        if self._is_node_relevant_to_context(node, context):
                            relevant_nodes.append(node_id)
                    
                    results["models_analyzed"].append({
                        "model_id": model_id,
                        "relevant_nodes": relevant_nodes,
                        "total_nodes": len(model.nodes)
                    })
                
                # Extract causal insights
                insights = await self._extract_causal_insights(model_id, context)
                results["causal_insights"].extend(insights)
        
        return results
    
    async def _execute_conceptual_reasoning(self, context: SharedContext,
                                         approach: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conceptual reasoning with context awareness"""
        results = {
            "spaces_analyzed": [],
            "blends_created": [],
            "conceptual_insights": []
        }
        
        # Get relevant spaces
        relevant_spaces = await self._get_contextually_relevant_spaces(context)
        
        if len(relevant_spaces) >= 2 and approach["strategy"] == "exploratory":
            # Create a blend
            space1_id = relevant_spaces[0]["space_id"]
            space2_id = relevant_spaces[1]["space_id"]
            
            self.active_spaces.add(space1_id)
            self.active_spaces.add(space2_id)
            
            # Find mappings between spaces
            mappings = await self._find_contextual_mappings(space1_id, space2_id, context)
            
            if mappings:
                # Create blend based on emotional context
                blend_type = self._determine_blend_type_from_context(context)
                
                blend_result = await self._create_contextual_blend(
                    space1_id, space2_id, mappings, blend_type
                )
                
                if blend_result:
                    results["blends_created"].append(blend_result)
        
        # Analyze spaces for insights
        for space_info in relevant_spaces[:3]:
            space_id = space_info["space_id"]
            self.active_spaces.add(space_id)
            
            insights = await self._extract_conceptual_insights(space_id, context)
            results["conceptual_insights"].extend(insights)
            
            results["spaces_analyzed"].append({
                "space_id": space_id,
                "insights_found": len(insights)
            })
        
        return results
    
    async def _execute_integrated_reasoning(self, context: SharedContext,
                                         approach: Dict[str, Any]) -> Dict[str, Any]:
        """Execute integrated causal and conceptual reasoning"""
        results = {
            "causal_results": {},
            "conceptual_results": {},
            "integration_insights": [],
            "creative_interventions": []
        }
        
        # First execute causal reasoning
        causal_results = await self._execute_causal_reasoning(context, approach)
        results["causal_results"] = causal_results
        
        # Then execute conceptual reasoning
        conceptual_results = await self._execute_conceptual_reasoning(context, approach)
        results["conceptual_results"] = conceptual_results
        
        # Integrate insights
        if causal_results["models_analyzed"] and conceptual_results["spaces_analyzed"]:
            # Find integration opportunities
            for model_info in causal_results["models_analyzed"]:
                model_id = model_info["model_id"]
                
                for space_info in conceptual_results["spaces_analyzed"]:
                    space_id = space_info["space_id"]
                    
                    # Check for integration potential
                    integration_score = await self._assess_integration_potential(
                        model_id, space_id, context
                    )
                    
                    if integration_score > 0.5:
                        # Create integrated model
                        integrated_result = await self.original_core.create_integrated_model(
                            domain=self._infer_domain_from_context(context),
                            base_on_causal=True
                        )
                        
                        results["integration_insights"].append({
                            "type": "integrated_model",
                            "result": integrated_result,
                            "integration_score": integration_score
                        })
                        
                        # If problem-solving strategy, suggest interventions
                        if approach["strategy"] == "problem_solving":
                            intervention = await self._suggest_creative_intervention(
                                model_id, context
                            )
                            if intervention:
                                results["creative_interventions"].append(intervention)
        
        return results
    
    # ========================================================================================
    # SYNTHESIS METHODS
    # ========================================================================================
    
    async def _synthesize_causal_insights(self, context: SharedContext, 
                                       messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize causal insights from active models"""
        insights = {
            "key_causal_factors": [],
            "causal_chains": [],
            "intervention_points": [],
            "new_discoveries": []
        }
        
        for model_id in self.active_models:
            if model_id not in self.original_core.causal_models:
                continue
                
            model = self.original_core.causal_models[model_id]
            
            # Find key causal factors (high centrality nodes)
            try:
                import networkx as nx
                centrality = nx.betweenness_centrality(model.graph)
                top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for node_id, centrality_score in top_nodes:
                    node = model.nodes.get(node_id)
                    if node:
                        insights["key_causal_factors"].append({
                            "node_name": node.name,
                            "centrality": centrality_score,
                            "model": model.name
                        })
            except:
                pass
            
            # Find causal chains relevant to context
            if context.user_input:
                relevant_chains = await self._find_relevant_causal_chains(model, context)
                insights["causal_chains"].extend(relevant_chains)
            
            # Identify intervention points
            intervention_points = await self._identify_intervention_points(model, context)
            insights["intervention_points"].extend(intervention_points)
        
        # Check for new discoveries from this session
        if hasattr(self, 'reasoning_context') and "discoveries" in self.reasoning_context:
            insights["new_discoveries"] = self.reasoning_context["discoveries"]
        
        return insights
    
    async def _synthesize_conceptual_insights(self, context: SharedContext,
                                           messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize conceptual insights from active spaces"""
        insights = {
            "key_concepts": [],
            "conceptual_patterns": [],
            "creative_possibilities": [],
            "blend_opportunities": []
        }
        
        for space_id in self.active_spaces:
            if space_id not in self.original_core.concept_spaces:
                continue
                
            space = self.original_core.concept_spaces[space_id]
            
            # Find key concepts
            key_concepts = await self._identify_key_concepts(space, context)
            insights["key_concepts"].extend(key_concepts)
            
            # Find conceptual patterns
            patterns = await self._identify_conceptual_patterns(space, context)
            insights["conceptual_patterns"].extend(patterns)
        
        # Find blend opportunities
        if len(self.active_spaces) >= 2:
            space_list = list(self.active_spaces)
            for i in range(len(space_list)):
                for j in range(i + 1, len(space_list)):
                    opportunity = await self._assess_blend_opportunity(
                        space_list[i], space_list[j], context
                    )
                    if opportunity["score"] > 0.5:
                        insights["blend_opportunities"].append(opportunity)
        
        return insights
    
    async def _synthesize_counterfactual_insights(self, context: SharedContext,
                                               messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize counterfactual insights"""
        insights = {
            "counterfactual_scenarios": [],
            "what_if_analyses": [],
            "alternative_paths": []
        }
        
        # Check if input suggests counterfactual reasoning
        if "what if" in context.user_input.lower() or "would have" in context.user_input.lower():
            for model_id in self.active_models:
                if model_id not in self.original_core.causal_models:
                    continue
                    
                # Generate counterfactual scenario
                scenario = await self._generate_counterfactual_scenario(model_id, context)
                if scenario:
                    insights["counterfactual_scenarios"].append(scenario)
                    
                    # Analyze alternative paths
                    alt_paths = await self._analyze_alternative_paths(model_id, scenario)
                    insights["alternative_paths"].extend(alt_paths)
        
        return insights
    
    async def _synthesize_intervention_suggestions(self, context: SharedContext,
                                               messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Synthesize intervention suggestions based on reasoning"""
        suggestions = []
        
        # Check goal context for intervention needs
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            
            for goal in active_goals:
                if goal.get("priority", 0) > 0.7:  # High priority goals
                    # Find causal model related to goal
                    for model_id in self.active_models:
                        model = self.original_core.causal_models.get(model_id)
                        if model and self._model_relates_to_goal(model, goal):
                            # Suggest intervention
                            suggestion = await self._generate_intervention_suggestion(
                                model_id, goal, context
                            )
                            if suggestion:
                                suggestions.append(suggestion)
        
        return suggestions
    
    async def _generate_reasoning_narrative(self, context: SharedContext,
                                         messages: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a narrative explaining the reasoning process"""
        narrative_parts = []
        
        # Describe what type of reasoning was performed
        if self.active_models:
            narrative_parts.append(f"I analyzed {len(self.active_models)} causal models")
        
        if self.active_spaces:
            narrative_parts.append(f"explored {len(self.active_spaces)} conceptual spaces")
        
        # Describe key findings
        if hasattr(self, 'reasoning_context') and "key_findings" in self.reasoning_context:
            findings = self.reasoning_context["key_findings"]
            if findings:
                narrative_parts.append(f"and discovered {len(findings)} key insights")
        
        # Create coherent narrative
        if narrative_parts:
            narrative = "Through integrated reasoning, " + ", ".join(narrative_parts) + "."
        else:
            narrative = "I'm analyzing the situation from multiple perspectives."
        
        return narrative
    
    async def _assess_reasoning_confidence(self, context: SharedContext) -> float:
        """Assess confidence in reasoning results"""
        confidence = 0.5  # Base confidence
        
        # More models/spaces analyzed = higher confidence
        if len(self.active_models) > 0:
            confidence += 0.1 * min(3, len(self.active_models))
        
        if len(self.active_spaces) > 0:
            confidence += 0.1 * min(3, len(self.active_spaces))
        
        # Cross-module integration increases confidence
        messages = await self.get_cross_module_messages()
        if len(messages) > 0:
            confidence += 0.05 * min(4, len(messages))
        
        # New discoveries increase confidence
        if hasattr(self, 'reasoning_context') and self.reasoning_context.get("new_relations_discovered", 0) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    # Replace simplified _extract_domain_keywords
    def _extract_domain_keywords(self, text: str) -> List[str]:
        """Extract domain-related keywords from text using multiple strategies"""
        keywords = []
        text_lower = text.lower()
        
        # Expanded domain vocabulary with subcategories
        domain_patterns = {
            "climate": ["climate", "weather", "temperature", "warming", "carbon", "emissions", 
                       "greenhouse", "environmental", "sustainability", "renewable"],
            "health": ["health", "medical", "disease", "treatment", "symptoms", "diagnosis",
                      "wellness", "medicine", "patient", "therapy", "clinical", "healthcare"],
            "economics": ["economics", "economy", "market", "finance", "trade", "investment",
                         "inflation", "gdp", "monetary", "fiscal", "business", "commerce"],
            "psychology": ["psychology", "mental", "cognitive", "behavior", "emotional", 
                          "personality", "consciousness", "perception", "motivation", "learning"],
            "technology": ["technology", "software", "hardware", "ai", "algorithm", "data",
                          "computing", "digital", "innovation", "automation", "cyber"],
            "relationships": ["relationship", "social", "interpersonal", "communication",
                             "family", "friendship", "conflict", "trust", "love", "attachment"],
            "education": ["education", "learning", "teaching", "academic", "curriculum",
                         "pedagogy", "student", "knowledge", "skill", "training"],
            "politics": ["politics", "government", "policy", "democracy", "election",
                        "legislation", "governance", "political", "civic", "administration"],
            "biology": ["biology", "biological", "organism", "evolution", "genetics",
                       "ecology", "cell", "species", "ecosystem", "life"],
            "physics": ["physics", "quantum", "energy", "force", "motion", "particle",
                       "wave", "field", "relativity", "mechanics", "thermodynamics"]
        }
        
        # Multi-word domain indicators
        compound_patterns = {
            "machine_learning": ["machine learning", "deep learning", "neural network"],
            "public_health": ["public health", "epidemiology", "population health"],
            "behavioral_economics": ["behavioral economics", "decision making", "choice architecture"],
            "climate_science": ["climate science", "global warming", "climate change"],
            "social_psychology": ["social psychology", "group dynamics", "social influence"]
        }
        
        # Check for domain matches
        domains_found = set()
        
        # Single word patterns
        for domain, patterns in domain_patterns.items():
            domain_score = sum(1 for pattern in patterns if pattern in text_lower)
            if domain_score >= 2:  # At least 2 keywords from domain
                domains_found.add(domain)
                keywords.append(domain)
            elif domain_score == 1 and len(text_lower.split()) < 20:
                # For short texts, even one keyword might be significant
                domains_found.add(domain)
                keywords.append(domain)
        
        # Compound patterns
        for domain, patterns in compound_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                keywords.append(domain)
        
        # Extract domain-specific technical terms
        technical_terms = self._extract_technical_terms(text_lower, domains_found)
        keywords.extend(technical_terms)
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_technical_terms(self, text: str, domains: Set[str]) -> List[str]:
        """Extract technical terms based on identified domains"""
        technical_terms = []
        
        # Domain-specific technical vocabularies
        domain_technical = {
            "climate": ["mitigation", "adaptation", "anthropogenic", "albedo", "feedback loop"],
            "health": ["pathogen", "etiology", "prognosis", "comorbidity", "epidemiological"],
            "economics": ["elasticity", "equilibrium", "arbitrage", "liquidity", "volatility"],
            "psychology": ["conditioning", "heuristic", "schema", "attribution", "metacognition"],
            "technology": ["scalability", "latency", "throughput", "architecture", "optimization"]
        }
        
        for domain in domains:
            if domain in domain_technical:
                for term in domain_technical[domain]:
                    if term in text:
                        technical_terms.append(term)
        
        return technical_terms
    
    def _infer_domain_from_context(self, context: SharedContext) -> str:
        """Infer domain from context"""
        keywords = self._extract_domain_keywords(context.user_input.lower())
        
        if keywords:
            return keywords[0]  # Use first keyword as domain
        
        # Check emotional context
        if context.emotional_state:
            return "emotional_dynamics"
        
        # Check goal context
        if context.goal_context:
            return "goal_achievement"
        
        return "general"
    
    def _is_node_relevant_to_context(self, node, context: SharedContext) -> bool:
        """Comprehensive node relevance assessment"""
        relevance_score = 0.0
        
        node_name_lower = node.name.lower()
        input_lower = context.user_input.lower()
        
        # 1. Direct name matching (with stemming considerations)
        input_words = set(input_lower.split())
        node_words = set(node_name_lower.split())
        
        # Exact word matches
        exact_matches = input_words.intersection(node_words)
        relevance_score += len(exact_matches) * 0.3
        
        # Partial word matches (prefix/suffix)
        for input_word in input_words:
            for node_word in node_words:
                if len(input_word) > 3 and len(node_word) > 3:
                    if input_word.startswith(node_word[:3]) or node_word.startswith(input_word[:3]):
                        relevance_score += 0.1
        
        # 2. Semantic similarity using word embeddings simulation
        semantic_sim = self._calculate_semantic_similarity(node_name_lower, input_lower)
        relevance_score += semantic_sim * 0.2
        
        # 3. Domain matching
        if hasattr(node, 'domain') and node.domain:
            domain_keywords = self._extract_domain_keywords(input_lower)
            if any(kw in node.domain.lower() for kw in domain_keywords):
                relevance_score += 0.2
        
        # 4. Property matching
        if hasattr(node, 'properties'):
            for prop_name, prop_value in node.properties.items():
                if isinstance(prop_value, str):
                    if any(word in prop_value.lower() for word in input_words):
                        relevance_score += 0.1
        
        # 5. Contextual factors
        # Goal relevance
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            for goal in goals:
                goal_desc = goal.get("description", "").lower()
                if any(word in node_name_lower for word in goal_desc.split()):
                    relevance_score += 0.15
        
        # Emotional relevance
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion")
            if emotion and self._node_aligns_with_emotion(node, emotion[0]):
                relevance_score += 0.1
        
        # 6. Graph structural relevance
        if hasattr(node, 'centrality_score'):
            # High centrality nodes are generally more relevant
            relevance_score += node.centrality_score * 0.1
        
        # 7. Temporal relevance
        if hasattr(node, 'timestamp'):
            # Recent nodes might be more relevant
            time_diff = (datetime.now() - node.timestamp).days
            if time_diff < 7:
                relevance_score += 0.1
            elif time_diff < 30:
                relevance_score += 0.05
        
        return relevance_score > 0.25  # Threshold for relevance
    

    def _calculate_semantic_similarity(self, text1: str, text2: str, 
                                     use_embeddings: bool = True,
                                     use_structure: bool = True) -> float:
        """
        Calculate semantic similarity between texts using multiple approaches.
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            use_embeddings: Whether to use transformer embeddings
            use_structure: Whether to consider syntactic structure
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        # Quick exact match check
        if text1 == text2:
            return 1.0
        
        similarity_scores = []
        weights = []
        
        # 1. Embedding-based similarity (most important)
        if use_embeddings and hasattr(self, 'semantic_model'):
            try:
                # Get embeddings
                embeddings = self.semantic_model.encode([text1, text2])
                embedding_similarity = cosine_similarity(
                    embeddings[0].reshape(1, -1), 
                    embeddings[1].reshape(1, -1)
                )[0][0]
                
                # Normalize to 0-1 range (cosine similarity can be negative)
                embedding_similarity = (embedding_similarity + 1) / 2
                
                similarity_scores.append(embedding_similarity)
                weights.append(0.5)  # High weight for embeddings
            except Exception as e:
                logger.warning(f"Embedding calculation failed: {e}")
        
        # 2. Token-based similarity with TF-IDF weighting
        token_similarity = self._calculate_token_similarity(text1, text2)
        similarity_scores.append(token_similarity)
        weights.append(0.2)
        
        # 3. N-gram similarity
        ngram_similarity = self._calculate_ngram_similarity(text1, text2)
        similarity_scores.append(ngram_similarity)
        weights.append(0.15)
        
        # 4. Syntactic structure similarity
        if use_structure and hasattr(self, 'nlp'):
            try:
                structure_similarity = self._calculate_structure_similarity(text1, text2)
                similarity_scores.append(structure_similarity)
                weights.append(0.15)
            except Exception as e:
                logger.warning(f"Structure similarity calculation failed: {e}")
        
        # Calculate weighted average
        if similarity_scores:
            # Normalize weights
            total_weight = sum(weights[:len(similarity_scores)])
            if total_weight > 0:
                weighted_similarity = sum(
                    score * weight for score, weight in zip(similarity_scores, weights)
                ) / total_weight
                return min(1.0, max(0.0, weighted_similarity))
        
        # Fallback to simple Jaccard similarity
        return self._calculate_token_similarity(text1, text2)

    def _calculate_ngram_similarity(self, text1: str, text2: str, n_range: Tuple[int, int] = (2, 3)) -> float:
        """Calculate character n-gram similarity"""
        def get_ngrams(text: str, n: int) -> Set[str]:
            """Extract character n-grams from text"""
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        similarity_scores = []
        
        for n in range(n_range[0], n_range[1] + 1):
            ngrams1 = get_ngrams(text1, n)
            ngrams2 = get_ngrams(text2, n)
            
            if ngrams1 and ngrams2:
                intersection = len(ngrams1.intersection(ngrams2))
                union = len(ngrams1.union(ngrams2))
                similarity = intersection / union if union > 0 else 0.0
                similarity_scores.append(similarity)
        
        return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

    def _calculate_token_similarity(self, text1: str, text2: str) -> float:
        """Calculate token-based similarity with TF-IDF-like weighting"""
        # Get stop words
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "but",
                         "in", "with", "to", "for", "of", "as", "by", "that", "this", "it"}
        
        # Tokenize and filter
        tokens1 = set(word for word in text1.split() if word not in stop_words and len(word) > 2)
        tokens2 = set(word for word in text2.split() if word not in stop_words and len(word) > 2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Calculate token frequencies (simple TF)
        all_tokens = tokens1.union(tokens2)
        freq1 = {token: text1.count(token) for token in all_tokens}
        freq2 = {token: text2.count(token) for token in all_tokens}
        
        # Calculate weighted Jaccard similarity
        intersection_weight = sum(min(freq1.get(token, 0), freq2.get(token, 0)) 
                                for token in all_tokens)
        union_weight = sum(max(freq1.get(token, 0), freq2.get(token, 0)) 
                          for token in all_tokens)
        
        return intersection_weight / union_weight if union_weight > 0 else 0.0

    def _calculate_structure_similarity(self, text1: str, text2: str) -> float:
        """Calculate syntactic structure similarity using spaCy"""
        try:
            # Parse texts
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Extract POS tag sequences
            pos1 = [token.pos_ for token in doc1]
            pos2 = [token.pos_ for token in doc2]
            
            # Calculate POS sequence similarity
            pos_similarity = self._sequence_similarity(pos1, pos2)
            
            # Extract dependency patterns
            dep_patterns1 = [(token.dep_, token.head.pos_) for token in doc1]
            dep_patterns2 = [(token.dep_, token.head.pos_) for token in doc2]
            
            # Calculate dependency pattern similarity
            dep_similarity = self._pattern_similarity(dep_patterns1, dep_patterns2)
            
            # Combine scores
            return 0.6 * pos_similarity + 0.4 * dep_similarity
            
        except Exception as e:
            logger.error(f"Structure similarity error: {e}")
            return 0.0
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences using LCS"""
        if not seq1 or not seq2:
            return 0.0
        
        # Longest Common Subsequence
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return 2 * lcs_length / (m + n) if (m + n) > 0 else 0.0
    
    def _pattern_similarity(self, patterns1: List[Tuple], patterns2: List[Tuple]) -> float:
        """Calculate similarity between dependency patterns"""
        if not patterns1 or not patterns2:
            return 0.0
        
        set1 = set(patterns1)
        set2 = set(patterns2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _node_aligns_with_emotion(self, node, emotion: str) -> bool:
        """Check if node aligns with emotional state"""
        node_name_lower = node.name.lower()
        
        emotion_associations = {
            "Curiosity": ["unknown", "mystery", "explore", "discover", "new", "novel", "interesting"],
            "Anxiety": ["threat", "danger", "risk", "worry", "concern", "fear", "uncertain"],
            "Joy": ["success", "achievement", "positive", "good", "happy", "benefit", "reward"],
            "Frustration": ["problem", "obstacle", "difficult", "blocked", "challenge", "stuck"],
            "Satisfaction": ["complete", "achieved", "solved", "finished", "successful", "done"]
        }
        
        associations = emotion_associations.get(emotion, [])
        return any(assoc in node_name_lower for assoc in associations)
    
    async def _extract_causal_insights(self, model_id: str, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract causal insights from a model"""
        insights = []
        model = self.original_core.causal_models.get(model_id)
        
        if not model:
            return insights
        
        # Find causal paths relevant to input
        input_keywords = context.user_input.lower().split()
        
        for node_id, node in model.nodes.items():
            if any(kw in node.name.lower() for kw in input_keywords):
                # Found relevant node - trace causal paths
                ancestors = model.get_ancestors(node_id)[:3]  # Top 3 causes
                descendants = model.get_descendants(node_id)[:3]  # Top 3 effects
                
                insight = {
                    "type": "causal_path",
                    "central_factor": node.name,
                    "causes": [model.nodes[a].name for a in ancestors if a in model.nodes],
                    "effects": [model.nodes[d].name for d in descendants if d in model.nodes],
                    "model": model.name
                }
                
                insights.append(insight)
        
        return insights
    
    async def _find_contextual_mappings(self, space1_id: str, space2_id: str, 
                                     context: SharedContext) -> List[Dict[str, Any]]:
        """Find mappings between concept spaces based on context"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return []
        
        mappings = []
        
        # Weight mappings based on context relevance
        for concept1_id, concept1 in space1.concepts.items():
            for concept2_id, concept2 in space2.concepts.items():
                similarity = self.original_core._calculate_concept_similarity(
                    concept1, concept2, space1, space2
                )
                
                # Boost similarity if concepts relate to context
                if self._concept_relates_to_context(concept1, context):
                    similarity += 0.1
                if self._concept_relates_to_context(concept2, context):
                    similarity += 0.1
                
                if similarity >= 0.5:
                    mappings.append({
                        "concept1": concept1_id,
                        "concept2": concept2_id,
                        "similarity": min(1.0, similarity),
                        "context_boosted": True
                    })
        
        return mappings
    
    def _determine_blend_type_from_context(self, context: SharedContext) -> str:
        """Determine blend type based on context"""
        # Check emotional state
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion:
                emotion_name = dominant_emotion[0]
                
                if emotion_name == "Curiosity":
                    return "elaboration"  # Elaborate on ideas
                elif emotion_name in ["Frustration", "Conflict"]:
                    return "contrast"  # Find contrasts
                elif emotion_name in ["Joy", "Satisfaction"]:
                    return "fusion"  # Deep integration
        
        # Check goal context
        if context.goal_context:
            goal_types = [g.get("associated_need", "") for g in context.goal_context.get("active_goals", [])]
            
            if "novelty" in goal_types:
                return "elaboration"
            elif "coherence" in goal_types:
                return "completion"
        
        return "composition"  # Default
    
    def _concept_relates_to_context(self, concept: Dict[str, Any], context: SharedContext) -> bool:
        """Check if concept relates to context"""
        concept_name = concept.get("name", "").lower()
        input_lower = context.user_input.lower()
        
        # Check name match
        if any(word in concept_name for word in input_lower.split()):
            return True
        
        # Check property matches
        for prop_value in concept.get("properties", {}).values():
            if isinstance(prop_value, str) and any(word in prop_value.lower() for word in input_lower.split()):
                return True
        
        return False
    
    async def _create_contextual_blend(self, space1_id: str, space2_id: str,
                                    mappings: List[Dict[str, Any]], 
                                    blend_type: str) -> Optional[Dict[str, Any]]:
        """Create blend with context awareness"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return None
        
        # Use appropriate blend generation method
        blend_data = None
        
        if blend_type == "composition":
            blend_data = self.original_core._generate_composition_blend(space1, space2, mappings)
        elif blend_type == "fusion":
            blend_data = self.original_core._generate_fusion_blend(space1, space2, mappings)
        elif blend_type == "elaboration":
            blend_data = self.original_core._generate_elaboration_blend(space1, space2, mappings)
        elif blend_type == "contrast":
            blend_data = self.original_core._generate_contrast_blend(space1, space2, mappings)
        elif blend_type == "completion":
            blend_data = self.original_core._generate_completion_blend(space1, space2, mappings)
        
        if blend_data:
            # Update statistics
            self.original_core.stats["blends_created"] += 1
            
            # Store in reasoning context
            if "blends_created" not in self.reasoning_context:
                self.reasoning_context["blends_created"] = []
            self.reasoning_context["blends_created"].append(blend_data["id"])
        
        return blend_data
    
    async def _extract_conceptual_insights(self, space_id: str, context: SharedContext) -> List[Dict[str, Any]]:
        """Extract conceptual insights from a space"""
        insights = []
        space = self.original_core.concept_spaces.get(space_id)
        
        if not space:
            return insights
        
        # Find concepts relevant to input
        input_keywords = context.user_input.lower().split()
        
        for concept_id, concept in space.concepts.items():
            if any(kw in concept["name"].lower() for kw in input_keywords):
                # Found relevant concept
                related = space.get_related_concepts(concept_id)
                
                insight = {
                    "type": "conceptual_network",
                    "central_concept": concept["name"],
                    "related_concepts": [r["concept"]["name"] for r in related[:5]],
                    "space": space.name,
                    "properties": list(concept.get("properties", {}).keys())[:5]
                }
                
                insights.append(insight)
        
        return insights
    
    async def _perform_contextual_blending(self, space_ids: List[str], blend_type: str) -> Dict[str, Any]:
        """Perform complete contextual blending with all details"""
        if len(space_ids) < 2:
            return {"error": "Need at least 2 spaces for blending", "success": False}
        
        result = {
            "success": False,
            "blend_id": None,
            "blend_type": blend_type,
            "input_spaces": space_ids[:2],
            "concepts_blended": 0,
            "novel_concepts": [],
            "emergent_properties": [],
            "integration_insights": []
        }
        
        try:
            space1 = self.original_core.concept_spaces.get(space_ids[0])
            space2 = self.original_core.concept_spaces.get(space_ids[1])
            
            if not space1 or not space2:
                result["error"] = "Invalid space IDs"
                return result
            
            # Find mappings between spaces
            mappings = []
            mapping_scores = {}
            
            for c1_id, c1 in space1.concepts.items():
                for c2_id, c2 in space2.concepts.items():
                    similarity = self.original_core._calculate_concept_similarity(c1, c2, space1, space2)
                    
                    if similarity >= 0.5:
                        mappings.append({
                            "concept1": c1_id,
                            "concept2": c2_id,
                            "similarity": similarity
                        })
                        mapping_scores[(c1_id, c2_id)] = similarity
            
            if not mappings:
                result["error"] = "No suitable mappings found"
                return result
            
            # Create blend based on type
            blend_name = f"{space1.name}_{space2.name}_{blend_type}_blend"
            blend_id = f"blend_{len(self.original_core.blends)}"
            
            # Generate blended concepts based on blend type
            blended_concepts = {}
            
            if blend_type == "composition":
                # Standard composition - combine properties
                for mapping in mappings[:10]:  # Limit to top 10 mappings
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    blended_concept = {
                        "name": f"{c1['name']}_{c2['name']}_composite",
                        "properties": {**c1.get("properties", {}), **c2.get("properties", {})},
                        "source_concepts": [c1["name"], c2["name"]],
                        "blend_strength": mapping["similarity"]
                    }
                    
                    blended_concepts[f"blend_{len(blended_concepts)}"] = blended_concept
                    result["concepts_blended"] += 1
            
            elif blend_type == "fusion":
                # Deep fusion - merge and transform properties
                for mapping in mappings[:7]:  # Fewer but deeper
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    # Fuse properties
                    fused_props = {}
                    for key in set(c1.get("properties", {}).keys()) | set(c2.get("properties", {}).keys()):
                        val1 = c1.get("properties", {}).get(key)
                        val2 = c2.get("properties", {}).get(key)
                        
                        if val1 and val2:
                            fused_props[key] = f"fused({val1}, {val2})"
                        else:
                            fused_props[key] = val1 or val2
                    
                    blended_concept = {
                        "name": f"{c1['name']}~{c2['name']}",
                        "properties": fused_props,
                        "source_concepts": [c1["name"], c2["name"]],
                        "blend_strength": mapping["similarity"],
                        "fusion_type": "deep"
                    }
                    
                    blended_concepts[f"fusion_{len(blended_concepts)}"] = blended_concept
                    result["concepts_blended"] += 1
                    result["novel_concepts"].append(blended_concept["name"])
            
            elif blend_type == "elaboration":
                # Elaboration - expand and explore
                for mapping in mappings[:5]:
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    # Generate elaborated variants
                    elaborations = [
                        {
                            "name": f"{c1['name']}_via_{c2['name']}",
                            "properties": {
                                **c1.get("properties", {}),
                                "elaborated_through": c2["name"],
                                "new_perspective": "cross_domain"
                            }
                        },
                        {
                            "name": f"{c2['name']}_as_{c1['name']}",
                            "properties": {
                                **c2.get("properties", {}),
                                "reframed_as": c1["name"],
                                "conceptual_shift": "metaphorical"
                            }
                        }
                    ]
                    
                    for i, elab in enumerate(elaborations):
                        blended_concepts[f"elab_{len(blended_concepts)}"] = elab
                        result["novel_concepts"].append(elab["name"])
                    
                    result["concepts_blended"] += 2
            
            elif blend_type == "contrast":
                # Contrast - highlight differences
                for mapping in mappings[:5]:
                    c1 = space1.concepts[mapping["concept1"]]
                    c2 = space2.concepts[mapping["concept2"]]
                    
                    # Find contrasting properties
                    contrasts = {}
                    all_props = set(c1.get("properties", {}).keys()) | set(c2.get("properties", {}).keys())
                    
                    for prop in all_props:
                        val1 = c1.get("properties", {}).get(prop)
                        val2 = c2.get("properties", {}).get(prop)
                        
                        if val1 and val2 and val1 != val2:
                            contrasts[f"contrast_{prop}"] = {"space1": val1, "space2": val2}
                    
                    if contrasts:
                        contrast_concept = {
                            "name": f"{c1['name']}_vs_{c2['name']}",
                            "properties": contrasts,
                            "type": "contrastive_analysis"
                        }
                        
                        blended_concepts[f"contrast_{len(blended_concepts)}"] = contrast_concept
                        result["concepts_blended"] += 1
            
            elif blend_type == "completion":
                # Completion - fill gaps
                # Find concepts in space1 without good matches in space2
                unmapped_space1 = set(space1.concepts.keys())
                unmapped_space2 = set(space2.concepts.keys())
                
                for mapping in mappings:
                    unmapped_space1.discard(mapping["concept1"])
                    unmapped_space2.discard(mapping["concept2"])
                
                # Create bridging concepts
                for unmapped_id in list(unmapped_space1)[:3]:
                    unmapped = space1.concepts[unmapped_id]
                    
                    bridge_concept = {
                        "name": f"{unmapped['name']}_bridge",
                        "properties": {
                            **unmapped.get("properties", {}),
                            "bridges_to": space2.name,
                            "completion_type": "gap_filler"
                        }
                    }
                    
                    blended_concepts[f"bridge_{len(blended_concepts)}"] = bridge_concept
                    result["novel_concepts"].append(bridge_concept["name"])
                    result["concepts_blended"] += 1
            
            # Identify emergent properties
            if len(blended_concepts) > 3:
                result["emergent_properties"].append("cross_domain_synthesis")
                
                if blend_type == "fusion":
                    result["emergent_properties"].append("unified_framework")
                elif blend_type == "elaboration":
                    result["emergent_properties"].append("expanded_possibility_space")
            
            # Generate integration insights
            result["integration_insights"] = [
                f"Successfully blended {result['concepts_blended']} concepts using {blend_type}",
                f"Discovered {len(result['novel_concepts'])} novel conceptual combinations",
                f"Blend reveals {len(result['emergent_properties'])} emergent properties"
            ]
            
            # Store the blend
            blend_data = {
                "id": blend_id,
                "name": blend_name,
                "type": blend_type,
                "input_spaces": [space1.name, space2.name],
                "concepts": blended_concepts,
                "mappings": mappings,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "concept_count": len(blended_concepts),
                    "emergent_properties": result["emergent_properties"]
                }
            }
            
            self.original_core.blends[blend_id] = blend_data
            self.original_core.stats["blends_created"] += 1
            
            result["success"] = True
            result["blend_id"] = blend_id
            
        except Exception as e:
            logger.error(f"Error in contextual blending: {e}")
            result["error"] = str(e)
        
        return result
    
    async def _adjust_reasoning_from_emotion(self, emotional_data: Dict[str, Any]):
        """Adjust reasoning based on emotional state"""
        dominant_emotion = emotional_data.get("dominant_emotion")
        if not dominant_emotion:
            return
        
        emotion_name, strength = dominant_emotion
        
        # Store emotional influence in reasoning context
        self.reasoning_context["emotional_influence"] = {
            "emotion": emotion_name,
            "strength": strength,
            "timestamp": datetime.now().isoformat()
        }
        
        # Adjust reasoning parameters based on emotion
        if emotion_name == "Curiosity" and strength > 0.5:
            # Increase exploration in reasoning
            self.original_core.causal_config["discovery_threshold"] *= 0.8  # Lower threshold
            self.original_core.blending_config["default_mapping_threshold"] *= 0.9  # More mappings
        elif emotion_name == "Anxiety" and strength > 0.6:
            # Focus on understanding and control
            self.original_core.causal_config["min_relation_strength"] *= 1.2  # Higher standards
    
    async def _align_reasoning_with_goals(self, goal_data: Dict[str, Any]):
        """Align reasoning with active goals"""
        active_goals = goal_data.get("active_goals", [])
        
        # Store goal alignment in context
        self.reasoning_context["goal_alignment"] = {
            "aligned_goals": [],
            "timestamp": datetime.now().isoformat()
        }
        
        for goal in active_goals:
            if goal.get("priority", 0) > 0.6:
                # High priority goal - align reasoning
                goal_desc = goal.get("description", "").lower()
                
                if "understand" in goal_desc or "knowledge" in goal_desc:
                    # Focus on causal discovery
                    self.reasoning_context["goal_alignment"]["aligned_goals"].append({
                        "goal_id": goal.get("id"),
                        "alignment": "causal_discovery"
                    })
                elif "creative" in goal_desc or "novel" in goal_desc:
                    # Focus on conceptual blending
                    self.reasoning_context["goal_alignment"]["aligned_goals"].append({
                        "goal_id": goal.get("id"),
                        "alignment": "conceptual_blending"
                    })
    
    async def _inform_reasoning_from_memory(self, memory_data: Dict[str, Any]):
        """Use memory to inform reasoning"""
        retrieved_memories = memory_data.get("retrieved_memories", [])
        
        # Extract patterns from memories
        memory_patterns = []
        for memory in retrieved_memories:
            if memory.get("memory_type") == "experience":
                # Experience memories can inform causal models
                memory_patterns.append({
                    "type": "experiential",
                    "content": memory.get("content", ""),
                    "relevance": "causal"
                })
            elif memory.get("memory_type") == "reflection":
                # Reflection memories can inform conceptual understanding
                memory_patterns.append({
                    "type": "reflective",
                    "content": memory.get("content", ""),
                    "relevance": "conceptual"
                })
        
        # Store in reasoning context
        self.reasoning_context["memory_patterns"] = memory_patterns
    
    # ========================================================================================
    # ANALYSIS AND SYNTHESIS HELPER METHODS
    # ========================================================================================
    
    async def _analyze_reasoning_potential(self, context: SharedContext, messages: Dict,
                                        available_models: List[str], 
                                        available_spaces: List[str]) -> Dict[str, Any]:
        """Analyze the potential for different types of reasoning"""
        potential = {
            "causal_potential": 0.0,
            "conceptual_potential": 0.0,
            "integrated_potential": 0.0,
            "factors": []
        }
        
        # Causal potential
        if available_models:
            potential["causal_potential"] += 0.3
            potential["factors"].append("existing_causal_models")
        
        if "knowledge_core" in messages:
            potential["causal_potential"] += 0.2
            potential["factors"].append("knowledge_available")
        
        # Conceptual potential
        if available_spaces:
            potential["conceptual_potential"] += 0.3
            potential["factors"].append("existing_concept_spaces")
        
        if context.emotional_state and context.emotional_state.get("dominant_emotion"):
            emotion = context.emotional_state["dominant_emotion"][0]
            if emotion in ["Curiosity", "Wonder", "Excitement"]:
                potential["conceptual_potential"] += 0.2
                potential["factors"].append("creative_emotional_state")
        
        # Integrated potential
        if available_models and available_spaces:
            potential["integrated_potential"] = (
                potential["causal_potential"] + potential["conceptual_potential"]
            ) / 2
            potential["factors"].append("both_systems_available")
        
        return potential
    
    async def _identify_reasoning_opportunities(self, context: SharedContext,
                                             messages: Dict) -> List[Dict[str, Any]]:
        """Identify specific reasoning opportunities"""
        opportunities = []
        
        # Check for causal discovery opportunity
        if "perception_input" in [m.get("type") for msgs in messages.values() for m in msgs]:
            opportunities.append({
                "type": "causal_discovery",
                "trigger": "new_perceptual_data",
                "priority": 0.7
            })
        
        # Check for conceptual blending opportunity
        if len(self.original_core.concept_spaces) >= 2:
            opportunities.append({
                "type": "conceptual_blending",
                "trigger": "multiple_concept_spaces",
                "priority": 0.6
            })
        
        # Check for counterfactual reasoning opportunity
        if "what if" in context.user_input.lower():
            opportunities.append({
                "type": "counterfactual_reasoning",
                "trigger": "counterfactual_query",
                "priority": 0.9
            })
        
        return opportunities
    
    async def _analyze_reasoning_coherence(self, context: SharedContext,
                                        messages: Dict) -> Dict[str, Any]:
        """Analyze coherence between reasoning and other modules"""
        coherence = {
            "overall_score": 1.0,
            "issues": [],
            "alignments": []
        }
        
        # Check alignment with emotional state
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion")
            if emotion and emotion[0] == "Confusion" and len(self.active_models) == 0:
                coherence["overall_score"] -= 0.2
                coherence["issues"].append("no_models_during_confusion")
            else:
                coherence["alignments"].append("emotion_reasoning_aligned")
        
        # Check alignment with goals
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            understanding_goals = [g for g in active_goals if "understand" in g.get("description", "").lower()]
            
            if understanding_goals and len(self.active_models) > 0:
                coherence["alignments"].append("goal_reasoning_aligned")
            elif understanding_goals and len(self.active_models) == 0:
                coherence["overall_score"] -= 0.15
                coherence["issues"].append("understanding_goal_without_models")
        
        return coherence
    
    # ========================================================================================
    # CAUSAL ANALYSIS HELPER METHODS
    # ========================================================================================
    
    async def _find_relevant_causal_chains(self, model, context: SharedContext) -> List[Dict[str, Any]]:
        """Find causal chains in model relevant to context"""
        chains = []
        
        # Extract key terms from input
        input_terms = set(context.user_input.lower().split())
        input_terms.update(self._extract_domain_keywords(context.user_input.lower()))
        
        # Find nodes matching input terms
        relevant_nodes = []
        for node_id, node in model.nodes.items():
            node_name_lower = node.name.lower()
            if any(term in node_name_lower for term in input_terms):
                relevant_nodes.append(node_id)
        
        # For each relevant node, trace causal chains
        for node_id in relevant_nodes[:3]:  # Limit to top 3 to avoid overwhelming
            # Get upstream chain (causes)
            upstream_chain = []
            current_nodes = [node_id]
            visited = set()
            
            for depth in range(3):  # Max depth of 3
                next_nodes = []
                for current in current_nodes:
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Find parent nodes (causes)
                    for relation in model.relations:
                        if relation.target == current and relation.source not in visited:
                            next_nodes.append(relation.source)
                            upstream_chain.append({
                                "from": relation.source,
                                "to": current,
                                "strength": relation.strength,
                                "mechanism": relation.mechanism
                            })
                
                current_nodes = next_nodes
                if not current_nodes:
                    break
            
            # Get downstream chain (effects)
            downstream_chain = []
            current_nodes = [node_id]
            visited = {node_id}
            
            for depth in range(3):  # Max depth of 3
                next_nodes = []
                for current in current_nodes:
                    # Find child nodes (effects)
                    for relation in model.relations:
                        if relation.source == current and relation.target not in visited:
                            next_nodes.append(relation.target)
                            visited.add(relation.target)
                            downstream_chain.append({
                                "from": current,
                                "to": relation.target,
                                "strength": relation.strength,
                                "mechanism": relation.mechanism
                            })
                
                current_nodes = next_nodes
                if not current_nodes:
                    break
            
            # Create chain summary
            if upstream_chain or downstream_chain:
                chain = {
                    "central_node": model.nodes[node_id].name,
                    "central_node_id": node_id,
                    "upstream_depth": len(set(r["from"] for r in upstream_chain)),
                    "downstream_depth": len(set(r["to"] for r in downstream_chain)),
                    "total_relations": len(upstream_chain) + len(downstream_chain),
                    "upstream_chain": upstream_chain,
                    "downstream_chain": downstream_chain,
                    "chain_type": self._classify_chain_type(upstream_chain, downstream_chain),
                    "relevance_score": self._calculate_chain_relevance(
                        upstream_chain + downstream_chain, context
                    )
                }
                
                chains.append(chain)
        
        # Sort by relevance
        chains.sort(key=lambda c: c["relevance_score"], reverse=True)
        
        return chains
    
    async def _identify_intervention_points(self, model, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify points in causal model where interventions could be effective"""
        intervention_points = []
        
        # Strategy 1: Find high-centrality nodes (influential points)
        try:
            import networkx as nx
            
            # Calculate various centrality measures
            betweenness = nx.betweenness_centrality(model.graph)
            eigenvector = nx.eigenvector_centrality(model.graph, max_iter=1000)
            degree = nx.degree_centrality(model.graph)
            
            # Combine centrality scores
            combined_scores = {}
            for node_id in model.nodes:
                combined_scores[node_id] = (
                    betweenness.get(node_id, 0) * 0.4 +
                    eigenvector.get(node_id, 0) * 0.3 +
                    degree.get(node_id, 0) * 0.3
                )
            
            # Get top intervention candidates
            top_nodes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for node_id, score in top_nodes:
                node = model.nodes[node_id]
                
                # Analyze intervention potential
                intervention_point = {
                    "node_id": node_id,
                    "node_name": node.name,
                    "intervention_score": score,
                    "intervention_type": self._determine_intervention_type(node, model),
                    "expected_impact": self._estimate_intervention_impact(node_id, model),
                    "feasibility": self._assess_intervention_feasibility(node, context),
                    "downstream_effects": len(model.get_descendants(node_id)),
                    "upstream_causes": len(model.get_ancestors(node_id))
                }
                
                # Add context-specific reasoning
                if context.goal_context:
                    intervention_point["goal_alignment"] = self._assess_goal_alignment(
                        node, context.goal_context
                    )
                
                intervention_points.append(intervention_point)
                
        except Exception as e:
            logger.warning(f"NetworkX analysis failed: {e}")
            
            # Fallback: Simple analysis based on node connectivity
            for node_id, node in model.nodes.items():
                in_degree = sum(1 for r in model.relations if r.target == node_id)
                out_degree = sum(1 for r in model.relations if r.source == node_id)
                
                if out_degree > 2:  # Nodes with multiple effects
                    intervention_points.append({
                        "node_id": node_id,
                        "node_name": node.name,
                        "intervention_score": out_degree / 10.0,
                        "intervention_type": "high_impact",
                        "expected_impact": "multiple_downstream_effects",
                        "feasibility": 0.5,
                        "downstream_effects": out_degree,
                        "upstream_causes": in_degree
                    })
        
        # Sort by intervention score
        intervention_points.sort(key=lambda p: p["intervention_score"], reverse=True)
        
        return intervention_points
    
    def _classify_chain_type(self, upstream_chain: List[Dict], downstream_chain: List[Dict]) -> str:
        """Classify the type of causal chain"""
        if len(upstream_chain) > len(downstream_chain) * 2:
            return "convergent"  # Many causes lead to one effect
        elif len(downstream_chain) > len(upstream_chain) * 2:
            return "divergent"  # One cause leads to many effects
        elif len(upstream_chain) == 0:
            return "source"  # Starting point
        elif len(downstream_chain) == 0:
            return "sink"  # End point
        else:
            return "pathway"  # Middle of chain
    
    def _calculate_chain_relevance(self, chain_relations: List[Dict], context: SharedContext) -> float:
        """Calculate relevance of a causal chain to context"""
        relevance = 0.0
        
        # Base relevance on chain length and strength
        if chain_relations:
            avg_strength = sum(r["strength"] for r in chain_relations) / len(chain_relations)
            relevance += avg_strength * 0.3
            
            # Bonus for longer coherent chains
            if len(chain_relations) > 3:
                relevance += 0.2
        
        # Check goal alignment
        if context.goal_context:
            active_goals = context.goal_context.get("active_goals", [])
            if any("understand" in g.get("description", "").lower() for g in active_goals):
                relevance += 0.2  # Understanding goals favor causal chains
        
        # Check emotional context
        if context.emotional_state:
            dominant_emotion = context.emotional_state.get("dominant_emotion")
            if dominant_emotion and dominant_emotion[0] == "Curiosity":
                relevance += 0.1  # Curiosity increases relevance of all chains
        
        return min(1.0, relevance)
    
    def _determine_intervention_type(self, node, model) -> str:
        """Determine what type of intervention would be appropriate for a node"""
        # Check node properties
        if hasattr(node, 'node_type'):
            if node.node_type == 'observable':
                return "measurement"
            elif node.node_type == 'latent':
                return "indirect"
        
        # Check based on relationships
        in_relations = [r for r in model.relations if r.target == node.id]
        out_relations = [r for r in model.relations if r.source == node.id]
        
        if len(in_relations) == 0:
            return "root_cause_modification"
        elif len(out_relations) == 0:
            return "outcome_optimization"
        elif len(out_relations) > 3:
            return "leverage_point"
        else:
            return "pathway_modulation"
    
    def _estimate_intervention_impact(self, node_id: str, model) -> str:
        """Estimate the impact of intervening at a node"""
        descendants = model.get_descendants(node_id)
        
        if len(descendants) > 10:
            return "very_high"
        elif len(descendants) > 5:
            return "high"
        elif len(descendants) > 2:
            return "moderate"
        elif len(descendants) > 0:
            return "low"
        else:
            return "minimal"
    
    def _assess_intervention_feasibility(self, node, context: SharedContext) -> float:
        """Assess how feasible it would be to intervene at this node"""
        feasibility = 0.5  # Base feasibility
        
        # Check if node represents something controllable
        node_name_lower = node.name.lower()
        
        # Highly feasible interventions
        if any(term in node_name_lower for term in ["policy", "decision", "choice", "action", "behavior"]):
            feasibility += 0.3
        
        # Moderately feasible
        elif any(term in node_name_lower for term in ["process", "method", "approach", "strategy"]):
            feasibility += 0.2
        
        # Less feasible
        elif any(term in node_name_lower for term in ["weather", "natural", "inherent", "genetic"]):
            feasibility -= 0.2
        
        # Check context for constraints
        if context.constraints:
            if "limited_resources" in context.constraints:
                feasibility -= 0.1
            if "time_sensitive" in context.constraints:
                feasibility -= 0.1
        
        return max(0.0, min(1.0, feasibility))
    
    def _assess_goal_alignment(self, node, goal_context: Dict[str, Any]) -> float:
        """Assess how well an intervention aligns with active goals"""
        alignment = 0.0
        active_goals = goal_context.get("active_goals", [])
        
        node_name_lower = node.name.lower()
        
        for goal in active_goals:
            goal_desc = goal.get("description", "").lower()
            goal_priority = goal.get("priority", 0.5)
            
            # Check for keyword matches
            goal_keywords = set(goal_desc.split())
            node_keywords = set(node_name_lower.split())
            
            overlap = len(goal_keywords.intersection(node_keywords))
            if overlap > 0:
                alignment += overlap * 0.1 * goal_priority
            
            # Check for semantic alignment
            if goal.get("associated_need") in node_name_lower:
                alignment += 0.2 * goal_priority
        
        return min(1.0, alignment)
    
    # ========================================================================================
    # CONCEPTUAL ANALYSIS HELPER METHODS
    # ========================================================================================
    
    async def _identify_key_concepts(self, space, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify key concepts in a conceptual space"""
        key_concepts = []
        
        # Calculate concept importance scores
        for concept_id, concept in space.concepts.items():
            importance_score = 0.0
            
            # Factor 1: Connectivity (how many relations)
            relations_count = sum(1 for r in space.relations if 
                                r["source"] == concept_id or r["target"] == concept_id)
            importance_score += min(relations_count / 10.0, 0.3)
            
            # Factor 2: Property richness
            property_count = len(concept.get("properties", {}))
            importance_score += min(property_count / 20.0, 0.2)
            
            # Factor 3: Context relevance
            relevance = self._calculate_concept_relevance_to_context(concept, context)
            importance_score += relevance * 0.5
            
            if importance_score > 0.3:  # Threshold for key concepts
                # Get related concepts for context
                related = space.get_related_concepts(concept_id)
                
                key_concept = {
                    "concept_id": concept_id,
                    "name": concept["name"],
                    "importance_score": importance_score,
                    "properties": self._get_salient_properties(concept),
                    "relation_count": relations_count,
                    "related_concepts": [r["concept"]["name"] for r in related[:3]],
                    "semantic_role": self._determine_semantic_role(concept, space),
                    "context_alignment": relevance
                }
                
                key_concepts.append(key_concept)
        
        # Sort by importance
        key_concepts.sort(key=lambda c: c["importance_score"], reverse=True)
        
        return key_concepts[:10]  # Top 10 concepts
    
    async def _identify_conceptual_patterns(self, space, context: SharedContext) -> List[Dict[str, Any]]:
        """Identify patterns in conceptual organization"""
        patterns = []
        
        # Pattern 1: Hierarchical structures
        hierarchies = self._find_hierarchical_patterns(space)
        for hierarchy in hierarchies:
            patterns.append({
                "type": "hierarchy",
                "root_concept": hierarchy["root"],
                "depth": hierarchy["depth"],
                "branch_count": hierarchy["branches"],
                "pattern_strength": hierarchy["consistency"],
                "example_path": hierarchy["example_path"]
            })
        
        # Pattern 2: Clusters (tightly connected groups)
        clusters = self._find_conceptual_clusters(space)
        for cluster in clusters:
            patterns.append({
                "type": "cluster",
                "central_concepts": cluster["centers"],
                "cluster_size": cluster["size"],
                "density": cluster["density"],
                "theme": self._infer_cluster_theme(cluster, space),
                "pattern_strength": cluster["cohesion"]
            })
        
        # Pattern 3: Bridge concepts (connect different areas)
        bridges = self._find_bridge_concepts(space)
        for bridge in bridges:
            patterns.append({
                "type": "bridge",
                "bridge_concept": bridge["concept"],
                "connected_regions": bridge["regions"],
                "bridge_strength": bridge["strength"],
                "integration_potential": bridge["potential"]
            })
        
        # Pattern 4: Conceptual gradients (smooth transitions)
        gradients = self._find_conceptual_gradients(space)
        for gradient in gradients:
            patterns.append({
                "type": "gradient",
                "dimension": gradient["dimension"],
                "start_concept": gradient["start"],
                "end_concept": gradient["end"],
                "intermediate_concepts": gradient["path"],
                "smoothness": gradient["smoothness"]
            })
        
        # Filter patterns by context relevance
        relevant_patterns = []
        for pattern in patterns:
            if self._pattern_relevant_to_context(pattern, context):
                relevant_patterns.append(pattern)
        
        return relevant_patterns
    
    def _calculate_concept_relevance_to_context(self, concept: Dict[str, Any], 
                                               context: SharedContext) -> float:
        """Calculate how relevant a concept is to the current context"""
        relevance = 0.0
        
        # Check name similarity
        concept_name = concept.get("name", "").lower()
        input_words = set(context.user_input.lower().split())
        
        # Direct word matches
        name_words = set(concept_name.split())
        word_overlap = len(name_words.intersection(input_words))
        relevance += word_overlap * 0.2
        
        # Property matches
        for prop_name, prop_value in concept.get("properties", {}).items():
            if isinstance(prop_value, str):
                prop_words = set(prop_value.lower().split())
                if prop_words.intersection(input_words):
                    relevance += 0.1
        
        # Domain alignment
        if hasattr(concept, "domain") and concept.domain:
            domain_keywords = self._extract_domain_keywords(context.user_input.lower())
            if any(kw in concept.domain.lower() for kw in domain_keywords):
                relevance += 0.2
        
        # Emotional context alignment
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion")
            if emotion and self._concept_aligns_with_emotion(concept, emotion[0]):
                relevance += 0.15
        
        return min(1.0, relevance)
    
    def _get_salient_properties(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the most salient properties of a concept"""
        properties = concept.get("properties", {})
        
        if len(properties) <= 5:
            return properties
        
        # Rank properties by salience
        salient_props = {}
        
        # Prioritize certain property types
        priority_keys = ["definition", "purpose", "function", "type", "category"]
        
        for key in priority_keys:
            if key in properties:
                salient_props[key] = properties[key]
        
        # Add other properties up to limit
        for key, value in properties.items():
            if key not in salient_props and len(salient_props) < 5:
                salient_props[key] = value
        
        return salient_props
    
    def _determine_semantic_role(self, concept: Dict[str, Any], space) -> str:
        """Determine the semantic role of a concept in the space"""
        concept_id = concept.get("id", "")
        
        # Count different types of relations
        incoming_relations = [r for r in space.relations if r["target"] == concept_id]
        outgoing_relations = [r for r in space.relations if r["source"] == concept_id]
        
        # Analyze relation types
        if len(incoming_relations) > len(outgoing_relations) * 2:
            return "terminal"  # End point of many relations
        elif len(outgoing_relations) > len(incoming_relations) * 2:
            return "generative"  # Source of many relations
        elif len(incoming_relations) > 5 and len(outgoing_relations) > 5:
            return "hub"  # Central connector
        elif len(incoming_relations) == 0 and len(outgoing_relations) > 0:
            return "primitive"  # Basic building block
        elif len(outgoing_relations) == 0 and len(incoming_relations) > 0:
            return "composite"  # Built from other concepts
        else:
            return "intermediate"  # Regular concept
    
    def _find_hierarchical_patterns(self, space) -> List[Dict[str, Any]]:
        """Comprehensive hierarchical pattern detection"""
        hierarchies = []
        
        # Extended hierarchical relations
        hierarchical_relations = [
            "is_a", "type_of", "kind_of", "instance_of", "example_of",
            "part_of", "component_of", "member_of", "subset_of", "belongs_to",
            "inherits_from", "derives_from", "extends", "implements",
            "generalizes", "specializes", "abstracts", "refines"
        ]
        
        # Build parent-child mappings
        parent_children = defaultdict(list)
        child_parents = defaultdict(list)
        
        for relation in space.relations:
            if relation.get("relation_type") in hierarchical_relations:
                parent = relation["source"]
                child = relation["target"]
                parent_children[parent].append(child)
                child_parents[child].append(parent)
        
        # Find roots (nodes with no hierarchical parents)
        all_nodes = set(space.concepts.keys())
        potential_roots = all_nodes - set(child_parents.keys())
        
        # Build hierarchies from each root
        for root_id in potential_roots:
            hierarchy = self._build_hierarchy_tree(root_id, parent_children, space)
            
            if hierarchy["total_nodes"] > 2:  # Meaningful hierarchy
                hierarchies.append(hierarchy)
        
        # Detect polyhierarchies (nodes with multiple parents)
        polyhierarchical_nodes = [
            node for node, parents in child_parents.items() 
            if len(parents) > 1
        ]
        
        # Analyze hierarchy characteristics
        for hierarchy in hierarchies:
            hierarchy["characteristics"] = self._analyze_hierarchy_characteristics(
                hierarchy, polyhierarchical_nodes, space
            )
        
        return hierarchies
    
    def _build_hierarchy_tree(self, root_id: str, parent_children: Dict, space) -> Dict[str, Any]:
        """Build complete hierarchy tree from root"""
        hierarchy = {
            "root": root_id,
            "root_name": space.concepts[root_id]["name"],
            "levels": defaultdict(list),
            "total_nodes": 1,
            "max_depth": 0,
            "avg_branching_factor": 0.0,
            "tree_structure": {}
        }
        
        # BFS to build levels
        queue = [(root_id, 0)]
        visited = {root_id}
        node_levels = {root_id: 0}
        
        while queue:
            current_id, level = queue.pop(0)
            hierarchy["levels"][level].append(current_id)
            hierarchy["max_depth"] = max(hierarchy["max_depth"], level)
            
            # Get children
            children = parent_children.get(current_id, [])
            hierarchy["tree_structure"][current_id] = children
            
            for child_id in children:
                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, level + 1))
                    node_levels[child_id] = level + 1
                    hierarchy["total_nodes"] += 1
        
        # Calculate average branching factor
        total_branches = sum(len(children) for children in parent_children.values())
        nodes_with_children = len([n for n in parent_children if parent_children[n]])
        
        if nodes_with_children > 0:
            hierarchy["avg_branching_factor"] = total_branches / nodes_with_children
        
        return hierarchy
    
    def _analyze_hierarchy_characteristics(self, hierarchy: Dict, 
                                         polyhierarchical_nodes: List, 
                                         space) -> Dict[str, Any]:
        """Analyze characteristics of a hierarchy"""
        characteristics = {
            "hierarchy_type": "unknown",
            "balance": "unknown",
            "specialization_pattern": "unknown",
            "semantic_consistency": 0.0
        }
        
        # Determine hierarchy type
        if hierarchy["avg_branching_factor"] > 5:
            characteristics["hierarchy_type"] = "flat"
        elif hierarchy["max_depth"] > 5:
            characteristics["hierarchy_type"] = "deep"
        else:
            characteristics["hierarchy_type"] = "balanced"
        
        # Analyze balance
        level_sizes = [len(nodes) for nodes in hierarchy["levels"].values()]
        if level_sizes:
            size_variance = np.var(level_sizes)
            if size_variance < 2:
                characteristics["balance"] = "highly_balanced"
            elif size_variance < 10:
                characteristics["balance"] = "moderately_balanced"
            else:
                characteristics["balance"] = "unbalanced"
        
        # Analyze specialization pattern
        if hierarchy["max_depth"] > 0:
            deeper_levels_avg = np.mean([len(hierarchy["levels"][i]) 
                                        for i in range(1, hierarchy["max_depth"] + 1)])
            if deeper_levels_avg > len(hierarchy["levels"][0]):
                characteristics["specialization_pattern"] = "expanding"
            else:
                characteristics["specialization_pattern"] = "converging"
        
        # Semantic consistency check
        consistency_scores = []
        for level_nodes in hierarchy["levels"].values():
            if len(level_nodes) > 1:
                # Check semantic similarity within level
                similarities = []
                for i in range(len(level_nodes)):
                    for j in range(i + 1, len(level_nodes)):
                        concept1 = space.concepts[level_nodes[i]]
                        concept2 = space.concepts[level_nodes[j]]
                        sim = self._calculate_concept_similarity_simple(concept1, concept2)
                        similarities.append(sim)
                
                if similarities:
                    consistency_scores.append(np.mean(similarities))
        
        if consistency_scores:
            characteristics["semantic_consistency"] = np.mean(consistency_scores)
        
        return characteristics
    
    def _trace_hierarchy(self, root_id: str, space, relation_types: List[str]) -> Dict[str, Any]:
        """Trace a hierarchical structure from a root concept"""
        hierarchy = {
            "root": space.concepts[root_id]["name"],
            "root_id": root_id,
            "depth": 0,
            "branches": 0,
            "consistency": 1.0,
            "example_path": []
        }
        
        # BFS to explore hierarchy
        queue = [(root_id, 0, [root_id])]
        visited = {root_id}
        max_depth = 0
        branch_count = 0
        
        while queue:
            current_id, depth, path = queue.pop(0)
            max_depth = max(max_depth, depth)
            
            # Find children
            children = [
                r["target"] for r in space.relations
                if r["source"] == current_id and 
                r.get("relation_type") in relation_types and
                r["target"] not in visited
            ]
            
            if children:
                branch_count += len(children) - 1  # -1 because first child continues branch
                
                for child_id in children:
                    visited.add(child_id)
                    new_path = path + [child_id]
                    queue.append((child_id, depth + 1, new_path))
                    
                    # Keep track of one example path
                    if depth + 1 > hierarchy["depth"]:
                        hierarchy["depth"] = depth + 1
                        hierarchy["example_path"] = [
                            space.concepts[cid]["name"] for cid in new_path
                        ]
        
        hierarchy["branches"] = branch_count
        
        return hierarchy
    
    def _find_conceptual_clusters(self, space) -> List[Dict[str, Any]]:
        """Advanced conceptual clustering with multiple strategies"""
        clusters = []
        
        # Strategy 1: Dense subgraph detection
        dense_clusters = self._find_dense_subgraphs(space)
        
        # Strategy 2: Semantic clustering
        semantic_clusters = self._find_semantic_clusters(space)
        
        # Strategy 3: Property-based clustering
        property_clusters = self._find_property_clusters(space)
        
        # Merge and reconcile clusters
        all_clusters = self._merge_cluster_results(
            dense_clusters, semantic_clusters, property_clusters, space
        )
        
        # Analyze each cluster
        for cluster in all_clusters:
            cluster["analysis"] = self._analyze_cluster_properties(cluster, space)
            cluster["quality_score"] = self._calculate_cluster_quality(cluster, space)
        
        # Filter high-quality clusters
        clusters = [c for c in all_clusters if c["quality_score"] > 0.4]
        
        return clusters
    
    def _find_dense_subgraphs(self, space) -> List[Dict[str, Any]]:
        """Find densely connected subgraphs"""
        clusters = []
        visited_globally = set()
        
        # For each unvisited node, try to grow a cluster
        for start_node in space.concepts:
            if start_node in visited_globally:
                continue
            
            # Grow cluster using density threshold
            cluster = self._grow_dense_cluster(start_node, space, visited_globally)
            
            if len(cluster["members"]) >= 3:
                clusters.append(cluster)
        
        return clusters
    
    def _grow_dense_cluster(self, start_node: str, space, visited_globally: set) -> Dict[str, Any]:
        """Grow a cluster based on connection density"""
        cluster_members = {start_node}
        boundary = {start_node}
        density_threshold = 0.4
        
        while boundary:
            new_boundary = set()
            
            for node in boundary:
                # Get neighbors
                neighbors = self._get_concept_neighbors(node, space)
                
                for neighbor in neighbors:
                    if neighbor not in cluster_members:
                        # Check if neighbor is densely connected to cluster
                        connections_to_cluster = sum(
                            1 for member in cluster_members 
                            if self._concepts_connected(neighbor, member, space)
                        )
                        
                        density = connections_to_cluster / len(cluster_members)
                        
                        if density >= density_threshold:
                            cluster_members.add(neighbor)
                            new_boundary.add(neighbor)
            
            boundary = new_boundary
        
        visited_globally.update(cluster_members)
        
        return {
            "members": list(cluster_members),
            "size": len(cluster_members),
            "density": self._calculate_cluster_density(cluster_members, space),
            "cohesion": self._calculate_cluster_cohesion(cluster_members, space),
            "cluster_type": "dense_subgraph"
        }
    
    def _find_semantic_clusters(self, space) -> List[Dict[str, Any]]:
        """Find clusters based on semantic similarity"""
        clusters = []
        similarity_threshold = 0.6
        min_cluster_size = 3
        
        # Build similarity matrix
        concepts = list(space.concepts.keys())
        n = len(concepts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                concept1 = space.concepts[concepts[i]]
                concept2 = space.concepts[concepts[j]]
                sim = self._calculate_enhanced_concept_similarity(concept1, concept2, space)
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Find clusters using similarity
        clustered = set()
        
        for i in range(n):
            if i in clustered:
                continue
            
            # Start new cluster
            cluster_indices = {i}
            
            # Add similar concepts
            for j in range(n):
                if j != i and similarity_matrix[i, j] >= similarity_threshold:
                    # Check average similarity to cluster
                    avg_sim = np.mean([similarity_matrix[j, k] for k in cluster_indices])
                    if avg_sim >= similarity_threshold * 0.8:  # Slightly relaxed threshold
                        cluster_indices.add(j)
            
            if len(cluster_indices) >= min_cluster_size:
                clustered.update(cluster_indices)
                cluster_concepts = [concepts[idx] for idx in cluster_indices]
                
                clusters.append({
                    "members": cluster_concepts,
                    "size": len(cluster_concepts),
                    "avg_similarity": np.mean([
                        similarity_matrix[i, j] 
                        for i in cluster_indices 
                        for j in cluster_indices 
                        if i != j
                    ]),
                    "cluster_type": "semantic"
                })
        
        return clusters
    
    def _find_property_clusters(self, space) -> List[Dict[str, Any]]:
        """Find clusters based on shared properties"""
        property_groups = defaultdict(list)
        
        # Group concepts by properties
        for concept_id, concept in space.concepts.items():
            properties = concept.get("properties", {})
            
            # Create property signature
            for prop_name, prop_value in properties.items():
                if isinstance(prop_value, str):
                    property_key = f"{prop_name}:{prop_value}"
                else:
                    property_key = f"{prop_name}:has_value"
                
                property_groups[property_key].append(concept_id)
        
        # Find concepts that share multiple properties
        concept_property_sets = defaultdict(set)
        for prop_key, concepts in property_groups.items():
            for concept in concepts:
                concept_property_sets[concept].add(prop_key)
        
        # Cluster based on property overlap
        clusters = []
        clustered = set()
        
        for concept1, props1 in concept_property_sets.items():
            if concept1 in clustered:
                continue
            
            cluster = {concept1}
            
            for concept2, props2 in concept_property_sets.items():
                if concept2 != concept1 and concept2 not in clustered:
                    overlap = len(props1.intersection(props2))
                    if overlap >= 3:  # Share at least 3 properties
                        cluster.add(concept2)
            
            if len(cluster) >= 3:
                clustered.update(cluster)
                clusters.append({
                    "members": list(cluster),
                    "size": len(cluster),
                    "shared_properties": len(props1),
                    "cluster_type": "property_based"
                })
        
        return clusters
    
    def _calculate_enhanced_concept_similarity(self, concept1: Dict, concept2: Dict, space) -> float:
        """Enhanced concept similarity calculation"""
        similarity = 0.0
        
        # Name similarity (with n-gram analysis)
        name1 = concept1["name"].lower()
        name2 = concept2["name"].lower()
        
        # Word overlap
        words1 = set(name1.split())
        words2 = set(name2.split())
        if words1 and words2:
            word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
            similarity += word_similarity * 0.3
        
        # Character n-gram similarity
        ngrams1 = set(name1[i:i+3] for i in range(len(name1)-2))
        ngrams2 = set(name2[i:i+3] for i in range(len(name2)-2))
        if ngrams1 and ngrams2:
            ngram_similarity = len(ngrams1.intersection(ngrams2)) / len(ngrams1.union(ngrams2))
            similarity += ngram_similarity * 0.2
        
        # Property similarity
        props1 = set(concept1.get("properties", {}).keys())
        props2 = set(concept2.get("properties", {}).keys())
        if props1 and props2:
            prop_similarity = len(props1.intersection(props2)) / len(props1.union(props2))
            similarity += prop_similarity * 0.3
        
        # Relation similarity (concepts connected to similar things)
        neighbors1 = set(self._get_concept_neighbors(concept1.get("id", ""), space))
        neighbors2 = set(self._get_concept_neighbors(concept2.get("id", ""), space))
        if neighbors1 and neighbors2:
            neighbor_similarity = len(neighbors1.intersection(neighbors2)) / len(neighbors1.union(neighbors2))
            similarity += neighbor_similarity * 0.2
        
        return min(1.0, similarity)
    
    def _find_connected_component(self, start_id: str, space, visited: set) -> List[str]:
        """Find all concepts connected to start_id"""
        component = []
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
                
            visited.add(current)
            component.append(current)
            
            # Find neighbors
            neighbors = set()
            for relation in space.relations:
                if relation["source"] == current:
                    neighbors.add(relation["target"])
                elif relation["target"] == current:
                    neighbors.add(relation["source"])
            
            queue.extend(neighbors - visited)
        
        return component
    
    def _calculate_cluster_density(self, cluster: List[str], space) -> float:
        """Calculate the density of connections within a cluster"""
        if len(cluster) < 2:
            return 0.0
        
        # Count internal edges
        internal_edges = 0
        for relation in space.relations:
            if relation["source"] in cluster and relation["target"] in cluster:
                internal_edges += 1
        
        # Calculate density (actual edges / possible edges)
        possible_edges = len(cluster) * (len(cluster) - 1)
        density = internal_edges / possible_edges if possible_edges > 0 else 0.0
        
        return density
    
    def _calculate_cluster_cohesion(self, cluster: List[str], space) -> float:
        """Calculate semantic cohesion of a cluster"""
        if len(cluster) < 2:
            return 0.0
        
        # Calculate average similarity between cluster members
        total_similarity = 0.0
        comparisons = 0
        
        for i, concept1_id in enumerate(cluster):
            for concept2_id in cluster[i+1:]:
                concept1 = space.concepts[concept1_id]
                concept2 = space.concepts[concept2_id]
                
                similarity = self._calculate_concept_similarity_simple(concept1, concept2)
                total_similarity += similarity
                comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0.0
    
    def _calculate_concept_similarity_simple(self, concept1: Dict, concept2: Dict) -> float:
        """Simple concept similarity calculation"""
        similarity = 0.0
        
        # Name similarity
        name1_words = set(concept1["name"].lower().split())
        name2_words = set(concept2["name"].lower().split())
        name_overlap = len(name1_words.intersection(name2_words))
        similarity += name_overlap * 0.2
        
        # Property overlap
        props1 = set(concept1.get("properties", {}).keys())
        props2 = set(concept2.get("properties", {}).keys())
        prop_overlap = len(props1.intersection(props2))
        similarity += min(prop_overlap * 0.1, 0.3)
        
        return min(1.0, similarity)
    
    def _find_cluster_centers(self, cluster: List[str], space) -> List[str]:
        """Find the most central concepts in a cluster"""
        centrality_scores = {}
        
        for concept_id in cluster:
            # Count connections within cluster
            connections = 0
            for relation in space.relations:
                if relation["source"] == concept_id and relation["target"] in cluster:
                    connections += 1
                elif relation["target"] == concept_id and relation["source"] in cluster:
                    connections += 1
            
            centrality_scores[concept_id] = connections
        
        # Get top 3 centers
        sorted_centers = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return [space.concepts[cid]["name"] for cid, _ in sorted_centers[:3]]
    
    def _infer_cluster_theme(self, cluster: Dict[str, Any], space) -> str:
        """Infer the thematic focus of a cluster"""
        # Get all concept names in cluster
        concept_names = [space.concepts[cid]["name"] for cid in cluster["members"]]
        
        # Extract common words
        word_freq = {}
        for name in concept_names:
            words = name.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find most common meaningful word
        if word_freq:
            theme_word = max(word_freq.items(), key=lambda x: x[1])[0]
            return f"{theme_word}_related"
        
        return "mixed_concepts"
    
    def _find_bridge_concepts(self, space) -> List[Dict[str, Any]]:
        """Comprehensive bridge concept detection"""
        bridges = []
        
        # First, identify distinct regions using community detection
        communities = self._detect_communities(space)
        
        if len(communities) < 2:
            return bridges
        
        # Analyze each concept for bridge potential
        for concept_id, concept in space.concepts.items():
            bridge_score = 0.0
            connected_communities = set()
            bridge_connections = defaultdict(list)
            
            # Check connections to different communities
            neighbors = self._get_concept_neighbors(concept_id, space)
            
            for neighbor in neighbors:
                # Find which community the neighbor belongs to
                for comm_id, community in enumerate(communities):
                    if neighbor in community["members"]:
                        connected_communities.add(comm_id)
                        bridge_connections[comm_id].append(neighbor)
            
            # A good bridge connects multiple communities
            if len(connected_communities) >= 2:
                # Calculate bridge importance
                importance = self._calculate_bridge_importance(
                    concept_id, connected_communities, communities, space
                )
                
                bridges.append({
                    "concept": concept["name"],
                    "concept_id": concept_id,
                    "regions": [communities[i]["name"] for i in connected_communities],
                    "connections_per_region": {
                        communities[i]["name"]: len(connections) 
                        for i, connections in bridge_connections.items()
                    },
                    "strength": len(connected_communities) / len(communities),
                    "importance": importance,
                    "potential": self._assess_bridge_potential(concept_id, connected_communities, communities, space),
                    "bridge_type": self._classify_bridge_type(concept_id, bridge_connections, space)
                })
        
        # Sort by importance
        bridges.sort(key=lambda b: b["importance"], reverse=True)
        
        return bridges
    
    def _detect_communities(self, space, resolution: float = 1.0, 
                           random_state: int = None) -> List[Dict[str, Any]]:
        """
        Detect communities using the Louvain algorithm with NetworkX.
        
        Args:
            space: ConceptSpace to analyze
            resolution: Resolution parameter for community detection (higher = smaller communities)
            random_state: Random seed for reproducibility
            
        Returns:
            List of community dictionaries with members and metadata
        """
        if not space.concepts:
            return []
        
        # Build NetworkX graph from concept space
        G = nx.Graph()
        
        # Add nodes
        for concept_id, concept in space.concepts.items():
            G.add_node(concept_id, **concept)
        
        # Add edges with weights
        for relation in space.relations:
            source = relation.get("source")
            target = relation.get("target")
            strength = relation.get("strength", 1.0)
            
            if source in G and target in G:
                G.add_edge(source, target, weight=strength)
        
        # Apply Louvain algorithm
        try:
            import community as community_louvain
            
            # Detect communities
            partition = community_louvain.best_partition(
                G, 
                weight='weight',
                resolution=resolution,
                random_state=random_state
            )
            
            # Calculate modularity
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
        except ImportError:
            # Fallback to NetworkX's built-in method
            from networkx.algorithms import community as nx_comm
            
            communities_generator = nx_comm.louvain_communities(
                G, 
                weight='weight', 
                resolution=resolution,
                seed=random_state
            )
            
            # Convert to partition format
            partition = {}
            for i, community in enumerate(communities_generator):
                for node in community:
                    partition[node] = i
            
            # Calculate modularity
            modularity = nx_comm.modularity(
                G, 
                [{n for n, c in partition.items() if c == i} 
                 for i in set(partition.values())],
                weight='weight'
            )
        
        # Organize communities
        communities_dict = defaultdict(list)
        for node, comm_id in partition.items():
            communities_dict[comm_id].append(node)
        
        # Build community objects with metadata
        communities = []
        for comm_id, members in communities_dict.items():
            if not members:
                continue
                
            # Calculate community properties
            subgraph = G.subgraph(members)
            
            # Internal density
            internal_edges = subgraph.number_of_edges()
            possible_edges = len(members) * (len(members) - 1) / 2
            density = internal_edges / possible_edges if possible_edges > 0 else 0
            
            # Find central nodes
            try:
                centrality = nx.degree_centrality(subgraph)
                central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            except:
                central_nodes = [(members[0], 1.0)] if members else []
            
            # Determine community theme
            theme = self._determine_community_theme(members, space)
            
            # Calculate cohesion
            cohesion = self._calculate_community_cohesion(members, space, G)
            
            community = {
                "id": f"community_{comm_id}",
                "members": members,
                "size": len(members),
                "density": density,
                "cohesion": cohesion,
                "central_nodes": [{"id": node, "centrality": cent} for node, cent in central_nodes],
                "theme": theme,
                "name": f"Community {comm_id}: {theme}",
                "metadata": {
                    "internal_edges": internal_edges,
                    "avg_degree": 2 * internal_edges / len(members) if members else 0,
                    "created_at": datetime.now().isoformat()
                }
            }
            
            communities.append(community)
        
        # Sort by size and cohesion
        communities.sort(key=lambda c: (c["size"], c["cohesion"]), reverse=True)
        
        # Add quality metrics
        for community in communities:
            community["quality_score"] = self._calculate_community_quality(
                community, G, modularity
            )
        
        return communities
    
    def _calculate_community_cohesion(self, members: List[str], space, graph: nx.Graph) -> float:
        """Calculate semantic and structural cohesion of a community"""
        if len(members) < 2:
            return 1.0
        
        cohesion_scores = []
        
        # 1. Structural cohesion (clustering coefficient)
        try:
            subgraph = graph.subgraph(members)
            clustering = nx.average_clustering(subgraph, weight='weight')
            cohesion_scores.append(clustering)
        except:
            cohesion_scores.append(0.5)
        
        # 2. Semantic cohesion (concept similarity)
        semantic_similarities = []
        for i, member1 in enumerate(members):
            for member2 in members[i+1:]:
                concept1 = space.concepts.get(member1, {})
                concept2 = space.concepts.get(member2, {})
                
                # Calculate concept name similarity
                if concept1.get("name") and concept2.get("name"):
                    similarity = self._calculate_semantic_similarity(
                        concept1["name"], 
                        concept2["name"],
                        use_embeddings=True,
                        use_structure=False  # Faster for many comparisons
                    )
                    semantic_similarities.append(similarity)
        
        if semantic_similarities:
            avg_semantic_similarity = sum(semantic_similarities) / len(semantic_similarities)
            cohesion_scores.append(avg_semantic_similarity)
        
        # 3. Property cohesion
        property_cohesion = self._calculate_property_cohesion(members, space)
        cohesion_scores.append(property_cohesion)
        
        # Weighted average
        weights = [0.4, 0.4, 0.2]  # Structural, semantic, property
        weighted_cohesion = sum(score * weight for score, weight in 
                              zip(cohesion_scores, weights[:len(cohesion_scores)]))
        
        return min(1.0, max(0.0, weighted_cohesion))
    
    def _calculate_property_cohesion(self, members: List[str], space) -> float:
        """Calculate cohesion based on shared properties"""
        if len(members) < 2:
            return 1.0
        
        # Collect all properties
        property_sets = []
        for member in members:
            concept = space.concepts.get(member, {})
            props = set(concept.get("properties", {}).keys())
            if props:
                property_sets.append(props)
        
        if len(property_sets) < 2:
            return 0.5
        
        # Calculate average Jaccard similarity
        similarities = []
        for i in range(len(property_sets)):
            for j in range(i + 1, len(property_sets)):
                intersection = len(property_sets[i].intersection(property_sets[j]))
                union = len(property_sets[i].union(property_sets[j]))
                if union > 0:
                    similarities.append(intersection / union)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _determine_community_theme(self, members: List[str], space) -> str:
        """Determine the theme of a community using NLP and statistical analysis"""
        # Collect text data from community
        texts = []
        all_properties = defaultdict(int)
        
        for member in members:
            concept = space.concepts.get(member, {})
            
            # Add concept name
            if concept.get("name"):
                texts.append(concept["name"])
            
            # Count properties
            for prop_name in concept.get("properties", {}):
                all_properties[prop_name] += 1
        
        # Find most common words across concept names
        if texts:
            # Tokenize and count
            word_freq = defaultdict(int)
            
            for text in texts:
                tokens = text.lower().split()
                for token in tokens:
                    if len(token) > 3:  # Skip short words
                        word_freq[token] += 1
            
            # Find most frequent meaningful word
            if word_freq:
                # Filter out common words
                common_words = {"this", "that", "with", "from", "have", "been"}
                filtered_words = {w: f for w, f in word_freq.items() 
                                if w not in common_words}
                
                if filtered_words:
                    theme_word = max(filtered_words.items(), key=lambda x: x[1])[0]
                    
                    # Check if it's a domain term
                    if theme_word in ["system", "process", "concept", "model"]:
                        # Look for second most common
                        sorted_words = sorted(filtered_words.items(), 
                                            key=lambda x: x[1], reverse=True)
                        if len(sorted_words) > 1:
                            theme_word = sorted_words[1][0]
                    
                    return theme_word.capitalize()
        
        # Fallback to most common property
        if all_properties:
            most_common_prop = max(all_properties.items(), key=lambda x: x[1])[0]
            return f"{most_common_prop}_focused"
        
        return "General"
    
    def _calculate_community_quality(self, community: Dict[str, Any], 
                                   graph: nx.Graph, global_modularity: float) -> float:
        """Calculate overall quality score for a community"""
        scores = []
        weights = []
        
        # 1. Size score (normalized by total nodes)
        size_ratio = community["size"] / graph.number_of_nodes()
        size_score = 1 - abs(0.15 - size_ratio) / 0.15  # Optimal around 15% of nodes
        scores.append(max(0, size_score))
        weights.append(0.2)
        
        # 2. Density score
        scores.append(community["density"])
        weights.append(0.25)
        
        # 3. Cohesion score
        scores.append(community["cohesion"])
        weights.append(0.3)
        
        # 4. Contribution to modularity
        modularity_contribution = global_modularity * (community["size"] / graph.number_of_nodes())
        scores.append(min(1.0, modularity_contribution * 2))  # Scale up
        weights.append(0.25)
        
        # Calculate weighted average
        quality = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, quality))
    
    def _calculate_bridge_importance(self, bridge_id: str, connected_communities: set,
                                   communities: List[Dict], space) -> float:
        """Calculate importance of a bridge concept"""
        importance = 0.0
        
        # Betweenness-like measure
        total_shortest_paths = 0
        paths_through_bridge = 0
        
        # Sample pairs from different communities
        for comm1_id in connected_communities:
            for comm2_id in connected_communities:
                if comm1_id >= comm2_id:
                    continue
                
                # Sample nodes from each community
                comm1_sample = list(communities[comm1_id]["members"])[:5]
                comm2_sample = list(communities[comm2_id]["members"])[:5]
                
                for node1 in comm1_sample:
                    for node2 in comm2_sample:
                        paths = self._find_all_short_paths(node1, node2, space, max_length=5)
                        total_shortest_paths += len(paths)
                        
                        # Count paths through bridge
                        for path in paths:
                            if bridge_id in path:
                                paths_through_bridge += 1
        
        if total_shortest_paths > 0:
            importance = paths_through_bridge / total_shortest_paths
        
        # Adjust by community sizes
        total_size = sum(len(communities[i]["members"]) for i in connected_communities)
        importance *= np.log(total_size + 1) / 10
        
        return min(1.0, importance)
    
    def _identify_gradient_endpoints(self, space) -> List[Tuple[str, str]]:
        """Identify potential gradient endpoints"""
        endpoints = []
        
        for concept_id, concept in space.concepts.items():
            properties = concept.get("properties", {})
            
            # Look for extreme values
            endpoint_type = None
            
            # Check for superlatives
            name_lower = concept["name"].lower()
            if any(word in name_lower for word in ["most", "least", "highest", "lowest", "maximum", "minimum"]):
                endpoint_type = "superlative"
            
            # Check for polar properties
            elif any(prop in properties for prop in ["polarity", "extreme", "boundary"]):
                endpoint_type = "polar"
            
            # Check for scale endpoints
            elif any(str(val) in ["0", "1", "100", "infinite", "none", "all"] 
                    for val in properties.values()):
                endpoint_type = "scale_endpoint"
            
            if endpoint_type:
                endpoints.append((concept_id, endpoint_type))
        
        return endpoints
    
    def _find_gradient_path(self, start_id: str, end_id: str, space) -> List[str]:
        """Find smooth gradient path between concepts"""
        # A* search with gradient heuristic
        from heapq import heappush, heappop
        
        # Get properties of endpoints for gradient guidance
        start_props = space.concepts[start_id].get("properties", {})
        end_props = space.concepts[end_id].get("properties", {})
        
        # Priority queue: (f_score, path)
        queue = [(0, [start_id])]
        visited = {start_id}
        best_paths = {}
        
        while queue:
            f_score, path = heappop(queue)
            current = path[-1]
            
            if current == end_id:
                return path
            
            neighbors = self._get_concept_neighbors(current, space)
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # Calculate gradient score
                    g_score = len(path)  # Path length
                    h_score = self._gradient_heuristic(neighbor, end_id, space)
                    
                    # Check if this maintains gradient property
                    if len(path) >= 2:
                        gradient_quality = self._check_local_gradient_quality(
                            path[-2], path[-1], neighbor, space
                        )
                        if gradient_quality < 0.3:
                            continue  # Skip if it breaks gradient
                    
                    new_path = path + [neighbor]
                    f = g_score + h_score
                    
                    heappush(queue, (f, new_path))
        
        return []  # No path found
    
    def _gradient_heuristic(self, current: str, target: str, space) -> float:
        """Heuristic for gradient path finding"""
        current_concept = space.concepts[current]
        target_concept = space.concepts[target]
        
        # Semantic distance
        semantic_dist = 1.0 - self._calculate_enhanced_concept_similarity(
            current_concept, target_concept, space
        )
        
        # Property distance
        current_props = set(current_concept.get("properties", {}).keys())
        target_props = set(target_concept.get("properties", {}).keys())
        
        if current_props and target_props:
            prop_overlap = len(current_props.intersection(target_props))
            prop_union = len(current_props.union(target_props))
            prop_dist = 1.0 - (prop_overlap / prop_union if prop_union > 0 else 0)
        else:
            prop_dist = 1.0
        
        return (semantic_dist + prop_dist) / 2
    
    def _assess_bridge_potential(self, concept_id: str, connected_clusters: set, 
                                clusters: List[Dict], space) -> float:
        """Assess the potential of a bridge concept for creating new connections"""
        potential = len(connected_clusters) / len(clusters)  # Base potential
        
        # Check if bridge has unique properties
        concept = space.concepts[concept_id]
        unique_props = len(concept.get("properties", {}))
        potential += min(unique_props / 10.0, 0.3)
        
        return min(1.0, potential)
    
    def _find_conceptual_gradients(self, space) -> List[Dict[str, Any]]:
        """Enhanced conceptual gradient detection"""
        gradients = []
        analyzed_pairs = set()
        
        # Find concepts that could be endpoints
        endpoint_candidates = self._identify_gradient_endpoints(space)
        
        for start_id, start_type in endpoint_candidates:
            for end_id, end_type in endpoint_candidates:
                if start_id == end_id or (start_id, end_id) in analyzed_pairs:
                    continue
                
                analyzed_pairs.add((start_id, end_id))
                analyzed_pairs.add((end_id, start_id))
                
                # Only look for gradients between different types
                if start_type != end_type:
                    # Find gradient path
                    gradient_path = self._find_gradient_path(start_id, end_id, space)
                    
                    if gradient_path and len(gradient_path) >= 3:
                        # Analyze gradient quality
                        quality = self._analyze_gradient_quality(gradient_path, space)
                        
                        if quality["smoothness"] > 0.6:
                            dimension = self._identify_gradient_dimension_advanced(
                                gradient_path, space
                            )
                            
                            gradients.append({
                                "dimension": dimension,
                                "start": space.concepts[start_id]["name"],
                                "end": space.concepts[end_id]["name"],
                                "path": [space.concepts[cid]["name"] for cid in gradient_path],
                                "smoothness": quality["smoothness"],
                                "monotonicity": quality["monotonicity"],
                                "length": len(gradient_path),
                                "gradient_type": self._classify_gradient_type(quality)
                            })
        
        return gradients
    
    def _find_conceptual_path(self, start_id: str, end_id: str, space) -> List[str]:
        """Find a path between two concepts"""
        # Simple BFS pathfinding
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == end_id:
                return path
            
            # Find neighbors
            neighbors = set()
            for relation in space.relations:
                if relation["source"] == current_id:
                    neighbors.add(relation["target"])
                elif relation["target"] == current_id:
                    neighbors.add(relation["source"])
            
            for neighbor in neighbors - visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _assess_path_smoothness(self, path: List[str], space) -> float:
        """Assess how smooth the transitions are along a path"""
        if len(path) < 2:
            return 0.0
        
        total_similarity = 0.0
        
        for i in range(len(path) - 1):
            concept1 = space.concepts[path[i]]
            concept2 = space.concepts[path[i + 1]]
            
            similarity = self._calculate_concept_similarity_simple(concept1, concept2)
            total_similarity += similarity
        
        # Average similarity between adjacent concepts
        smoothness = total_similarity / (len(path) - 1)
        
        return smoothness
    
    def _identify_gradient_dimension(self, path: List[str], space) -> str:
        """Identify what dimension changes along a gradient"""
        # Look for systematic property changes
        changing_properties = {}
        
        for i in range(len(path) - 1):
            concept1 = space.concepts[path[i]]
            concept2 = space.concepts[path[i + 1]]
            
            props1 = set(concept1.get("properties", {}).keys())
            props2 = set(concept2.get("properties", {}).keys())
            
            # Track which properties change
            changed = props1.symmetric_difference(props2)
            for prop in changed:
                changing_properties[prop] = changing_properties.get(prop, 0) + 1
        
        if changing_properties:
            # Most frequently changing property
            dimension_prop = max(changing_properties.items(), key=lambda x: x[1])[0]
            return f"{dimension_prop}_dimension"
        
        return "abstract_dimension"
    
    def _pattern_relevant_to_context(self, pattern: Dict[str, Any], context: SharedContext) -> bool:
        """Check if a conceptual pattern is relevant to context"""
        pattern_type = pattern.get("type", "")
        
        # Check based on pattern type
        if pattern_type == "hierarchy":
            # Hierarchies relevant for understanding relationships
            if "relationship" in context.user_input.lower() or "structure" in context.user_input.lower():
                return True
        
        elif pattern_type == "cluster":
            # Clusters relevant for grouping/categorization
            if "group" in context.user_input.lower() or "category" in context.user_input.lower():
                return True
        
        elif pattern_type == "bridge":
            # Bridges relevant for integration/connection
            if "connect" in context.user_input.lower() or "integrate" in context.user_input.lower():
                return True
        
        elif pattern_type == "gradient":
            # Gradients relevant for transitions/progressions
            if "progress" in context.user_input.lower() or "transition" in context.user_input.lower():
                return True
        
        # General relevance check
        pattern_desc = str(pattern).lower()
        input_words = set(context.user_input.lower().split())
        
        return any(word in pattern_desc for word in input_words)
    
    # ========================================================================================
    # BLEND OPPORTUNITY ASSESSMENT
    # ========================================================================================
    
    async def _assess_blend_opportunity(self, space1_id: str, space2_id: str,
                                      context: SharedContext) -> Dict[str, Any]:
        """Assess the opportunity for blending two conceptual spaces"""
        space1 = self.original_core.concept_spaces.get(space1_id)
        space2 = self.original_core.concept_spaces.get(space2_id)
        
        if not space1 or not space2:
            return {"score": 0.0}
        
        opportunity = {
            "space1": space1.name,
            "space2": space2.name,
            "score": 0.0,
            "blend_type": "none",
            "potential_insights": [],
            "mapping_quality": 0.0,
            "context_alignment": 0.0
        }
        
        # Find potential mappings
        mappings = await self._find_contextual_mappings(space1_id, space2_id, context)
        
        if not mappings:
            return opportunity
        
        # Assess mapping quality
        mapping_quality = sum(m["similarity"] for m in mappings) / len(mappings)
        opportunity["mapping_quality"] = mapping_quality
        
        # Base score on mapping quality
        opportunity["score"] = mapping_quality * 0.4
        
        # Check for complementary structures
        space1_patterns = await self._identify_conceptual_patterns(space1, context)
        space2_patterns = await self._identify_conceptual_patterns(space2, context)
        
        complementarity = self._assess_pattern_complementarity(space1_patterns, space2_patterns)
        opportunity["score"] += complementarity * 0.3
        
        # Context alignment
        context_alignment = self._assess_blend_context_alignment(space1, space2, context)
        opportunity["context_alignment"] = context_alignment
        opportunity["score"] += context_alignment * 0.3
        
        # Determine best blend type
        if mapping_quality > 0.7:
            opportunity["blend_type"] = "fusion"  # Deep integration
        elif complementarity > 0.6:
            opportunity["blend_type"] = "completion"  # Fill gaps
        elif context.emotional_state and context.emotional_state.get("dominant_emotion", [""])[0] == "Curiosity":
            opportunity["blend_type"] = "elaboration"  # Explore possibilities
        else:
            opportunity["blend_type"] = "composition"  # Standard blend
        
        # Identify potential insights
        opportunity["potential_insights"] = self._identify_blend_insights(
            space1, space2, mappings, opportunity["blend_type"]
        )
        
        return opportunity
    
    def _assess_pattern_complementarity(self, patterns1: List[Dict], patterns2: List[Dict]) -> float:
        """Assess how well two sets of patterns complement each other"""
        complementarity = 0.0
        
        # Get pattern type distributions
        types1 = {p["type"] for p in patterns1}
        types2 = {p["type"] for p in patterns2}
        
        # Different pattern types suggest complementarity
        unique_to_1 = types1 - types2
        unique_to_2 = types2 - types1
        
        if unique_to_1 or unique_to_2:
            complementarity += 0.4
        
        # Check for hierarchies + clusters (good combination)
        if "hierarchy" in types1 and "cluster" in types2:
            complementarity += 0.3
        elif "cluster" in types1 and "hierarchy" in types2:
            complementarity += 0.3
        
        # Check for bridges in one and not the other (integration opportunity)
        if "bridge" in types1 and "bridge" not in types2:
            complementarity += 0.2
        
        return min(1.0, complementarity)
    
    def _assess_blend_context_alignment(self, space1, space2, context: SharedContext) -> float:
        """Assess how well a potential blend aligns with context"""
        alignment = 0.0
        
        # Check if both spaces relate to user query
        input_keywords = set(context.user_input.lower().split())
        
        space1_relevance = self._calculate_space_relevance(space1, input_keywords)
        space2_relevance = self._calculate_space_relevance(space2, input_keywords)
        
        # Both spaces should be somewhat relevant
        min_relevance = min(space1_relevance, space2_relevance)
        alignment += min_relevance * 0.5
        
        # Check goal alignment
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            
            for goal in goals:
                if "creative" in goal.get("description", "").lower():
                    alignment += 0.2  # Creativity goals favor blending
                elif "integrate" in goal.get("description", "").lower():
                    alignment += 0.3  # Integration goals strongly favor blending
        
        return min(1.0, alignment)
    
    def _calculate_space_relevance(self, space, keywords: set) -> float:
        """Calculate relevance of a conceptual space to keywords"""
        relevance = 0.0
        
        # Check space name and domain
        if any(kw in space.name.lower() for kw in keywords):
            relevance += 0.3
        
        if space.domain and any(kw in space.domain.lower() for kw in keywords):
            relevance += 0.2
        
        # Check concepts in space
        matching_concepts = 0
        for concept in space.concepts.values():
            if any(kw in concept["name"].lower() for kw in keywords):
                matching_concepts += 1
        
        relevance += min(matching_concepts / 10.0, 0.5)
        
        return min(1.0, relevance)
    
    def _identify_blend_insights(self, space1, space2, mappings: List[Dict], 
                               blend_type: str) -> List[str]:
        """Identify potential insights from blending"""
        insights = []
        
        if blend_type == "fusion":
            insights.append("Deep structural alignment could reveal hidden isomorphisms")
            insights.append("Unified framework might emerge from merged concepts")
        
        elif blend_type == "completion":
            insights.append("Gap-filling could reveal missing conceptual links")
            insights.append("Complementary structures might form complete picture")
        
        elif blend_type == "elaboration":
            insights.append("Novel concept combinations could generate creative solutions")
            insights.append("Exploratory blending might reveal unexpected connections")
        
        else:  # composition
            insights.append("Standard blending could create useful hybrid concepts")
            insights.append("Cross-domain mapping might enable knowledge transfer")
        
        # Add specific insights based on mappings
        if len(mappings) > 5:
            insights.append(f"Rich mapping structure ({len(mappings)} connections) suggests high integration potential")
        
        return insights
    
    # ========================================================================================
    # COUNTERFACTUAL AND INTERVENTION METHODS
    # ========================================================================================
    
    async def _generate_counterfactual_scenario(self, model_id: str, 
                                              context: SharedContext) -> Optional[Dict[str, Any]]:
        """Generate a counterfactual scenario based on context"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        # Parse counterfactual query
        query_lower = context.user_input.lower()
        
        # Extract the counterfactual condition
        condition = None
        if "what if" in query_lower:
            condition = query_lower.split("what if")[1].strip()
        elif "would have" in query_lower:
            parts = query_lower.split("would have")
            if len(parts) > 1:
                condition = parts[0].strip()
        
        if not condition:
            return None
        
        # Find relevant nodes for the condition
        relevant_nodes = []
        condition_words = set(condition.split())
        
        for node_id, node in model.nodes.items():
            node_words = set(node.name.lower().split())
            if node_words.intersection(condition_words):
                relevant_nodes.append(node_id)
        
        if not relevant_nodes:
            return None
        
        # Generate scenario
        scenario = {
            "condition": condition,
            "model": model.name,
            "intervention_nodes": relevant_nodes[:2],  # Limit to 2 nodes
            "original_state": {},
            "counterfactual_state": {},
            "changes": [],
            "confidence": 0.7
        }
        
        # For each intervention node, trace effects
        for node_id in scenario["intervention_nodes"]:
            node = model.nodes[node_id]
            
            # Record original state
            scenario["original_state"][node.name] = {
                "type": getattr(node, "node_type", "standard"),
                "properties": getattr(node, "properties", {})
            }
            
            # Create counterfactual state
            scenario["counterfactual_state"][node.name] = {
                "type": "intervened",
                "properties": {"counterfactual": True}
            }
            
            # Trace downstream effects
            effects = self._trace_intervention_effects(node_id, model)
            scenario["changes"].extend(effects)
        
        return scenario
    
    def _trace_intervention_effects(self, node_id: str, model) -> List[Dict[str, Any]]:
        """Trace the effects of intervening on a node"""
        effects = []
        
        # Get immediate effects
        immediate_effects = model.get_descendants(node_id, max_depth=1)
        
        for effect_id in immediate_effects:
            effect_node = model.nodes.get(effect_id)
            if effect_node:
                # Find the relation strength
                relation_strength = 0.5  # Default
                for relation in model.relations:
                    if relation.source == node_id and relation.target == effect_id:
                        relation_strength = relation.strength
                        break
                
                effects.append({
                    "affected_node": effect_node.name,
                    "effect_type": "direct",
                    "strength": relation_strength,
                    "mechanism": "causal_influence"
                })
        
        # Get secondary effects
        secondary_effects = model.get_descendants(node_id, max_depth=2)
        secondary_effects = set(secondary_effects) - set(immediate_effects) - {node_id}
        
        for effect_id in list(secondary_effects)[:3]:  # Limit secondary effects
            effect_node = model.nodes.get(effect_id)
            if effect_node:
                effects.append({
                    "affected_node": effect_node.name,
                    "effect_type": "indirect",
                    "strength": 0.3,  # Weaker for indirect
                    "mechanism": "propagated_influence"
                })
        
        return effects
    
    async def _analyze_alternative_paths(self, model_id: str, 
                                       scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze alternative causal paths given a counterfactual scenario"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return []
        
        alternative_paths = []
        
        # For each intervention node, find alternative paths to key outcomes
        for node_id in scenario["intervention_nodes"]:
            # Find important outcome nodes (high centrality or user-mentioned)
            outcome_nodes = self._identify_outcome_nodes(model, scenario)
            
            for outcome_id in outcome_nodes:
                # Find paths that don't go through intervention node
                paths = self._find_alternative_causal_paths(
                    model, node_id, outcome_id
                )
                
                for path in paths:
                    alternative_paths.append({
                        "blocked_node": model.nodes[node_id].name,
                        "outcome": model.nodes[outcome_id].name,
                        "alternative_route": [model.nodes[n].name for n in path],
                        "path_strength": self._calculate_path_strength(path, model),
                        "viability": self._assess_path_viability(path, model)
                    })
        
        # Sort by viability
        alternative_paths.sort(key=lambda p: p["viability"], reverse=True)
        
        return alternative_paths[:5]  # Top 5 alternatives
    
    def _identify_outcome_nodes(self, model, scenario: Dict[str, Any]) -> List[str]:
        """Identify important outcome nodes in the model"""
        outcome_nodes = []
        
        # High out-degree nodes (consequences)
        for node_id, node in model.nodes.items():
            out_degree = sum(1 for r in model.relations if r.source == node_id)
            if out_degree == 0:  # Leaf nodes
                outcome_nodes.append(node_id)
        
        # Limit to top 3 by centrality
        if len(outcome_nodes) > 3:
            # Simple centrality: nodes with most incoming paths
            centrality = {}
            for node_id in outcome_nodes:
                ancestors = model.get_ancestors(node_id)
                centrality[node_id] = len(ancestors)
            
            sorted_outcomes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            outcome_nodes = [node_id for node_id, _ in sorted_outcomes[:3]]
        
        return outcome_nodes
    
    def _find_alternative_causal_paths(self, model, blocked_node: str, 
                                     target_node: str) -> List[List[str]]:
        """Find causal paths that don't go through blocked node"""
        alternative_paths = []
        
        # Modified BFS that avoids blocked node
        queue = [(target_node, [target_node])]
        visited = {target_node}
        
        while queue and len(alternative_paths) < 3:  # Find up to 3 paths
            current, path = queue.pop(0)
            
            # Find predecessors (going backwards from target)
            predecessors = []
            for relation in model.relations:
                if relation.target == current and relation.source != blocked_node:
                    predecessors.append(relation.source)
            
            for pred in predecessors:
                if pred in visited:
                    continue
                    
                visited.add(pred)
                new_path = [pred] + path
                
                # Check if we've found a complete path (no more predecessors)
                pred_predecessors = [r.source for r in model.relations if r.target == pred]
                if not pred_predecessors:
                    alternative_paths.append(new_path)
                else:
                    queue.append((pred, new_path))
        
        return alternative_paths
    
    def _calculate_path_strength(self, path: List[str], model) -> float:
        """Calculate the strength of a causal path"""
        if len(path) < 2:
            return 0.0
        
        total_strength = 1.0
        
        for i in range(len(path) - 1):
            # Find relation between consecutive nodes
            source = path[i]
            target = path[i + 1]
            
            relation_strength = 0.5  # Default
            for relation in model.relations:
                if relation.source == source and relation.target == target:
                    relation_strength = relation.strength
                    break
            
            # Multiply strengths (assuming independence)
            total_strength *= relation_strength
        
        return total_strength
    
    def _assess_path_viability(self, path: List[str], model) -> float:
        """Assess how viable an alternative path is"""
        # Base viability on path strength
        viability = self._calculate_path_strength(path, model)
        
        # Penalize very long paths
        if len(path) > 4:
            viability *= 0.7
        elif len(path) > 6:
            viability *= 0.5
        
        # Bonus for paths through controllable nodes
        for node_id in path[1:-1]:  # Intermediate nodes
            node = model.nodes[node_id]
            if any(term in node.name.lower() for term in ["policy", "decision", "action"]):
                viability *= 1.2
        
        return min(1.0, viability)
    
    # ========================================================================================
    # GOAL AND INTERVENTION METHODS
    # ========================================================================================
    
    def _model_relates_to_goal(self, model, goal: Dict[str, Any]) -> bool:
        """Check if a causal model relates to a goal"""
        goal_desc = goal.get("description", "").lower()
        goal_keywords = set(goal_desc.split())
        
        # Check model name and domain
        model_text = f"{model.name} {model.domain}".lower()
        if any(kw in model_text for kw in goal_keywords):
            return True
        
        # Check node names
        for node in model.nodes.values():
            if any(kw in node.name.lower() for kw in goal_keywords):
                return True
        
        # Check associated need
        associated_need = goal.get("associated_need", "").lower()
        if associated_need and associated_need in model_text:
            return True
        
        return False
    
    async def _generate_intervention_suggestion(self, model_id: str, goal: Dict[str, Any],
                                             context: SharedContext) -> Optional[Dict[str, Any]]:
        """Generate an intervention suggestion to achieve a goal"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        # Find nodes related to goal
        goal_nodes = []
        goal_keywords = set(goal.get("description", "").lower().split())
        
        for node_id, node in model.nodes.items():
            node_words = set(node.name.lower().split())
            if node_words.intersection(goal_keywords):
                goal_nodes.append(node_id)
        
        if not goal_nodes:
            return None
        
        # Find best intervention point for goal
        best_intervention = None
        best_score = 0.0
        
        for goal_node in goal_nodes:
            # Find nodes that influence this goal node
            influencers = model.get_ancestors(goal_node, max_depth=2)
            
            for influencer_id in influencers:
                influencer = model.nodes.get(influencer_id)
                if not influencer:
                    continue
                
                # Score this intervention point
                score = 0.0
                
                # Controllability
                if any(term in influencer.name.lower() for term in ["action", "decision", "behavior"]):
                    score += 0.4
                
                # Direct influence
                for relation in model.relations:
                    if relation.source == influencer_id and relation.target == goal_node:
                        score += relation.strength * 0.3
                        break
                
                # Feasibility
                feasibility = self._assess_intervention_feasibility(influencer, context)
                score += feasibility * 0.3
                
                if score > best_score:
                    best_score = score
                    best_intervention = {
                        "intervention_node": influencer.name,
                        "goal_node": model.nodes[goal_node].name,
                        "intervention_type": self._determine_intervention_type(influencer, model),
                        "expected_impact": self._estimate_intervention_impact(influencer_id, model),
                        "confidence": best_score,
                        "implementation_suggestions": self._generate_implementation_suggestions(
                            influencer, goal, context
                        )
                    }
        
        return best_intervention
    
    def _generate_implementation_suggestions(self, intervention_node, goal: Dict[str, Any],
                                           context: SharedContext) -> List[str]:
        """Generate specific suggestions for implementing an intervention"""
        suggestions = []
        node_name_lower = intervention_node.name.lower()
        
        # Generate suggestions based on intervention type
        if "behavior" in node_name_lower:
            suggestions.append("Establish clear behavioral targets and tracking metrics")
            suggestions.append("Use positive reinforcement to encourage desired behaviors")
        
        elif "decision" in node_name_lower:
            suggestions.append("Create decision frameworks or criteria")
            suggestions.append("Implement systematic decision review processes")
        
        elif "process" in node_name_lower:
            suggestions.append("Map current process and identify improvement points")
            suggestions.append("Implement incremental process changes with monitoring")
        
        elif "policy" in node_name_lower:
            suggestions.append("Draft clear policy guidelines with specific objectives")
            suggestions.append("Ensure stakeholder buy-in before implementation")
        
        # Add goal-specific suggestions
        if "understand" in goal.get("description", "").lower():
            suggestions.append("Document learnings and insights systematically")
        elif "improve" in goal.get("description", "").lower():
            suggestions.append("Set measurable improvement targets")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    # ========================================================================================
    # INTEGRATION AND CREATIVE METHODS
    # ========================================================================================
    
    async def _assess_integration_potential(self, model_id: str, space_id: str,
                                          context: SharedContext) -> float:
        """Assess potential for integrating causal model with conceptual space"""
        model = self.original_core.causal_models.get(model_id)
        space = self.original_core.concept_spaces.get(space_id)
        
        if not model or not space:
            return 0.0
        
        potential = 0.0
        
        # Domain compatibility
        if model.domain and space.domain:
            if model.domain.lower() == space.domain.lower():
                potential += 0.3
            elif any(word in space.domain.lower() for word in model.domain.lower().split()):
                potential += 0.15
        
        # Concept-node overlap
        overlap_count = 0
        for node in model.nodes.values():
            for concept in space.concepts.values():
                if self._concepts_match(node.name, concept["name"]):
                    overlap_count += 1
        
        potential += min(overlap_count * 0.1, 0.4)
        
        # Structural compatibility
        if len(model.nodes) > 5 and len(space.concepts) > 5:
            potential += 0.1  # Both have rich structure
        
        # Context alignment
        if context.goal_context:
            goals = context.goal_context.get("active_goals", [])
            if any("integrate" in g.get("description", "").lower() for g in goals):
                potential += 0.2
        
        return min(1.0, potential)
    
    def _concepts_match(self, name1: str, name2: str) -> bool:
        """Check if two concept names match"""
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Exact match
        if name1_lower == name2_lower:
            return True
        
        # Word overlap
        words1 = set(name1_lower.split())
        words2 = set(name2_lower.split())
        
        overlap = len(words1.intersection(words2))
        if overlap >= min(len(words1), len(words2)) * 0.5:
            return True
        
        return False
    
    async def _suggest_creative_intervention(self, model_id: str, 
                                           context: SharedContext) -> Optional[Dict[str, Any]]:
        """Suggest a creative intervention based on causal understanding"""
        model = self.original_core.causal_models.get(model_id)
        if not model:
            return None
        
        # Find leverage points (nodes with high impact)
        leverage_points = []
        
        for node_id, node in model.nodes.items():
            descendants = model.get_descendants(node_id)
            if len(descendants) >= 3:  # Influences multiple outcomes
                leverage_points.append({
                    "node_id": node_id,
                    "node": node,
                    "impact_breadth": len(descendants)
                })
        
        if not leverage_points:
            return None
        
        # Select best leverage point
        best_point = max(leverage_points, key=lambda p: p["impact_breadth"])
        
        # Generate creative intervention
        intervention = {
            "target_node": best_point["node"].name,
            "intervention_class": "creative_leverage",
            "approach": self._generate_creative_approach(best_point["node"], context),
            "expected_outcomes": [
                model.nodes[desc].name 
                for desc in model.get_descendants(best_point["node_id"])[:3]
                if desc in model.nodes
            ],
            "creativity_factors": self._identify_creativity_factors(best_point["node"], model, context),
            "implementation_phases": self._design_intervention_phases(best_point["node"], context)
        }
        
        return intervention
    
    def _generate_creative_approach(self, node, context: SharedContext) -> str:
        """Generate a creative approach for intervention"""
        node_name_lower = node.name.lower()
        
        # Context-aware creative strategies
        if context.emotional_state:
            emotion = context.emotional_state.get("dominant_emotion", [""])[0]
            
            if emotion == "Curiosity":
                return f"Experimental exploration of {node.name} through systematic variation"
            elif emotion == "Frustration":
                return f"Radical reimagining of {node.name} to bypass current constraints"
        
        # Default creative strategies based on node type
        if "process" in node_name_lower:
            return "Process gamification to enhance engagement and outcomes"
        elif "behavior" in node_name_lower:
            return "Behavioral nudges using environmental design"
        elif "system" in node_name_lower:
            return "Systems thinking workshop to identify hidden connections"
        else:
            return f"Design thinking approach to reimagine {node.name}"
    
    def _identify_creativity_factors(self, node, model, context: SharedContext) -> List[str]:
        """Identify factors that enhance creative intervention potential"""
        factors = []
        
        # Node has many weak connections (opportunity for strengthening)
        weak_relations = [r for r in model.relations 
                         if (r.source == node.id or r.target == node.id) and r.strength < 0.3]
        if len(weak_relations) > 2:
            factors.append("multiple_weak_connections_to_strengthen")
        
        # Node bridges different domains
        connected_nodes = set()
        for relation in model.relations:
            if relation.source == node.id:
                connected_nodes.add(relation.target)
            elif relation.target == node.id:
                connected_nodes.add(relation.source)
        
        domains = set()
        for node_id in connected_nodes:
            if node_id in model.nodes and hasattr(model.nodes[node_id], 'domain'):
                domains.add(model.nodes[node_id].domain)
        
        if len(domains) > 1:
            factors.append("cross_domain_bridge_potential")
        
        # Context suggests innovation need
        if "new" in context.user_input.lower() or "innovative" in context.user_input.lower():
            factors.append("explicit_innovation_requirement")
        
        return factors
    
    def _design_intervention_phases(self, node, context: SharedContext) -> List[Dict[str, str]]:
        """Design phases for implementing creative intervention"""
        phases = [
            {
                "phase": "Discovery",
                "duration": "1-2 weeks",
                "activities": f"Map current state of {node.name} and identify constraints"
            },
            {
                "phase": "Ideation",
                "duration": "1 week",
                "activities": "Generate diverse intervention ideas using creative techniques"
            },
            {
                "phase": "Prototyping",
                "duration": "2-3 weeks",
                "activities": "Test small-scale versions of most promising interventions"
            },
            {
                "phase": "Implementation",
                "duration": "4-6 weeks",
                "activities": "Roll out refined intervention with continuous monitoring"
            },
            {
                "phase": "Evolution",
                "duration": "Ongoing",
                "activities": "Iterate based on outcomes and emergent opportunities"
            }
        ]
        
        # Adjust based on context
        if context.constraints and "time_sensitive" in context.constraints:
            # Compress timeline
            for phase in phases:
                phase["duration"] = "Accelerated"
        
        return phases
    
    # ========================================================================================
    # ADDITIONAL HELPER METHOD FOR BLENDING
    # ========================================================================================
    
    def _concept_aligns_with_emotion(self, concept: Dict[str, Any], emotion: str) -> bool:
        """Check if a concept aligns with an emotional state"""
        concept_name = concept.get("name", "").lower()
        properties = concept.get("properties", {})
        
        emotion_keywords = {
            "Curiosity": ["unknown", "explore", "discover", "mystery", "question"],
            "Joy": ["positive", "success", "achievement", "happiness", "reward"],
            "Frustration": ["obstacle", "challenge", "difficulty", "problem", "constraint"],
            "Anxiety": ["threat", "risk", "uncertainty", "danger", "concern"],
            "Satisfaction": ["complete", "fulfilled", "achieved", "resolved", "success"]
        }
        
        keywords = emotion_keywords.get(emotion, [])
        
        # Check concept name
        if any(kw in concept_name for kw in keywords):
            return True
        
        # Check properties
        for prop_value in properties.values():
            if isinstance(prop_value, str) and any(kw in prop_value.lower() for kw in keywords):
                return True
        
        return False
    
    def _get_concept_neighbors(self, concept_id: str, space) -> List[str]:
        """Get all concepts connected to a given concept"""
        neighbors = set()
        
        # Check all relations in the space
        for relation in space.relations:
            if relation.get("source") == concept_id:
                neighbors.add(relation.get("target"))
            elif relation.get("target") == concept_id:
                neighbors.add(relation.get("source"))
        
        return list(neighbors)
    
    def _concepts_connected(self, concept1_id: str, concept2_id: str, space) -> bool:
        """Check if two concepts are directly connected"""
        for relation in space.relations:
            if (relation.get("source") == concept1_id and relation.get("target") == concept2_id) or \
               (relation.get("source") == concept2_id and relation.get("target") == concept1_id):
                return True
        return False
    
    def _merge_cluster_results(self, dense_clusters: List[Dict[str, Any]], 
                             semantic_clusters: List[Dict[str, Any]], 
                             property_clusters: List[Dict[str, Any]], 
                             space) -> List[Dict[str, Any]]:
        """Merge results from different clustering strategies"""
        merged_clusters = []
        all_clusters = dense_clusters + semantic_clusters + property_clusters
        
        # Track which concepts have been assigned to merged clusters
        assigned_concepts = set()
        
        # Sort clusters by size and quality
        all_clusters.sort(key=lambda c: c.get("size", 0) * c.get("density", 0.5), reverse=True)
        
        for cluster in all_clusters:
            cluster_members = set(cluster["members"])
            
            # Check overlap with existing merged clusters
            best_merge_candidate = None
            best_overlap = 0
            
            for i, merged in enumerate(merged_clusters):
                merged_members = set(merged["members"])
                overlap = len(cluster_members.intersection(merged_members))
                
                # If significant overlap, consider merging
                if overlap >= min(len(cluster_members), len(merged_members)) * 0.5:
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_merge_candidate = i
            
            if best_merge_candidate is not None:
                # Merge with existing cluster
                merged = merged_clusters[best_merge_candidate]
                merged["members"] = list(set(merged["members"]).union(cluster_members))
                merged["size"] = len(merged["members"])
                
                # Update cluster type to reflect merge
                if merged.get("cluster_type") != cluster.get("cluster_type"):
                    merged["cluster_type"] = "hybrid"
                
                # Recalculate metrics
                merged["density"] = self._calculate_cluster_density(merged["members"], space)
                merged["cohesion"] = self._calculate_cluster_cohesion(merged["members"], space)
                
            else:
                # Create new merged cluster
                new_merged = {
                    "members": cluster["members"],
                    "size": cluster["size"],
                    "density": cluster.get("density", 0),
                    "cohesion": cluster.get("cohesion", 0),
                    "cluster_type": cluster.get("cluster_type", "unknown"),
                    "source_methods": [cluster.get("cluster_type", "unknown")]
                }
                merged_clusters.append(new_merged)
                assigned_concepts.update(cluster["members"])
        
        # Post-process: ensure no overlapping clusters
        final_clusters = []
        processed_concepts = set()
        
        for cluster in merged_clusters:
            # Remove already processed concepts
            unique_members = [m for m in cluster["members"] if m not in processed_concepts]
            
            if len(unique_members) >= 3:  # Minimum cluster size
                cluster["members"] = unique_members
                cluster["size"] = len(unique_members)
                final_clusters.append(cluster)
                processed_concepts.update(unique_members)
        
        return final_clusters
    
    def _analyze_cluster_properties(self, cluster: Dict[str, Any], space) -> Dict[str, Any]:
        """Analyze detailed properties of a cluster"""
        analysis = {
            "central_concepts": [],
            "peripheral_concepts": [],
            "cluster_theme": "",
            "internal_structure": "",
            "boundary_strength": 0.0,
            "homogeneity": 0.0
        }
        
        members = cluster["members"]
        
        # Find central vs peripheral concepts
        centrality_scores = {}
        for member in members:
            # Count internal connections
            internal_connections = sum(1 for other in members 
                                     if other != member and 
                                     self._concepts_connected(member, other, space))
            centrality_scores[member] = internal_connections
        
        # Sort by centrality
        sorted_members = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 20% are central
        central_count = max(1, len(members) // 5)
        analysis["central_concepts"] = [m[0] for m in sorted_members[:central_count]]
        analysis["peripheral_concepts"] = [m[0] for m in sorted_members[-central_count:]]
        
        # Determine cluster theme
        analysis["cluster_theme"] = self._determine_cluster_theme(members, space)
        
        # Analyze internal structure
        avg_connections = sum(centrality_scores.values()) / len(members) if members else 0
        if avg_connections > len(members) * 0.7:
            analysis["internal_structure"] = "highly_connected"
        elif avg_connections > len(members) * 0.4:
            analysis["internal_structure"] = "moderately_connected"
        else:
            analysis["internal_structure"] = "loosely_connected"
        
        # Calculate boundary strength
        internal_edges = sum(centrality_scores.values()) / 2  # Each edge counted twice
        external_edges = 0
        
        for member in members:
            neighbors = self._get_concept_neighbors(member, space)
            external_edges += sum(1 for n in neighbors if n not in members)
        
        if internal_edges + external_edges > 0:
            analysis["boundary_strength"] = internal_edges / (internal_edges + external_edges)
        
        # Calculate homogeneity
        property_overlap_scores = []
        for i, member1 in enumerate(members):
            for member2 in members[i+1:]:
                concept1 = space.concepts.get(member1, {})
                concept2 = space.concepts.get(member2, {})
                
                props1 = set(concept1.get("properties", {}).keys())
                props2 = set(concept2.get("properties", {}).keys())
                
                if props1 or props2:
                    overlap = len(props1.intersection(props2)) / len(props1.union(props2))
                    property_overlap_scores.append(overlap)
        
        if property_overlap_scores:
            analysis["homogeneity"] = sum(property_overlap_scores) / len(property_overlap_scores)
        
        return analysis
    
    def _calculate_cluster_quality(self, cluster: Dict[str, Any], space) -> float:
        """Calculate overall quality score for a cluster"""
        quality = 0.0
        
        # Size factor (larger clusters up to a point)
        size = cluster.get("size", 0)
        if size >= 3 and size <= 10:
            quality += 0.2
        elif size > 10 and size <= 20:
            quality += 0.15
        elif size > 20:
            quality += 0.1
        
        # Density factor
        density = cluster.get("density", 0)
        quality += density * 0.3
        
        # Cohesion factor
        cohesion = cluster.get("cohesion", 0)
        quality += cohesion * 0.3
        
        # Boundary strength (from analysis if available)
        if "analysis" in cluster:
            boundary_strength = cluster["analysis"].get("boundary_strength", 0)
            quality += boundary_strength * 0.2
        
        return min(1.0, quality)
    
    def _calculate_modularity_gain(self, node: str, community: Set[str], 
                                 space, graph: nx.Graph = None) -> float:
        """
        Calculate the modularity gain from adding a node to a community.
        Uses the standard modularity formula: ΔQ = [Σin + ki,in]/2m - [(Σtot + ki)/2m]²
        
        Args:
            node: Node to potentially add
            community: Current community members
            space: Concept space
            graph: Pre-built NetworkX graph (optional, for efficiency)
            
        Returns:
            Modularity gain value
        """
        # Build graph if not provided
        if graph is None:
            graph = nx.Graph()
            for concept_id in space.concepts:
                graph.add_node(concept_id)
            
            for relation in space.relations:
                source = relation.get("source")
                target = relation.get("target")
                weight = relation.get("strength", 1.0)
                if source in graph and target in graph:
                    graph.add_edge(source, target, weight=weight)
        
        # Check if node exists
        if node not in graph:
            return 0.0
        
        # Calculate key values
        m = graph.size(weight='weight')  # Total weight of all edges
        if m == 0:
            return 0.0
        
        # ki: degree of node i
        ki = graph.degree(node, weight='weight')
        
        # ki,in: sum of weights from node to community
        ki_in = sum(graph[node][neighbor].get('weight', 1.0) 
                    for neighbor in community 
                    if neighbor in graph[node])
        
        # Σin: sum of weights inside community
        sigma_in = sum(graph[u][v].get('weight', 1.0)
                       for u in community for v in community
                       if u < v and u in graph and v in graph[u])
        
        # Σtot: sum of degrees of nodes in community
        sigma_tot = sum(graph.degree(n, weight='weight') for n in community)
        
        # Calculate modularity gain
        # ΔQ = [ki,in - ki * Σtot / 2m] / m
        delta_q = (ki_in - ki * sigma_tot / (2 * m)) / m
        
        return delta_q
    
    def _find_all_short_paths(self, start: str, end: str, space, max_length: int = 5) -> List[List[str]]:
        """Find all short paths between two concepts"""
        if start == end:
            return [[start]]
        
        paths = []
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_length:
                continue
            
            neighbors = self._get_concept_neighbors(current, space)
            
            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    
                    if neighbor == end:
                        paths.append(new_path)
                    else:
                        queue.append((neighbor, new_path))
        
        return paths
    
    def _classify_bridge_type(self, bridge_id: str, bridge_connections: Dict[int, List[str]], 
                            space) -> str:
        """Classify the type of bridge concept"""
        bridge_concept = space.concepts.get(bridge_id, {})
        
        # Analyze connection patterns
        connection_counts = [len(connections) for connections in bridge_connections.values()]
        total_connections = sum(connection_counts)
        
        if len(connection_counts) == 2 and all(c == 1 for c in connection_counts):
            return "simple_bridge"  # Connects exactly 2 communities with 1 connection each
        
        elif len(connection_counts) >= 3:
            return "hub_bridge"  # Connects 3+ communities
        
        elif max(connection_counts) > 3:
            return "strong_bridge"  # Many connections to at least one community
        
        elif total_connections > len(connection_counts) * 2:
            return "integrator_bridge"  # Multiple connections per community
        
        else:
            return "weak_bridge"  # Default
    
    def _check_local_gradient_quality(self, prev: str, current: str, next: str, space) -> float:
        """Check if three consecutive concepts maintain gradient property"""
        # Get concepts
        prev_concept = space.concepts.get(prev, {})
        current_concept = space.concepts.get(current, {})
        next_concept = space.concepts.get(next, {})
        
        # Check if properties change monotonically
        monotonic_score = 0.0
        property_changes = 0
        
        # Get common properties
        all_props = set()
        all_props.update(prev_concept.get("properties", {}).keys())
        all_props.update(current_concept.get("properties", {}).keys())
        all_props.update(next_concept.get("properties", {}).keys())
        
        for prop in all_props:
            prev_val = prev_concept.get("properties", {}).get(prop)
            curr_val = current_concept.get("properties", {}).get(prop)
            next_val = next_concept.get("properties", {}).get(prop)
            
            if prev_val is not None and curr_val is not None and next_val is not None:
                # Check for monotonic change
                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)) and isinstance(next_val, (int, float)):
                    if (prev_val <= curr_val <= next_val) or (prev_val >= curr_val >= next_val):
                        monotonic_score += 1
                    property_changes += 1
                elif isinstance(prev_val, str) and isinstance(curr_val, str) and isinstance(next_val, str):
                    # For strings, check if there's a progression
                    if prev_val != curr_val and curr_val != next_val:
                        monotonic_score += 0.5
                    property_changes += 1
        
        if property_changes > 0:
            quality = monotonic_score / property_changes
        else:
            # Check semantic gradient
            sim1 = self._calculate_concept_similarity_simple(prev_concept, current_concept)
            sim2 = self._calculate_concept_similarity_simple(current_concept, next_concept)
            
            # Good gradient has similar step sizes
            quality = 1.0 - abs(sim1 - sim2)
        
        return quality
    
    def _analyze_gradient_quality(self, path: List[str], space) -> Dict[str, Any]:
        """Analyze the quality of a gradient path"""
        quality = {
            "smoothness": 0.0,
            "monotonicity": 0.0,
            "step_uniformity": 0.0,
            "semantic_coherence": 0.0
        }
        
        if len(path) < 2:
            return quality
        
        # Calculate step similarities
        step_similarities = []
        for i in range(len(path) - 1):
            concept1 = space.concepts.get(path[i], {})
            concept2 = space.concepts.get(path[i + 1], {})
            similarity = self._calculate_concept_similarity_simple(concept1, concept2)
            step_similarities.append(similarity)
        
        # Smoothness: average similarity between consecutive steps
        quality["smoothness"] = sum(step_similarities) / len(step_similarities) if step_similarities else 0
        
        # Step uniformity: how similar are the step sizes
        if len(step_similarities) > 1:
            step_variance = np.var(step_similarities)
            quality["step_uniformity"] = 1.0 / (1.0 + step_variance)
        else:
            quality["step_uniformity"] = 1.0
        
        # Monotonicity: check property changes
        monotonic_props = 0
        total_props_checked = 0
        
        # Get all properties across path
        all_properties = set()
        for node_id in path:
            concept = space.concepts.get(node_id, {})
            all_properties.update(concept.get("properties", {}).keys())
        
        for prop in all_properties:
            values = []
            for node_id in path:
                concept = space.concepts.get(node_id, {})
                if prop in concept.get("properties", {}):
                    values.append(concept["properties"][prop])
            
            if len(values) >= 3:
                # Check if values show monotonic trend
                if self._is_monotonic_sequence(values):
                    monotonic_props += 1
                total_props_checked += 1
        
        if total_props_checked > 0:
            quality["monotonicity"] = monotonic_props / total_props_checked
        
        # Semantic coherence: no sudden jumps in meaning
        max_step_diff = max(step_similarities) - min(step_similarities) if step_similarities else 0
        quality["semantic_coherence"] = 1.0 - max_step_diff
        
        return quality
    
    def _identify_gradient_dimension_advanced(self, path: List[str], space) -> str:
        """Advanced identification of gradient dimension"""
        if len(path) < 2:
            return "undefined_dimension"
        
        # Analyze property changes along path
        property_trajectories = {}
        
        for prop_name in self._get_all_properties_in_path(path, space):
            values = []
            for node_id in path:
                concept = space.concepts.get(node_id, {})
                if prop_name in concept.get("properties", {}):
                    values.append(concept["properties"][prop_name])
            
            if len(values) >= len(path) * 0.7:  # Property present in most nodes
                property_trajectories[prop_name] = values
        
        # Find property with most consistent change
        best_property = None
        best_score = 0
        
        for prop_name, values in property_trajectories.items():
            # Calculate consistency score
            if all(isinstance(v, (int, float)) for v in values):
                # Numeric property - check for linear trend
                if len(values) >= 3:
                    # Simple linear regression
                    x = list(range(len(values)))
                    correlation = abs(np.corrcoef(x, values)[0, 1]) if len(set(values)) > 1 else 0
                    if correlation > best_score:
                        best_score = correlation
                        best_property = prop_name
            
            elif all(isinstance(v, str) for v in values):
                # String property - check for systematic changes
                unique_ratio = len(set(values)) / len(values)
                if 0.5 < unique_ratio < 1.0:  # Some change but not random
                    if unique_ratio > best_score:
                        best_score = unique_ratio
                        best_property = prop_name
        
        # Determine dimension name
        if best_property:
            return f"{best_property}_gradient"
        
        # Fallback: analyze concept names
        name_patterns = self._analyze_name_patterns(path, space)
        if name_patterns:
            return f"{name_patterns}_dimension"
        
        return "abstract_dimension"
    
    def _classify_gradient_type(self, quality: Dict[str, Any]) -> str:
        """Classify the type of gradient based on quality metrics"""
        smoothness = quality.get("smoothness", 0)
        monotonicity = quality.get("monotonicity", 0)
        uniformity = quality.get("step_uniformity", 0)
        coherence = quality.get("semantic_coherence", 0)
        
        # Calculate overall quality
        overall_quality = (smoothness + monotonicity + uniformity + coherence) / 4
        
        if overall_quality > 0.8:
            if monotonicity > 0.8:
                return "linear_gradient"
            else:
                return "smooth_gradient"
        
        elif overall_quality > 0.6:
            if uniformity < 0.5:
                return "variable_gradient"
            else:
                return "moderate_gradient"
        
        elif smoothness > 0.7 and monotonicity < 0.3:
            return "oscillating_gradient"
        
        elif coherence < 0.4:
            return "discontinuous_gradient"
        
        else:
            return "weak_gradient"
    
    # Additional helper methods for the above functions
    
    def _determine_cluster_theme(self, members: List[str], space) -> str:
        """Determine the thematic focus of a cluster"""
        # Collect all concept names and properties
        words = []
        properties = []
        
        for member_id in members:
            concept = space.concepts.get(member_id, {})
            
            # Add concept name words
            name_words = concept.get("name", "").lower().split()
            words.extend(name_words)
            
            # Add property names
            properties.extend(concept.get("properties", {}).keys())
        
        # Find most common words (excluding stop words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        word_freq = {}
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find most common meaningful word
        if word_freq:
            theme_word = max(word_freq.items(), key=lambda x: x[1])[0]
            
            # Check if it's a domain term
            domain_terms = {
                "system": "systems_and_processes",
                "process": "procedural_elements", 
                "state": "states_and_conditions",
                "action": "actions_and_behaviors",
                "property": "properties_and_attributes",
                "relation": "relationships_and_connections"
            }
            
            for term, theme in domain_terms.items():
                if term in theme_word:
                    return theme
            
            return f"{theme_word}_related"
        
        # Fallback to property analysis
        if properties:
            common_prop = max(set(properties), key=properties.count)
            return f"{common_prop}_focused"
        
        return "mixed_theme"
    
    def _is_monotonic_sequence(self, values: List[Any]) -> bool:
        """Check if a sequence of values is monotonic"""
        if len(values) < 2:
            return True
        
        # For numeric values
        if all(isinstance(v, (int, float)) for v in values):
            increasing = all(values[i] <= values[i+1] for i in range(len(values)-1))
            decreasing = all(values[i] >= values[i+1] for i in range(len(values)-1))
            return increasing or decreasing
        
        # For string values, check if there's a pattern
        if all(isinstance(v, str) for v in values):
            # Check if all different (indicating progression)
            return len(set(values)) == len(values)
        
        return False
    
    def _get_all_properties_in_path(self, path: List[str], space) -> Set[str]:
        """Get all properties present in concepts along a path"""
        all_properties = set()
        
        for node_id in path:
            concept = space.concepts.get(node_id, {})
            all_properties.update(concept.get("properties", {}).keys())
        
        return all_properties
    
    def _analyze_name_patterns(self, path: List[str], space) -> str:
        """Analyze naming patterns in a path"""
        names = []
        for node_id in path:
            concept = space.concepts.get(node_id, {})
            names.append(concept.get("name", "").lower())
        
        if not names:
            return ""
        
        # Look for common prefixes/suffixes
        if len(names) >= 3:
            # Check prefixes
            common_prefix = ""
            for i in range(min(len(n) for n in names)):
                if all(name[i] == names[0][i] for name in names):
                    common_prefix += names[0][i]
                else:
                    break
            
            if len(common_prefix) >= 3:
                return common_prefix.strip()
            
            # Check suffixes
            reversed_names = [name[::-1] for name in names]
            common_suffix = ""
            for i in range(min(len(n) for n in reversed_names)):
                if all(name[i] == reversed_names[0][i] for name in reversed_names):
                    common_suffix += reversed_names[0][i]
                else:
                    break
            
            if len(common_suffix) >= 3:
                return common_suffix[::-1].strip()
        
        # Look for numeric patterns
        numbers = []
        for name in names:
            import re
            nums = re.findall(r'\d+', name)
            if nums:
                numbers.extend([int(n) for n in nums])
        
        if numbers and len(numbers) == len(names):
            if self._is_monotonic_sequence(numbers):
                return "numeric_progression"
        
        return ""
    
    # ========================================================================================
    # DELEGATE TO ORIGINAL CORE
    # ========================================================================================
    
    def __getattr__(self, name):
        """Delegate any missing methods to the original reasoning core"""
        return getattr(self.original_core, name)
