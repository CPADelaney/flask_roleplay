# nyx/core/integration/multimodal_attention_bridge.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class MultimodalAttentionBridge:
    """
    Integrates multimodal perception with attention and expectation systems.
    Coordinates bidirectional flow between raw perception, attentional focus,
    and expectation generation.
    """
    
    def __init__(self, 
                multimodal_integrator=None,
                dynamic_attention_system=None,
                perceptual_integration_layer=None,
                reasoning_core=None):
        """Initialize the multimodal-attention bridge."""
        self.multimodal_integrator = multimodal_integrator
        self.attention_system = dynamic_attention_system
        self.perceptual_layer = perceptual_integration_layer
        self.reasoning_core = reasoning_core
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration state tracking
        self.active_expectations = []
        self.attention_queue = []
        self.recent_percepts = {}
        self._subscribed = False
        
        logger.info("MultimodalAttentionBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("sensory_input", self._handle_sensory_input)
                self.event_bus.subscribe("attention_shift", self._handle_attention_shift)
                self.event_bus.subscribe("expectation_generated", self._handle_expectation)
                self._subscribed = True
            
            logger.info("MultimodalAttentionBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing MultimodalAttentionBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="MultimodalAttention")
    async def process_percept_with_attention(self, 
                                         input_data: Dict[str, Any], 
                                         modality: str) -> Dict[str, Any]:
        """
        Process perceptual input with attention and expectation integration.
        
        Args:
            input_data: Input data to process
            modality: Input modality
            
        Returns:
            Processing results
        """
        if not self.multimodal_integrator or not self.attention_system:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get current expectations from active attention focus
            expectations = await self._get_current_expectations(modality)
            
            # 2. Process through multimodal integrator
            # Prepare input for integrator
            if hasattr(self.multimodal_integrator, 'process_sensory_input'):
                # Create proper input structure
                from nyx.core.multimodal_integrator import SensoryInput, Modality
                
                # Convert string modality to Modality enum if needed
                if isinstance(modality, str):
                    try:
                        mod_enum = Modality(modality)
                    except ValueError:
                        mod_enum = Modality.TEXT  # Default fallback
                else:
                    mod_enum = modality
                
                # Create input object
                sensory_input = SensoryInput(
                    modality=mod_enum,
                    data=input_data,
                    timestamp=datetime.datetime.now().isoformat(),
                    metadata={"source": "multimodal_attention_bridge"}
                )
                
                # Process with expectations
                percept = await self.multimodal_integrator.process_sensory_input(
                    sensory_input, expectations
                )
                
                # Save recent percept
                self.recent_percepts[modality] = {
                    "percept": percept,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # 3. Update attention based on percept significance
                if hasattr(self.attention_system, 'focus_attention'):
                    # Convert significance to attention level
                    content_summary = self._summarize_percept(percept)
                    confidence = getattr(percept, 'bottom_up_confidence', 0.5)
                    attention_level = confidence * 0.8  # Scale
                    
                    # Request attention focus
                    attention_result = await self.attention_system.focus_attention(
                        target=content_summary,
                        target_type=f"percept_{modality}",
                        attention_level=attention_level,
                        source="multimodal_bridge"
                    )
                
                return {
                    "status": "success",
                    "percept": percept,
                    "expectations_applied": len(expectations),
                    "attention_level": attention_level if 'attention_level' in locals() else None,
                    "modality": modality
                }
            else:
                logger.error("Multimodal integrator missing process_sensory_input method")
                return {"status": "error", "message": "Multimodal integrator API mismatch"}
        except Exception as e:
            logger.error(f"Error processing percept with attention: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="MultimodalAttention")
    async def generate_cross_modal_expectations(self) -> Dict[str, Any]:
        """
        Generate expectations across modalities based on current attentional focus.
        """
        if not self.multimodal_integrator or not self.attention_system:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # 1. Get current attentional focus
            attention_state = await self.attention_system.get_current_focus()
            if not attention_state or not attention_state.get("focus"):
                return {"status": "no_focus", "message": "No current attentional focus"}
            
            focus = attention_state["focus"]
            
            # 2. Generate expectations for other modalities based on current focus
            generated_expectations = []
            
            # If focused on text, generate visual expectations
            if "text" in focus.get("target_type", ""):
                # Text -> visual expectations
                visual_exp = await self._create_expectation(
                    target_modality="image",
                    pattern={"text_content": focus["target"]},
                    strength=0.6
                )
                generated_expectations.append(visual_exp)
            
            # If focused on image, generate text and audio expectations
            elif "image" in focus.get("target_type", ""):
                # Image -> text expectations
                text_exp = await self._create_expectation(
                    target_modality="text",
                    pattern=focus["target"],
                    strength=0.7
                )
                generated_expectations.append(text_exp)
                
                # Image -> audio expectations
                audio_exp = await self._create_expectation(
                    target_modality="audio_speech",
                    pattern={"content_related_to": focus["target"]},
                    strength=0.4
                )
                generated_expectations.append(audio_exp)
            
            # 3. Register expectations with multimodal integrator
            for exp in generated_expectations:
                if hasattr(self.multimodal_integrator, 'add_expectation'):
                    await self.multimodal_integrator.add_expectation(exp)
            
            # 4. Also register with perceptual integration layer if available
            if self.perceptual_layer and hasattr(self.perceptual_layer, 'add_expectation'):
                for exp in generated_expectations:
                    # Convert expectation format if needed
                    await self.perceptual_layer.add_expectation(
                        exp.target_modality,
                        exp.pattern,
                        exp.strength
                    )
            
            # Save active expectations
            self.active_expectations = generated_expectations
            
            return {
                "status": "success",
                "expectations_created": len(generated_expectations),
                "current_focus": focus["target"],
                "expectations": [
                    {
                        "modality": exp.target_modality,
                        "strength": exp.strength,
                        "pattern_type": type(exp.pattern).__name__
                    }
                    for exp in generated_expectations
                ]
            }
        except Exception as e:
            logger.error(f"Error generating cross-modal expectations: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_current_expectations(self, modality: str) -> List[Any]:
        """Get current expectations for a modality."""
        expectations = []
        
        # Filter active expectations for this modality
        for exp in self.active_expectations:
            if exp.target_modality == modality:
                expectations.append(exp)
        
        # If reasoning core can provide expectations, query it
        if self.reasoning_core and hasattr(self.reasoning_core, 'generate_perceptual_expectations'):
            try:
                additional_expectations = await self.reasoning_core.generate_perceptual_expectations(modality)
                if additional_expectations:
                    expectations.extend(additional_expectations)
            except Exception as e:
                logger.error(f"Error getting expectations from reasoning core: {e}")
        
        return expectations
    
    async def _create_expectation(self, 
                               target_modality: str, 
                               pattern: Any, 
                               strength: float = 0.5) -> Any:
        """Create an expectation object."""
        # Import ExpectationSignal from multimodal_integrator if available
        try:
            from nyx.core.multimodal_integrator import ExpectationSignal
            
            return ExpectationSignal(
                target_modality=target_modality,
                pattern=pattern,
                strength=strength,
                source="multimodal_attention_bridge",
                priority=0.7
            )
        except ImportError:
            # Fallback to dict if import fails
            return {
                "target_modality": target_modality,
                "pattern": pattern,
                "strength": strength,
                "source": "multimodal_attention_bridge",
                "priority": 0.7
            }
    
    def _summarize_percept(self, percept: Any) -> str:
        """Generate a simple summary of a percept for attention."""
        # Extract useful summary based on percept structure
        content = getattr(percept, 'content', None)
        
        if content is None:
            return "Unknown percept"
        
        if isinstance(content, str):
            return content[:50]  # Truncate long text
        elif isinstance(content, dict):
            if "description" in content:
                return content["description"][:50]
            elif "summary" in content:
                return content["summary"][:50]
            else:
                return str(list(content.keys()))[:50]
        else:
            return str(content)[:50]
    
    async def _handle_sensory_input(self, event: Event) -> None:
        """
        Handle sensory input events.
        
        Args:
            event: Sensory input event
        """
        try:
            # Extract data
            modality = event.data.get("modality")
            content = event.data.get("content")
            
            if not modality or not content:
                return
            
            # Process with attention
            asyncio.create_task(
                self.process_percept_with_attention(content, modality)
            )
        except Exception as e:
            logger.error(f"Error handling sensory input: {e}")
    
    async def _handle_attention_shift(self, event: Event) -> None:
        """
        Handle attention shift events.
        
        Args:
            event: Attention shift event
        """
        try:
            # When attention shifts, generate new expectations
            asyncio.create_task(self.generate_cross_modal_expectations())
        except Exception as e:
            logger.error(f"Error handling attention shift: {e}")
    
    async def _handle_expectation(self, event: Event) -> None:
        """
        Handle expectation events.
        
        Args:
            event: Expectation event
        """
        try:
            # Extract data
            target_modality = event.data.get("target_modality")
            pattern = event.data.get("pattern")
            strength = event.data.get("strength", 0.5)
            
            if not target_modality or not pattern:
                return
            
            # Create expectation
            expectation = await self._create_expectation(
                target_modality=target_modality,
                pattern=pattern,
                strength=strength
            )
            
            # Add to active expectations
            self.active_expectations.append(expectation)
            
            # Register with multimodal integrator
            if self.multimodal_integrator and hasattr(self.multimodal_integrator, 'add_expectation'):
                await self.multimodal_integrator.add_expectation(expectation)
        except Exception as e:
            logger.error(f"Error handling expectation event: {e}")

# Function to create the bridge
def create_multimodal_attention_bridge(nyx_brain):
    """Create a multimodal-attention bridge for the given brain."""
    return MultimodalAttentionBridge(
        multimodal_integrator=nyx_brain.multimodal_integrator if hasattr(nyx_brain, "multimodal_integrator") else None,
        dynamic_attention_system=nyx_brain.dynamic_attention_system if hasattr(nyx_brain, "dynamic_attention_system") else None,
        perceptual_integration_layer=nyx_brain.perceptual_integration_layer if hasattr(nyx_brain, "perceptual_integration_layer") else None,
        reasoning_core=nyx_brain.reasoning_core if hasattr(nyx_brain, "reasoning_core") else None
    )
