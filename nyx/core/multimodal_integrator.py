# nyx/core/multimodal_integrator.py

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, RunContextWrapper

class SensoryInput(BaseModel):
    """Schema for raw sensory input"""
    modality: str = Field(..., description="Input modality (visual, auditory, text, etc.)")
    data: Any = Field(..., description="Raw input data")
    confidence: float = Field(1.0, description="Input confidence (0.0-1.0)", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="Input timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ExpectationSignal(BaseModel):
    """Schema for top-down expectation signal"""
    target_modality: str = Field(..., description="Target modality to influence")
    pattern: Any = Field(..., description="Expected pattern or feature")
    strength: float = Field(0.5, description="Signal strength (0.0-1.0)", ge=0.0, le=1.0)
    source: str = Field(..., description="Source of expectation (reasoning, memory, etc.)")
    priority: float = Field(0.5, description="Priority level (0.0-1.0)", ge=0.0, le=1.0)

class IntegratedPercept(BaseModel):
    """Schema for integrated percept after bottom-up and top-down processing"""
    modality: str = Field(..., description="Percept modality")
    content: Any = Field(..., description="Processed content")
    bottom_up_confidence: float = Field(..., description="Confidence from bottom-up processing")
    top_down_influence: float = Field(..., description="Degree of top-down influence")
    attention_weight: float = Field(..., description="Attentional weight applied")
    timestamp: str = Field(..., description="Processing timestamp")

class EnhancedMultiModalIntegrator:
    """
    Processes sensory inputs using both bottom-up and top-down pathways.
    Bottom-up: Processes raw data from sensory systems.
    Top-down: Applies expectations from reasoning to modulate processing.
    """
    
    def __init__(self, reasoning_core=None, attentional_controller=None):
        self.reasoning_core = reasoning_core
        self.attentional_controller = attentional_controller
        
        # Processing stages
        self.feature_extractors = {}  # For bottom-up processing
        self.expectation_modulators = {}  # For top-down processing
        self.integration_strategies = {}  # For combining both pathways
        
        # Buffer for recent perceptions
        self.perception_buffer = []
        self.max_buffer_size = 50
        
        # Current active expectations
        self.active_expectations = []
        
        self.logger = logging.getLogger(__name__)
    
    async def register_feature_extractor(self, modality: str, extractor_function):
        """Register a feature extraction function for a specific modality"""
        self.feature_extractors[modality] = extractor_function
        
    async def register_expectation_modulator(self, modality: str, modulator_function):
        """Register a function that applies top-down expectations to a modality"""
        self.expectation_modulators[modality] = modulator_function
    
    async def register_integration_strategy(self, modality: str, integration_function):
        """Register a function that integrates bottom-up and top-down processing"""
        self.integration_strategies[modality] = integration_function
    
    async def process_sensory_input(self, 
                                   input_data: SensoryInput, 
                                   expectations: List[ExpectationSignal] = None) -> IntegratedPercept:
        """
        Process sensory input using both bottom-up and top-down pathways
        
        Args:
            input_data: Raw sensory input
            expectations: Optional list of top-down expectations
            
        Returns:
            Integrated percept combining bottom-up and top-down processing
        """
        modality = input_data.modality
        
        # 1. Bottom-up processing (data-driven)
        bottom_up_result = await self._perform_bottom_up_processing(input_data)
        
        # 2. Get or use provided top-down expectations
        if expectations is None:
            expectations = await self._get_active_expectations(modality)
        
        # 3. Apply top-down modulation
        modulated_result = await self._apply_top_down_modulation(bottom_up_result, expectations, modality)
        
        # 4. Integrate bottom-up and top-down pathways
        integrated_result = await self._integrate_pathways(bottom_up_result, modulated_result, modality)
        
        # 5. Apply attentional filtering if available
        if self.attentional_controller:
            attentional_weight = await self.attentional_controller.calculate_attention_weight(
                integrated_result, modality, expectations
            )
        else:
            attentional_weight = 1.0  # Default full attention
        
        # 6. Create final percept
        percept = IntegratedPercept(
            modality=modality,
            content=integrated_result["content"],
            bottom_up_confidence=bottom_up_result["confidence"],
            top_down_influence=modulated_result["influence_strength"],
            attention_weight=attentional_weight,
            timestamp=input_data.timestamp
        )
        
        # 7. Add to perception buffer
        self._add_to_perception_buffer(percept)
        
        # 8. Update reasoning core with new perception (if significant)
        if self.reasoning_core and attentional_weight > 0.5:
            await self._update_reasoning_with_perception(percept)
        
        return percept
    
    async def _perform_bottom_up_processing(self, input_data: SensoryInput) -> Dict[str, Any]:
        """Extract features from raw sensory input (bottom-up processing)"""
        modality = input_data.modality
        
        if modality in self.feature_extractors:
            try:
                # Use registered feature extractor for this modality
                extractor = self.feature_extractors[modality]
                features = await extractor(input_data.data)
                
                return {
                    "modality": modality,
                    "features": features,
                    "confidence": input_data.confidence,
                    "metadata": input_data.metadata
                }
            except Exception as e:
                self.logger.error(f"Error in bottom-up processing for {modality}: {str(e)}")
        
        # Default simple processing if no extractor available
        return {
            "modality": modality,
            "features": input_data.data,  # Just pass through
            "confidence": input_data.confidence,
            "metadata": input_data.metadata
        }
    
    async def _get_active_expectations(self, modality: str) -> List[ExpectationSignal]:
        """Get current active expectations for the specified modality"""
        # Filter existing expectations for this modality
        relevant_expectations = [exp for exp in self.active_expectations 
                               if exp.target_modality == modality]
        
        # If reasoning core is available, get additional expectations
        if self.reasoning_core:
            try:
                # Request expectations from reasoning core
                new_expectations = await self.reasoning_core.generate_perceptual_expectations(modality)
                
                # Add to active expectations
                for exp in new_expectations:
                    if exp not in self.active_expectations:
                        self.active_expectations.append(exp)
                
                # Update relevant expectations
                relevant_expectations.extend([exp for exp in new_expectations 
                                           if exp.target_modality == modality])
                
                # Prune old expectations (keeping only recent ones)
                if len(self.active_expectations) > 30:
                    # Sort by priority and keep top 30
                    self.active_expectations.sort(key=lambda x: x.priority, reverse=True)
                    self.active_expectations = self.active_expectations[:30]
            except Exception as e:
                self.logger.error(f"Error getting expectations from reasoning core: {str(e)}")
        
        return relevant_expectations
    
    async def _apply_top_down_modulation(self, 
                                       bottom_up_result: Dict[str, Any],
                                       expectations: List[ExpectationSignal],
                                       modality: str) -> Dict[str, Any]:
        """Apply top-down expectations to modulate perception"""
        # If no expectations or no modulator for this modality, return bottom-up unchanged
        if not expectations or modality not in self.expectation_modulators:
            return {
                "modality": modality,
                "features": bottom_up_result["features"],
                "influence_strength": 0.0,
                "influenced_by": []
            }
        
        try:
            # Get modulator for this modality
            modulator = self.expectation_modulators[modality]
            
            # Apply modulator
            modulation_result = await modulator(bottom_up_result["features"], expectations)
            
            return {
                "modality": modality,
                "features": modulation_result["features"],
                "influence_strength": modulation_result["influence_strength"],
                "influenced_by": modulation_result["influenced_by"]
            }
        except Exception as e:
            self.logger.error(f"Error in top-down modulation for {modality}: {str(e)}")
            
            # Fallback to unmodulated result
            return {
                "modality": modality,
                "features": bottom_up_result["features"],
                "influence_strength": 0.0,
                "influenced_by": []
            }
    
    async def _integrate_pathways(self,
                                bottom_up_result: Dict[str, Any],
                                top_down_result: Dict[str, Any],
                                modality: str) -> Dict[str, Any]:
        """Integrate bottom-up and top-down processing pathways"""
        # If integration strategy exists for this modality, use it
        if modality in self.integration_strategies:
            try:
                integration_func = self.integration_strategies[modality]
                integrated = await integration_func(bottom_up_result, top_down_result)
                
                return integrated
            except Exception as e:
                self.logger.error(f"Error in pathway integration for {modality}: {str(e)}")
        
        # Default integration strategy - weighted by confidence and influence
        bottom_up_weight = bottom_up_result["confidence"]
        top_down_weight = top_down_result["influence_strength"]
        
        total_weight = bottom_up_weight + top_down_weight
        if total_weight == 0:
            # Avoid division by zero
            bottom_up_ratio = 1.0
            top_down_ratio = 0.0
        else:
            bottom_up_ratio = bottom_up_weight / total_weight
            top_down_ratio = top_down_weight / total_weight
        
        # For simple features, use weighted average
        if isinstance(bottom_up_result["features"], (int, float)) and \
           isinstance(top_down_result["features"], (int, float)):
            integrated_content = (bottom_up_result["features"] * bottom_up_ratio) + \
                              (top_down_result["features"] * top_down_ratio)
        else:
            # For complex features, prioritize by weight
            if bottom_up_ratio >= top_down_ratio:
                integrated_content = bottom_up_result["features"]
            else:
                integrated_content = top_down_result["features"]
        
        return {
            "content": integrated_content,
            "bottom_up_ratio": bottom_up_ratio,
            "top_down_ratio": top_down_ratio,
            "bottom_up_features": bottom_up_result["features"],
            "top_down_features": top_down_result["features"]
        }
    
    def _add_to_perception_buffer(self, percept: IntegratedPercept):
        """Add percept to buffer, removing oldest if full"""
        self.perception_buffer.append(percept)
        
        if len(self.perception_buffer) > self.max_buffer_size:
            self.perception_buffer.pop(0)
    
    async def _update_reasoning_with_perception(self, percept: IntegratedPercept):
        """Update reasoning core with significant perception"""
        if self.reasoning_core:
            try:
                await self.reasoning_core.update_with_perception(percept)
            except Exception as e:
                self.logger.error(f"Error updating reasoning with perception: {str(e)}")
    
    async def add_expectation(self, expectation: ExpectationSignal):
        """Add a new top-down expectation"""
        self.active_expectations.append(expectation)
    
    async def clear_expectations(self, modality: str = None):
        """Clear active expectations, optionally for a specific modality"""
        if modality:
            self.active_expectations = [exp for exp in self.active_expectations 
                                      if exp.target_modality != modality]
        else:
            self.active_expectations = []
    
    async def get_recent_percepts(self, modality: str = None, limit: int = 10) -> List[IntegratedPercept]:
        """Get recent percepts, optionally filtered by modality"""
        if modality:
            filtered = [p for p in self.perception_buffer if p.modality == modality]
            return filtered[-limit:]
        else:
            return self.perception_buffer[-limit:]
