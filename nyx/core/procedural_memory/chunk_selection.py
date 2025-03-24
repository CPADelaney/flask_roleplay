# nyx/core/procedural_memory/chunk_selection.py

import datetime
from typing import Dict, List, Any, Optional, Set
from collections import Counter
from .models import ContextPattern, ChunkPrediction

class ContextAwareChunkSelector:
    """Enhanced selection system for chunks based on execution context"""
    
    def __init__(self):
        self.context_patterns = {}  # pattern_id -> ContextPattern
        self.recent_contexts = []  # List of recent execution contexts
        self.recent_selections = []  # List of recent chunk selections
        self.max_history = 50  # Max number of historical contexts to keep
        self.domain_specific_patterns = {}  # domain -> [pattern_ids]
        
    def register_context_pattern(self, pattern: ContextPattern) -> str:
        """Register a new context pattern"""
        self.context_patterns[pattern.id] = pattern
        
        # Update domain index
        if pattern.domain not in self.domain_specific_patterns:
            self.domain_specific_patterns[pattern.domain] = []
        self.domain_specific_patterns[pattern.domain].append(pattern.id)
        
        return pattern.id
        
    def select_chunk(self, 
                    available_chunks: Dict[str, List[Dict[str, Any]]], 
                    context: Dict[str, Any],
                    procedure_domain: str) -> ChunkPrediction:
        """
        Select the most appropriate chunk based on context
        
        Args:
            available_chunks: Dictionary of chunk_id -> steps
            context: Current execution context
            procedure_domain: Domain of the procedure
            
        Returns:
            Prediction of which chunk to use
        """
        # Store context for future learning
        self._record_context(context)
        
        # Calculate match scores for each context pattern
        pattern_scores = {}
        
        # Get relevant patterns for this domain
        domain_patterns = self.domain_specific_patterns.get(procedure_domain, [])
        
        for pattern_id in domain_patterns:
            pattern = self.context_patterns.get(pattern_id)
            if not pattern:
                continue
                
            # Calculate match score
            match_score = self._calculate_pattern_match(pattern, context)
            
            if match_score >= pattern.confidence_threshold:
                pattern_scores[pattern_id] = match_score
                
                # Update pattern statistics
                pattern.match_count += 1
                pattern.last_matched = datetime.datetime.now().isoformat()
        
        # Find patterns associated with chunks
        chunk_scores = {}
        reasoning = {}
        
        for chunk_id in available_chunks.keys():
            chunk_scores[chunk_id] = 0.0
            reasoning[chunk_id] = []
            
            # Check direct indicators in context
            if f"near_{chunk_id}" in context and context[f"near_{chunk_id}"]:
                chunk_scores[chunk_id] += 0.5
                reasoning[chunk_id].append(f"Direct context indicator: near_{chunk_id}")
                
            # Check command intent
            if "command_intent" in context and chunk_id in context["command_intent"]:
                chunk_scores[chunk_id] += 0.4
                reasoning[chunk_id].append(f"Command intent includes {chunk_id}")
                
            # Check pattern matches
            for pattern_id, score in pattern_scores.items():
                pattern = self.context_patterns[pattern_id]
                
                # See if this pattern is associated with this chunk
                for indicator, values in pattern.indicators.items():
                    if indicator.startswith(f"chunk_{chunk_id}_suitable") and values:
                        chunk_scores[chunk_id] += score * 0.7
                        reasoning[chunk_id].append(f"Pattern {pattern.name} matched with score {score:.2f}")
            
            # Check recent context history for similar situations
            history_score = self._check_history_for_chunk(chunk_id, context)
            if history_score > 0:
                chunk_scores[chunk_id] += history_score * 0.3
                reasoning[chunk_id].append(f"Similar historical context used this chunk with score {history_score:.2f}")
        
        # Select best chunk
        if not chunk_scores:
            # No good matches, return first chunk with low confidence
            first_chunk_id = next(iter(available_chunks.keys())) if available_chunks else None
            return ChunkPrediction(
                chunk_id=first_chunk_id,
                confidence=0.1,
                context_match_score=0.1,
                reasoning=["No context patterns matched"]
            )
        
        # Get top chunk and confidence
        best_chunk_id = max(chunk_scores.items(), key=lambda x: x[1])[0]
        best_score = chunk_scores[best_chunk_id]
        
        # Get alternative chunks
        alternatives = [
            {"chunk_id": c_id, "score": score}
            for c_id, score in chunk_scores.items()
            if c_id != best_chunk_id and score > 0
        ]
        
        # Create prediction
        prediction = ChunkPrediction(
            chunk_id=best_chunk_id,
            confidence=min(1.0, best_score),
            context_match_score=best_score,
            reasoning=reasoning.get(best_chunk_id, []),
            alternative_chunks=alternatives
        )
        
        # Record selection for future learning
        self._record_selection(prediction, context)
        
        return prediction
    
    def _calculate_pattern_match(self, pattern: ContextPattern, context: Dict[str, Any]) -> float:
        """Calculate how well a context pattern matches the current context"""
        # Check each indicator in the pattern
        indicator_matches = 0
        total_indicators = len(pattern.indicators)
        
        if total_indicators == 0:
            return 0.0
            
        for indicator, expected_value in pattern.indicators.items():
            if indicator not in context:
                continue
                
            actual_value = context[indicator]
            
            # Different comparison based on type
            if isinstance(expected_value, (list, tuple, set)):
                # Check if value is in list
                if actual_value in expected_value:
                    indicator_matches += 1
            elif isinstance(expected_value, dict) and "min" in expected_value and "max" in expected_value:
                # Range check
                if expected_value["min"] <= actual_value <= expected_value["max"]:
                    indicator_matches += 1
            else:
                # Direct equality check
                if actual_value == expected_value:
                    indicator_matches += 1
        
        # Base match score on percentage of matching indicators
        base_score = indicator_matches / total_indicators
        
        # Check temporal pattern if defined
        temporal_score = 0.0
        if pattern.temporal_pattern and "action_history" in context:
            action_history = context["action_history"]
            temporal_score = self._match_temporal_pattern(pattern.temporal_pattern, action_history)
            
            # Combine scores, giving more weight to direct indicators
            return base_score * 0.7 + temporal_score * 0.3
        
        return base_score
    
    def _match_temporal_pattern(self, pattern: List[Dict[str, Any]], history: List[Dict[str, Any]]) -> float:
        """Match a temporal pattern against action history"""
        # Skip if history is too short
        if len(history) < len(pattern):
            return 0.0
            
        # Check pattern against most recent history
        recent_history = history[-len(pattern):]
        
        # Count matching items
        matches = 0
        for i, expected in enumerate(pattern):
            if i >= len(recent_history):
                break
                
            actual = recent_history[i]
            
            # Count matching keys
            matching_keys = 0
            total_keys = len(expected)
            
            for key, expected_value in expected.items():
                if key in actual and actual[key] == expected_value:
                    matching_keys += 1
            
            # Consider a step matching if most keys match
            if matching_keys / total_keys >= 0.7:
                matches += 1
        
        # Calculate match percentage
        return matches / len(pattern)
    
    def _check_history_for_chunk(self, chunk_id: str, context: Dict[str, Any]) -> float:
        """Check if similar contexts in history used this chunk"""
        if not self.recent_contexts or not self.recent_selections:
            return 0.0
            
        # Find similarity scores between current context and historical contexts
        similarity_scores = []
        
        for i, historical_context in enumerate(self.recent_contexts):
            if i >= len(self.recent_selections):
                break
                
            # Calculate context similarity
            similarity = self._calculate_context_similarity(context, historical_context)
            
            # Check if this chunk was selected
            selection = self.recent_selections[i]
            if selection.chunk_id == chunk_id:
                similarity_scores.append(similarity * selection.confidence)
        
        # Return max similarity if any
        return max(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        # Get common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        
        if not common_keys:
            return 0.0
            
        # Count matching values
        matching_values = 0
        
        for key in common_keys:
            if context1[key] == context2[key]:
                matching_values += 1
        
        return matching_values / len(common_keys)
    
    def _record_context(self, context: Dict[str, Any]):
        """Record context for future reference"""
        self.recent_contexts.append(context.copy())
        
        # Trim history if needed
        if len(self.recent_contexts) > self.max_history:
            self.recent_contexts = self.recent_contexts[-self.max_history:]
    
    def _record_selection(self, prediction: ChunkPrediction, context: Dict[str, Any]):
        """Record chunk selection for future reference"""
        self.recent_selections.append(prediction)
        
        # Trim history if needed
        if len(self.recent_selections) > self.max_history:
            self.recent_selections = self.recent_selections[-self.max_history:]
    
    def create_context_pattern_from_history(self, chunk_id: str, domain: str) -> Optional[ContextPattern]:
        """Automatically create a context pattern from historical data"""
        # Need sufficient history
        if len(self.recent_contexts) < 5 or len(self.recent_selections) < 5:
            return None
            
        # Find instances where this chunk was selected
        instances = []
        
        for i, selection in enumerate(self.recent_selections):
            if selection.chunk_id == chunk_id and i < len(self.recent_contexts):
                instances.append(self.recent_contexts[i])
        
        if len(instances) < 3:
            return None
            
        # Find common indicators across contexts
        common_indicators = {}
        
        # Get all keys present in at least half of instances
        all_keys = set()
        for context in instances:
            all_keys.update(context.keys())
            
        for key in all_keys:
            # Get all values for this key
            values = [context.get(key) for context in instances if key in context]
            
            # Skip if not present in most instances
            if len(values) < len(instances) / 2:
                continue
                
            # Check if values are consistent
            if all(v == values[0] for v in values):
                # All values the same
                common_indicators[key] = values[0]
            elif all(isinstance(v, (int, float)) for v in values):
                # Numeric values - use range
                common_indicators[key] = {
                    "min": min(values),
                    "max": max(values)
                }
            elif len(set(values)) < len(values) / 2:
                # Some consistent values - use set of common values
                counter = Counter(values)
                common = [v for v, count in counter.items() if count > 1]
                if common:
                    common_indicators[key] = common
        
        # Create pattern ID and name
        pattern_id = f"auto_pattern_{chunk_id}_{len(self.context_patterns)}"
        pattern_name = f"Auto-generated pattern for {chunk_id}"
        
        # Create pattern
        pattern = ContextPattern(
            id=pattern_id,
            name=pattern_name,
            domain=domain,
            indicators=common_indicators
        )
        
        # Register pattern
        self.register_context_pattern(pattern)
        
        return pattern
