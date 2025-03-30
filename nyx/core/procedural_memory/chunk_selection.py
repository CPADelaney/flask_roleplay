# nyx/core/procedural_memory/chunk_selection.py

import datetime
import logging
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from collections import Counter, defaultdict
import numpy as np

# OpenAI Agents SDK imports
from agents import function_tool, custom_span, trace, RunContextWrapper
from agents.tracing import TraceProvider
from agents.exceptions import UserError

from .models import ContextPattern, ChunkPrediction

logger = logging.getLogger(__name__)

class ContextAwareChunkSelector:
    """Enhanced selection system for chunks based on execution context"""
    
    def __init__(self):
        self.context_patterns = {}  # pattern_id -> ContextPattern
        self.recent_contexts = []  # List of recent execution contexts
        self.recent_selections = []  # List of recent chunk selections
        self.max_history = 50  # Max number of historical contexts to keep
        self.domain_specific_patterns = {}  # domain -> [pattern_ids]
        self.similarity_cache = {}  # Cache for similarity calculations
        self.pattern_performance = {}  # pattern_id -> performance metrics
        self.session_id = str(uuid.uuid4())  # Unique session ID for tracing
        
    def register_context_pattern(self, pattern: ContextPattern) -> str:
        """Register a new context pattern"""
        # Create trace span for pattern registration
        with custom_span(
            name="register_context_pattern",
            data={
                "pattern_id": pattern.id,
                "pattern_name": pattern.name,
                "domain": pattern.domain,
                "indicators_count": len(pattern.indicators)
            }
        ):
            self.context_patterns[pattern.id] = pattern
            
            # Update domain index
            if pattern.domain not in self.domain_specific_patterns:
                self.domain_specific_patterns[pattern.domain] = []
            self.domain_specific_patterns[pattern.domain].append(pattern.id)
            
            # Initialize performance metrics
            self.pattern_performance[pattern.id] = {
                "match_count": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "last_used": None
            }
            
            logger.info(f"Registered context pattern: {pattern.id} ({pattern.name}) with {len(pattern.indicators)} indicators")
            
            return pattern.id
        
    async def select_chunk(self, 
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
        # Generate trace ID for selection
        trace_id = f"trace_{uuid.uuid4().hex}"
        
        # Create trace for chunk selection
        with trace(
            workflow_name=f"chunk_selection",
            trace_id=trace_id,
            metadata={
                "procedure_domain": procedure_domain,
                "available_chunks": list(available_chunks.keys()),
                "context_summary": str({k: v for k, v in context.items() if not isinstance(v, (dict, list))})[:500],
                "session_id": self.session_id
            }
        ):
            # Store context for future learning
            self._record_context(context)
            
            # Calculate match scores for each context pattern
            pattern_scores = {}
            
            # Create span for pattern matching
            with custom_span(
                name="pattern_matching",
                data={
                    "procedure_domain": procedure_domain,
                    "available_chunks": list(available_chunks.keys())
                }
            ) as pattern_span:
                # Get relevant patterns for this domain
                domain_patterns = self.domain_specific_patterns.get(procedure_domain, [])
                
                # Process patterns in parallel for better performance
                pattern_match_tasks = []
                for pattern_id in domain_patterns:
                    pattern = self.context_patterns.get(pattern_id)
                    if pattern:
                        # Create task for pattern matching
                        task = asyncio.create_task(
                            self._calculate_pattern_match_async(pattern, context, parent_span=pattern_span)
                        )
                        pattern_match_tasks.append((pattern_id, pattern, task))
                
                # Wait for all pattern matching tasks to complete
                for pattern_id, pattern, task in pattern_match_tasks:
                    try:
                        match_score = await task
                        
                        if match_score >= pattern.confidence_threshold:
                            pattern_scores[pattern_id] = match_score
                            
                            # Update pattern statistics
                            pattern.match_count += 1
                            pattern.last_matched = datetime.datetime.now().isoformat()
                            
                            # Update performance metrics
                            self.pattern_performance[pattern_id]["match_count"] += 1
                            self.pattern_performance[pattern_id]["last_used"] = datetime.datetime.now().isoformat()
                            
                            # Log match in trace
                            with custom_span(
                                name=f"pattern_match_{pattern_id}",
                                data={
                                    "pattern_id": pattern_id,
                                    "pattern_name": pattern.name,
                                    "match_score": match_score,
                                    "threshold": pattern.confidence_threshold
                                },
                                parent=pattern_span
                            ):
                                pass
                    except Exception as e:
                        # Log error in trace
                        with custom_span(
                            name=f"pattern_match_error_{pattern_id}",
                            data={
                                "pattern_id": pattern_id,
                                "error": str(e),
                                "error_type": type(e).__name__
                            },
                            parent=pattern_span
                        ):
                            pass
                        
                        logger.error(f"Error calculating pattern match for {pattern_id}: {str(e)}")
            
            # Find patterns associated with chunks
            chunk_scores = {}
            reasoning = {}
            
            # Create span for chunk scoring
            with custom_span(
                name="chunk_scoring",
                data={
                    "chunks_count": len(available_chunks),
                    "pattern_scores_count": len(pattern_scores)
                }
            ) as chunk_span:
                for chunk_id in available_chunks.keys():
                    chunk_scores[chunk_id] = 0.0
                    reasoning[chunk_id] = []
                    
                    # Create span for individual chunk scoring
                    with custom_span(
                        name=f"score_chunk_{chunk_id}",
                        data={"chunk_id": chunk_id},
                        parent=chunk_span
                    ) as individual_chunk_span:
                        # Check direct indicators in context
                        if f"near_{chunk_id}" in context and context[f"near_{chunk_id}"]:
                            chunk_scores[chunk_id] += 0.5
                            reasoning[chunk_id].append(f"Direct context indicator: near_{chunk_id}")
                            
                            # Log direct indicator in trace
                            with custom_span(
                                name=f"direct_indicator_{chunk_id}",
                                data={
                                    "indicator": f"near_{chunk_id}",
                                    "value": context[f"near_{chunk_id}"],
                                    "score_increase": 0.5
                                },
                                parent=individual_chunk_span
                            ):
                                pass
                            
                        # Check command intent
                        if "command_intent" in context and chunk_id in context["command_intent"]:
                            chunk_scores[chunk_id] += 0.4
                            reasoning[chunk_id].append(f"Command intent includes {chunk_id}")
                            
                            # Log command intent in trace
                            with custom_span(
                                name=f"command_intent_{chunk_id}",
                                data={
                                    "command_intent": context["command_intent"],
                                    "score_increase": 0.4
                                },
                                parent=individual_chunk_span
                            ):
                                pass
                            
                        # Check pattern matches
                        for pattern_id, score in pattern_scores.items():
                            pattern = self.context_patterns[pattern_id]
                            
                            # See if this pattern is associated with this chunk
                            for indicator, values in pattern.indicators.items():
                                if indicator.startswith(f"chunk_{chunk_id}_suitable") and values:
                                    score_increase = score * 0.7
                                    chunk_scores[chunk_id] += score_increase
                                    reasoning[chunk_id].append(f"Pattern {pattern.name} matched with score {score:.2f}")
                                    
                                    # Log pattern match in trace
                                    with custom_span(
                                        name=f"pattern_association_{chunk_id}_{pattern_id}",
                                        data={
                                            "pattern_id": pattern_id,
                                            "pattern_name": pattern.name,
                                            "pattern_score": score,
                                            "score_increase": score_increase
                                        },
                                        parent=individual_chunk_span
                                    ):
                                        pass
                        
                        # Check recent context history for similar situations
                        history_score = await self._check_history_for_chunk(chunk_id, context)
                        if history_score > 0:
                            score_increase = history_score * 0.3
                            chunk_scores[chunk_id] += score_increase
                            reasoning[chunk_id].append(f"Similar historical context used this chunk with score {history_score:.2f}")
                            
                            # Log history match in trace
                            with custom_span(
                                name=f"history_match_{chunk_id}",
                                data={
                                    "history_score": history_score,
                                    "score_increase": score_increase
                                },
                                parent=individual_chunk_span
                            ):
                                pass
            
            # Select best chunk
            if not chunk_scores:
                # No good matches, return first chunk with low confidence
                first_chunk_id = next(iter(available_chunks.keys())) if available_chunks else None
                
                # Log no matches in trace
                with custom_span(
                    name="no_chunk_matches",
                    data={"first_chunk_id": first_chunk_id}
                ):
                    pass
                
                prediction = ChunkPrediction(
                    chunk_id=first_chunk_id,
                    confidence=0.1,
                    context_match_score=0.1,
                    reasoning=["No context patterns matched"]
                )
            else:
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
                
                # Log selection result in trace
                with custom_span(
                    name=f"selected_chunk_{best_chunk_id}",
                    data={
                        "chunk_id": best_chunk_id,
                        "score": best_score,
                        "confidence": min(1.0, best_score),
                        "alternatives_count": len(alternatives)
                    }
                ):
                    pass
            
            # Record selection for future learning
            self._record_selection(prediction, context)
            
            return prediction
    
    async def _calculate_pattern_match_async(
        self, 
        pattern: ContextPattern, 
        context: Dict[str, Any],
        parent_span: Any = None
    ) -> float:
        """Asynchronous version of pattern matching for parallel processing"""
        # Create trace span for pattern match calculation
        with custom_span(
            name=f"calculate_pattern_match_{pattern.id}",
            data={
                "pattern_id": pattern.id,
                "pattern_name": pattern.name,
                "indicators_count": len(pattern.indicators)
            },
            parent=parent_span
        ) as match_span:
            # Check each indicator in the pattern
            indicator_matches = 0
            total_indicators = len(pattern.indicators)
            
            if total_indicators == 0:
                return 0.0
                
            for indicator, expected_value in pattern.indicators.items():
                # Skip if indicator not in context
                if indicator not in context:
                    continue
                    
                actual_value = context[indicator]
                matched = False
                
                # Different comparison based on type
                if isinstance(expected_value, (list, tuple, set)):
                    # Check if value is in list
                    if actual_value in expected_value:
                        indicator_matches += 1
                        matched = True
                elif isinstance(expected_value, dict) and "min" in expected_value and "max" in expected_value:
                    # Range check
                    if expected_value["min"] <= actual_value <= expected_value["max"]:
                        indicator_matches += 1
                        matched = True
                else:
                    # Direct equality check
                    if actual_value == expected_value:
                        indicator_matches += 1
                        matched = True
                
                # Log indicator match in trace
                with custom_span(
                    name=f"indicator_match_{indicator}",
                    data={
                        "indicator": indicator,
                        "expected": str(expected_value)[:100],
                        "actual": str(actual_value)[:100],
                        "matched": matched
                    },
                    parent=match_span
                ):
                    pass
            
            # Base match score on percentage of matching indicators
            base_score = indicator_matches / total_indicators
            
            # Check temporal pattern if defined
            temporal_score = 0.0
            if pattern.temporal_pattern and "action_history" in context:
                action_history = context["action_history"]
                
                # Create span for temporal pattern matching
                with custom_span(
                    name=f"temporal_pattern_match_{pattern.id}",
                    data={
                        "pattern_id": pattern.id,
                        "temporal_items_count": len(pattern.temporal_pattern),
                        "history_items_count": len(action_history)
                    },
                    parent=match_span
                ) as temporal_span:
                    temporal_score = await self._match_temporal_pattern(
                        pattern.temporal_pattern, 
                        action_history,
                        parent_span=temporal_span
                    )
                
                # Combine scores, giving more weight to direct indicators
                final_score = base_score * 0.7 + temporal_score * 0.3
            else:
                final_score = base_score
            
            # Log final score in trace
            with custom_span(
                name=f"pattern_match_score_{pattern.id}",
                data={
                    "base_score": base_score,
                    "temporal_score": temporal_score,
                    "final_score": final_score
                },
                parent=match_span
            ):
                pass
            
            return final_score
    
    async def _match_temporal_pattern(
        self, 
        pattern: List[Dict[str, Any]], 
        history: List[Dict[str, Any]],
        parent_span: Any = None
    ) -> float:
        """Match a temporal pattern against action history"""
        # Skip if history is too short
        if len(history) < len(pattern):
            return 0.0
            
        # Check pattern against most recent history
        recent_history = history[-len(pattern):]
        
        # Create span for temporal pattern matching
        with custom_span(
            name="temporal_pattern_matching",
            data={
                "pattern_length": len(pattern),
                "history_length": len(recent_history)
            },
            parent=parent_span
        ) as temporal_span:
            # Count matching items
            matches = 0
            match_details = []
            
            for i, expected in enumerate(pattern):
                if i >= len(recent_history):
                    break
                    
                actual = recent_history[i]
                
                # Count matching keys
                matching_keys = 0
                total_keys = len(expected)
                
                # Skip if no keys to match
                if total_keys == 0:
                    continue
                
                for key, expected_value in expected.items():
                    key_matched = False
                    if key in actual and actual[key] == expected_value:
                        matching_keys += 1
                        key_matched = True
                    
                    # Log key match in trace
                    with custom_span(
                        name=f"temporal_key_match_{i}_{key}",
                        data={
                            "key": key,
                            "expected": str(expected_value)[:100],
                            "actual": str(actual.get(key, "missing"))[:100],
                            "matched": key_matched
                        },
                        parent=temporal_span
                    ):
                        pass
                
                # Consider a step matching if most keys match
                step_match_ratio = matching_keys / total_keys
                step_matched = step_match_ratio >= 0.7
                
                if step_matched:
                    matches += 1
                
                match_details.append({
                    "step": i,
                    "matching_keys": matching_keys,
                    "total_keys": total_keys,
                    "matched": step_matched,
                    "match_ratio": step_match_ratio
                })
            
            # Calculate match percentage
            match_percentage = matches / len(pattern) if len(pattern) > 0 else 0.0
            
            # Log match percentage in trace
            with custom_span(
                name="temporal_pattern_match_result",
                data={
                    "matches": matches,
                    "pattern_length": len(pattern),
                    "match_percentage": match_percentage,
                    "match_details": match_details
                },
                parent=temporal_span
            ):
                pass
            
            return match_percentage
    
    async def _check_history_for_chunk(self, chunk_id: str, context: Dict[str, Any]) -> float:
        """Check if similar contexts in history used this chunk"""
        if not self.recent_contexts or not self.recent_selections:
            return 0.0
            
        # Create span for history check
        with custom_span(
            name=f"history_check_{chunk_id}",
            data={
                "chunk_id": chunk_id,
                "contexts_count": len(self.recent_contexts),
                "selections_count": len(self.recent_selections)
            }
        ) as history_span:
            # Find similarity scores between current context and historical contexts
            similarity_scores = []
            
            for i, historical_context in enumerate(self.recent_contexts):
                if i >= len(self.recent_selections):
                    break
                    
                # Calculate context similarity
                similarity = await self._calculate_context_similarity(
                    context, 
                    historical_context,
                    parent_span=history_span
                )
                
                # Check if this chunk was selected
                selection = self.recent_selections[i]
                if selection.chunk_id == chunk_id:
                    weighted_similarity = similarity * selection.confidence
                    similarity_scores.append(weighted_similarity)
                    
                    # Log similarity in trace
                    with custom_span(
                        name=f"historical_similarity_{chunk_id}_{i}",
                        data={
                            "raw_similarity": similarity,
                            "selection_confidence": selection.confidence,
                            "weighted_similarity": weighted_similarity
                        },
                        parent=history_span
                    ):
                        pass
            
            # Return max similarity if any
            max_similarity = max(similarity_scores) if similarity_scores else 0.0
            
            # Log result in trace
            with custom_span(
                name=f"history_check_result_{chunk_id}",
                data={
                    "max_similarity": max_similarity,
                    "similarity_scores_count": len(similarity_scores)
                },
                parent=history_span
            ):
                pass
            
            return max_similarity
    
    async def _calculate_context_similarity(
        self, 
        context1: Dict[str, Any], 
        context2: Dict[str, Any],
        parent_span: Any = None
    ) -> float:
        """Calculate similarity between two contexts"""
        # Create cache key
        cache_key = (
            hash(frozenset(context1.items())),
            hash(frozenset(context2.items()))
        )
        
        # Check cache first
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        # Create span for context similarity calculation
        with custom_span(
            name="calculate_context_similarity",
            data={
                "context1_keys": len(context1),
                "context2_keys": len(context2)
            },
            parent=parent_span
        ) as similarity_span:
            # Get common keys
            common_keys = set(context1.keys()) & set(context2.keys())
            
            if not common_keys:
                return 0.0
                
            # Count matching values
            matching_values = 0
            
            for key in common_keys:
                key_match = context1[key] == context2[key]
                if key_match:
                    matching_values += 1
                
                # Log key match in trace
                with custom_span(
                    name=f"context_key_match_{key}",
                    data={
                        "key": key,
                        "matched": key_match
                    },
                    parent=similarity_span
                ):
                    pass
            
            # Calculate similarity
            similarity = matching_values / len(common_keys) if common_keys else 0.0
            
            # Cache the result
            self.similarity_cache[cache_key] = similarity
            
            # Log similarity result in trace
            with custom_span(
                name="context_similarity_result",
                data={
                    "common_keys": len(common_keys),
                    "matching_values": matching_values,
                    "similarity": similarity
                },
                parent=similarity_span
            ):
                pass
            
            return similarity
    
    def _record_context(self, context: Dict[str, Any]):
        """Record context for future reference"""
        # Create trace span for recording context
        with custom_span(
            name="record_context",
            data={
                "context_keys": len(context),
                "contexts_history_size": len(self.recent_contexts)
            }
        ):
            self.recent_contexts.append(context.copy())
            
            # Trim history if needed
            if len(self.recent_contexts) > self.max_history:
                self.recent_contexts = self.recent_contexts[-self.max_history:]
    
    def _record_selection(self, prediction: ChunkPrediction, context: Dict[str, Any]):
        """Record chunk selection for future reference"""
        # Create trace span for recording selection
        with custom_span(
            name=f"record_selection_{prediction.chunk_id}",
            data={
                "chunk_id": prediction.chunk_id,
                "confidence": prediction.confidence,
                "context_match_score": prediction.context_match_score,
                "selections_history_size": len(self.recent_selections)
            }
        ):
            self.recent_selections.append(prediction)
            
            # Trim history if needed
            if len(self.recent_selections) > self.max_history:
                self.recent_selections = self.recent_selections[-self.max_history:]
    
    async def create_context_pattern_from_history(
        self, 
        chunk_id: str, 
        domain: str
    ) -> Optional[ContextPattern]:
        """Automatically create a context pattern from historical data"""
        # Create trace span for pattern creation
        with custom_span(
            name=f"create_pattern_from_history_{chunk_id}",
            data={
                "chunk_id": chunk_id,
                "domain": domain,
                "contexts_count": len(self.recent_contexts),
                "selections_count": len(self.recent_selections)
            }
        ) as creation_span:
            # Need sufficient history
            if len(self.recent_contexts) < 5 or len(self.recent_selections) < 5:
                # Log insufficient history in trace
                with custom_span(
                    name="insufficient_history",
                    data={
                        "required_min": 5,
                        "contexts_count": len(self.recent_contexts),
                        "selections_count": len(self.recent_selections)
                    },
                    parent=creation_span
                ):
                    pass
                
                return None
                
            # Find instances where this chunk was selected
            instances = []
            
            for i, selection in enumerate(self.recent_selections):
                if selection.chunk_id == chunk_id and i < len(self.recent_contexts):
                    instances.append(self.recent_contexts[i])
            
            if len(instances) < 3:
                # Log insufficient instances in trace
                with custom_span(
                    name="insufficient_instances",
                    data={
                        "required_min": 3,
                        "instances_count": len(instances)
                    },
                    parent=creation_span
                ):
                    pass
                
                return None
                
            # Find common indicators across contexts
            common_indicators = {}
            
            # Get all keys present in at least half of instances
            all_keys = set()
            for context in instances:
                all_keys.update(context.keys())
                
            # Create span for indicator extraction
            with custom_span(
                name="extract_common_indicators",
                data={
                    "all_keys_count": len(all_keys),
                    "instances_count": len(instances)
                },
                parent=creation_span
            ) as indicators_span:
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
                        
                        # Log exact match indicator in trace
                        with custom_span(
                            name=f"exact_match_indicator_{key}",
                            data={
                                "key": key,
                                "value": str(values[0])[:100],
                                "match_type": "exact"
                            },
                            parent=indicators_span
                        ):
                            pass
                    elif all(isinstance(v, (int, float)) for v in values):
                        # Numeric values - use range
                        common_indicators[key] = {
                            "min": min(values),
                            "max": max(values)
                        }
                        
                        # Log range indicator in trace
                        with custom_span(
                            name=f"range_indicator_{key}",
                            data={
                                "key": key,
                                "min": min(values),
                                "max": max(values),
                                "match_type": "range"
                            },
                            parent=indicators_span
                        ):
                            pass
                    elif len(set(values)) < len(values) / 2:
                        # Some consistent values - use set of common values
                        counter = Counter(values)
                        common = [v for v, count in counter.items() if count > 1]
                        if common:
                            common_indicators[key] = common
                            
                            # Log set indicator in trace
                            with custom_span(
                                name=f"set_indicator_{key}",
                                data={
                                    "key": key,
                                    "values": [str(v)[:50] for v in common],
                                    "values_count": len(common),
                                    "match_type": "set"
                                },
                                parent=indicators_span
                            ):
                                pass
            
            # Extract temporal pattern if action history is present
            temporal_pattern = None
            if all("action_history" in context for context in instances):
                # Create span for temporal pattern extraction
                with custom_span(
                    name="extract_temporal_pattern",
                    data={"instances_count": len(instances)},
                    parent=creation_span
                ) as temporal_span:
                    # Get minimum sequence length that works for all instances
                    min_seq_length = min(len(context.get("action_history", [])) for context in instances)
                    
                    # Skip if sequences are too short
                    if min_seq_length >= 3:
                        # Analyze the last N steps before chunk selection
                        last_n = 3  # Start with 3 steps
                        
                        # Get last N actions from each instance
                        action_sequences = [
                            context["action_history"][-last_n:] 
                            for context in instances
                            if len(context.get("action_history", [])) >= last_n
                        ]
                        
                        # Find common action patterns
                        if action_sequences:
                            # Start with first sequence as template
                            template = action_sequences[0]
                            
                            # Find common elements across all sequences
                            common_elements = []
                            
                            for i in range(last_n):
                                # Get elements at this position from all sequences
                                position_elements = [seq[i] for seq in action_sequences]
                                
                                # Find common keys and values
                                common_keys = set.intersection(*(set(elem.keys()) for elem in position_elements))
                                
                                if common_keys:
                                    # Build common element with matching values
                                    common_element = {}
                                    
                                    for key in common_keys:
                                        # Get all values for this key
                                        key_values = [elem[key] for elem in position_elements]
                                        
                                        # Check if values are consistent
                                        if all(v == key_values[0] for v in key_values):
                                            # All values the same
                                            common_element[key] = key_values[0]
                                    
                                    if common_element:
                                        common_elements.append(common_element)
                            
                            # Use as temporal pattern if we found common elements
                            if common_elements:
                                temporal_pattern = common_elements
                                
                                # Log temporal pattern in trace
                                with custom_span(
                                    name="extracted_temporal_pattern",
                                    data={
                                        "elements_count": len(common_elements),
                                        "keys_per_element": [len(elem) for elem in common_elements]
                                    },
                                    parent=temporal_span
                                ):
                                    pass
            
            # Create pattern ID and name
            pattern_id = f"auto_pattern_{chunk_id}_{len(self.context_patterns)}"
            pattern_name = f"Auto-generated pattern for {chunk_id}"
            
            # Make sure we have the chunk association indicator
            common_indicators[f"chunk_{chunk_id}_suitable"] = True
            
            # Create pattern
            pattern = ContextPattern(
                id=pattern_id,
                name=pattern_name,
                domain=domain,
                indicators=common_indicators,
                temporal_pattern=temporal_pattern or [],
                confidence_threshold=0.7,
                match_count=0,
                last_matched=None
            )
            
            # Register pattern
            self.register_context_pattern(pattern)
            
            # Log created pattern in trace
            with custom_span(
                name="pattern_created",
                data={
                    "pattern_id": pattern_id,
                    "indicators_count": len(common_indicators),
                    "has_temporal_pattern": temporal_pattern is not None
                },
                parent=creation_span
            ):
                pass
            
            return pattern
    
    def update_pattern_performance(self, pattern_id: str, success: bool) -> None:
        """Update performance metrics for a pattern"""
        if pattern_id not in self.pattern_performance:
            return
            
        # Create trace span for updating performance
        with custom_span(
            name=f"update_pattern_performance_{pattern_id}",
            data={
                "pattern_id": pattern_id,
                "success": success
            }
        ):
            performance = self.pattern_performance[pattern_id]
            
            # Update success count
            if success:
                performance["success_count"] += 1
            
            # Update success rate
            if performance["match_count"] > 0:
                performance["success_rate"] = performance["success_count"] / performance["match_count"]
            
            # Update timestamp
            performance["last_used"] = datetime.datetime.now().isoformat()
            
            logger.debug(f"Updated pattern performance for {pattern_id}: success_rate={performance['success_rate']:.2f}")
    
    def get_pattern_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all patterns"""
        # Create trace span for getting performance
        with custom_span(
            name="get_pattern_performance",
            data={"patterns_count": len(self.pattern_performance)}
        ):
            return self.pattern_performance.copy()

# Function tools for use with Agent SDK
@function_tool
async def create_context_pattern(
    ctx,
    name: str,
    domain: str,
    indicators: Dict[str, Any],
    chunk_id: str,
    temporal_pattern: Optional[List[Dict[str, Any]]] = None,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Create a new context pattern for chunk selection
    
    Args:
        name: Name for the pattern
        domain: Domain where the pattern applies
        indicators: Dictionary of indicators that trigger the pattern
        chunk_id: ID of the chunk this pattern is associated with
        temporal_pattern: Optional temporal pattern of actions
        confidence_threshold: Minimum confidence required to activate pattern
        
    Returns:
        Information about the created pattern
    """
    # Access chunk selector from context
    chunk_selector = ctx.context.manager.chunk_selector if hasattr(ctx.context, "manager") else None
    
    if not chunk_selector:
        return {
            "success": False,
            "error": "Chunk selector not available in context"
        }
    
    # Create trace span for pattern creation
    with custom_span(
        name="create_context_pattern",
        data={
            "name": name,
            "domain": domain,
            "indicators_count": len(indicators),
            "chunk_id": chunk_id
        }
    ):
        # Add chunk association indicator
        indicators[f"chunk_{chunk_id}_suitable"] = True
        
        # Generate pattern ID
        pattern_id = f"pattern_{int(datetime.datetime.now().timestamp())}_{chunk_id}"
        
        # Create pattern
        pattern = ContextPattern(
            id=pattern_id,
            name=name,
            domain=domain,
            indicators=indicators,
            temporal_pattern=temporal_pattern or [],
            confidence_threshold=confidence_threshold,
            match_count=0,
            last_matched=None
        )
        
        # Register pattern
        chunk_selector.register_context_pattern(pattern)
        
        return {
            "success": True,
            "pattern_id": pattern_id,
            "name": name,
            "domain": domain,
            "indicators_count": len(indicators),
            "has_temporal_pattern": temporal_pattern is not None,
            "chunk_id": chunk_id
        }

@function_tool
async def get_pattern_performance_metrics(ctx, pattern_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get performance metrics for patterns
    
    Args:
        pattern_id: Optional ID of a specific pattern to get metrics for
        
    Returns:
        Performance metrics for the pattern(s)
    """
    # Access chunk selector from context
    chunk_selector = ctx.context.manager.chunk_selector if hasattr(ctx.context, "manager") else None
    
    if not chunk_selector:
        return {
            "success": False,
            "error": "Chunk selector not available in context"
        }
    
    # Create trace span for getting metrics
    with custom_span(
        name="get_pattern_performance_metrics",
        data={"pattern_id": pattern_id}
    ):
        if pattern_id:
            # Get metrics for specific pattern
            if pattern_id not in chunk_selector.pattern_performance:
                return {
                    "success": False,
                    "error": f"Pattern '{pattern_id}' not found"
                }
                
            return {
                "success": True,
                "pattern_id": pattern_id,
                "metrics": chunk_selector.pattern_performance[pattern_id]
            }
        else:
            # Get metrics for all patterns
            all_metrics = chunk_selector.get_pattern_performance()
            
            # Calculate summary metrics
            total_match_count = sum(metrics["match_count"] for metrics in all_metrics.values())
            total_success_count = sum(metrics["success_count"] for metrics in all_metrics.values())
            
            avg_success_rate = 0.0
            if total_match_count > 0:
                avg_success_rate = total_success_count / total_match_count
            
            return {
                "success": True,
                "patterns_count": len(all_metrics),
                "total_match_count": total_match_count,
                "total_success_count": total_success_count,
                "avg_success_rate": avg_success_rate,
                "patterns": {
                    pattern_id: {
                        "match_count": metrics["match_count"],
                        "success_count": metrics["success_count"],
                        "success_rate": metrics["success_rate"],
                        "last_used": metrics["last_used"]
                    }
                    for pattern_id, metrics in all_metrics.items()
                }
            }

@function_tool
async def analyze_context_patterns(ctx, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze context patterns for efficiency and effectiveness
    
    Args:
        domain: Optional domain to filter patterns by
        
    Returns:
        Analysis of patterns including usage and success metrics
    """
    # Access chunk selector from context
    chunk_selector = ctx.context.manager.chunk_selector if hasattr(ctx.context, "manager") else None
    
    if not chunk_selector:
        return {
            "success": False,
            "error": "Chunk selector not available in context"
        }
    
    # Create trace span for analysis
    with custom_span(
        name="analyze_context_patterns",
        data={"domain": domain}
    ):
        # Filter patterns by domain if specified
        if domain:
            pattern_ids = chunk_selector.domain_specific_patterns.get(domain, [])
            patterns = {
                pid: chunk_selector.context_patterns[pid] 
                for pid in pattern_ids 
                if pid in chunk_selector.context_patterns
            }
        else:
            patterns = chunk_selector.context_patterns
        
        if not patterns:
            return {
                "success": True,
                "patterns_count": 0,
                "message": f"No patterns found for domain: {domain}" if domain else "No patterns found"
            }
        
        # Get performance metrics
        all_metrics = chunk_selector.get_pattern_performance()
        
        # Filter metrics for selected patterns
        filtered_metrics = {
            pid: all_metrics.get(pid, {})
            for pid in patterns.keys()
        }
        
        # Calculate efficiency metrics
        efficiency_metrics = {}
        
        for pid, pattern in patterns.items():
            metrics = filtered_metrics.get(pid, {})
            indicators_count = len(pattern.indicators)
            has_temporal = bool(pattern.temporal_pattern)
            success_rate = metrics.get("success_rate", 0.0)
            match_count = metrics.get("match_count", 0)
            
            # Calculate effectiveness score
            effectiveness = success_rate * 0.7 + (match_count / 100 if match_count <= 100 else 1.0) * 0.3
            
            # Calculate complexity score (lower is better)
            complexity = (indicators_count / 20 if indicators_count <= 20 else 1.0) * 0.7
            complexity += 0.3 if has_temporal else 0.0
            
            # Calculate overall efficiency (higher is better)
            efficiency = effectiveness * 0.8 - complexity * 0.2
            
            efficiency_metrics[pid] = {
                "name": pattern.name,
                "effectiveness": effectiveness,
                "complexity": complexity,
                "efficiency": efficiency,
                "success_rate": success_rate,
                "match_count": match_count,
                "indicators_count": indicators_count,
                "has_temporal_pattern": has_temporal
            }
        
        # Sort patterns by efficiency
        sorted_patterns = sorted(
            efficiency_metrics.items(),
            key=lambda x: x[1]["efficiency"],
            reverse=True
        )
        
        return {
            "success": True,
            "patterns_count": len(patterns),
            "domain": domain,
            "top_patterns": [
                {
                    "pattern_id": pid,
                    "name": metrics["name"],
                    "efficiency": metrics["efficiency"],
                    "success_rate": metrics["success_rate"],
                    "match_count": metrics["match_count"]
                }
                for pid, metrics in sorted_patterns[:5]
            ],
            "inefficient_patterns": [
                {
                    "pattern_id": pid,
                    "name": metrics["name"],
                    "efficiency": metrics["efficiency"],
                    "success_rate": metrics["success_rate"],
                    "match_count": metrics["match_count"]
                }
                for pid, metrics in sorted_patterns[-5:] if metrics["efficiency"] < 0.3
            ],
            "avg_efficiency": sum(m["efficiency"] for m in efficiency_metrics.values()) / len(efficiency_metrics) if efficiency_metrics else 0.0,
            "avg_success_rate": sum(m["success_rate"] for m in efficiency_metrics.values()) / len(efficiency_metrics) if efficiency_metrics else 0.0
        }
