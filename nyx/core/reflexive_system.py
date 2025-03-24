# nyx/core/reflexive_system.py

import asyncio
import logging
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import datetime
import numpy as np

from agents import RunContextWrapper, Agent, function_tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ReflexPattern:
    """Pattern that can trigger a reflexive response"""
    
    def __init__(self, 
                 name: str,
                 pattern_data: Dict[str, Any],
                 procedure_name: str,
                 threshold: float = 0.7,
                 priority: int = 1,
                 context_template: Dict[str, Any] = None):
        """
        Initialize a reflex pattern
        
        Args:
            name: Unique name for this pattern
            pattern_data: Pattern definition (features that should trigger this reflex)
            procedure_name: Name of procedure to execute when triggered
            threshold: Matching threshold (0.0-1.0)
            priority: Priority level (higher values take precedence)
            context_template: Template for context to pass to procedure
        """
        self.name = name
        self.pattern_data = pattern_data
        self.procedure_name = procedure_name
        self.threshold = threshold
        self.priority = priority
        self.context_template = context_template or {}
        
        # Performance tracking
        self.execution_count = 0
        self.success_count = 0
        self.avg_execution_time = 0.0
        self.last_executed = None
        self.response_times = []  # Store last 10 response times in ms
        
    def track_execution(self, success: bool, execution_time: float):
        """Track execution statistics"""
        self.execution_count += 1
        if success:
            self.success_count += 1
        
        # Update average execution time
        if self.execution_count == 1:
            self.avg_execution_time = execution_time
        else:
            self.avg_execution_time = (self.avg_execution_time * (self.execution_count - 1) + execution_time) / self.execution_count
        
        self.last_executed = datetime.datetime.now()
        
        # Store response time (in milliseconds)
        self.response_times.append(execution_time * 1000)
        if len(self.response_times) > 10:
            self.response_times.pop(0)
    
    def get_success_rate(self) -> float:
        """Get success rate of this reflex pattern"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def get_avg_response_time(self) -> float:
        """Get average response time in milliseconds"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "procedure_name": self.procedure_name,
            "threshold": self.threshold,
            "priority": self.priority,
            "stats": {
                "execution_count": self.execution_count,
                "success_rate": self.get_success_rate(),
                "avg_response_time_ms": self.get_avg_response_time(),
                "last_executed": self.last_executed.isoformat() if self.last_executed else None
            }
        }

class ReflexDecisionSystem:
    """System to determine when reflexes should be used vs. deliberate thinking"""
    
    def __init__(self):
        """Initialize the decision system"""
        self.context_patterns = {
            "urgent": 0.9,  # High likelihood of using reflexes
            "time_sensitive": 0.8,
            "gaming": 0.9,
            "conversation": 0.3,
            "creative": 0.1,
            "analytical": 0.1,
            "emotional": 0.7
        }
        
        # Learn from experience
        self.context_history = {}
        self.success_rate_by_context = {}
    
    def should_use_reflex(self, stimulus, context=None, domain=None) -> tuple[bool, float]:
        """
        Determine whether to use reflexes or deliberate thinking
        
        Args:
            stimulus: The input stimulus
            context: Additional context information
            domain: Optional domain to limit patterns
            
        Returns:
            Tuple of (use_reflex, confidence)
        """
        # Default confidence threshold
        confidence_threshold = 0.6
        
        # Domain-specific considerations
        if domain == "gaming":
            confidence_threshold = 0.4  # Lower threshold for gaming
        elif domain == "analytical":
            confidence_threshold = 0.8  # Higher threshold for analytical tasks
        
        # Context modifiers
        if context:
            if "urgent" in context and context["urgent"]:
                confidence_threshold -= 0.2
            
            if "creative" in context and context["creative"]:
                confidence_threshold += 0.3
        
        # Stimulus features that suggest reflexes
        reflex_indicators = {
            "time_sensitive": "timing" in stimulus or "_timing" in stimulus,
            "pattern_match": len(stimulus) > 2,
            "simple_structure": isinstance(stimulus, dict) and all(isinstance(v, (str, int, float, bool)) for v in stimulus.values()),
            "familiar": False  # Will be set based on history
        }
        
        # Check history for similar stimuli
        stimulus_key = str(sorted(stimulus.items()))
        if stimulus_key in self.context_history:
            reflex_indicators["familiar"] = True
            # If reflexes worked well historically, lower threshold
            if self.success_rate_by_context.get(stimulus_key, 0) > 0.7:
                confidence_threshold -= 0.1
        
        # Calculate confidence score
        confidence_score = 0.0
        active_indicators = 0
        
        for indicator, active in reflex_indicators.items():
            if active:
                confidence_score += 0.25  # Each indicator adds 25%
                active_indicators += 1
        
        if active_indicators > 0:
            # Average the score
            confidence_score = confidence_score / active_indicators
        
        # Decision
        use_reflex = confidence_score >= confidence_threshold
        
        return (use_reflex, confidence_score)
    
    def update_from_result(self, stimulus, used_reflex, success):
        """Update decision system based on result"""
        stimulus_key = str(sorted(stimulus.items()))
        if stimulus_key not in self.context_history:
            self.context_history[stimulus_key] = {"reflex": 0, "deliberate": 0}
            self.success_rate_by_context[stimulus_key] = 0.0
        
        # Track usage
        if used_reflex:
            self.context_history[stimulus_key]["reflex"] += 1
        else:
            self.context_history[stimulus_key]["deliberate"] += 1
        
        # Update success rate
        history = self.context_history[stimulus_key]
        total = history["reflex"] + history["deliberate"]
        success_count = total * self.success_rate_by_context.get(stimulus_key, 0.5)
        
        if success:
            success_count += 1
        
        self.success_rate_by_context[stimulus_key] = success_count / (total + 1)

class ReflexiveSystem:
    """
    System for rapid, reflexive responses without deliberate thinking.
    
    This system manages stimulus-response patterns that can trigger immediate
    execution of procedures, bypassing the normal thinking process for extremely
    fast reactions - similar to muscle memory or reflexes in humans.
    """
    
    def __init__(self, agent_enhanced_memory):
        """
        Initialize the reflexive system
        
        Args:
            agent_enhanced_memory: Reference to the agent enhanced memory system
        """
        self.memory_manager = agent_enhanced_memory
        self.reflex_patterns = {}  # name -> ReflexPattern
        
        # Reaction time tracking
        self.reaction_times = []  # Track last 100 reaction times in ms
        self._is_active = True
        
        # Specialized reaction libraries (domain-specific)
        self.domain_libraries = {
            "gaming": {},
            "conversation": {},
            "physical": {},
            "decision": {}
        }
        
        # Response modes
        self.response_mode = "normal"  # normal, hyper, relaxed, learning
        
        # Pattern recognition system
        self.pattern_recognition = SimplePatternRecognition()
        
        # Decision system
        self.decision_system = ReflexDecisionSystem()
        
        # For training
        self.training_data = []  # Store stimulus-response pairs for training
        self.training_in_progress = False
        
        logger.info("Reflexive system initialized")
    
    async def register_reflex(self, 
                             name: str,
                             pattern_data: Dict[str, Any],
                             procedure_name: str,
                             threshold: float = 0.7,
                             priority: int = 1,
                             domain: str = None,
                             context_template: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register a new reflex pattern
        
        Args:
            name: Unique name for this pattern
            pattern_data: Pattern definition (features that should trigger this reflex)
            procedure_name: Name of procedure to execute when triggered
            threshold: Matching threshold (0.0-1.0)
            priority: Priority level (higher values take precedence)
            domain: Optional domain for specialized responses
            context_template: Template for context to pass to procedure
            
        Returns:
            Registration result
        """
        # Validate procedure exists
        procedures = await self.memory_manager.list_procedures()
        if procedure_name not in [p["name"] for p in procedures]:
            return {
                "success": False,
                "error": f"Procedure '{procedure_name}' not found"
            }
        
        # Create reflex pattern
        reflex = ReflexPattern(
            name=name,
            pattern_data=pattern_data,
            procedure_name=procedure_name,
            threshold=threshold,
            priority=priority,
            context_template=context_template
        )
        
        # Store in main registry
        self.reflex_patterns[name] = reflex
        
        # If domain specified, add to domain library
        if domain and domain in self.domain_libraries:
            self.domain_libraries[domain][name] = reflex
        
        logger.info(f"Registered reflex pattern '{name}' for procedure '{procedure_name}'")
        
        return {
            "success": True,
            "reflex": reflex.to_dict()
        }
    
    async def process_stimulus_fast(self, 
                                  stimulus: Dict[str, Any],
                                  domain: str = None,
                                  context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process stimulus with minimal overhead for fastest possible reaction
        
        This method uses a streamlined matching process for absolute minimum latency.
        
        Args:
            stimulus: The stimulus data requiring immediate reaction
            domain: Optional domain to limit reflex patterns
            context: Additional context information
            
        Returns:
            Reaction result
        """
        if not self._is_active:
            return {"success": False, "reason": "reflexive_system_inactive"}
        
        start_time = time.time()
        matched_pattern = None
        highest_match = 0.0
        highest_priority = -1
        
        # Determine patterns to check
        patterns_to_check = self.reflex_patterns
        if domain and domain in self.domain_libraries:
            patterns_to_check = self.domain_libraries[domain]
        
        # Fast pattern matching
        for name, pattern in patterns_to_check.items():
            match_score = self.pattern_recognition.fast_match(stimulus, pattern.pattern_data)
            
            if match_score >= pattern.threshold:
                # Check if this is higher priority or better match
                if (pattern.priority > highest_priority or 
                    (pattern.priority == highest_priority and match_score > highest_match)):
                    matched_pattern = pattern
                    highest_match = match_score
                    highest_priority = pattern.priority
        
        # If no match found, return quickly
        if not matched_pattern:
            end_time = time.time()
            self.reaction_times.append((end_time - start_time) * 1000)
            if len(self.reaction_times) > 100:
                self.reaction_times.pop(0)
                
            return {
                "success": False,
                "reason": "no_matching_pattern",
                "reaction_time_ms": (end_time - start_time) * 1000
            }
        
        # Prepare execution context
        execution_context = matched_pattern.context_template.copy() if matched_pattern.context_template else {}
        if context:
            execution_context.update(context)
        
        # Add stimulus data and match details
        execution_context["stimulus"] = stimulus
        execution_context["match_score"] = highest_match
        execution_context["reflexive"] = True
        execution_context["reaction_start_time"] = start_time
        
        # Execute procedure with minimal overhead
        try:
            result = await self.memory_manager.execute_procedure(
                matched_pattern.procedure_name,
                context=execution_context,
                force_conscious=False,  # Ensure non-conscious execution
                priority="critical"  # Set highest priority
            )
            
            # Track execution time
            end_time = time.time()
            execution_time = end_time - start_time
            matched_pattern.track_execution(
                success=result.get("success", False),
                execution_time=execution_time
            )
            
            # Record reaction time
            reaction_time_ms = execution_time * 1000
            self.reaction_times.append(reaction_time_ms)
            if len(self.reaction_times) > 100:
                self.reaction_times.pop(0)
            
            # For training purposes
            if self.response_mode == "learning":
                self.training_data.append({
                    "stimulus": stimulus,
                    "pattern_name": matched_pattern.name,
                    "reaction_time_ms": reaction_time_ms,
                    "success": result.get("success", False)
                })
            
            # Add reaction metadata to result
            result["reaction"] = {
                "pattern_name": matched_pattern.name,
                "reaction_time_ms": reaction_time_ms,
                "match_score": highest_match
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing reflex pattern '{matched_pattern.name}': {e}")
            
            # Still track execution for failed attempts
            end_time = time.time()
            matched_pattern.track_execution(
                success=False,
                execution_time=end_time - start_time
            )
            
            return {
                "success": False,
                "error": str(e),
                "pattern_name": matched_pattern.name,
                "reaction_time_ms": (end_time - start_time) * 1000
            }
    
    async def train_reflexes(self, 
                           training_iterations: int = 100,
                           domain: str = None) -> Dict[str, Any]:
        """
        Train reflexes to improve response time and accuracy
        
        Args:
            training_iterations: Number of training iterations to perform
            domain: Optional domain to limit training
            
        Returns:
            Training results
        """
        if self.training_in_progress:
            return {"success": False, "error": "Training already in progress"}
        
        self.training_in_progress = True
        
        try:
            # Get patterns to train
            patterns_to_train = self.reflex_patterns
            if domain and domain in self.domain_libraries:
                patterns_to_train = self.domain_libraries[domain]
            
            if not patterns_to_train:
                return {"success": False, "error": "No patterns to train"}
            
            # Track improvements
            original_stats = {}
            for name, pattern in patterns_to_train.items():
                original_stats[name] = {
                    "avg_response_time": pattern.get_avg_response_time(),
                    "success_rate": pattern.get_success_rate()
                }
            
            # Generate training stimuli
            training_stimuli = []
            for _ in range(training_iterations):
                # Generate stimulus similar to patterns we're training
                pattern = random.choice(list(patterns_to_train.values()))
                stimulus = self.pattern_recognition.generate_similar_stimulus(pattern.pattern_data)
                training_stimuli.append({
                    "stimulus": stimulus,
                    "target_pattern": pattern.name
                })
            
            # Run training iterations
            results = []
            original_mode = self.response_mode
            self.response_mode = "learning"
            
            for training_item in training_stimuli:
                stimulus = training_item["stimulus"]
                target_pattern = training_item["target_pattern"]
                
                # Process stimulus
                result = await self.process_stimulus_fast(stimulus)
                
                # Record result
                results.append({
                    "target_pattern": target_pattern,
                    "matched_pattern": result.get("pattern_name"),
                    "reaction_time_ms": result.get("reaction_time_ms"),
                    "success": result.get("success", False)
                })
            
            # Restore original mode
            self.response_mode = original_mode
            
            # Calculate improvements
            improvements = {}
            for name, pattern in patterns_to_train.items():
                current_stats = {
                    "avg_response_time": pattern.get_avg_response_time(),
                    "success_rate": pattern.get_success_rate()
                }
                
                # Calculate improvements
                if name in original_stats:
                    time_improvement = original_stats[name]["avg_response_time"] - current_stats["avg_response_time"]
                    success_improvement = current_stats["success_rate"] - original_stats[name]["success_rate"]
                    
                    improvements[name] = {
                        "response_time_improvement_ms": time_improvement,
                        "success_rate_improvement": success_improvement,
                        "current_stats": current_stats
                    }
            
            # Apply automatic threshold adjustments based on training
            for name, pattern in patterns_to_train.items():
                # Adjust threshold based on success rate
                if pattern.execution_count >= 10:
                    success_rate = pattern.get_success_rate()
                    if success_rate < 0.6 and pattern.threshold > 0.6:
                        # Lower threshold slightly if success rate is low
                        pattern.threshold = max(0.6, pattern.threshold - 0.05)
                    elif success_rate > 0.9 and pattern.threshold < 0.9:
                        # Increase threshold if success rate is very high
                        pattern.threshold = min(0.9, pattern.threshold + 0.03)
            
            return {
                "success": True,
                "iterations": training_iterations,
                "improvements": improvements,
                "training_accuracy": sum(1 for r in results if r["target_pattern"] == r["matched_pattern"]) / len(results),
                "avg_reaction_time": sum(r["reaction_time_ms"] for r in results) / len(results)
            }
            
        except Exception as e:
            logger.error(f"Error during reflex training: {e}")
            return {"success": False, "error": str(e)}
            
        finally:
            self.training_in_progress = False
    
    async def add_gaming_reflex(self, 
                              game_name: str,
                              action_type: str,
                              trigger_pattern: Dict[str, Any],
                              response_procedure: str,
                              reaction_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Add a specialized gaming reflex
        
        Args:
            game_name: Name of the game
            action_type: Type of action (e.g., "block", "attack", "dodge")
            trigger_pattern: Pattern to recognize
            response_procedure: Procedure to execute
            reaction_threshold: Recognition threshold
            
        Returns:
            Registration result
        """
        reflex_name = f"gaming_{game_name}_{action_type}_{len(self.domain_libraries['gaming'])+1}"
        
        # Add game-specific context to template
        context_template = {
            "game": game_name,
            "action_type": action_type,
            "gaming_reflex": True
        }
        
        # Gaming reflexes get high priority
        priority = 3
        
        # Register the reflex
        result = await self.register_reflex(
            name=reflex_name,
            pattern_data=trigger_pattern,
            procedure_name=response_procedure,
            threshold=reaction_threshold,
            priority=priority,
            domain="gaming",
            context_template=context_template
        )
        
        return result
    
    async def simulate_gaming_scenarios(self, game_name: str, scenario_count: int = 10) -> Dict[str, Any]:
        """
        Simulate gaming scenarios to test and improve reaction time
        
        Args:
            game_name: Game to simulate
            scenario_count: Number of scenarios to run
            
        Returns:
            Simulation results
        """
        # Get game-specific patterns
        game_patterns = {
            name: pattern for name, pattern in self.domain_libraries.get("gaming", {}).items()
            if pattern.context_template.get("game") == game_name
        }
        
        if not game_patterns:
            return {
                "success": False,
                "error": f"No reflexes registered for game '{game_name}'"
            }
        
        # Run simulations
        results = []
        
        for i in range(scenario_count):
            # Select a random pattern to test
            pattern_name = random.choice(list(game_patterns.keys()))
            pattern = game_patterns[pattern_name]
            
            # Generate a stimulus based on the pattern
            stimulus = self.pattern_recognition.generate_gaming_stimulus(
                pattern.pattern_data,
                difficulty=min(1.0, 0.5 + (i/scenario_count))  # Increase difficulty gradually
            )
            
            # Add some distractor elements
            if random.random() < 0.3:
                stimulus["distractors"] = [
                    {"type": "visual", "intensity": random.random()},
                    {"type": "audio", "intensity": random.random() * 0.5}
                ]
            
            # Process stimulus
            start_time = time.time()
            reaction = await self.process_stimulus_fast(
                stimulus=stimulus,
                domain="gaming",
                context={"simulation": True, "difficulty": min(1.0, 0.5 + (i/scenario_count))}
            )
            reaction_time = (time.time() - start_time) * 1000  # ms
            
            results.append({
                "scenario": i+1,
                "pattern_tested": pattern_name,
                "reaction_time_ms": reaction_time,
                "success": reaction.get("success", False),
                "matched_pattern": reaction.get("pattern_name")
            })
        
        # Calculate statistics
        success_count = sum(1 for r in results if r["success"])
        avg_reaction_time = sum(r["reaction_time_ms"] for r in results) / len(results)
        
        return {
            "success": True,
            "game": game_name,
            "scenarios_run": scenario_count,
            "success_rate": success_count / scenario_count,
            "avg_reaction_time_ms": avg_reaction_time,
            "results": results
        }
    
    def set_response_mode(self, mode: str) -> Dict[str, Any]:
        """
        Set the response mode
        
        Args:
            mode: Response mode (normal, hyper, relaxed, learning)
            
        Returns:
            Mode change result
        """
        valid_modes = ["normal", "hyper", "relaxed", "learning"]
        
        if mode not in valid_modes:
            return {
                "success": False,
                "error": f"Invalid mode. Valid modes are: {', '.join(valid_modes)}"
            }
        
        # Apply mode-specific adjustments
        if mode == "hyper":
            # Hyper mode lowers thresholds for quicker reactions
            for pattern in self.reflex_patterns.values():
                pattern._original_threshold = pattern.threshold
                pattern.threshold = max(0.5, pattern.threshold - 0.15)
        elif mode == "relaxed":
            # Relaxed mode increases thresholds for more deliberate reactions
            for pattern in self.reflex_patterns.values():
                pattern._original_threshold = pattern.threshold
                pattern.threshold = min(0.95, pattern.threshold + 0.15)
        elif mode == "normal" and hasattr(self, "_original_thresholds"):
            # Restore original thresholds
            for name, pattern in self.reflex_patterns.items():
                if hasattr(pattern, "_original_threshold"):
                    pattern.threshold = pattern._original_threshold
                    delattr(pattern, "_original_threshold")
        
        # Set mode
        self.response_mode = mode
        
        return {
            "success": True,
            "mode": mode
        }
    
    async def get_reflexive_stats(self) -> Dict[str, Any]:
        """Get statistics about reflexive system performance"""
        stats = {
            "total_patterns": len(self.reflex_patterns),
            "domain_counts": {
                domain: len(patterns) for domain, patterns in self.domain_libraries.items()
                if patterns
            },
            "response_mode": self.response_mode,
            "overall_avg_reaction_time_ms": sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else 0,
            "min_reaction_time_ms": min(self.reaction_times) if self.reaction_times else 0,
            "max_reaction_time_ms": max(self.reaction_times) if self.reaction_times else 0,
            "active_status": self._is_active,
            "top_patterns": []
        }
        
        # Add top patterns by execution count
        top_patterns = sorted(
            self.reflex_patterns.values(),
            key=lambda p: p.execution_count,
            reverse=True
        )[:5]
        
        stats["top_patterns"] = [p.to_dict() for p in top_patterns]
        
    async def optimize_reflexes(self) -> Dict[str, Any]:
            """
            Optimize reflexes through analysis and pattern refinement
            
            Returns:
                Optimization results
            """
            if not self.reflex_patterns:
                return {"success": False, "error": "No reflex patterns to optimize"}
            
            optimization_results = {}
            
            # Sort patterns by execution count to focus on most used
            sorted_patterns = sorted(
                self.reflex_patterns.values(),
                key=lambda p: p.execution_count,
                reverse=True
            )
            
            for pattern in sorted_patterns:
                if pattern.execution_count < 5:
                    continue  # Skip patterns without enough data
                
                # Calculate success rate
                success_rate = pattern.get_success_rate()
                
                # If success rate is low, try to improve pattern
                if success_rate < 0.7:
                    if self.pattern_recognition.can_optimize(pattern.pattern_data):
                        # Try to optimize pattern
                        old_pattern = pattern.pattern_data.copy()
                        optimized_pattern = self.pattern_recognition.optimize_pattern(pattern.pattern_data)
                        
                        if optimized_pattern:
                            pattern.pattern_data = optimized_pattern
                            optimization_results[pattern.name] = {
                                "status": "optimized",
                                "original_success_rate": success_rate,
                                "optimization_type": "pattern_refinement"
                            }
                            
                            logger.info(f"Optimized pattern '{pattern.name}' with success rate {success_rate:.2f}")
                    else:
                        # If pattern can't be optimized further, adjust threshold
                        old_threshold = pattern.threshold
                        if success_rate < 0.5 and pattern.threshold > 0.6:
                            pattern.threshold = max(0.6, pattern.threshold - 0.1)
                        elif success_rate > 0.9 and pattern.threshold < 0.9:
                            pattern.threshold = min(0.9, pattern.threshold + 0.05)
                        
                        if old_threshold != pattern.threshold:
                            optimization_results[pattern.name] = {
                                "status": "threshold_adjusted",
                                "original_success_rate": success_rate,
                                "old_threshold": old_threshold,
                                "new_threshold": pattern.threshold
                            }
                            
                            logger.info(f"Adjusted threshold for pattern '{pattern.name}' from {old_threshold:.2f} to {pattern.threshold:.2f}")
                
                # For high success rate but slow reaction time, try simplifying pattern
                elif success_rate > 0.8 and pattern.get_avg_response_time() > 50:  # 50ms threshold
                    if self.pattern_recognition.can_simplify(pattern.pattern_data):
                        old_pattern = pattern.pattern_data.copy()
                        simplified_pattern = self.pattern_recognition.simplify_pattern(pattern.pattern_data)
                        
                        if simplified_pattern:
                            pattern.pattern_data = simplified_pattern
                            optimization_results[pattern.name] = {
                                "status": "simplified",
                                "original_avg_response_time": pattern.get_avg_response_time(),
                                "optimization_type": "pattern_simplification"
                            }
                            
                            logger.info(f"Simplified pattern '{pattern.name}' for faster reaction time")
            
            return {
                "success": True,
                "optimizations": optimization_results,
                "patterns_examined": len(sorted_patterns),
                "patterns_optimized": len(optimization_results)
            }


class SimplePatternRecognition:
    """Simple pattern recognition system for the reflexive system"""
    
    def __init__(self):
        """Initialize the pattern recognition system"""
        pass
    
    def fast_match(self, stimulus: Dict[str, Any], pattern: Dict[str, Any]) -> float:
        """
        Perform fast pattern matching
        
        Args:
            stimulus: Stimulus data
            pattern: Pattern to match against
            
        Returns:
            Match score (0.0-1.0)
        """
        # Handle empty cases
        if not stimulus or not pattern:
            return 0.0
        
        # Get keys present in both
        common_keys = set(stimulus.keys()) & set(pattern.keys())
        if not common_keys:
            return 0.0
        
        # Calculate match score
        match_points = 0
        total_points = 0
        
        for key in common_keys:
            # Skip metadata keys
            if key.startswith('_'):
                continue
                
            total_points += 1
            
            # Simple equality check for strings
            if isinstance(pattern[key], str) and isinstance(stimulus[key], str):
                if stimulus[key].lower() == pattern[key].lower():
                    match_points += 1
                elif pattern[key].lower() in stimulus[key].lower():
                    match_points += 0.7
                elif stimulus[key].lower() in pattern[key].lower():
                    match_points += 0.5
            
            # Numeric range check
            elif isinstance(pattern[key], dict) and 'min' in pattern[key] and 'max' in pattern[key] and isinstance(stimulus[key], (int, float)):
                min_val = pattern[key]['min']
                max_val = pattern[key]['max']
                
                if min_val <= stimulus[key] <= max_val:
                    match_points += 1
                else:
                    # Partial match based on distance to range
                    distance = min(abs(stimulus[key] - min_val), abs(stimulus[key] - max_val))
                    range_size = max_val - min_val
                    if range_size > 0:
                        match_points += max(0, 1 - (distance / range_size))
            
            # Array/list contains check
            elif isinstance(pattern[key], list) and isinstance(stimulus[key], list):
                # Calculate overlap
                pattern_set = set(pattern[key])
                stimulus_set = set(stimulus[key])
                
                if pattern_set and stimulus_set:
                    overlap = len(pattern_set & stimulus_set)
                    match_points += overlap / max(len(pattern_set), len(stimulus_set))
            
            # Simple equality for other types
            elif pattern[key] == stimulus[key]:
                match_points += 1
        
        # Add special handling for required keys
        required_keys = {k for k, v in pattern.items() if isinstance(v, dict) and v.get('required', False)}
        if required_keys:
            for key in required_keys:
                if key not in stimulus:
                    return 0.0  # Required key missing, no match
        
        # If we have significant keys to check
        if total_points > 0:
            return match_points / total_points
        
        return 0.0
    
    def generate_similar_stimulus(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a stimulus similar to the given pattern for training
        
        Args:
            pattern: Pattern to generate similar stimulus for
            
        Returns:
            Generated stimulus
        """
        stimulus = {}
        
        for key, value in pattern.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # String handling
            if isinstance(value, str):
                if random.random() < 0.8:  # 80% keep the same
                    stimulus[key] = value
                else:
                    # Generate variation
                    stimulus[key] = value + " (variant)"
            
            # Numeric range handling
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                min_val = value['min']
                max_val = value['max']
                
                # Generate random value in range
                stimulus[key] = min_val + random.random() * (max_val - min_val)
            
            # List handling
            elif isinstance(value, list):
                # Use subset of list items
                use_count = max(1, len(value) - random.randint(0, min(2, len(value))))
                stimulus[key] = random.sample(value, use_count)
            
            # Other types
            else:
                stimulus[key] = value
        
        return stimulus
    
    def generate_gaming_stimulus(self, pattern: Dict[str, Any], difficulty: float = 0.5) -> Dict[str, Any]:
        """
        Generate a gaming-specific stimulus with appropriate timing and visual elements
        
        Args:
            pattern: Base pattern
            difficulty: Difficulty level (0.0-1.0)
            
        Returns:
            Gaming-specific stimulus
        """
        # Start with base stimulus
        stimulus = self.generate_similar_stimulus(pattern)
        
        # Add gaming-specific elements
        stimulus["_timing"] = {
            "frame_number": random.randint(1, 1000),
            "time_to_impact": max(100, 500 - int(300 * difficulty)),  # ms, less time at higher difficulty
            "frame_window": max(3, 10 - int(difficulty * 7))  # Fewer frames at higher difficulty
        }
        
        # Add visual elements
        visual_noise = difficulty * 0.5  # More visual noise at higher difficulty
        stimulus["visual_clarity"] = max(0.2, 1.0 - visual_noise)
        
        # Add opponent state
        stimulus["opponent"] = {
            "state": random.choice(["attacking", "defending", "neutral", "special"]),
            "position": random.choice(["close", "mid", "far"]),
            "orientation": random.choice(["facing", "side", "away"])
        }
        
        return stimulus
    
    def can_optimize(self, pattern: Dict[str, Any]) -> bool:
        """
        Determine if a pattern can be optimized
        
        Args:
            pattern: Pattern to check
            
        Returns:
            Whether pattern can be optimized
        """
        # Check if pattern has enough complexity to optimize
        return len(pattern) >= 3
    
    def optimize_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a pattern for better recognition
        
        Args:
            pattern: Pattern to optimize
            
        Returns:
            Optimized pattern
        """
        optimized = pattern.copy()
        
        # Find most distinctive features
        for key, value in optimized.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # Enhance string matching with variants
            if isinstance(value, str) and len(value) > 5:
                # Add variations or keywords for more robust matching
                optimized[key] = {
                    "primary": value,
                    "variants": [value.lower(), value.upper(), value.capitalize()],
                    "weight": 1.2  # Give this feature more weight
                }
            
            # Enhance numeric ranges
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                # Expand range slightly for better matching
                min_val = value['min']
                max_val = value['max']
                range_size = max_val - min_val
                
                optimized[key] = {
                    'min': min_val - (range_size * 0.1),
                    'max': max_val + (range_size * 0.1),
                    'optimal': (min_val + max_val) / 2,
                    'weight': 1.5
                }
        
        return optimized
    
    def can_simplify(self, pattern: Dict[str, Any]) -> bool:
        """
        Determine if a pattern can be simplified
        
        Args:
            pattern: Pattern to check
            
        Returns:
            Whether pattern can be simplified
        """
        # Check if pattern has enough elements to simplify
        return len(pattern) > 3
    
    def simplify_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplify a pattern for faster recognition
        
        Args:
            pattern: Pattern to simplify
            
        Returns:
            Simplified pattern
        """
        simplified = {}
        
        # Find most important features
        feature_scores = {}
        for key, value in pattern.items():
            # Skip metadata keys
            if key.startswith('_'):
                continue
            
            # Score features by expected distinctiveness
            if isinstance(value, dict) and value.get('weight', 0) > 0:
                feature_scores[key] = value.get('weight', 1)
            elif isinstance(value, dict) and ('min' in value or 'max' in value):
                feature_scores[key] = 1.2  # Numeric ranges are distinctive
            elif isinstance(value, str) and len(value) > 3:
                feature_scores[key] = 1.0  # Strings are useful
            elif isinstance(value, list) and len(value) > 0:
                feature_scores[key] = 0.8  # Lists are somewhat useful
            else:
                feature_scores[key] = 0.5  # Other features
        
        # Keep only top 3 most distinctive features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for key, _ in top_features:
            simplified[key] = pattern[key]
        
        # Ensure we keep metadata
        for key, value in pattern.items():
            if key.startswith('_'):
                simplified[key] = value
        
        return simplified
