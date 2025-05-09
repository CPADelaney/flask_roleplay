# nyx/core/integration/knowledge_curiosity_exploration_bridge.py

import logging
import asyncio
import datetime
import random
from typing import Dict, List, Any, Optional, Tuple

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class KnowledgeCuriosityExplorationBridge:
    """
    Integrates knowledge core with curiosity and exploration systems.
    
    This bridge enables:
    1. Identifying knowledge gaps for exploration
    2. Managing knowledge acquisition through exploration
    3. Prioritizing curiosity targets based on need and potential value
    4. Tracking exploration outcomes and knowledge growth
    5. Coordinating between knowledge and exploration subsystems
    """
    
    def __init__(self, 
                knowledge_core=None,
                exploration_engine=None,
                goal_manager=None,
                need_system=None):
        """Initialize the knowledge-curiosity-exploration bridge."""
        self.knowledge_core = knowledge_core
        self.exploration_engine = exploration_engine
        self.goal_manager = goal_manager
        self.need_system = need_system
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.exploration_threshold = 0.6  # Minimum importance for exploration
        self.knowledge_acquisition_target = 0.7  # Target knowledge level
        self.exploration_budget = 0.5  # Fraction of resources for exploration
        self.knowledge_exploration_interval = 24  # Hours between auto-exploration cycles
        
        # Tracking variables
        self.active_explorations = {}  # target_id -> exploration info
        self.exploration_history = []  # List of past explorations
        self.knowledge_domain_priorities = {}  # domain -> priority
        self.next_exploration_id = 1
        self.last_exploration_cycle = datetime.datetime.now()
        
        # Integration event subscriptions
        self._subscribed = False
        
        logger.info("KnowledgeCuriosityExplorationBridge initialized")
    
    async def initialize(self) -> bool:
        """Initialize the bridge and establish connections to systems."""
        try:
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("knowledge_gap_detected", self._handle_knowledge_gap)
                self.event_bus.subscribe("goal_completed", self._handle_goal_completed)
                self.event_bus.subscribe("need_state_change", self._handle_need_change)
                self._subscribed = True
            
            logger.info("KnowledgeCuriosityExplorationBridge successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing KnowledgeCuriosityExplorationBridge: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeCuriosityExploration")
    async def identify_exploration_targets(self, 
                                        limit: int = 5,
                                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Identify potential exploration targets from knowledge gaps.
        
        Args:
            limit: Maximum number of targets to identify
            context: Optional context information for prioritization
            
        Returns:
            List of exploration targets
        """
        if not self.knowledge_core:
            return {"status": "error", "message": "Knowledge core not available"}
        
        try:
            # Identify knowledge gaps
            knowledge_gaps = await self.knowledge_core.identify_knowledge_gaps()
            
            if not knowledge_gaps:
                return {
                    "status": "no_gaps",
                    "message": "No significant knowledge gaps identified"
                }
            
            # Apply context-specific prioritization if provided
            if context:
                knowledge_gaps = await self._apply_context_prioritization(knowledge_gaps, context)
            
            # Filter by exploration threshold
            valid_targets = [
                gap for gap in knowledge_gaps 
                if gap.get("importance", 0) * gap.get("gap_size", 0) >= self.exploration_threshold
            ]
            
            if not valid_targets:
                return {
                    "status": "no_viable_targets",
                    "message": "No knowledge gaps meet the exploration threshold"
                }
            
            # Apply limit
            if limit and limit < len(valid_targets):
                targets = valid_targets[:limit]
            else:
                targets = valid_targets
            
            # Create exploration targets
            exploration_targets = []
            for target in targets:
                # Create a target
                target_id = await self.knowledge_core.create_exploration_target(
                    domain=target.get("domain"),
                    topic=target.get("topic"),
                    importance=target.get("importance", 0.5),
                    urgency=target.get("priority", 0.5) / 2 + 0.25,  # Map to 0.25-0.75 range
                    knowledge_gap=target.get("gap_size", 0.5)
                )
                
                if target_id:
                    exploration_targets.append({
                        "target_id": target_id,
                        "domain": target.get("domain"),
                        "topic": target.get("topic"),
                        "importance": target.get("importance", 0.5),
                        "gap_size": target.get("gap_size", 0.5),
                        "priority": target.get("priority", 0.5)
                    })
            
            return {
                "status": "success",
                "targets": exploration_targets,
                "total_gaps": len(knowledge_gaps),
                "viable_targets": len(valid_targets)
            }
            
        except Exception as e:
            logger.error(f"Error identifying exploration targets: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeCuriosityExploration")
    async def explore_knowledge_target(self, 
                                   target_id: str,
                                   exploration_depth: float = 0.5) -> Dict[str, Any]:
        """
        Actively explore a knowledge target.
        
        Args:
            target_id: ID of the exploration target
            exploration_depth: How deeply to explore (0.0-1.0)
            
        Returns:
            Results of the exploration
        """
        if not self.knowledge_core or not self.exploration_engine:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get the target
            exploration_targets = await self.knowledge_core.get_exploration_targets(
                limit=1, 
                min_priority=0
            )
            
            target = None
            for t in exploration_targets:
                if t.get("id") == target_id:
                    target = t
                    break
            
            if not target:
                return {"status": "error", "message": f"Exploration target {target_id} not found"}
            
            # Generate exploration questions
            questions = await self.knowledge_core.generate_questions(target_id)
            
            if not questions:
                return {
                    "status": "error",
                    "message": "Could not generate exploration questions"
                }
            
            # Create exploration ID
            exploration_id = f"exploration_{self.next_exploration_id}"
            self.next_exploration_id += 1
            
            # Mark as active exploration
            self.active_explorations[target_id] = {
                "exploration_id": exploration_id,
                "start_time": datetime.datetime.now().isoformat(),
                "domain": target.get("domain"),
                "topic": target.get("topic"),
                "questions": questions,
                "status": "in_progress"
            }
            
            # Run exploration
            exploration_results = await self.exploration_engine.explore_topic(
                topic=target.get("topic"),
                domain=target.get("domain"),
                questions=questions,
                depth=exploration_depth
            )
            
            # Process the results
            knowledge_gained = 0.0
            acquired_facts = []
            
            if exploration_results.get("status") == "success":
                # Extract discovered facts
                facts = exploration_results.get("facts", [])
                
                # Add each fact to knowledge
                for fact in facts:
                    content = {
                        "fact": fact.get("statement"),
                        "domain": target.get("domain"),
                        "topic": target.get("topic"),
                        "source": "exploration",
                        "exploration_id": exploration_id
                    }
                    
                    # Add context if available
                    if "context" in fact:
                        content["context"] = fact["context"]
                    
                    # Add to knowledge with confidence from fact
                    confidence = fact.get("confidence", 0.7)
                    node_id = await self.knowledge_core.add_knowledge(
                        type="fact",
                        content=content,
                        source="exploration",
                        confidence=confidence
                    )
                    
                    if node_id:
                        acquired_facts.append({
                            "node_id": node_id,
                            "fact": fact.get("statement"),
                            "confidence": confidence
                        })
                
                # Calculate knowledge gained
                if facts:
                    knowledge_gained = len(acquired_facts) / len(facts)
                    
                    # Normalize to 0.0-1.0 range with diminishing returns
                    knowledge_gained = min(1.0, 0.2 + (knowledge_gained * 0.8))
            
            # Record exploration outcome
            outcome = {
                "success": exploration_results.get("status") == "success",
                "knowledge_gained": knowledge_gained,
                "facts_discovered": len(acquired_facts),
                "questions_explored": len(questions)
            }
            
            await self.knowledge_core.record_exploration(target_id, outcome)
            
            # Update exploration history
            self.exploration_history.append({
                "exploration_id": exploration_id,
                "target_id": target_id,
                "domain": target.get("domain"),
                "topic": target.get("topic"),
                "timestamp": datetime.datetime.now().isoformat(),
                "knowledge_gained": knowledge_gained,
                "facts_acquired": len(acquired_facts)
            })
            
            # Remove from active explorations
            if target_id in self.active_explorations:
                self.active_explorations[target_id]["status"] = "completed"
            
            return {
                "status": "success",
                "exploration_id": exploration_id,
                "target_id": target_id,
                "knowledge_gained": knowledge_gained,
                "facts_acquired": acquired_facts,
                "questions_explored": questions
            }
            
        except Exception as e:
            logger.error(f"Error exploring knowledge target: {e}")
            
            # Update status of active exploration if it exists
            if target_id in self.active_explorations:
                self.active_explorations[target_id]["status"] = "failed"
                self.active_explorations[target_id]["error"] = str(e)
            
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeCuriosityExploration")
    async def create_exploration_goal(self, 
                                   target_id: str,
                                   priority: float = 0.5) -> Dict[str, Any]:
        """
        Create a goal for exploring a knowledge target.
        
        Args:
            target_id: ID of the exploration target
            priority: Priority of the goal
            
        Returns:
            Results of goal creation
        """
        if not self.knowledge_core or not self.goal_manager:
            return {"status": "error", "message": "Required systems not available"}
        
        try:
            # Get the target
            exploration_targets = await self.knowledge_core.get_exploration_targets(
                limit=1, 
                min_priority=0
            )
            
            target = None
            for t in exploration_targets:
                if t.get("id") == target_id:
                    target = t
                    break
            
            if not target:
                return {"status": "error", "message": f"Exploration target {target_id} not found"}
            
            # Formulate goal description
            domain = target.get("domain")
            topic = target.get("topic")
            description = f"Explore knowledge about {topic} in the domain of {domain}"
            
            # Enhance with questions
            questions = await self.knowledge_core.generate_questions(target_id)
            if questions:
                question_text = "\n- " + "\n- ".join(questions[:3])
                description += f"\nKey questions to explore:{question_text}"
            
            # Create goal
            goal_id = await self.goal_manager.add_goal(
                description=description,
                priority=priority,
                source="KnowledgeCuriosityExplorationBridge",
                metadata={
                    "type": "knowledge_exploration",
                    "target_id": target_id,
                    "domain": domain,
                    "topic": topic
                }
            )
            
            if not goal_id:
                return {"status": "error", "message": "Failed to create exploration goal"}
            
            # Add steps to the goal if possible
            if hasattr(self.goal_manager, "add_goal_step"):
                # Add a research step
                await self.goal_manager.add_goal_step(
                    goal_id=goal_id,
                    step_description=f"Research {topic} in {domain}",
                    step_index=0,
                    action="explore_knowledge",
                    parameters={"target_id": target_id, "depth": 0.7}
                )
                
                # Add a knowledge integration step
                await self.goal_manager.add_goal_step(
                    goal_id=goal_id,
                    step_description=f"Integrate new knowledge about {topic}",
                    step_index=1,
                    action="run_knowledge_integration",
                    parameters={"domain": domain, "topic": topic}
                )
            
            return {
                "status": "success",
                "goal_id": goal_id,
                "target_id": target_id,
                "description": description,
                "priority": priority
            }
            
        except Exception as e:
            logger.error(f"Error creating exploration goal: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeCuriosityExploration")
    async def prioritize_knowledge_domains(self, 
                                        limit: int = 5) -> Dict[str, Any]:
        """
        Prioritize knowledge domains for exploration.
        
        Args:
            limit: Maximum number of domains to prioritize
            
        Returns:
            Prioritized knowledge domains
        """
        if not self.knowledge_core:
            return {"status": "error", "message": "Knowledge core not available"}
        
        try:
            # Get need states if available
            need_weights = {}
            if self.need_system:
                need_states = await self.need_system.get_needs_state_async()
                
                # Map needs to knowledge domains
                for need_name, need_data in need_states.items():
                    if need_name == "knowledge":
                        # Knowledge need directly maps to overall exploration priority
                        need_weights["knowledge_general"] = need_data.get("drive_strength", 0.5)
                    elif need_name == "novelty":
                        # Novelty need maps to exploration of new domains
                        need_weights["knowledge_novelty"] = need_data.get("drive_strength", 0.5)
            
            # Get domain statistics from knowledge core
            stats = await self.knowledge_core.get_knowledge_statistics()
            
            # Extract domain information if available
            domains = []
            if "curiosity_system" in stats:
                curiosity_stats = stats["curiosity_system"]
                
                if "knowledge_map" in curiosity_stats:
                    knowledge_map = curiosity_stats["knowledge_map"]
                    
                    # Get domain names if available
                    if "domains" in knowledge_map:
                        domains = list(knowledge_map["domains"].keys())
            
            # Score each domain
            domain_scores = []
            for domain in domains:
                # Calculate base score
                score = random.random() * 0.3 + 0.35  # Random base score between 0.35-0.65
                
                # Apply need weights
                knowledge_need_factor = need_weights.get("knowledge_general", 0.5)
                novelty_need_factor = need_weights.get("knowledge_novelty", 0.5)
                
                # Check if domain was recently explored
                recently_explored = False
                for history in self.exploration_history[-10:]:
                    if history["domain"] == domain and history["knowledge_gained"] > 0.3:
                        recently_explored = True
                        break
                
                # Recently explored domains get lower score if novelty need is high
                if recently_explored and novelty_need_factor > 0.6:
                    score *= 0.7
                
                # Domains not explored recently get higher score if knowledge need is high
                if not recently_explored and knowledge_need_factor > 0.6:
                    score *= 1.3
                
                # Add to list
                domain_scores.append({
                    "domain": domain,
                    "score": min(1.0, score),  # Cap at 1.0
                    "recently_explored": recently_explored
                })
            
            # Sort by score
            domain_scores.sort(key=lambda x: x["score"], reverse=True)
            
            # Apply limit
            prioritized_domains = domain_scores[:limit]
            
            # Update stored priorities
            for domain in prioritized_domains:
                self.knowledge_domain_priorities[domain["domain"]] = domain["score"]
            
            return {
                "status": "success",
                "prioritized_domains": prioritized_domains,
                "total_domains": len(domains),
                "knowledge_need": need_weights.get("knowledge_general", 0.5),
                "novelty_need": need_weights.get("knowledge_novelty", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error prioritizing knowledge domains: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeCuriosityExploration")
    async def run_exploration_cycle(self) -> Dict[str, Any]:
        """
        Run an automatic exploration cycle to add knowledge.
        
        Returns:
            Results of the exploration cycle
        """
        self.last_exploration_cycle = datetime.datetime.now()
        
        results = {
            "targets_identified": 0,
            "explorations_performed": 0,
            "knowledge_gained": 0.0,
            "start_time": self.last_exploration_cycle.isoformat()
        }
        
        try:
            # Determine available exploration budget
            exploration_budget = 2  # Default budget (number of explorations)
            
            # Adjust based on needs if available
            if self.need_system:
                need_states = await self.need_system.get_needs_state_async()
                
                knowledge_need = 0.5  # Default
                if "knowledge" in need_states:
                    knowledge_need = need_states["knowledge"].get("drive_strength", 0.5)
                
                # Adjust budget based on need (more need = more budget)
                if knowledge_need > 0.7:
                    exploration_budget = 3
                elif knowledge_need < 0.3:
                    exploration_budget = 1
            
            # Identify exploration targets
            targets_result = await self.identify_exploration_targets(limit=5)
            
            if targets_result.get("status") != "success":
                return {
                    "status": "no_targets",
                    "message": targets_result.get("message", "No valid exploration targets found")
                }
            
            targets = targets_result.get("targets", [])
            results["targets_identified"] = len(targets)
            
            # Explore highest priority targets up to budget
            for i, target in enumerate(targets[:exploration_budget]):
                target_id = target.get("target_id")
                
                # Skip if already active
                if target_id in self.active_explorations:
                    continue
                
                # Explore the target
                exploration_result = await self.explore_knowledge_target(
                    target_id=target_id,
                    exploration_depth=0.6 + (random.random() * 0.3)  # Random depth between 0.6-0.9
                )
                
                if exploration_result.get("status") == "success":
                    results["explorations_performed"] += 1
                    results["knowledge_gained"] += exploration_result.get("knowledge_gained", 0.0)
            
            # Calculate average knowledge gained
            if results["explorations_performed"] > 0:
                results["knowledge_gained"] /= results["explorations_performed"]
            
            # Update stats
            results["end_time"] = datetime.datetime.now().isoformat()
            results["duration_seconds"] = (
                datetime.datetime.now() - self.last_exploration_cycle
            ).total_seconds()
            
            logger.info(f"Exploration cycle complete: {results['explorations_performed']} explorations, "
                       f"{results['knowledge_gained']:.2f} avg knowledge gained")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in exploration cycle: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    @trace_method(level=TraceLevel.INFO, group_id="KnowledgeCuriosityExploration")
    async def get_exploration_status(self) -> Dict[str, Any]:
        """
        Get the current status of knowledge exploration.
        
        Returns:
            Current exploration status
        """
        status = {
            "active_explorations": len(self.active_explorations),
            "total_explorations": len(self.exploration_history),
            "knowledge_domains_prioritized": len(self.knowledge_domain_priorities),
            "last_exploration_cycle": self.last_exploration_cycle.isoformat()
        }
        
        # Get details of active explorations
        active_details = []
        for target_id, exploration in self.active_explorations.items():
            if exploration["status"] == "in_progress":
                active_details.append({
                    "target_id": target_id,
                    "exploration_id": exploration["exploration_id"],
                    "domain": exploration["domain"],
                    "topic": exploration["topic"],
                    "start_time": exploration["start_time"]
                })
        
        status["active_details"] = active_details
        
        # Calculate historical performance metrics
        if self.exploration_history:
            # Average knowledge gained
            knowledge_gained = [h["knowledge_gained"] for h in self.exploration_history]
            status["avg_knowledge_gained"] = sum(knowledge_gained) / len(knowledge_gained)
            
            # Success rate
            success_count = sum(1 for h in self.exploration_history if h.get("facts_acquired", 0) > 0)
            status["success_rate"] = success_count / len(self.exploration_history)
            
            # Top domains explored
            domain_counts = {}
            for history in self.exploration_history:
                domain = history["domain"]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
            top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            status["top_domains"] = [{"domain": d, "explorations": c} for d, c in top_domains]
        
        return status
    
    async def _handle_knowledge_gap(self, event: Event) -> None:
        """
        Handle knowledge gap events.
        
        Args:
            event: Knowledge gap event
        """
        try:
            # Extract event data
            domain = event.data.get("domain")
            topic = event.data.get("topic")
            importance = event.data.get("importance", 0.5)
            
            if not domain or not topic:
                return
                
            # Create exploration target for the gap
            target_id = await self.knowledge_core.create_exploration_target(
                domain=domain,
                topic=topic,
                importance=importance,
                urgency=0.6  # Default to moderately high urgency for explicit gaps
            )
            
            if target_id and importance > 0.7:
                # High importance gap - create a goal for it
                await self.create_exploration_goal(
                    target_id=target_id,
                    priority=importance
                )
                
        except Exception as e:
            logger.error(f"Error handling knowledge gap event: {e}")
    
    async def _handle_goal_completed(self, event: Event) -> None:
        """
        Handle goal completed events.
        
        Args:
            event: Goal completed event
        """
        try:
            # Extract event data
            goal_id = event.data.get("goal_id")
            
            if not goal_id or not self.goal_manager:
                return
                
            # Get goal details
            goal = await self.goal_manager.get_goal_status(goal_id)
            
            if not goal:
                return
                
            # Check if this was an exploration goal
            metadata = goal.get("metadata", {})
            if metadata.get("type") == "knowledge_exploration":
                target_id = metadata.get("target_id")
                
                if target_id:
                    # Run a targeted exploration to utilize the success
                    asyncio.create_task(self.explore_knowledge_target(
                        target_id=target_id,
                        exploration_depth=0.8  # Deep exploration for explicit goal
                    ))
                
        except Exception as e:
            logger.error(f"Error handling goal completed event: {e}")
    
    async def _handle_need_change(self, event: Event) -> None:
        """
        Handle need state change events.
        
        Args:
            event: Need state change event
        """
        try:
            # Extract event data
            need_name = event.data.get("need_name")
            drive_strength = event.data.get("drive_strength", 0.0)
            
            if not need_name:
                return
                
            # Check if knowledge-related need has high drive
            if (need_name in ["knowledge", "novelty", "curiosity"] and 
                drive_strength > 0.7):
                
                # Check if we're due for an exploration cycle
                now = datetime.datetime.now()
                hours_since_cycle = (now - self.last_exploration_cycle).total_seconds() / 3600
                
                if hours_since_cycle >= self.knowledge_exploration_interval:
                    # Run an exploration cycle to satisfy the need
                    asyncio.create_task(self.run_exploration_cycle())
                
        except Exception as e:
            logger.error(f"Error handling need change event: {e}")
    
    async def _apply_context_prioritization(self, knowledge_gaps: List[Dict[str, Any]], 
                                         context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply context-specific prioritization to knowledge gaps."""
        # Create a copy of the gaps to avoid modifying the original
        prioritized = knowledge_gaps.copy()
        
        # Check for current goal context
        if "current_goal" in context:
            goal = context["current_goal"]
            goal_domain = goal.get("domain")
            goal_topics = goal.get("topics", [])
            
            # Boost priority for gaps related to current goal
            if goal_domain or goal_topics:
                for gap in prioritized:
                    # Domain match
                    if goal_domain and gap.get("domain") == goal_domain:
                        gap["priority"] = gap.get("priority", 0.5) * 1.3  # 30% boost
                    
                    # Topic match
                    if goal_topics and gap.get("topic") in goal_topics:
                        gap["priority"] = gap.get("priority", 0.5) * 1.5  # 50% boost
        
        # Check for recent topic context
        if "recent_topics" in context:
            recent_topics = context["recent_topics"]
            
            for gap in prioritized:
                if gap.get("topic") in recent_topics:
                    gap["priority"] = gap.get("priority", 0.5) * 1.2  # 20% boost
        
        # Sort by updated priority
        prioritized.sort(key=lambda x: x.get("priority", 0.5), reverse=True)
        
        return prioritized

# Function to create the bridge
def create_knowledge_curiosity_exploration_bridge(nyx_brain):
    """Create a knowledge-curiosity-exploration bridge for the given brain."""
    return KnowledgeCuriosityExplorationBridge(
        knowledge_core=nyx_brain.knowledge_core if hasattr(nyx_brain, "knowledge_core") else None,
        exploration_engine=nyx_brain.exploration_engine if hasattr(nyx_brain, "exploration_engine") else None,
        goal_manager=nyx_brain.goal_manager if hasattr(nyx_brain, "goal_manager") else None,
        need_system=nyx_brain.needs_system if hasattr(nyx_brain, "needs_system") else None
    )
