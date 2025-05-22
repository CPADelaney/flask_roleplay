# nyx/core/a2a/context_aware_experience_consolidation.py

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareExperienceConsolidation(ContextAwareModule):
    """
    Advanced Experience Consolidation System with full context distribution capabilities
    """
    
    def __init__(self, original_consolidation_system):
        super().__init__("experience_consolidation")
        self.original_system = original_consolidation_system
        self.context_subscriptions = [
            "experience_created", "experience_retrieved", "memory_formation",
            "emotional_experience", "goal_related_experience", "identity_impact",
            "consolidation_trigger", "experience_pattern_detected", "user_feedback"
        ]
        
        # Track consolidation state
        self.pending_consolidations = {}
        self.consolidation_patterns = {}
        self.active_consolidation_groups = []
    
    async def on_context_received(self, context: SharedContext):
        """Initialize consolidation processing for this context"""
        logger.debug(f"ExperienceConsolidation received context for user: {context.user_id}")
        
        # Check if consolidation is due
        consolidation_status = await self._check_consolidation_readiness(context)
        
        # Analyze current experience landscape
        experience_analysis = await self._analyze_experience_landscape(context)
        
        # Identify potential consolidation opportunities
        consolidation_opportunities = await self._identify_consolidation_opportunities(
            context, experience_analysis
        )
        
        # Send initial consolidation context to other modules
        await self.send_context_update(
            update_type="consolidation_context_available",
            data={
                "consolidation_ready": consolidation_status["ready"],
                "next_consolidation_in": consolidation_status.get("hours_until_next", 0),
                "experience_analysis": experience_analysis,
                "consolidation_opportunities": len(consolidation_opportunities),
                "pending_groups": consolidation_opportunities[:3]  # Top 3 opportunities
            },
            priority=ContextPriority.NORMAL
        )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules that affect consolidation"""
        
        if update.update_type == "experience_created":
            # New experience created - check if it fits existing patterns
            experience_data = update.data
            experience_id = experience_data.get("memory_id")
            
            # Check against pending consolidation groups
            await self._check_experience_for_consolidation_fit(experience_id, experience_data)
            
            # Update pattern detection
            await self._update_experience_patterns(experience_data)
        
        elif update.update_type == "emotional_experience":
            # Strong emotional experiences may trigger consolidation
            emotional_data = update.data
            intensity = emotional_data.get("intensity", 0.5)
            
            if intensity > 0.8:
                # High intensity - mark for priority consolidation
                await self._mark_for_priority_consolidation(emotional_data)
        
        elif update.update_type == "goal_related_experience":
            # Goal-related experiences should be consolidated together
            goal_data = update.data
            goal_id = goal_data.get("goal_id")
            
            await self._group_goal_related_experiences(goal_id, goal_data)
        
        elif update.update_type == "identity_impact":
            # Experiences with identity impact need special consolidation
            identity_data = update.data
            await self._handle_identity_impacting_experience(identity_data)
        
        elif update.update_type == "consolidation_trigger":
            # External trigger for consolidation
            trigger_data = update.data
            await self._process_consolidation_trigger(trigger_data)
        
        elif update.update_type == "experience_pattern_detected":
            # Pattern detected across experiences
            pattern_data = update.data
            await self._incorporate_detected_pattern(pattern_data)
        
        elif update.update_type == "user_feedback":
            # User feedback affects consolidation priorities
            feedback_data = update.data
            await self._adjust_consolidation_from_feedback(feedback_data)
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input with consolidation awareness"""
        # Check if input references consolidated experiences
        consolidation_references = await self._check_input_for_consolidation_references(context)
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        # Check if immediate consolidation is triggered
        immediate_trigger = await self._check_immediate_consolidation_trigger(context, messages)
        
        if immediate_trigger:
            # Run immediate consolidation
            consolidation_result = await self._execute_immediate_consolidation(context, immediate_trigger)
            
            # Send consolidation update
            await self.send_context_update(
                update_type="immediate_consolidation_executed",
                data={
                    "trigger": immediate_trigger,
                    "result": consolidation_result,
                    "consolidated_count": consolidation_result.get("consolidations_created", 0)
                },
                priority=ContextPriority.HIGH
            )
            
            return {
                "consolidation_processed": True,
                "immediate_consolidation": True,
                "result": consolidation_result,
                "references": consolidation_references
            }
        
        # Regular processing with consolidation monitoring
        monitoring_result = await self._monitor_consolidation_opportunities(context, messages)
        
        return {
            "consolidation_processed": True,
            "immediate_consolidation": False,
            "monitoring": monitoring_result,
            "references": consolidation_references
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze consolidation needs and patterns"""
        # Comprehensive experience analysis
        experience_landscape = await self._comprehensive_experience_analysis(context)
        
        # Pattern analysis across experiences
        pattern_analysis = await self._analyze_experience_patterns(experience_landscape)
        
        # Consolidation effectiveness analysis
        effectiveness_analysis = await self._analyze_consolidation_effectiveness()
        
        # Cross-module impact analysis
        messages = await self.get_cross_module_messages()
        cross_module_impact = await self._analyze_cross_module_experience_impact(messages)
        
        # Predict future consolidation needs
        future_needs = await self._predict_consolidation_needs(
            experience_landscape, pattern_analysis, cross_module_impact
        )
        
        return {
            "experience_landscape": experience_landscape,
            "pattern_analysis": pattern_analysis,
            "effectiveness_analysis": effectiveness_analysis,
            "cross_module_impact": cross_module_impact,
            "future_consolidation_needs": future_needs,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize consolidation insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Create consolidation synthesis
        consolidation_synthesis = {
            "consolidation_status": await self._get_consolidation_status_summary(),
            "recent_consolidations": await self._get_recent_consolidation_summary(),
            "pattern_insights": await self._synthesize_pattern_insights(context),
            "consolidation_recommendations": await self._generate_consolidation_recommendations(context, messages),
            "memory_optimization": await self._analyze_memory_optimization_potential()
        }
        
        # Check if proactive consolidation is recommended
        proactive_consolidation = await self._evaluate_proactive_consolidation(consolidation_synthesis)
        
        if proactive_consolidation:
            await self.send_context_update(
                update_type="proactive_consolidation_recommended",
                data={
                    "recommendation": proactive_consolidation,
                    "rationale": consolidation_synthesis["consolidation_recommendations"],
                    "expected_benefit": proactive_consolidation.get("expected_benefit", "memory_optimization")
                },
                priority=ContextPriority.NORMAL
            )
        
        return consolidation_synthesis
    
    # ========================================================================================
    # ADVANCED HELPER METHODS
    # ========================================================================================
    
    async def _check_consolidation_readiness(self, context: SharedContext) -> Dict[str, Any]:
        """Check if system is ready for consolidation"""
        now = datetime.now()
        last_consolidation = self.original_system.last_consolidation
        time_since = (now - last_consolidation).total_seconds() / 3600  # hours
        
        readiness = {
            "ready": time_since >= self.original_system.consolidation_interval,
            "hours_since_last": time_since,
            "hours_until_next": max(0, self.original_system.consolidation_interval - time_since),
            "pending_candidates": len(self.pending_consolidations),
            "system_load": await self._assess_system_load_for_consolidation(context)
        }
        
        # Check if forced readiness due to memory pressure
        if await self._check_memory_pressure():
            readiness["ready"] = True
            readiness["forced_reason"] = "memory_pressure"
        
        return readiness
    
    async def _analyze_experience_landscape(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze the current landscape of experiences"""
        landscape = {
            "total_experiences": 0,
            "unconsolidated_experiences": 0,
            "experience_types": {},
            "scenario_distribution": {},
            "temporal_distribution": {},
            "user_distribution": {}
        }
        
        # Get experience statistics from memory core
        if self.original_system.memory_core:
            try:
                # Get all experiences
                all_experiences = await self.original_system.memory_core.retrieve_memories(
                    query="*",
                    memory_types=["experience", "observation", "episodic"],
                    limit=1000
                )
                
                landscape["total_experiences"] = len(all_experiences)
                
                # Analyze distribution
                for exp in all_experiences:
                    # Check if unconsolidated
                    metadata = exp.get("metadata", {})
                    if not metadata.get("is_consolidation", False) and not metadata.get("consolidated_into"):
                        landscape["unconsolidated_experiences"] += 1
                    
                    # Type distribution
                    exp_type = exp.get("memory_type", "unknown")
                    landscape["experience_types"][exp_type] = landscape["experience_types"].get(exp_type, 0) + 1
                    
                    # Scenario distribution
                    scenario = metadata.get("scenario_type", "general")
                    landscape["scenario_distribution"][scenario] = landscape["scenario_distribution"].get(scenario, 0) + 1
                    
                    # User distribution
                    user_id = metadata.get("user_id", "unknown")
                    landscape["user_distribution"][user_id] = landscape["user_distribution"].get(user_id, 0) + 1
                    
                    # Temporal distribution (by day)
                    if "timestamp" in exp:
                        try:
                            timestamp = datetime.fromisoformat(exp["timestamp"].replace("Z", "+00:00"))
                            day_key = timestamp.strftime("%Y-%m-%d")
                            landscape["temporal_distribution"][day_key] = \
                                landscape["temporal_distribution"].get(day_key, 0) + 1
                        except:
                            pass
                
            except Exception as e:
                logger.error(f"Error analyzing experience landscape: {e}")
        
        # Calculate consolidation potential
        if landscape["total_experiences"] > 0:
            landscape["consolidation_ratio"] = 1.0 - (landscape["unconsolidated_experiences"] / landscape["total_experiences"])
        else:
            landscape["consolidation_ratio"] = 1.0
        
        return landscape
    
    async def _identify_consolidation_opportunities(self, 
                                                  context: SharedContext,
                                                  experience_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify opportunities for consolidation"""
        opportunities = []
        
        # Check for scenario-based opportunities
        scenario_distribution = experience_analysis.get("scenario_distribution", {})
        for scenario, count in scenario_distribution.items():
            if count >= self.original_system.min_group_size:
                opportunities.append({
                    "type": "scenario_based",
                    "scenario": scenario,
                    "experience_count": count,
                    "priority": self._calculate_opportunity_priority(scenario, count)
                })
        
        # Check for temporal clustering opportunities
        temporal_distribution = experience_analysis.get("temporal_distribution", {})
        temporal_clusters = self._identify_temporal_clusters(temporal_distribution)
        
        for cluster in temporal_clusters:
            opportunities.append({
                "type": "temporal_cluster",
                "date_range": cluster["range"],
                "experience_count": cluster["count"],
                "priority": cluster["priority"]
            })
        
        # Check for cross-user pattern opportunities
        if len(experience_analysis.get("user_distribution", {})) > 1:
            cross_user_patterns = await self._identify_cross_user_patterns(experience_analysis)
            opportunities.extend(cross_user_patterns)
        
        # Sort by priority
        opportunities.sort(key=lambda x: x["priority"], reverse=True)
        
        return opportunities
    
    async def _check_experience_for_consolidation_fit(self, 
                                                     experience_id: str,
                                                     experience_data: Dict[str, Any]):
        """Check if new experience fits existing consolidation groups"""
        scenario_type = experience_data.get("scenario_type", "general")
        tags = experience_data.get("tags", [])
        significance = experience_data.get("significance", 5)
        
        # Check each pending consolidation group
        for group_id, group_data in self.pending_consolidations.items():
            # Check scenario match
            if group_data.get("scenario_type") == scenario_type:
                # Check tag overlap
                group_tags = set(group_data.get("common_tags", []))
                exp_tags = set(tags)
                tag_overlap = len(group_tags.intersection(exp_tags)) / max(1, len(group_tags))
                
                if tag_overlap > 0.5:
                    # Good fit - add to group
                    if "experience_ids" not in group_data:
                        group_data["experience_ids"] = []
                    
                    group_data["experience_ids"].append(experience_id)
                    group_data["total_significance"] = group_data.get("total_significance", 0) + significance
                    
                    # Update common tags
                    group_data["common_tags"] = list(group_tags.intersection(exp_tags))
                    
                    logger.debug(f"Added experience {experience_id} to consolidation group {group_id}")
                    
                    # Check if group is ready for consolidation
                    if len(group_data["experience_ids"]) >= self.original_system.min_group_size:
                        await self._mark_group_ready_for_consolidation(group_id)
                    
                    break
    
    async def _update_experience_patterns(self, experience_data: Dict[str, Any]):
        """Update pattern tracking with new experience"""
        scenario_type = experience_data.get("scenario_type", "general")
        emotional_context = experience_data.get("emotional_context", {})
        user_id = experience_data.get("user_id", "unknown")
        
        # Create pattern key
        pattern_key = f"{scenario_type}_{emotional_context.get('primary_emotion', 'neutral')}"
        
        if pattern_key not in self.consolidation_patterns:
            self.consolidation_patterns[pattern_key] = {
                "occurrences": 0,
                "users": set(),
                "timestamps": [],
                "average_significance": 0.0
            }
        
        pattern = self.consolidation_patterns[pattern_key]
        pattern["occurrences"] += 1
        pattern["users"].add(user_id)
        pattern["timestamps"].append(datetime.now())
        
        # Update average significance
        current_avg = pattern["average_significance"]
        new_significance = experience_data.get("significance", 5)
        pattern["average_significance"] = (
            (current_avg * (pattern["occurrences"] - 1) + new_significance) / 
            pattern["occurrences"]
        )
        
        # Check if pattern is strong enough to trigger consolidation
        if pattern["occurrences"] >= self.original_system.min_group_size:
            await self._check_pattern_consolidation_trigger(pattern_key, pattern)
    
    async def _mark_for_priority_consolidation(self, emotional_data: Dict[str, Any]):
        """Mark high-intensity emotional experiences for priority consolidation"""
        experience_ids = emotional_data.get("experience_ids", [])
        emotion = emotional_data.get("primary_emotion", "unknown")
        intensity = emotional_data.get("intensity", 0.5)
        
        if not experience_ids:
            return
        
        # Create priority group
        group_id = f"emotional_{emotion}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.pending_consolidations[group_id] = {
            "type": "emotional_priority",
            "emotion": emotion,
            "intensity": intensity,
            "experience_ids": experience_ids,
            "priority": 0.8 + (intensity * 0.2),  # High priority
            "created_at": datetime.now()
        }
        
        # Send notification
        await self.send_context_update(
            update_type="priority_consolidation_queued",
            data={
                "group_id": group_id,
                "reason": "high_emotional_intensity",
                "emotion": emotion,
                "intensity": intensity
            }
        )
    
    async def _group_goal_related_experiences(self, goal_id: str, goal_data: Dict[str, Any]):
        """Group experiences related to the same goal"""
        experience_ids = goal_data.get("related_experiences", [])
        
        if not experience_ids:
            return
        
        group_key = f"goal_{goal_id}"
        
        if group_key not in self.pending_consolidations:
            self.pending_consolidations[group_key] = {
                "type": "goal_related",
                "goal_id": goal_id,
                "goal_description": goal_data.get("goal_description", ""),
                "experience_ids": [],
                "priority": 0.7,  # Moderate-high priority
                "created_at": datetime.now()
            }
        
        # Add new experiences
        group = self.pending_consolidations[group_key]
        for exp_id in experience_ids:
            if exp_id not in group["experience_ids"]:
                group["experience_ids"].append(exp_id)
        
        # Check if ready for consolidation
        if len(group["experience_ids"]) >= self.original_system.min_group_size:
            await self._mark_group_ready_for_consolidation(group_key)
    
    async def _handle_identity_impacting_experience(self, identity_data: Dict[str, Any]):
        """Handle experiences that impact identity for special consolidation"""
        experience_id = identity_data.get("experience_id")
        impact_type = identity_data.get("impact_type", "general")
        impact_strength = identity_data.get("impact_strength", 0.5)
        
        if not experience_id:
            return
        
        # Group by impact type
        group_key = f"identity_{impact_type}"
        
        if group_key not in self.pending_consolidations:
            self.pending_consolidations[group_key] = {
                "type": "identity_impact",
                "impact_type": impact_type,
                "experience_ids": [],
                "total_impact": 0.0,
                "priority": 0.6,  # Moderate priority by default
                "created_at": datetime.now()
            }
        
        group = self.pending_consolidations[group_key]
        group["experience_ids"].append(experience_id)
        group["total_impact"] += impact_strength
        
        # Adjust priority based on total impact
        group["priority"] = min(0.9, 0.6 + (group["total_impact"] / 10))
    
    async def _process_consolidation_trigger(self, trigger_data: Dict[str, Any]):
        """Process external consolidation trigger"""
        trigger_type = trigger_data.get("type", "manual")
        target = trigger_data.get("target", "all")
        
        logger.info(f"Processing consolidation trigger: {trigger_type} for {target}")
        
        if trigger_type == "manual":
            # Manual trigger - run consolidation immediately
            await self._execute_triggered_consolidation(target, trigger_data)
        
        elif trigger_type == "memory_pressure":
            # Memory pressure - prioritize largest groups
            await self._consolidate_for_memory_optimization()
        
        elif trigger_type == "pattern_detected":
            # Pattern-based trigger
            pattern_info = trigger_data.get("pattern_info", {})
            await self._consolidate_detected_pattern(pattern_info)
    
    async def _incorporate_detected_pattern(self, pattern_data: Dict[str, Any]):
        """Incorporate newly detected pattern into consolidation planning"""
        pattern_type = pattern_data.get("pattern_type", "unknown")
        experience_ids = pattern_data.get("experience_ids", [])
        confidence = pattern_data.get("confidence", 0.5)
        
        if len(experience_ids) < self.original_system.min_group_size:
            return
        
        # Create pattern-based group
        group_id = f"pattern_{pattern_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.pending_consolidations[group_id] = {
            "type": "detected_pattern",
            "pattern_type": pattern_type,
            "experience_ids": experience_ids,
            "confidence": confidence,
            "priority": 0.5 + (confidence * 0.4),  # Priority based on confidence
            "created_at": datetime.now()
        }
        
        # If high confidence, mark for consolidation
        if confidence > 0.8:
            await self._mark_group_ready_for_consolidation(group_id)
    
    async def _adjust_consolidation_from_feedback(self, feedback_data: Dict[str, Any]):
        """Adjust consolidation priorities based on user feedback"""
        feedback_type = feedback_data.get("type", "general")
        satisfaction = feedback_data.get("satisfaction", 0.5)
        
        if feedback_type == "consolidation_quality":
            # Feedback on consolidation quality
            consolidation_id = feedback_data.get("consolidation_id")
            
            if satisfaction < 0.3:
                # Poor feedback - adjust strategy
                logger.warning(f"Poor feedback on consolidation {consolidation_id}")
                
                # Reduce similarity threshold for future consolidations
                self.original_system.similarity_threshold = max(
                    0.6, 
                    self.original_system.similarity_threshold - 0.05
                )
            
            elif satisfaction > 0.8:
                # Good feedback - can be more aggressive
                self.original_system.similarity_threshold = min(
                    0.8,
                    self.original_system.similarity_threshold + 0.02
                )
        
        elif feedback_type == "memory_recall":
            # Feedback on memory recall effectiveness
            if satisfaction < 0.4:
                # Poor recall - may need better consolidation
                await self.send_context_update(
                    update_type="recall_improvement_needed",
                    data={
                        "satisfaction": satisfaction,
                        "suggestion": "increase_consolidation_frequency"
                    }
                )
    
    async def _check_input_for_consolidation_references(self, context: SharedContext) -> Dict[str, Any]:
        """Check if input references consolidated experiences"""
        references = {
            "references_consolidation": False,
            "consolidation_keywords": [],
            "potential_expansions": []
        }
        
        input_lower = context.user_input.lower()
        
        # Check for consolidation keywords
        consolidation_keywords = [
            "pattern", "tendency", "usually", "often", "generally",
            "experiences show", "learned that", "noticed that"
        ]
        
        found_keywords = [kw for kw in consolidation_keywords if kw in input_lower]
        references["consolidation_keywords"] = found_keywords
        references["references_consolidation"] = len(found_keywords) > 0
        
        # Check for requests to expand consolidations
        expansion_keywords = ["tell me more about", "expand on", "details about", "specific examples"]
        
        if any(kw in input_lower for kw in expansion_keywords):
            references["potential_expansions"].append("user_requests_expansion")
        
        return references
    
    async def _check_immediate_consolidation_trigger(self, 
                                                    context: SharedContext,
                                                    messages: Dict[str, List[Dict]]) -> Optional[str]:
        """Check if immediate consolidation is needed"""
        # Check for explicit trigger in input
        if "consolidate" in context.user_input.lower() or "summarize experiences" in context.user_input.lower():
            return "explicit_user_request"
        
        # Check for memory pressure signals
        for module_messages in messages.values():
            for msg in module_messages:
                if msg.get("type") == "memory_pressure_high":
                    return "memory_pressure"
        
        # Check for critical mass in pending consolidations
        large_pending_groups = [
            group for group in self.pending_consolidations.values()
            if len(group.get("experience_ids", [])) >= self.original_system.max_group_size
        ]
        
        if large_pending_groups:
            return "group_size_limit_reached"
        
        # Check for time-based trigger
        readiness = await self._check_consolidation_readiness(context)
        if readiness["ready"] and readiness.get("forced_reason"):
            return readiness["forced_reason"]
        
        return None
    
    async def _execute_immediate_consolidation(self, 
                                             context: SharedContext,
                                             trigger: str) -> Dict[str, Any]:
        """Execute immediate consolidation based on trigger"""
        logger.info(f"Executing immediate consolidation due to: {trigger}")
        
        # Select groups based on trigger
        if trigger == "memory_pressure":
            # Prioritize largest groups
            selected_groups = sorted(
                self.pending_consolidations.items(),
                key=lambda x: len(x[1].get("experience_ids", [])),
                reverse=True
            )[:5]  # Top 5 largest
        
        elif trigger == "explicit_user_request":
            # Use all ready groups
            selected_groups = [
                (gid, gdata) for gid, gdata in self.pending_consolidations.items()
                if len(gdata.get("experience_ids", [])) >= self.original_system.min_group_size
            ]
        
        else:
            # Default: highest priority groups
            selected_groups = sorted(
                self.pending_consolidations.items(),
                key=lambda x: x[1].get("priority", 0.5),
                reverse=True
            )[:3]  # Top 3 priority
        
        # Execute consolidation
        consolidation_results = {
            "trigger": trigger,
            "groups_processed": 0,
            "consolidations_created": 0,
            "experiences_consolidated": 0,
            "errors": []
        }
        
        for group_id, group_data in selected_groups:
            try:
                # Create consolidation candidate
                candidate = await self._create_consolidation_candidate(group_id, group_data)
                
                if candidate:
                    # Execute consolidation
                    result = await self.original_system.create_consolidated_experience(candidate)
                    
                    if result:
                        consolidation_results["consolidations_created"] += 1
                        consolidation_results["experiences_consolidated"] += len(
                            group_data.get("experience_ids", [])
                        )
                        
                        # Remove from pending
                        del self.pending_consolidations[group_id]
                    
                    consolidation_results["groups_processed"] += 1
                    
            except Exception as e:
                logger.error(f"Error consolidating group {group_id}: {e}")
                consolidation_results["errors"].append({
                    "group_id": group_id,
                    "error": str(e)
                })
        
        # Update last consolidation time
        self.original_system.last_consolidation = datetime.now()
        
        return consolidation_results
    
    async def _monitor_consolidation_opportunities(self, 
                                                 context: SharedContext,
                                                 messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Monitor for consolidation opportunities"""
        monitoring = {
            "pending_groups": len(self.pending_consolidations),
            "ready_groups": 0,
            "total_pending_experiences": 0,
            "high_priority_groups": 0,
            "pattern_strength": {}
        }
        
        # Analyze pending groups
        for group_id, group_data in self.pending_consolidations.items():
            exp_count = len(group_data.get("experience_ids", []))
            monitoring["total_pending_experiences"] += exp_count
            
            if exp_count >= self.original_system.min_group_size:
                monitoring["ready_groups"] += 1
            
            if group_data.get("priority", 0.5) > 0.7:
                monitoring["high_priority_groups"] += 1
        
        # Analyze pattern strength
        for pattern_key, pattern_data in self.consolidation_patterns.items():
            if pattern_data["occurrences"] >= 3:  # Minimum for pattern
                monitoring["pattern_strength"][pattern_key] = {
                    "occurrences": pattern_data["occurrences"],
                    "users": len(pattern_data["users"]),
                    "significance": pattern_data["average_significance"]
                }
        
        # Check for emerging patterns from cross-module messages
        emerging_patterns = self._detect_emerging_patterns(messages)
        if emerging_patterns:
            monitoring["emerging_patterns"] = emerging_patterns
        
        return monitoring
    
    async def _comprehensive_experience_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Comprehensive analysis of all experiences"""
        analysis = {
            "total_experiences": 0,
            "consolidation_coverage": 0.0,
            "experience_distribution": {},
            "consolidation_effectiveness": {},
            "memory_efficiency": 0.0,
            "pattern_coverage": {}
        }
        
        if not self.original_system.memory_core:
            return analysis
        
        try:
            # Get all experiences
            all_experiences = await self.original_system.memory_core.retrieve_memories(
                query="*",
                memory_types=["experience", "observation", "episodic", "consolidated"],
                limit=2000
            )
            
            analysis["total_experiences"] = len(all_experiences)
            
            # Categorize experiences
            consolidated = []
            unconsolidated = []
            
            for exp in all_experiences:
                metadata = exp.get("metadata", {})
                if metadata.get("is_consolidation", False):
                    consolidated.append(exp)
                elif not metadata.get("consolidated_into"):
                    unconsolidated.append(exp)
            
            # Calculate coverage
            total_represented = len(consolidated) * 3  # Assume each consolidation represents ~3 experiences
            analysis["consolidation_coverage"] = min(
                1.0, 
                total_represented / max(1, len(all_experiences))
            )
            
            # Analyze distribution by type
            for exp in all_experiences:
                exp_type = exp.get("memory_type", "unknown")
                if exp_type not in analysis["experience_distribution"]:
                    analysis["experience_distribution"][exp_type] = {
                        "total": 0,
                        "consolidated": 0,
                        "unconsolidated": 0
                    }
                
                analysis["experience_distribution"][exp_type]["total"] += 1
                
                metadata = exp.get("metadata", {})
                if metadata.get("is_consolidation", False):
                    analysis["experience_distribution"][exp_type]["consolidated"] += 1
                elif not metadata.get("consolidated_into"):
                    analysis["experience_distribution"][exp_type]["unconsolidated"] += 1
            
            # Calculate memory efficiency
            # Efficiency = reduction in memory items while maintaining information
            if len(all_experiences) > 0:
                analysis["memory_efficiency"] = 1.0 - (len(unconsolidated) / len(all_experiences))
            
            # Analyze pattern coverage
            for pattern_key, pattern_data in self.consolidation_patterns.items():
                if pattern_data["occurrences"] >= 3:
                    analysis["pattern_coverage"][pattern_key] = {
                        "coverage": min(1.0, pattern_data["occurrences"] / 10),
                        "cross_user": len(pattern_data["users"]) > 1,
                        "temporal_spread": self._calculate_temporal_spread(pattern_data["timestamps"])
                    }
            
        except Exception as e:
            logger.error(f"Error in comprehensive experience analysis: {e}")
        
        return analysis
    
    async def _analyze_experience_patterns(self, experience_landscape: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns across experiences"""
        patterns = {
            "identified_patterns": [],
            "pattern_strength": {},
            "cross_user_patterns": [],
            "temporal_patterns": [],
            "scenario_patterns": []
        }
        
        # Analyze scenario patterns
        scenario_dist = experience_landscape.get("scenario_distribution", {})
        for scenario, count in scenario_dist.items():
            if count >= 5:  # Minimum for pattern
                patterns["scenario_patterns"].append({
                    "scenario": scenario,
                    "frequency": count,
                    "percentage": count / max(1, experience_landscape.get("total_experiences", 1))
                })
        
        # Analyze temporal patterns
        temporal_dist = experience_landscape.get("temporal_distribution", {})
        temporal_clusters = self._identify_temporal_clusters(temporal_dist)
        patterns["temporal_patterns"] = temporal_clusters
        
        # Analyze cross-user patterns
        if len(experience_landscape.get("user_distribution", {})) > 1:
            # Look for experiences that appear across multiple users
            for pattern_key, pattern_data in self.consolidation_patterns.items():
                if len(pattern_data["users"]) > 1:
                    patterns["cross_user_patterns"].append({
                        "pattern": pattern_key,
                        "users": list(pattern_data["users"]),
                        "occurrences": pattern_data["occurrences"],
                        "significance": pattern_data["average_significance"]
                    })
        
        # Calculate pattern strength
        for pattern_key, pattern_data in self.consolidation_patterns.items():
            strength = self._calculate_pattern_strength(pattern_data)
            patterns["pattern_strength"][pattern_key] = strength
            
            if strength > 0.7:
                patterns["identified_patterns"].append({
                    "pattern": pattern_key,
                    "strength": strength,
                    "ready_for_consolidation": pattern_data["occurrences"] >= self.original_system.min_group_size
                })
        
        return patterns
    
    async def _analyze_consolidation_effectiveness(self) -> Dict[str, Any]:
        """Analyze how effective consolidations have been"""
        effectiveness = {
            "total_consolidations": len(self.original_system.consolidation_history),
            "average_quality": 0.0,
            "quality_trend": "stable",
            "consolidation_types": {},
            "user_satisfaction": 0.5  # Default
        }
        
        if not self.original_system.consolidation_history:
            return effectiveness
        
        # Calculate average quality
        quality_scores = [
            entry.get("quality_score", 0.5) 
            for entry in self.original_system.consolidation_history
        ]
        
        if quality_scores:
            effectiveness["average_quality"] = sum(quality_scores) / len(quality_scores)
        
        # Analyze quality trend
        if len(quality_scores) >= 5:
            recent = quality_scores[-5:]
            older = quality_scores[:-5]
            
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older) if older else recent_avg
            
            if recent_avg > older_avg * 1.1:
                effectiveness["quality_trend"] = "improving"
            elif recent_avg < older_avg * 0.9:
                effectiveness["quality_trend"] = "declining"
        
        # Analyze by consolidation type
        for entry in self.original_system.consolidation_history:
            c_type = entry.get("consolidation_type", "unknown")
            if c_type not in effectiveness["consolidation_types"]:
                effectiveness["consolidation_types"][c_type] = {
                    "count": 0,
                    "average_quality": 0.0,
                    "total_quality": 0.0
                }
            
            type_stats = effectiveness["consolidation_types"][c_type]
            type_stats["count"] += 1
            type_stats["total_quality"] += entry.get("quality_score", 0.5)
            type_stats["average_quality"] = type_stats["total_quality"] / type_stats["count"]
        
        return effectiveness
    
    async def _analyze_cross_module_experience_impact(self, messages: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze how experiences impact other modules"""
        impact = {
            "experience_references": 0,
            "module_experience_usage": {},
            "experience_driven_decisions": [],
            "experience_effectiveness": {}
        }
        
        # Analyze messages for experience references
        for module_name, module_messages in messages.items():
            module_refs = 0
            
            for msg in module_messages:
                msg_type = msg.get("type", "")
                msg_data = msg.get("data", {})
                
                # Check for experience references
                if any(exp_word in msg_type.lower() for exp_word in ["experience", "memory", "recall"]):
                    module_refs += 1
                    impact["experience_references"] += 1
                    
                    # Track experience-driven decisions
                    if "decision" in msg_type or "action" in msg_type:
                        impact["experience_driven_decisions"].append({
                            "module": module_name,
                            "decision_type": msg_type,
                            "timestamp": msg.get("timestamp")
                        })
                
                # Check for experience effectiveness feedback
                if "experience_utility" in msg_data:
                    utility = msg_data["experience_utility"]
                    if module_name not in impact["experience_effectiveness"]:
                        impact["experience_effectiveness"][module_name] = []
                    impact["experience_effectiveness"][module_name].append(utility)
            
            if module_refs > 0:
                impact["module_experience_usage"][module_name] = module_refs
        
        # Calculate average effectiveness per module
        for module_name, utilities in impact["experience_effectiveness"].items():
            if utilities:
                avg_utility = sum(utilities) / len(utilities)
                impact["experience_effectiveness"][module_name] = avg_utility
        
        return impact
    
    async def _predict_consolidation_needs(self,
                                         experience_landscape: Dict[str, Any],
                                         pattern_analysis: Dict[str, Any],
                                         cross_module_impact: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future consolidation needs"""
        predictions = {
            "next_consolidation_needed": "unknown",
            "estimated_groups": 0,
            "priority_areas": [],
            "memory_pressure_risk": "low",
            "recommendations": []
        }
        
        # Calculate experience accumulation rate
        temporal_dist = experience_landscape.get("temporal_distribution", {})
        if temporal_dist:
            dates = sorted(temporal_dist.keys())
            if len(dates) >= 7:  # Need at least a week of data
                recent_dates = dates[-7:]
                recent_counts = [temporal_dist[d] for d in recent_dates]
                daily_rate = sum(recent_counts) / len(recent_counts)
                
                # Predict when consolidation will be needed
                unconsolidated = experience_landscape.get("unconsolidated_experiences", 0)
                threshold = 50  # Threshold for consolidation need
                
                if daily_rate > 0:
                    days_until_threshold = (threshold - unconsolidated) / daily_rate
                    
                    if days_until_threshold < 1:
                        predictions["next_consolidation_needed"] = "immediate"
                    elif days_until_threshold < 3:
                        predictions["next_consolidation_needed"] = "soon"
                    elif days_until_threshold < 7:
                        predictions["next_consolidation_needed"] = "this_week"
                    else:
                        predictions["next_consolidation_needed"] = "not_urgent"
        
        # Estimate number of groups
        ready_patterns = [
            p for p in pattern_analysis.get("identified_patterns", [])
            if p.get("ready_for_consolidation", False)
        ]
        predictions["estimated_groups"] = len(ready_patterns) + len(self.pending_consolidations)
        
        # Identify priority areas
        if pattern_analysis.get("scenario_patterns"):
            top_scenarios = sorted(
                pattern_analysis["scenario_patterns"],
                key=lambda x: x["frequency"],
                reverse=True
            )[:3]
            
            for scenario in top_scenarios:
                predictions["priority_areas"].append({
                    "area": scenario["scenario"],
                    "reason": f"High frequency ({scenario['frequency']} experiences)"
                })
        
        # Assess memory pressure risk
        total_experiences = experience_landscape.get("total_experiences", 0)
        consolidation_ratio = experience_landscape.get("consolidation_ratio", 1.0)
        
        if total_experiences > 1000 and consolidation_ratio < 0.3:
            predictions["memory_pressure_risk"] = "high"
            predictions["recommendations"].append("Urgent consolidation needed to reduce memory load")
        elif total_experiences > 500 and consolidation_ratio < 0.5:
            predictions["memory_pressure_risk"] = "medium"
            predictions["recommendations"].append("Regular consolidation recommended")
        
        # Generate recommendations based on patterns
        if len(pattern_analysis.get("cross_user_patterns", [])) > 3:
            predictions["recommendations"].append(
                "Multiple cross-user patterns detected - consider cross-user consolidation"
            )
        
        if cross_module_impact.get("experience_effectiveness"):
            low_effectiveness_modules = [
                m for m, eff in cross_module_impact["experience_effectiveness"].items()
                if eff < 0.4
            ]
            if low_effectiveness_modules:
                predictions["recommendations"].append(
                    f"Improve consolidation quality for modules: {', '.join(low_effectiveness_modules)}"
                )
        
        return predictions
    
    async def _assess_system_load_for_consolidation(self, context: SharedContext) -> float:
        """Assess if system load permits consolidation"""
        load_factors = []
        
        # Check active modules
        active_module_ratio = len(context.active_modules) / 15  # Assume 15 max
        load_factors.append(active_module_ratio)
        
        # Check processing stage
        if context.processing_stage in ["synthesis", "output"]:
            load_factors.append(0.8)  # High load during synthesis
        else:
            load_factors.append(0.3)
        
        # Check context update rate
        update_rate = len(context.context_updates) / max(1, (datetime.now() - context.created_at).total_seconds())
        load_factors.append(min(1.0, update_rate / 10))  # Normalize to 10 updates/sec
        
        return sum(load_factors) / len(load_factors) if load_factors else 0.5
    
    async def _check_memory_pressure(self) -> bool:
        """Check if there's memory pressure requiring consolidation"""
        if not self.original_system.memory_core:
            return False
        
        try:
            # Simple heuristic: if we have too many unconsolidated experiences
            stats = await self.original_system.memory_core.get_statistics()
            
            total_memories = stats.get("total_memories", 0)
            if total_memories > 1000:  # Threshold for memory pressure
                return True
                
        except:
            pass
        
        return False
    
    def _identify_temporal_clusters(self, temporal_distribution: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify temporal clusters in experience distribution"""
        if not temporal_distribution:
            return []
        
        clusters = []
        sorted_dates = sorted(temporal_distribution.keys())
        
        current_cluster = {
            "start": sorted_dates[0],
            "end": sorted_dates[0],
            "count": temporal_distribution[sorted_dates[0]],
            "dates": [sorted_dates[0]]
        }
        
        for i in range(1, len(sorted_dates)):
            current_date = datetime.strptime(sorted_dates[i], "%Y-%m-%d")
            prev_date = datetime.strptime(sorted_dates[i-1], "%Y-%m-%d")
            
            # If dates are consecutive (within 2 days), add to cluster
            if (current_date - prev_date).days <= 2:
                current_cluster["end"] = sorted_dates[i]
                current_cluster["count"] += temporal_distribution[sorted_dates[i]]
                current_cluster["dates"].append(sorted_dates[i])
            else:
                # Start new cluster
                if current_cluster["count"] >= self.original_system.min_group_size:
                    current_cluster["range"] = f"{current_cluster['start']} to {current_cluster['end']}"
                    current_cluster["priority"] = min(0.9, current_cluster["count"] / 20)
                    clusters.append(current_cluster)
                
                current_cluster = {
                    "start": sorted_dates[i],
                    "end": sorted_dates[i],
                    "count": temporal_distribution[sorted_dates[i]],
                    "dates": [sorted_dates[i]]
                }
        
        # Don't forget the last cluster
        if current_cluster["count"] >= self.original_system.min_group_size:
            current_cluster["range"] = f"{current_cluster['start']} to {current_cluster['end']}"
            current_cluster["priority"] = min(0.9, current_cluster["count"] / 20)
            clusters.append(current_cluster)
        
        return clusters
    
    async def _identify_cross_user_patterns(self, experience_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify patterns that appear across multiple users"""
        cross_user_opportunities = []
        
        # Group experiences by scenario across users
        scenario_dist = experience_analysis.get("scenario_distribution", {})
        user_dist = experience_analysis.get("user_distribution", {})
        
        if len(user_dist) < 2:
            return []  # Need at least 2 users
        
        for scenario, total_count in scenario_dist.items():
            if total_count >= self.original_system.min_group_size * 2:  # Higher threshold for cross-user
                # Check if this scenario appears for multiple users
                # This is simplified - in reality would need to query per-user data
                cross_user_opportunities.append({
                    "type": "cross_user_scenario",
                    "scenario": scenario,
                    "experience_count": total_count,
                    "estimated_users": min(len(user_dist), total_count // 3),  # Rough estimate
                    "priority": 0.6 + min(0.3, total_count / 50)  # Higher priority for more experiences
                })
        
        return cross_user_opportunities
    
    def _calculate_opportunity_priority(self, scenario: str, count: int) -> float:
        """Calculate priority for a consolidation opportunity"""
        base_priority = 0.5
        
        # Count factor
        count_factor = min(0.3, count / 30)  # Max 0.3 for 30+ experiences
        
        # Scenario importance factor
        important_scenarios = ["emotional", "goal", "identity", "relationship"]
        scenario_factor = 0.2 if any(s in scenario.lower() for s in important_scenarios) else 0.0
        
        # Recency factor (would need timestamp data)
        recency_factor = 0.1  # Default
        
        return base_priority + count_factor + scenario_factor + recency_factor
    
    async def _mark_group_ready_for_consolidation(self, group_id: str):
        """Mark a group as ready for consolidation"""
        if group_id in self.pending_consolidations:
            group = self.pending_consolidations[group_id]
            group["ready"] = True
            group["marked_ready_at"] = datetime.now()
            
            # Send notification
            await self.send_context_update(
                update_type="consolidation_group_ready",
                data={
                    "group_id": group_id,
                    "group_type": group.get("type", "unknown"),
                    "experience_count": len(group.get("experience_ids", [])),
                    "priority": group.get("priority", 0.5)
                }
            )
    
    async def _check_pattern_consolidation_trigger(self, pattern_key: str, pattern: Dict[str, Any]):
        """Check if a pattern should trigger consolidation"""
        # Strong pattern criteria
        if (pattern["occurrences"] >= self.original_system.min_group_size and
            pattern["average_significance"] > 6 and
            len(pattern["users"]) > 1):
            
            # Create consolidation group from pattern
            group_id = f"pattern_{pattern_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Would need to find actual experience IDs matching this pattern
            # This is a simplified version
            self.pending_consolidations[group_id] = {
                "type": "pattern_based",
                "pattern": pattern_key,
                "priority": 0.7,
                "created_at": datetime.now(),
                "auto_triggered": True,
                "pattern_strength": pattern["occurrences"] / 10
            }
            
            logger.info(f"Pattern {pattern_key} triggered consolidation group {group_id}")
    
    async def _execute_triggered_consolidation(self, target: str, trigger_data: Dict[str, Any]):
        """Execute consolidation based on trigger"""
        if target == "all":
            # Run full consolidation cycle
            result = await self.original_system.run_consolidation_cycle()
        else:
            # Target specific group or pattern
            if target in self.pending_consolidations:
                # Consolidate specific group
                group_data = self.pending_consolidations[target]
                candidate = await self._create_consolidation_candidate(target, group_data)
                
                if candidate:
                    result = await self.original_system.create_consolidated_experience(candidate)
                    
                    if result:
                        del self.pending_consolidations[target]
                else:
                    result = {"error": "Failed to create candidate"}
            else:
                result = {"error": f"Target group {target} not found"}
        
        # Send result notification
        await self.send_context_update(
            update_type="triggered_consolidation_complete",
            data={
                "trigger": trigger_data,
                "target": target,
                "result": result
            }
        )
    
    async def _consolidate_for_memory_optimization(self):
        """Consolidate specifically for memory optimization"""
        # Sort groups by size (largest first for maximum memory reduction)
        sorted_groups = sorted(
            self.pending_consolidations.items(),
            key=lambda x: len(x[1].get("experience_ids", [])),
            reverse=True
        )
        
        consolidated_count = 0
        total_experiences_consolidated = 0
        
        for group_id, group_data in sorted_groups[:10]:  # Process up to 10 largest groups
            if len(group_data.get("experience_ids", [])) >= self.original_system.min_group_size:
                try:
                    candidate = await self._create_consolidation_candidate(group_id, group_data)
                    
                    if candidate:
                        result = await self.original_system.create_consolidated_experience(candidate)
                        
                        if result:
                            consolidated_count += 1
                            total_experiences_consolidated += len(group_data.get("experience_ids", []))
                            del self.pending_consolidations[group_id]
                            
                except Exception as e:
                    logger.error(f"Error in memory optimization consolidation: {e}")
        
        logger.info(f"Memory optimization: consolidated {consolidated_count} groups, {total_experiences_consolidated} experiences")
    
    async def _consolidate_detected_pattern(self, pattern_info: Dict[str, Any]):
        """Consolidate based on detected pattern"""
        pattern_type = pattern_info.get("pattern_type")
        experience_ids = pattern_info.get("experience_ids", [])
        
        if len(experience_ids) >= self.original_system.min_group_size:
            # Create candidate from pattern
            from nyx.core.experience_consolidation import ConsolidationCandidate
            
            candidate = ConsolidationCandidate(
                source_ids=experience_ids,
                similarity_score=pattern_info.get("confidence", 0.7),
                scenario_type=pattern_info.get("scenario_type", "pattern"),
                theme=f"Pattern: {pattern_type}",
                consolidation_type="pattern"
            )
            
            # Execute consolidation
            result = await self.original_system.create_consolidated_experience(candidate)
            
            if result:
                logger.info(f"Successfully consolidated pattern {pattern_type}")
    
    async def _create_consolidation_candidate(self, group_id: str, group_data: Dict[str, Any]):
        """Create a consolidation candidate from group data"""
        from nyx.core.experience_consolidation import ConsolidationCandidate
        
        experience_ids = group_data.get("experience_ids", [])
        
        if len(experience_ids) < self.original_system.min_group_size:
            return None
        
        # Determine consolidation type
        group_type = group_data.get("type", "general")
        if group_type == "emotional_priority":
            consolidation_type = "emotional"
        elif group_type == "goal_related":
            consolidation_type = "goal"
        elif group_type == "identity_impact":
            consolidation_type = "identity"
        elif group_type == "pattern_based":
            consolidation_type = "pattern"
        else:
            consolidation_type = "abstraction"
        
        # Create theme
        if group_type == "emotional_priority":
            theme = f"{group_data.get('emotion', 'emotional')} experiences"
        elif group_type == "goal_related":
            theme = f"Goal: {group_data.get('goal_description', 'achievement')}"
        elif group_type == "scenario_based":
            theme = f"{group_data.get('scenario', 'general')} scenario experiences"
        else:
            theme = group_data.get("pattern", "General experiences")
        
        # Get user IDs (would need to query from actual experiences)
        user_ids = []
        if "users" in group_data:
            user_ids = list(group_data["users"])
        
        candidate = ConsolidationCandidate(
            source_ids=experience_ids,
            similarity_score=group_data.get("priority", 0.7),  # Use priority as proxy for similarity
            scenario_type=group_data.get("scenario", "general"),
            theme=theme,
            user_ids=user_ids,
            consolidation_type=consolidation_type
        )
        
        return candidate
    
    def _detect_emerging_patterns(self, messages: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Detect emerging patterns from cross-module messages"""
        emerging = []
        
        # Count message types across modules
        message_type_counts = {}
        
        for module_messages in messages.values():
            for msg in module_messages:
                msg_type = msg.get("type", "")
                
                # Look for experience-related messages
                if "experience" in msg_type or "memory" in msg_type:
                    if msg_type not in message_type_counts:
                        message_type_counts[msg_type] = 0
                    message_type_counts[msg_type] += 1
        
        # Identify patterns
        for msg_type, count in message_type_counts.items():
            if count >= 3:  # Pattern threshold
                emerging.append({
                    "pattern_type": "message_pattern",
                    "message_type": msg_type,
                    "frequency": count,
                    "significance": "emerging"
                })
        
        return emerging
    
    def _calculate_pattern_strength(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate the strength of a pattern"""
        strength = 0.0
        
        # Occurrence factor
        occurrences = pattern_data.get("occurrences", 0)
        strength += min(0.4, occurrences / 20)  # Max 0.4 for 20+ occurrences
        
        # User diversity factor
        users = pattern_data.get("users", set())
        if isinstance(users, set):
            user_count = len(users)
        else:
            user_count = users
        strength += min(0.3, user_count / 5)  # Max 0.3 for 5+ users
        
        # Significance factor
        avg_significance = pattern_data.get("average_significance", 5)
        strength += (avg_significance / 10) * 0.2  # Max 0.2
        
        # Temporal consistency factor
        if "timestamps" in pattern_data and len(pattern_data["timestamps"]) >= 3:
            temporal_spread = self._calculate_temporal_spread(pattern_data["timestamps"])
            if temporal_spread > 0.5:  # Pattern spans significant time
                strength += 0.1
        
        return min(1.0, strength)
    
    def _calculate_temporal_spread(self, timestamps: List[datetime]) -> float:
        """Calculate temporal spread of timestamps"""
        if len(timestamps) < 2:
            return 0.0
        
        sorted_times = sorted(timestamps)
        total_span = (sorted_times[-1] - sorted_times[0]).total_seconds()
        
        if total_span == 0:
            return 0.0
        
        # Calculate average gap
        gaps = []
        for i in range(1, len(sorted_times)):
            gap = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps) if gaps else 0
        
        # Normalize spread (1.0 = perfectly even distribution)
        expected_gap = total_span / (len(timestamps) - 1)
        
        if expected_gap > 0:
            spread = 1.0 - min(1.0, abs(avg_gap - expected_gap) / expected_gap)
        else:
            spread = 0.0
        
        return spread
    
    async def _get_consolidation_status_summary(self) -> Dict[str, Any]:
        """Get summary of consolidation status"""
        return {
            "last_consolidation": self.original_system.last_consolidation.isoformat(),
            "pending_groups": len(self.pending_consolidations),
            "ready_groups": sum(1 for g in self.pending_consolidations.values() if g.get("ready", False)),
            "total_pending_experiences": sum(
                len(g.get("experience_ids", [])) for g in self.pending_consolidations.values()
            ),
            "active_patterns": len([p for p in self.consolidation_patterns.values() if p["occurrences"] >= 3]),
            "consolidation_interval": self.original_system.consolidation_interval,
            "similarity_threshold": self.original_system.similarity_threshold
        }
    
    async def _get_recent_consolidation_summary(self) -> List[Dict[str, Any]]:
        """Get summary of recent consolidations"""
        recent = self.original_system.consolidation_history[-5:]  # Last 5
        
        summaries = []
        for entry in recent:
            summaries.append({
                "timestamp": entry.get("timestamp"),
                "consolidated_id": entry.get("consolidated_id"),
                "source_count": entry.get("source_count", 0),
                "quality_score": entry.get("quality_score", 0.5),
                "consolidation_type": entry.get("consolidation_type", "unknown")
            })
        
        return summaries
    
    async def _synthesize_pattern_insights(self, context: SharedContext) -> List[str]:
        """Synthesize insights from patterns"""
        insights = []
        
        # Strong patterns
        strong_patterns = [
            (k, v) for k, v in self.consolidation_patterns.items()
            if self._calculate_pattern_strength(v) > 0.7
        ]
        
        if strong_patterns:
            insights.append(f"Identified {len(strong_patterns)} strong experience patterns")
            
            # Most common pattern
            if strong_patterns:
                most_common = max(strong_patterns, key=lambda x: x[1]["occurrences"])
                insights.append(f"Most frequent pattern: {most_common[0]} ({most_common[1]['occurrences']} occurrences)")
        
        # Cross-user patterns
        cross_user_patterns = [
            p for p in self.consolidation_patterns.values()
            if isinstance(p.get("users", set()), set) and len(p["users"]) > 1
        ]
        
        if cross_user_patterns:
            insights.append(f"{len(cross_user_patterns)} patterns appear across multiple users")
        
        # Pending consolidations
        if self.pending_consolidations:
            high_priority = sum(1 for g in self.pending_consolidations.values() if g.get("priority", 0.5) > 0.7)
            if high_priority > 0:
                insights.append(f"{high_priority} high-priority consolidation groups pending")
        
        return insights
    
    async def _generate_consolidation_recommendations(self, 
                                                    context: SharedContext,
                                                    messages: Dict[str, List[Dict]]) -> List[str]:
        """Generate consolidation recommendations"""
        recommendations = []
        
        # Check pending groups
        ready_groups = [g for g in self.pending_consolidations.values() if g.get("ready", False)]
        
        if len(ready_groups) >= 5:
            recommendations.append(f"Run consolidation cycle - {len(ready_groups)} groups ready")
        
        # Check pattern strength
        strong_patterns = sum(1 for p in self.consolidation_patterns.values() 
                            if self._calculate_pattern_strength(p) > 0.8)
        
        if strong_patterns > 3:
            recommendations.append("Strong patterns detected - consider pattern-based consolidation")
        
        # Check memory pressure
        if await self._check_memory_pressure():
            recommendations.append("Memory pressure detected - prioritize consolidation")
        
        # Check cross-module feedback
        low_utility_modules = []
        for module_name, module_messages in messages.items():
            for msg in module_messages:
                if msg.get("type") == "experience_utility_low":
                    low_utility_modules.append(module_name)
        
        if low_utility_modules:
            recommendations.append(
                f"Improve consolidation quality for: {', '.join(set(low_utility_modules))}"
            )
        
        # Time-based recommendation
        readiness = await self._check_consolidation_readiness(context)
        if readiness["ready"]:
            recommendations.append("Scheduled consolidation interval reached")
        
        if not recommendations:
            recommendations.append("No immediate consolidation actions needed")
        
        return recommendations
    
    async def _analyze_memory_optimization_potential(self) -> Dict[str, Any]:
        """Analyze potential for memory optimization through consolidation"""
        optimization = {
            "current_memory_items": 0,
            "potential_reduction": 0,
            "optimization_ratio": 0.0,
            "recommendations": []
        }
        
        # Calculate current state
        landscape = await self._analyze_experience_landscape(SharedContext())
        
        optimization["current_memory_items"] = landscape.get("total_experiences", 0)
        unconsolidated = landscape.get("unconsolidated_experiences", 0)
        
        # Estimate reduction potential
        # Assume each consolidation reduces 3-5 experiences to 1
        avg_reduction_factor = 4
        potential_consolidations = unconsolidated // self.original_system.min_group_size
        
        optimization["potential_reduction"] = potential_consolidations * (avg_reduction_factor - 1)
        
        if optimization["current_memory_items"] > 0:
            optimization["optimization_ratio"] = optimization["potential_reduction"] / optimization["current_memory_items"]
        
        # Generate recommendations
        if optimization["optimization_ratio"] > 0.3:
            optimization["recommendations"].append("Significant memory optimization possible")
        
        if unconsolidated > 100:
            optimization["recommendations"].append(f"Consolidate {unconsolidated} unconsolidated experiences")
        
        return optimization
    
    async def _evaluate_proactive_consolidation(self, synthesis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if proactive consolidation should be recommended"""
        # Check multiple factors
        status = synthesis.get("consolidation_status", {})
        recommendations = synthesis.get("consolidation_recommendations", [])
        memory_opt = synthesis.get("memory_optimization", {})
        
        # Don't recommend if recently consolidated
        if status.get("last_consolidation"):
            try:
                last_time = datetime.fromisoformat(status["last_consolidation"])
                hours_since = (datetime.now() - last_time).total_seconds() / 3600
                
                if hours_since < 6:  # Less than 6 hours
                    return None
            except:
                pass
        
        # Check for strong indicators
        if status.get("ready_groups", 0) >= 3:
            return {
                "action": "proactive_consolidation",
                "reason": f"{status['ready_groups']} groups ready for consolidation",
                "expected_benefit": "memory_optimization",
                "priority": "medium"
            }
        
        if memory_opt.get("optimization_ratio", 0) > 0.3:
            return {
                "action": "proactive_consolidation",
                "reason": f"Can reduce memory by {memory_opt['optimization_ratio']:.1%}",
                "expected_benefit": "significant_memory_reduction",
                "priority": "high"
            }
        
        # Check recommendations
        urgent_keywords = ["memory pressure", "prioritize", "run consolidation"]
        recommendation_text = " ".join(recommendations).lower()
        
        if any(keyword in recommendation_text for keyword in urgent_keywords):
            return {
                "action": "proactive_consolidation",
                "reason": recommendations[0],
                "expected_benefit": "system_optimization",
                "priority": "medium"
            }
        
        return None
    
    # Delegate all other methods to the original system
    def __getattr__(self, name):
        """Delegate any missing methods to the original system"""
        return getattr(self.original_system, name)
