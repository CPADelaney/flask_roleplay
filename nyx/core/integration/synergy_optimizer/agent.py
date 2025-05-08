# nyx/core/integration/synergy_optimizer/agent.py

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set
from agents import Agent, function_tool

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel

logger = logging.getLogger(__name__)

class SynergyOptimizerAgent:
    """
    Agent that monitors the event bus system to identify and suggest
    integration improvements between modules.
    """
    
    def __init__(self, brain):
        self.brain = brain
        self.event_bus = get_event_bus()
        self.tracer = get_tracer()
        self.module_interactions = {}
        self.event_patterns = {}
        self.recommendations = []
        self.active = False
        self.last_analysis_time = None
        
        # Initialize the agent
        self.agent = self._create_agent()
    
    async def initialize(self):
        """Initialize the synergy optimizer."""
        try:
            self.event_bus.subscribe("*", self._handle_event) # Assuming _handle_event is async or safe to call from sync
            
            if self.brain.integration_manager:
                # This await is now fine because IntegrationManager.register_bridge is async def
                await self.brain.integration_manager.register_bridge(
                    "synergy_optimizer", 
                    self,
                    ["event_bus"] 
                )
            
            
            # Start monitoring
            self.active = True
            # Ensure run_periodic_analysis is also an async method
            asyncio.create_task(self.run_periodic_analysis()) 
            
            logger.info("SynergyOptimizerAgent initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Error initializing SynergyOptimizerAgent: {e}", exc_info=True)
            return False

    
    async def _handle_event(self, event: Event):
        """Process an event from the event bus."""
        if not self.active:
            return
            
        # Record the event flow for later analysis
        source = event.source
        event_type = event.event_type
        
        # Update module interaction model
        if source not in self.module_interactions:
            self.module_interactions[source] = {"event_types": set(), "targets": set()}
        
        self.module_interactions[source]["event_types"].add(event_type)
        
        # Update event patterns
        pattern_key = f"{source}:{event_type}"
        if pattern_key not in self.event_patterns:
            self.event_patterns[pattern_key] = {
                "count": 0,
                "recent_timestamps": [],
                "related_events": {}
            }
        
        pattern = self.event_patterns[pattern_key]
        pattern["count"] += 1
        pattern["recent_timestamps"].append(event.timestamp)
        
        # Keep only recent timestamps (last 100)
        if len(pattern["recent_timestamps"]) > 100:
            pattern["recent_timestamps"] = pattern["recent_timestamps"][-100:]
    
    async def run_periodic_analysis(self):
        """Run periodic analysis of event patterns."""
        while self.active:
            await asyncio.sleep(300)  # Analyze every 5 minutes
            
            try:
                recommendations = await self.analyze_synergy_opportunities()
                
                # Store recommendations
                if recommendations:
                    self.recommendations.extend(recommendations)
                    
                    # Publish to event bus
                    await self.event_bus.publish(Event(
                        event_type="synergy_recommendations",
                        source="synergy_optimizer",
                        data={"recommendations": recommendations}
                    ))
            except Exception as e:
                logger.error(f"Error in synergy analysis: {e}")
    
    async def analyze_synergy_opportunities(self):
        """
        Analyze event patterns and module interactions to identify
        synergy opportunities.
        """
        from nyx.core.integration.synergy_optimizer.analyzers import (
            analyze_event_flow,
            detect_missing_connections,
            identify_redundant_events
        )
        
        with self.tracer.trace(
            source_module="synergy_optimizer",
            operation="analyze_synergy",
            level=TraceLevel.INFO
        ):
            # Get current integration status
            integration_status = await self.brain.get_integration_status()
            
            # Run various analyses
            flow_analysis = await analyze_event_flow(self.event_patterns)
            missing_connections = await detect_missing_connections(
                self.module_interactions, 
                integration_status
            )
            redundant_events = await identify_redundant_events(self.event_patterns)
            
            # Generate recommendations
            from nyx.core.integration.synergy_optimizer.recommendation_engine import (
                generate_recommendations
            )
            
            recommendations = await generate_recommendations(
                flow_analysis,
                missing_connections,
                redundant_events,
                integration_status
            )
            
            self.last_analysis_time = self.brain.system_context.current_time

            from dev_log.api import add_synergy_recommendation
            
            # Log recommendations to dev log
            for recommendation in recommendations:
                await add_synergy_recommendation(
                    title=recommendation["description"],
                    content=f"Priority: {recommendation['priority']}\nExpected impact: {recommendation['expected_impact']}",
                    from_module=recommendation.get("from_module"),
                    to_module=recommendation.get("to_module"),
                    priority=recommendation["priority"],
                    expected_impact=recommendation["expected_impact"],
                    source_module="synergy_optimizer",
                    metadata={
                        "recommendation_type": recommendation["type"],
                        "event_type": recommendation.get("event_type"),
                        "sources": recommendation.get("sources", [])
                    }
                )
            
            return recommendations
    
    async def get_status(self):
        """Get the status of the synergy optimizer."""
        return {
            "active": self.active,
            "monitored_modules": len(self.module_interactions),
            "event_patterns": len(self.event_patterns),
            "recommendations": len(self.recommendations),
            "last_analysis_time": self.last_analysis_time
        }
    
    def _create_agent(self):
        """Create the agent for this module."""
        return Agent(
            name="Synergy Optimizer Agent",
            instructions="""
            You are an agent responsible for analyzing system integration patterns
            and suggesting improvements to increase synergy between modules.
            
            Your tasks include:
            1. Monitoring events flowing through the event bus
            2. Identifying patterns of communication between modules
            3. Detecting potential gaps or inefficiencies in integration
            4. Generating recommendations for improving system cohesion
            """,
            tools=[
                function_tool(self.get_recommendations),
                function_tool(self.get_module_interactions),
                function_tool(self.get_event_patterns),
                function_tool(self.apply_recommendation)
            ]
        )
    
    @function_tool
    async def get_recommendations(self, limit: int = 10):
        """
        Get recent synergy recommendations.
        
        Args:
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommendations
        """
        return self.recommendations[-limit:]
    
    @function_tool
    async def get_module_interactions(self):
        """
        Get current module interaction model.
        
        Returns:
            Module interaction data
        """
        return self.module_interactions
    
    @function_tool
    async def get_event_patterns(self):
        """
        Get event pattern data.
        
        Returns:
            Event pattern statistics
        """
        return self.event_patterns
    
    @function_tool
    async def apply_recommendation(self, recommendation_id: str, approved: bool = False):
        """
        Apply a synergy recommendation.
        
        Args:
            recommendation_id: ID of the recommendation to apply
            approved: Whether the recommendation was approved
            
        Returns:
            Application result
        """
        # Find the recommendation
        recommendation = next(
            (r for r in self.recommendations if r["id"] == recommendation_id),
            None
        )
        
        if not recommendation:
            return {"success": False, "error": "Recommendation not found"}
        
        if not approved:
            return {"success": False, "error": "Recommendation not approved"}
        
        # Apply the recommendation based on its type
        rec_type = recommendation.get("type")
        
        try:
            if rec_type == "new_bridge":
                # Create a new bridge between modules
                # This would require extending the integration manager
                return {"success": False, "error": "Implementation not available yet"}
            
            elif rec_type == "event_subscription":
                # Add a new event subscription
                from_module = recommendation.get("from_module")
                to_module = recommendation.get("to_module")
                event_type = recommendation.get("event_type")
                
                # This would require implementing a subscription mechanism
                return {"success": False, "error": "Implementation not available yet"}
            
            else:
                return {"success": False, "error": f"Unknown recommendation type: {rec_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
