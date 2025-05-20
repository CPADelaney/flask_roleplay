from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import random

from .nyx_profile_agents import TeasingAgent, ProfilingAgent, ResponseAnalysisAgent

logger = logging.getLogger("nyx_profile_integration")

class ProfileIntegration:  # Remove BaseModel inheritance
    
    def __init__(self):
        self.teasing_agent = TeasingAgent()
        self.profiling_agent = ProfilingAgent()
        self.response_agent = ResponseAnalysisAgent()
        self.last_profile_update = datetime.now()
        self.profile_update_threshold = 0.1
        
    async def process_interaction(self, event: Dict[str, Any], current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Process a new interaction through all profile agents"""
        result = {
            "profile_updates": {},
            "teasing_strategy": None,
            "response_analysis": None,
            "recommendations": []
        }
        
        # Analyze response
        response_analysis = await self.response_agent.analyze_response(event, current_profile)
        result["response_analysis"] = response_analysis
        
        # Update profile
        profile_analysis = await self.profiling_agent.analyze_interaction(event, current_profile)
        if profile_analysis.get("new_insights"):
            result["profile_updates"] = profile_analysis["new_insights"]
            
        # Generate teasing strategy if appropriate
        if self._should_generate_teasing(response_analysis, profile_analysis):
            teasing_strategy = await self.teasing_agent.generate_teasing_strategy(
                current_profile,
                {"response_analysis": response_analysis, "profile_analysis": profile_analysis}
            )
            result["teasing_strategy"] = teasing_strategy
            
        # Add recommendations
        if profile_analysis.get("recommendations"):
            result["recommendations"].extend(profile_analysis["recommendations"])
            
        if response_analysis.get("suggested_actions"):
            result["recommendations"].extend(response_analysis["suggested_actions"])
            
        return result
        
    def _should_generate_teasing(self, response_analysis: Dict[str, Any], profile_analysis: Dict[str, Any]) -> bool:
        """Determine if teasing strategy should be generated"""
        # Check response intensity
        if response_analysis.get("reaction_intensity", 0) > 0.7:
            return True
            
        # Check for triggered preferences
        if response_analysis.get("triggered_preferences"):
            return True
            
        # Check for emotional state
        emotional_state = response_analysis.get("emotional_state", {})
        if emotional_state.get("arousal", 0) > 0.6 or emotional_state.get("desire", 0) > 0.6:
            return True
            
        # Check for new profile insights
        if profile_analysis.get("new_insights"):
            return True
            
        return False
        
    async def get_teasing_suggestions(self, context: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
        """Get teasing suggestions based on context and profile"""
        # Generate teasing strategy
        strategy = await self.teasing_agent.generate_teasing_strategy(profile, context)
        
        if not strategy or not strategy.get("elements"):
            return []
            
        # Extract phrases from strategy elements
        suggestions = []
        for element in strategy["elements"]:
            if element.get("phrases"):
                suggestions.extend(element["phrases"])
                
        # Add emotional trigger phrases
        for trigger in strategy.get("emotional_triggers", []):
            if trigger.get("phrases"):
                suggestions.extend(trigger["phrases"])
                
        return suggestions
        
    async def get_profile_insights(self, profile: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced profile insights"""
        insights = {
            "preferences": {},
            "patterns": {},
            "emotional_state": {},
            "confidence_levels": {}
        }
        
        # Analyze preferences
        for category in ["kink_preferences", "physical_preferences", "personality_preferences"]:
            if category in profile:
                insights["preferences"][category] = profile[category]
                
        # Analyze patterns
        if "observed_patterns" in profile:
            insights["patterns"] = self._analyze_patterns(profile["observed_patterns"])
            
        # Get emotional state from response analysis
        if context.get("response_analysis"):
            insights["emotional_state"] = context["response_analysis"].get("emotional_state", {})
            
        # Add confidence levels
        if "confidence_levels" in profile:
            insights["confidence_levels"] = profile["confidence_levels"]
            
        return insights
        
    def _analyze_patterns(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze observed patterns for insights"""
        analysis = {
            "frequency": {},
            "trends": {},
            "correlations": {}
        }
        
        if not patterns:
            return analysis
            
        # Analyze frequency
        for pattern in patterns:
            timestamp = pattern.get("timestamp")
            observations = pattern.get("observations", {})
            
            for category, values in observations.items():
                if category not in analysis["frequency"]:
                    analysis["frequency"][category] = {}
                    
                for key, value in values.items():
                    if key not in analysis["frequency"][category]:
                        analysis["frequency"][category][key] = 0
                    analysis["frequency"][category][key] += 1
                    
        # Analyze trends
        sorted_patterns = sorted(patterns, key=lambda x: x.get("timestamp", datetime.min))
        if len(sorted_patterns) > 1:
            for category in ["kink_preferences", "physical_preferences", "personality_preferences"]:
                if category in sorted_patterns[-1].get("observations", {}):
                    analysis["trends"][category] = self._calculate_trends(
                        sorted_patterns,
                        category
                    )
                    
        # Analyze correlations
        analysis["correlations"] = self._find_correlations(patterns)
        
        return analysis
        
    def _calculate_trends(self, patterns: List[Dict[str, Any]], category: str) -> Dict[str, float]:
        """Calculate trends for a category"""
        trends = {}
        
        if len(patterns) < 2:
            return trends
            
        # Get values from first and last pattern
        first_values = patterns[0].get("observations", {}).get(category, {})
        last_values = patterns[-1].get("observations", {}).get(category, {})
        
        # Calculate trends
        for key in set(first_values.keys()) | set(last_values.keys()):
            first_value = first_values.get(key, 0)
            last_value = last_values.get(key, 0)
            trends[key] = last_value - first_value
            
        return trends
        
    def _find_correlations(self, patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Find correlations between different preferences"""
        correlations = {}
        
        if len(patterns) < 2:
            return correlations
            
        # Collect values for each category
        category_values = {}
        for pattern in patterns:
            observations = pattern.get("observations", {})
            for category, values in observations.items():
                if category not in category_values:
                    category_values[category] = {}
                for key, value in values.items():
                    if key not in category_values[category]:
                        category_values[category][key] = []
                    category_values[category][key].append(value)
                    
        # Calculate correlations
        categories = list(category_values.keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
                for key1 in category_values[cat1]:
                    for key2 in category_values[cat2]:
                        if len(category_values[cat1][key1]) == len(category_values[cat2][key2]):
                            correlation = self._calculate_correlation(
                                category_values[cat1][key1],
                                category_values[cat2][key2]
                            )
                            if abs(correlation) > 0.5:  # Only store strong correlations
                                correlations[f"{cat1}_{key1}_{cat2}_{key2}"] = correlation
                                
        return correlations
        
    def _calculate_correlation(self, values1: List[float], values2: List[float]) -> float:
        """Calculate correlation between two lists of values"""
        if len(values1) != len(values2):
            return 0.0
            
        # Calculate means
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        # Calculate covariance
        covariance = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2)) / len(values1)
        
        # Calculate standard deviations
        std1 = (sum((v - mean1) ** 2 for v in values1) / len(values1)) ** 0.5
        std2 = (sum((v - mean2) ** 2 for v in values2) / len(values2)) ** 0.5
        
        # Calculate correlation
        if std1 == 0 or std2 == 0:
            return 0.0
            
        return covariance / (std1 * std2) 
