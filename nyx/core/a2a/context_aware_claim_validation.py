# nyx/core/a2a/context_aware_claim_validation.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareClaimValidation(ContextAwareModule):
    """
    Advanced Claim Validation System with full context distribution capabilities
    """
    
    def __init__(self, blacklisted_sources=None, computer_use_agent=None):
        super().__init__("claim_validation")
        self.blacklisted_sources = blacklisted_sources or []
        self.computer_use_agent = computer_use_agent
        self.context_subscriptions = [
            "content_validation_request", "misinformation_alert", "fact_check_request",
            "source_credibility_check", "claim_detected", "verification_needed",
            "social_content_flagged", "news_item_shared"
        ]
        self.validation_history = []
        self.source_credibility_cache = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize claim validation for this context"""
        logger.debug(f"ClaimValidation received context for user: {context.user_id}")
        
        # Check if input contains claims needing validation
        if await self._contains_verifiable_claims(context):
            await self.send_context_update(
                update_type="claim_validation_ready",
                data={
                    "capabilities": [
                        "fact_checking", "source_verification",
                        "misinformation_detection", "credibility_assessment"
                    ],
                    "blacklist_active": bool(self.blacklisted_sources),
                    "automated_checking": True
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "content_validation_request":
            # Validate content from any module
            validation_data = update.data
            result = await self._validate_content(validation_data)
            
            await self.send_context_update(
                update_type="content_validation_complete",
                data=result,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED,
                priority=ContextPriority.HIGH if result["verdict"] == "false" else ContextPriority.NORMAL
            )
        
        elif update.update_type == "social_content_flagged":
            # Validate flagged social media content
            social_data = update.data
            result = await self._validate_social_claim(
                social_data.get("claim", ""),
                social_data.get("source", "unknown")
            )
            
            if result["verdict"] == "false":
                await self.send_context_update(
                    update_type="misinformation_detected",
                    data={
                        "source": social_data.get("source"),
                        "claim": social_data.get("claim"),
                        "verdict": result["verdict"],
                        "explanation": result["explanation"],
                        "action_recommended": "flag_or_report"
                    },
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "source_credibility_check":
            # Check source credibility
            source_data = update.data
            credibility = await self._assess_source_credibility(source_data.get("source", ""))
            
            await self.send_context_update(
                update_type="source_credibility_assessed",
                data=credibility,
                target_modules=[update.source_module],
                scope=ContextScope.TARGETED
            )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for claims needing validation"""
        user_input = context.user_input
        
        # Extract potential claims
        claims = self._extract_claims(user_input)
        
        validation_results = []
        misinformation_found = False
        
        for claim in claims:
            # Validate each claim
            result = await self._validate_claim_with_context(claim, context)
            validation_results.append(result)
            
            if result["verdict"] == "false":
                misinformation_found = True
        
        # Send validation results if claims were found
        if validation_results:
            await self.send_context_update(
                update_type="input_claims_validated",
                data={
                    "claims_found": len(claims),
                    "validation_results": validation_results,
                    "misinformation_detected": misinformation_found
                },
                priority=ContextPriority.HIGH if misinformation_found else ContextPriority.NORMAL
            )
        
        # Check if user is asking about fact-checking
        if self._is_fact_check_request(user_input):
            fact_check_info = await self._provide_fact_check_guidance(context)
            
            await self.send_context_update(
                update_type="fact_check_guidance",
                data=fact_check_info,
                priority=ContextPriority.NORMAL
            )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "claims_detected": len(claims),
            "validation_performed": len(validation_results),
            "misinformation_found": misinformation_found,
            "fact_check_requested": self._is_fact_check_request(user_input),
            "cross_module_informed": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze validation patterns and misinformation trends"""
        messages = await self.get_cross_module_messages()
        
        # Analyze validation patterns
        validation_patterns = self._analyze_validation_patterns()
        
        # Analyze misinformation trends
        misinformation_trends = await self._analyze_misinformation_trends()
        
        # Analyze source reliability
        source_analysis = self._analyze_source_reliability()
        
        # Identify verification challenges
        verification_challenges = await self._identify_verification_challenges(context, messages)
        
        return {
            "validation_patterns": validation_patterns,
            "misinformation_trends": misinformation_trends,
            "source_analysis": source_analysis,
            "verification_challenges": verification_challenges,
            "analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize claim validation insights"""
        messages = await self.get_cross_module_messages()
        
        # Generate synthesis
        synthesis = {
            "validation_recommendations": await self._generate_validation_recommendations(context),
            "credibility_guidelines": self._compile_credibility_guidelines(),
            "fact_checking_tips": self._generate_fact_checking_tips(),
            "misinformation_summary": self._summarize_misinformation_encounters()
        }
        
        # Check if we should warn about misinformation
        if self._should_warn_about_misinformation(context, messages):
            synthesis["misinformation_warning"] = await self._generate_misinformation_warning(context)
            synthesis["issue_warning"] = True
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _contains_verifiable_claims(self, context: SharedContext) -> bool:
        """Check if context contains claims that can be verified"""
        claim_indicators = [
            "according to", "studies show", "research indicates",
            "statistics say", "% of", "data shows", "fact",
            "proven", "scientifically", "historically"
        ]
        
        user_input_lower = context.user_input.lower()
        return any(indicator in user_input_lower for indicator in claim_indicators)
    
    async def _validate_content(self, validation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content for accuracy"""
        content = validation_data.get("content", "")
        source = validation_data.get("source", "unknown")
        content_type = validation_data.get("type", "general")
        
        # Check if source is blacklisted
        if self._is_blacklisted_source(source):
            return {
                "verdict": "false",
                "explanation": f"Source {source} is known for misinformation",
                "confidence": 0.95,
                "source_credibility": "very_low"
            }
        
        # Use computer agent to fact-check if available
        if self.computer_use_agent and content_type in ["claim", "news", "statistic"]:
            try:
                fact_check_result = await self._automated_fact_check(content, source)
                return fact_check_result
            except Exception as e:
                logger.error(f"Automated fact-check failed: {e}")
        
        # Fallback to heuristic analysis
        return self._heuristic_validation(content, source)
    
    async def _validate_social_claim(self, claim: str, source: str) -> Dict[str, Any]:
        """Validate claim from social media with extra scrutiny"""
        # Import the original validation function
        from nyx.tools.claim_validation import validate_social_claim
        
        # Create a mock object with necessary attributes
        class MockContext:
            def __init__(self, computer_agent):
                self.creative_system = type('obj', (object,), {
                    'computer_user': computer_agent,
                    'logger': type('obj', (object,), {
                        'log_thought': lambda *args, **kwargs: None
                    })
                })
        
        mock_context = MockContext(self.computer_use_agent)
        
        # Use original validation
        result = await validate_social_claim(mock_context, claim, source)
        
        # Add to history
        self.validation_history.append({
            "claim": claim,
            "source": source,
            "verdict": result["verdict"],
            "timestamp": datetime.now().isoformat(),
            "type": "social_media"
        })
        
        return result
    
    async def _assess_source_credibility(self, source: str) -> Dict[str, Any]:
        """Assess credibility of a source"""
        # Check cache first
        if source in self.source_credibility_cache:
            return self.source_credibility_cache[source]
        
        credibility = {
            "source": source,
            "credibility_score": 0.5,
            "factors": [],
            "recommendation": "verify_claims"
        }
        
        # Check blacklist
        if self._is_blacklisted_source(source):
            credibility["credibility_score"] = 0.1
            credibility["factors"].append("known_misinformation_source")
            credibility["recommendation"] = "avoid"
        else:
            # Check for credibility indicators
            credible_indicators = [
                ".edu", ".gov", "reuters", "apnews", "bbc",
                "nature.com", "science.org", "pubmed"
            ]
            
            for indicator in credible_indicators:
                if indicator in source.lower():
                    credibility["credibility_score"] = 0.8
                    credibility["factors"].append(f"credible_domain:{indicator}")
                    credibility["recommendation"] = "generally_trustworthy"
                    break
        
        # Cache result
        self.source_credibility_cache[source] = credibility
        
        return credibility
    
    def _extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract verifiable claims from text"""
        claims = []
        
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', text)
        
        claim_patterns = [
            r'\d+%',  # Percentages
            r'\d+ out of \d+',  # Ratios
            r'studies show',
            r'research indicates',
            r'according to',
            r'data shows',
            r'proven',
            r'fact'
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if sentence contains claim patterns
            for pattern in claim_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    claims.append({
                        "text": sentence,
                        "type": "statistical" if re.search(r'\d+', sentence) else "factual",
                        "confidence_level": "medium"
                    })
                    break
        
        return claims
    
    async def _validate_claim_with_context(self, claim: Dict[str, Any], context: SharedContext) -> Dict[str, Any]:
        """Validate claim considering context"""
        claim_text = claim.get("text", "")
        claim_type = claim.get("type", "general")
        
        # Basic validation
        result = await self._validate_content({
            "content": claim_text,
            "type": claim_type,
            "source": "user_input"
        })
        
        # Enhance with context
        if context.memory_context:
            # Check if we've seen similar claims before
            similar_validations = self._find_similar_validations(claim_text)
            if similar_validations:
                result["previous_validations"] = similar_validations
                result["pattern_detected"] = True
        
        return result
    
    def _is_fact_check_request(self, text: str) -> bool:
        """Check if user is asking about fact-checking"""
        fact_check_phrases = [
            "is this true", "fact check", "verify this",
            "is it true that", "can you confirm", "check if"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in fact_check_phrases)
    
    async def _provide_fact_check_guidance(self, context: SharedContext) -> Dict[str, Any]:
        """Provide guidance on fact-checking"""
        return {
            "fact_check_tips": [
                "Check multiple reputable sources",
                "Look for primary sources and data",
                "Be wary of emotional language",
                "Check publication dates",
                "Verify author credentials"
            ],
            "reliable_fact_checkers": [
                "snopes.com",
                "factcheck.org",
                "politifact.com",
                "apnews.com/APFactCheck"
            ],
            "red_flags": [
                "No author attribution",
                "No date or old date",
                "Sensational headlines",
                "No sources cited",
                "Grammar and spelling errors"
            ]
        }
    
    def _is_blacklisted_source(self, source: str) -> bool:
        """Check if source is blacklisted"""
        from nyx.tools.claim_validation import BLACKLISTED_SOURCES
        
        # Use imported blacklist plus any additional ones
        all_blacklisted = BLACKLISTED_SOURCES + self.blacklisted_sources
        
        source_lower = source.lower()
        return any(blacklisted in source_lower for blacklisted in all_blacklisted)
    
    async def _automated_fact_check(self, content: str, source: str) -> Dict[str, Any]:
        """Perform automated fact-checking using computer agent"""
        if not self.computer_use_agent:
            return self._heuristic_validation(content, source)
        
        try:
            # Use computer agent to search for verification
            prompt = f"Fact check this claim: '{content}'. Look for credible sources that either confirm or refute this."
            
            result = self.computer_use_agent.run_task(
                url="https://www.google.com",
                prompt=prompt
            )
            
            # Analyze results
            if result:
                result_lower = result.lower()
                
                # Check for confirmation/refutation signals
                if any(word in result_lower for word in ["false", "misleading", "debunked", "myth"]):
                    return {
                        "verdict": "false",
                        "explanation": "Multiple sources indicate this claim is false",
                        "confidence": 0.8,
                        "evidence": result[:200]
                    }
                elif any(word in result_lower for word in ["true", "confirmed", "verified", "accurate"]):
                    return {
                        "verdict": "true",
                        "explanation": "Multiple sources confirm this claim",
                        "confidence": 0.8,
                        "evidence": result[:200]
                    }
            
            return {
                "verdict": "unverified",
                "explanation": "Unable to find definitive verification",
                "confidence": 0.5,
                "evidence": result[:200] if result else None
            }
            
        except Exception as e:
            logger.error(f"Automated fact-check error: {e}")
            return self._heuristic_validation(content, source)
    
    def _heuristic_validation(self, content: str, source: str) -> Dict[str, Any]:
        """Fallback heuristic validation"""
        content_lower = content.lower()
        
        # Check for obvious red flags
        red_flags = [
            "everyone knows", "they don't want you to know",
            "miracle cure", "one weird trick", "shocking truth"
        ]
        
        red_flag_count = sum(1 for flag in red_flags if flag in content_lower)
        
        if red_flag_count >= 2:
            return {
                "verdict": "likely_false",
                "explanation": "Content contains multiple credibility red flags",
                "confidence": 0.7,
                "heuristic_based": True
            }
        
        # Check for hedging language (more credible)
        hedge_words = ["might", "could", "possibly", "suggests", "indicates"]
        hedge_count = sum(1 for word in hedge_words if word in content_lower)
        
        if hedge_count >= 2:
            return {
                "verdict": "possibly_true",
                "explanation": "Content uses appropriate hedging language",
                "confidence": 0.6,
                "heuristic_based": True
            }
        
        return {
            "verdict": "unverified",
            "explanation": "Unable to verify claim with available methods",
            "confidence": 0.5,
            "heuristic_based": True
        }
    
    def _find_similar_validations(self, claim_text: str) -> List[Dict[str, Any]]:
        """Find similar previous validations"""
        similar = []
        claim_words = set(claim_text.lower().split())
        
        for validation in self.validation_history[-20:]:  # Check last 20
            prev_claim = validation.get("claim", "")
            prev_words = set(prev_claim.lower().split())
            
            # Calculate similarity
            overlap = len(claim_words.intersection(prev_words))
            similarity = overlap / max(len(claim_words), len(prev_words)) if claim_words or prev_words else 0
            
            if similarity > 0.5:
                similar.append({
                    "claim": prev_claim,
                    "verdict": validation.get("verdict"),
                    "timestamp": validation.get("timestamp"),
                    "similarity": similarity
                })
        
        return similar
    
    def _analyze_validation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in validation history"""
        if not self.validation_history:
            return {"patterns_found": False}
        
        patterns = {
            "total_validations": len(self.validation_history),
            "verdict_distribution": {"true": 0, "false": 0, "unverified": 0},
            "source_types": {},
            "claim_types": {}
        }
        
        for validation in self.validation_history:
            # Count verdicts
            verdict = validation.get("verdict", "unverified")
            if verdict in patterns["verdict_distribution"]:
                patterns["verdict_distribution"][verdict] += 1
            else:
                patterns["verdict_distribution"]["unverified"] += 1
            
            # Count source types
            source_type = validation.get("type", "unknown")
            patterns["source_types"][source_type] = patterns["source_types"].get(source_type, 0) + 1
        
        # Calculate false claim rate
        total = patterns["total_validations"]
        false_count = patterns["verdict_distribution"]["false"]
        patterns["false_claim_rate"] = false_count / total if total > 0 else 0
        
        return patterns
    
    async def _analyze_misinformation_trends(self) -> Dict[str, Any]:
        """Analyze trends in misinformation"""
        trends = {
            "common_false_claims": [],
            "trending_topics": [],
            "source_patterns": {}
        }
        
        # Analyze false claims
        false_claims = [v for v in self.validation_history if v.get("verdict") == "false"]
        
        if false_claims:
            # Find common themes
            theme_counts = {}
            for claim_data in false_claims:
                claim = claim_data.get("claim", "").lower()
                
                # Simple theme extraction
                themes = ["health", "politics", "science", "technology", "finance"]
                for theme in themes:
                    if theme in claim:
                        theme_counts[theme] = theme_counts.get(theme, 0) + 1
            
            if theme_counts:
                trends["trending_topics"] = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return trends
    
    def _analyze_source_reliability(self) -> Dict[str, Any]:
        """Analyze reliability of sources encountered"""
        source_stats = {}
        
        for validation in self.validation_history:
            source = validation.get("source", "unknown")
            verdict = validation.get("verdict", "unverified")
            
            if source not in source_stats:
                source_stats[source] = {"true": 0, "false": 0, "unverified": 0}
            
            source_stats[source][verdict] = source_stats[source].get(verdict, 0) + 1
        
        # Calculate reliability scores
        reliability_scores = {}
        for source, stats in source_stats.items():
            total = sum(stats.values())
            if total > 0:
                reliability_scores[source] = {
                    "reliability": stats["true"] / total,
                    "sample_size": total
                }
        
        return {
            "source_reliability": reliability_scores,
            "most_reliable": max(reliability_scores.items(), key=lambda x: x[1]["reliability"])[0] if reliability_scores else None,
            "least_reliable": min(reliability_scores.items(), key=lambda x: x[1]["reliability"])[0] if reliability_scores else None
        }
    
    async def _identify_verification_challenges(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Identify challenges in verification"""
        challenges = []
        
        # Check for complex claims
        unverified = [v for v in self.validation_history if v.get("verdict") == "unverified"]
        if len(unverified) > len(self.validation_history) * 0.3:
            challenges.append({
                "type": "high_unverified_rate",
                "description": "Many claims cannot be definitively verified",
                "suggestion": "Consider additional verification methods"
            })
        
        # Check for rapid information
        recent_validations = [v for v in self.validation_history[-10:] if 
                            datetime.fromisoformat(v["timestamp"]) > 
                            datetime.now().replace(microsecond=0) - datetime.timedelta(minutes=10)]
        
        if len(recent_validations) > 5:
            challenges.append({
                "type": "rapid_information_flow",
                "description": "High volume of claims requiring validation",
                "suggestion": "Implement automated pre-screening"
            })
        
        return challenges
    
    async def _generate_validation_recommendations(self, context: SharedContext) -> List[str]:
        """Generate recommendations for claim validation"""
        recommendations = []
        
        patterns = self._analyze_validation_patterns()
        
        if patterns.get("false_claim_rate", 0) > 0.3:
            recommendations.append("High rate of false claims detected - increase skepticism")
        
        if self.source_credibility_cache:
            low_cred_sources = [s for s, c in self.source_credibility_cache.items() 
                              if c["credibility_score"] < 0.3]
            if low_cred_sources:
                recommendations.append(f"Avoid these low-credibility sources: {', '.join(low_cred_sources[:3])}")
        
        return recommendations
    
    def _compile_credibility_guidelines(self) -> Dict[str, List[str]]:
        """Compile credibility assessment guidelines"""
        return {
            "high_credibility_indicators": [
                "Primary sources cited",
                "Author expertise evident",
                "Peer-reviewed publication",
                "Multiple corroborating sources",
                "Transparent methodology"
            ],
            "low_credibility_indicators": [
                "Anonymous or missing author",
                "Emotional/sensational language",
                "No sources cited",
                "Known misinformation outlet",
                "Conspiracy theory language"
            ],
            "verification_steps": [
                "Check the source",
                "Verify the author",
                "Check the date",
                "Cross-reference claims",
                "Look for bias"
            ]
        }
    
    def _generate_fact_checking_tips(self) -> List[str]:
        """Generate practical fact-checking tips"""
        return [
            "Use reverse image search for viral photos",
            "Check if quotes are taken out of context",
            "Look for the original source of statistics",
            "Be suspicious of perfect round numbers",
            "Check if old events are being presented as new"
        ]
    
    def _summarize_misinformation_encounters(self) -> Dict[str, Any]:
        """Summarize misinformation encounters"""
        false_claims = [v for v in self.validation_history if v.get("verdict") == "false"]
        
        summary = {
            "total_false_claims": len(false_claims),
            "common_types": [],
            "recommended_actions": []
        }
        
        if false_claims:
            # Identify common types
            type_counts = {}
            for claim in false_claims:
                claim_type = claim.get("type", "unknown")
                type_counts[claim_type] = type_counts.get(claim_type, 0) + 1
            
            summary["common_types"] = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Add recommendations
            if summary["total_false_claims"] > 5:
                summary["recommended_actions"].append("Report persistent misinformation sources")
                summary["recommended_actions"].append("Educate contacts about fact-checking")
        
        return summary
    
    def _should_warn_about_misinformation(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> bool:
        """Determine if we should warn about misinformation"""
        # Warn if recent false claims detected
        recent_false = [v for v in self.validation_history[-5:] if v.get("verdict") == "false"]
        if recent_false:
            return True
        
        # Warn if user is discussing unreliable sources
        if self._mentions_unreliable_sources(context.user_input):
            return True
        
        return False
    
    def _mentions_unreliable_sources(self, text: str) -> bool:
        """Check if text mentions unreliable sources"""
        from nyx.tools.claim_validation import BLACKLISTED_SOURCES
        
        text_lower = text.lower()
        return any(source in text_lower for source in BLACKLISTED_SOURCES)
    
    async def _generate_misinformation_warning(self, context: SharedContext) -> Dict[str, Any]:
        """Generate appropriate misinformation warning"""
        recent_false = [v for v in self.validation_history[-5:] if v.get("verdict") == "false"]
        
        warning = {
            "severity": "high" if len(recent_false) > 2 else "medium",
            "message": "Recent misinformation detected in discussed content",
            "false_claims": [
                {
                    "claim": v.get("claim", "")[:100],
                    "source": v.get("source", "unknown")
                }
                for v in recent_false[:3]
            ],
            "recommendations": [
                "Verify claims before sharing",
                "Check multiple reputable sources",
                "Be skeptical of sensational claims"
            ]
        }
        
        return warning
    
    # Delegation for any methods not explicitly defined
    def __getattr__(self, name):
        """Delegate to computer use agent if available"""
        if self.computer_use_agent and hasattr(self.computer_use_agent, name):
            return getattr(self.computer_use_agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
