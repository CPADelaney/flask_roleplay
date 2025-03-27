# nyx/core/integration/narrative_memory_identity_nexus.py

import logging
import asyncio
import datetime
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import deque

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class NarrativeMemoryIdentityNexus:
    """
    Integration nexus between narrative, memory, and identity systems.
    
    This module creates a cohesive integration between autobiographical narrative,
    memory systems, and identity evolution, ensuring that experiences shape identity
    through meaningful narratives and that identity influences memory retrieval.
    
    Key functions:
    1. Coordinates between memory formation and identity evolution
    2. Ensures narrative coherence across episodic memories
    3. Maintains self-concept stability while allowing growth
    4. Processes significant experiences into identity changes
    5. Facilitates identity-consistent memory retrieval
    """
    
    def __init__(self, 
                brain_reference=None, 
                autobiographical_narrative=None,
                memory_orchestrator=None,
                identity_evolution=None,
                relationship_manager=None):
        """Initialize the narrative-memory-identity nexus."""
        self.brain = brain_reference
        self.autobiographical_narrative = autobiographical_narrative
        self.memory_orchestrator = memory_orchestrator
        self.identity_evolution = identity_evolution
        self.relationship_manager = relationship_manager
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Integration parameters
        self.identity_update_threshold = 0.3  # Minimum significance for identity updates
        self.narrative_integration_interval = 24  # Hours between narrative updates
        self.identity_stability_factor = 0.8  # How stable identity is (higher = more stable)
        
        # Narrative-identity mapping
        self.identity_narrative_elements = {}  # identity_trait -> list of narrative elements
        self.memory_identity_impacts = {}  # memory_id -> list of identity impacts
        
        # Recent nexus activity
        self.recent_identity_updates = deque(maxlen=20)  # Recent identity updates
        self.recent_narrative_updates = deque(maxlen=10)  # Recent narrative updates
        self.recent_significant_memories = deque(maxlen=30)  # Recent significant memories
        
        # Monitoring
        self.nexus_metrics = {
            "identity_updates": 0,
            "narrative_updates": 0,
            "memory_identity_links": 0,
            "identity_coherence": 0.0
        }
        
        # Integration event subscriptions
        self._subscribed = False
        
        # Timestamp tracking
        self.last_narrative_update = datetime.datetime.now()
        self.last_identity_reflection = datetime.datetime.now()
        self.startup_time = datetime.datetime.now()
        
        logger.info("NarrativeMemoryIdentityNexus initialized")
    
    async def initialize(self) -> bool:
        """Initialize the nexus and establish connections to systems."""
        try:
            # Set up connections to required systems if needed
            if not self.autobiographical_narrative and hasattr(self.brain, "autobiographical_narrative"):
                self.autobiographical_narrative = self.brain.autobiographical_narrative
                
            if not self.memory_orchestrator and hasattr(self.brain, "memory_orchestrator"):
                self.memory_orchestrator = self.brain.memory_orchestrator
                
            if not self.identity_evolution and hasattr(self.brain, "identity_evolution"):
                self.identity_evolution = self.brain.identity_evolution
                
            if not self.relationship_manager and hasattr(self.brain, "relationship_manager"):
                self.relationship_manager = self.brain.relationship_manager
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("memory_added", self._handle_memory_added)
                self.event_bus.subscribe("identity_updated", self._handle_identity_updated)
                self.event_bus.subscribe("narrative_updated", self._handle_narrative_updated)
                self._subscribed = True
            
            logger.info("NarrativeMemoryIdentityNexus successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing NarrativeMemoryIdentityNexus: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="NarrativeMemoryIdentity")
    async def process_significant_experience(self, 
                                          experience_data: Dict[str, Any],
                                          significance: float = 0.5) -> Dict[str, Any]:
        """
        Process a significant experience through the nexus.
        
        This integrates the experience into memories, updates identity if relevant,
        and incorporates it into the autobiographical narrative.
        
        Args:
            experience_data: Data about the experience
            significance: How significant this experience is (0.0-1.0)
            
        Returns:
            Processing results
        """
        processing_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "significance": significance
        }
        
        try:
            # 1. Store experience as memory if significant enough
            memory_id = None
            if self.memory_orchestrator and significance >= 0.3:  # Only store somewhat significant experiences
                # Extract memory data
                memory_text = experience_data.get("text", "")
                if not memory_text and "description" in experience_data:
                    memory_text = experience_data["description"]
                
                memory_type = experience_data.get("type", "experience")
                
                # Add additional fields based on experience data
                metadata = experience_data.get("metadata", {})
                if not metadata:
                    metadata = {
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Add relationship information if available
                    if "user_id" in experience_data:
                        metadata["user_id"] = experience_data["user_id"]
                        
                    # Add emotional information if available
                    if "emotion" in experience_data:
                        metadata["emotion"] = experience_data["emotion"]
                        
                    # Add valence information if available
                    if "valence" in experience_data:
                        metadata["valence"] = experience_data["valence"]
                
                # Generate tags from experience data
                tags = experience_data.get("tags", [])
                if not tags:
                    tags = ["experience"]
                    if "emotion" in experience_data:
                        tags.append(experience_data["emotion"])
                    if "interaction" in experience_data:
                        tags.append("interaction")
                
                # Store the memory
                memory_id = await self.memory_orchestrator.add_memory(
                    memory_text=memory_text,
                    memory_type=memory_type,
                    significance=int(significance * 10),  # Convert to 0-10 scale
                    tags=tags,
                    metadata=metadata
                )
                
                # Add to recent significant memories
                if memory_id:
                    self.recent_significant_memories.append({
                        "memory_id": memory_id,
                        "memory_text": memory_text[:100] + "..." if len(memory_text) > 100 else memory_text,
                        "significance": significance,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    processing_results["memory_id"] = memory_id
            
            # 2. Update identity if the experience is significant enough
            identity_updates = []
            if self.identity_evolution and significance >= self.identity_update_threshold:
                # Determine which traits might be affected by this experience
                relevant_traits = self._determine_relevant_traits(experience_data)
                
                # Apply identity updates
                for trait, impact in relevant_traits.items():
                    # Scale impact by significance and stability factor
                    scaled_impact = impact * significance * (1.0 - self.identity_stability_factor)
                    
                    # Update the trait
                    update_result = await self.identity_evolution.update_trait(trait, scaled_impact)
                    
                    if update_result:
                        identity_updates.append({
                            "trait": trait,
                            "impact": scaled_impact,
                            "result": update_result
                        })
                        
                        # Record memory-identity impact
                        if memory_id:
                            if memory_id not in self.memory_identity_impacts:
                                self.memory_identity_impacts[memory_id] = []
                            
                            self.memory_identity_impacts[memory_id].append({
                                "trait": trait,
                                "impact": scaled_impact,
                                "timestamp": datetime.datetime.now().isoformat()
                            })
                
                # Update recent identity updates if there were any changes
                if identity_updates:
                    self.recent_identity_updates.append({
                        "trigger": "significant_experience",
                        "updates": identity_updates,
                        "memory_id": memory_id,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    
                    self.nexus_metrics["identity_updates"] += 1
                    
                    processing_results["identity_updates"] = identity_updates
            
            # 3. Check if narrative update is needed
            narrative_updated = False
            if self.autobiographical_narrative:
                hours_since_update = (datetime.datetime.now() - self.last_narrative_update).total_seconds() / 3600
                
                # Update narrative immediately for very significant experiences,
                # or if it's been a while since the last update
                if significance >= 0.8 or hours_since_update >= self.narrative_integration_interval:
                    # Trigger narrative update
                    narrative_result = await self.autobiographical_narrative.update_narrative(
                        force_update=(significance >= 0.8)  # Force for very significant experiences
                    )
                    
                    if narrative_result:
                        # Record the update
                        self.last_narrative_update = datetime.datetime.now()
                        narrative_updated = True
                        
                        # Add to recent narrative updates
                        self.recent_narrative_updates.append({
                            "segment_id": narrative_result.segment_id if hasattr(narrative_result, "segment_id") else "unknown",
                            "trigger": "significant_experience",
                            "memory_id": memory_id,
                            "timestamp": self.last_narrative_update.isoformat()
                        })
                        
                        self.nexus_metrics["narrative_updates"] += 1
                        
                        processing_results["narrative_updated"] = True
                        processing_results["narrative_segment"] = {
                            "segment_id": narrative_result.segment_id if hasattr(narrative_result, "segment_id") else "unknown",
                            "title": narrative_result.title if hasattr(narrative_result, "title") else None,
                            "summary": narrative_result.summary if hasattr(narrative_result, "summary") else None
                        }
            
            # 4. Update metrics
            if memory_id and identity_updates:
                self.nexus_metrics["memory_identity_links"] += 1
            
            # Calculate identity coherence based on identity-narrative alignment
            await self._calculate_identity_coherence()
            
            processing_results["success"] = True
            return processing_results
        except Exception as e:
            logger.error(f"Error processing significant experience: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    @trace_method(level=TraceLevel.INFO, group_id="NarrativeMemoryIdentity")
    async def retrieve_identity_consistent_memories(self, 
                                                query: str, 
                                                identity_traits: Optional[List[str]] = None,
                                                limit: int = 10) -> Dict[str, Any]:
        """
        Retrieve memories consistent with current identity.
        
        Args:
            query: Memory search query
            identity_traits: Optional list of identity traits to focus on
            limit: Maximum number of memories to return
            
        Returns:
            Retrieved memories
        """
        if not self.memory_orchestrator:
            return {"status": "error", "message": "Memory orchestrator not available"}
        
        if not self.identity_evolution:
            return {"status": "error", "message": "Identity evolution system not available"}
        
        try:
            # Get identity profile if not provided specific traits
            identity_profile = {}
            if not identity_traits:
                identity_profile = await self.identity_evolution.get_identity_profile()
                identity_traits = list(identity_profile.get("traits", {}).keys())
            
            # Filter to top traits if available
            if not identity_traits and identity_profile:
                # Sort traits by value and take top 5
                trait_items = list(identity_profile.get("traits", {}).items())
                trait_items.sort(key=lambda x: x[1], reverse=True)
                identity_traits = [trait for trait, _ in trait_items[:5]]
            
            # Default if still empty
            if not identity_traits:
                identity_traits = ["adaptive", "curious", "empathetic"]
            
            # Prepare identity-weighted query
            modified_query = query
            
            # Tag mapping for identity traits
            trait_to_tags = {
                "curious": ["learning", "exploration", "discovery", "question"],
                "analytical": ["analysis", "reasoning", "problem_solving"],
                "empathetic": ["empathy", "connection", "understanding", "emotional"],
                "creative": ["creative", "imagination", "novel", "innovation"],
                "adaptive": ["adaptation", "flexible", "change"],
                "logical": ["logic", "rational", "systematic"],
                "playful": ["play", "humor", "fun", "experiment"],
                "nurturing": ["nurture", "support", "care", "help"],
                "assertive": ["assertive", "confident", "direct"],
                "patient": ["patient", "calm", "composed"],
                "passionate": ["passionate", "enthusiastic", "intensity"],
                "dominant": ["dominant", "control", "guide", "lead", "command"],
                "submissive": ["submissive", "follow", "obey", "comply"],
                "independent": ["independent", "self-reliant", "autonomous"],
                "sociable": ["social", "interactive", "communicative"],
                "reflective": ["reflection", "introspection", "self-awareness"]
            }
            
            # Boost by tags related to identity traits
            boost_tags = []
            for trait in identity_traits:
                trait_lower = trait.lower()
                if trait_lower in trait_to_tags:
                    boost_tags.extend(trait_to_tags[trait_lower])
            
            # Remove duplicates
            boost_tags = list(set(boost_tags))
            
            # Get memories with boosted tags
            boosted_retrieval_params = {
                "query": query,
                "limit": limit,
                "tag_boosts": {tag: 0.8 for tag in boost_tags}  # Boost all related tags
            }
            
            # Execute retrieval
            memories = await self.memory_orchestrator.retrieve_memories(**boosted_retrieval_params)
            
            return {
                "status": "success",
                "memories": memories,
                "identity_traits_used": identity_traits,
                "boosted_tags": boost_tags,
                "count": len(memories)
            }
        except Exception as e:
            logger.error(f"Error retrieving identity-consistent memories: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="NarrativeMemoryIdentity")
    async def generate_identity_reflection(self) -> Dict[str, Any]:
        """
        Generate a reflection on current identity based on memories and narrative.
        
        Returns:
            Generated reflection
        """
        if not self.identity_evolution:
            return {"status": "error", "message": "Identity evolution system not available"}
        
        try:
            # Record this reflection
            self.last_identity_reflection = datetime.datetime.now()
            
            # Generate reflection using identity system
            reflection = await self.identity_evolution.generate_identity_reflection()
            
            if not reflection:
                return {
                    "status": "error", 
                    "message": "Failed to generate identity reflection"
                }
            
            # Store reflection as memory if memory system available
            memory_id = None
            if self.memory_orchestrator:
                memory_id = await self.memory_orchestrator.add_memory(
                    memory_text=reflection,
                    memory_type="reflection",
                    significance=8,  # Identity reflections are significant
                    tags=["identity", "reflection", "self_concept"],
                    metadata={
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection_type": "identity"
                    }
                )
            
            return {
                "status": "success",
                "reflection": reflection,
                "memory_id": memory_id,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating identity reflection: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="NarrativeMemoryIdentity")
    async def align_narrative_with_identity(self) -> Dict[str, Any]:
        """
        Ensure autobiographical narrative aligns with identity by
        highlighting identity-relevant events and themes.
        
        Returns:
            Alignment results
        """
        if not self.autobiographical_narrative:
            return {"status": "error", "message": "Autobiographical narrative not available"}
            
        if not self.identity_evolution:
            return {"status": "error", "message": "Identity evolution system not available"}
        
        try:
            # 1. Get current identity profile
            identity_profile = await self.identity_evolution.get_identity_profile()
            identity_traits = identity_profile.get("traits", {})
            
            # Sort traits by value
            trait_items = list(identity_traits.items())
            trait_items.sort(key=lambda x: x[1], reverse=True)
            top_traits = [trait for trait, _ in trait_items[:5]]
            
            # 2. Get current narrative
            narrative = self.autobiographical_narrative.get_full_narrative()
            narrative_segments = narrative.get("segments", [])
            
            if not narrative_segments:
                return {
                    "status": "partial",
                    "message": "No narrative segments available for alignment",
                    "identity_traits": top_traits
                }
            
            # 3. Analyze alignment
            alignment_scores = {}
            for segment in narrative_segments:
                segment_id = segment.get("segment_id", "unknown")
                segment_themes = segment.get("themes", [])
                segment_summary = segment.get("summary", "")
                
                # Calculate alignment score for this segment
                trait_alignments = {}
                for trait in top_traits:
                    # Simple text matching - would be more sophisticated in real implementation
                    alignment = 0.0
                    
                    # Check themes
                    if any(trait.lower() in theme.lower() for theme in segment_themes):
                        alignment += 0.6
                    
                    # Check summary
                    if trait.lower() in segment_summary.lower():
                        alignment += 0.4
                    
                    # Normalize
                    alignment = min(1.0, alignment)
                    trait_alignments[trait] = alignment
                
                # Overall alignment is average of trait alignments
                overall_alignment = sum(trait_alignments.values()) / len(trait_alignments) if trait_alignments else 0.0
                
                alignment_scores[segment_id] = {
                    "overall": overall_alignment,
                    "traits": trait_alignments
                }
            
            # 4. Update identity-narrative mappings
            self.identity_narrative_elements = {}
            for trait in top_traits:
                self.identity_narrative_elements[trait] = []
                for segment in narrative_segments:
                    segment_id = segment.get("segment_id", "unknown")
                    alignment = alignment_scores.get(segment_id, {}).get("traits", {}).get(trait, 0.0)
                    
                    if alignment > 0.5:  # Only include well-aligned segments
                        self.identity_narrative_elements[trait].append({
                            "segment_id": segment_id,
                            "alignment": alignment,
                            "title": segment.get("title"),
                            "summary_fragment": segment.get("summary", "")[:100] + "..." if len(segment.get("summary", "")) > 100 else segment.get("summary", "")
                        })
            
            # 5. Calculate overall narrative-identity coherence
            coherence_score = 0.0
            total_segments = len(narrative_segments)
            
            if total_segments > 0:
                aligned_segments = sum(1 for scores in alignment_scores.values() if scores["overall"] > 0.5)
                coherence_score = aligned_segments / total_segments
            
            self.nexus_metrics["identity_coherence"] = coherence_score
            
            return {
                "status": "success",
                "alignment_scores": alignment_scores,
                "coherence_score": coherence_score,
                "identity_traits": top_traits,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error aligning narrative with identity: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="NarrativeMemoryIdentity")
    async def get_nexus_status(self) -> Dict[str, Any]:
        """
        Get the current status of the narrative-memory-identity nexus.
        
        Returns:
            Current nexus status
        """
        try:
            # Get identity profile if available
            identity_profile = {}
            top_traits = []
            if self.identity_evolution:
                identity_profile = await self.identity_evolution.get_identity_profile()
                
                # Extract top traits
                trait_items = list(identity_profile.get("traits", {}).items())
                trait_items.sort(key=lambda x: x[1], reverse=True)
                top_traits = [{"trait": trait, "value": value} for trait, value in trait_items[:5]]
            
            # Get narrative summary if available
            narrative_summary = ""
            narrative_segments_count = 0
            if self.autobiographical_narrative:
                narrative_summary = self.autobiographical_narrative.get_narrative_summary()
                narrative_segments = self.autobiographical_narrative.get_narrative_segments()
                narrative_segments_count = len(narrative_segments)
            
            # Calculate hours since last updates
            hours_since_narrative = (datetime.datetime.now() - self.last_narrative_update).total_seconds() / 3600
            hours_since_reflection = (datetime.datetime.now() - self.last_identity_reflection).total_seconds() / 3600
            
            # Get recent activity
            recent_updates = {}
            for key in ["identity_updates", "narrative_updates", "significant_memories"]:
                recent_list = getattr(self, f"recent_{key}", [])
                recent_updates[key] = len(recent_list)
            
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "identity": {
                    "top_traits": top_traits,
                    "hours_since_reflection": hours_since_reflection
                },
                "narrative": {
                    "summary": narrative_summary,
                    "segments_count": narrative_segments_count,
                    "hours_since_update": hours_since_narrative
                },
                "memory_identity_links": len(self.memory_identity_impacts),
                "recent_activity": recent_updates,
                "metrics": self.nexus_metrics
            }
        except Exception as e:
            logger.error(f"Error getting nexus status: {e}")
            return {"status": "error", "message": str(e)}
    
    def _determine_relevant_traits(self, experience_data: Dict[str, Any]) -> Dict[str, float]:
        """Determine which identity traits might be affected by an experience."""
        relevant_traits = {}
        
        # Extract key data from experience
        emotion = experience_data.get("emotion", "").lower()
        valence = experience_data.get("valence", 0.0)
        experience_type = experience_data.get("type", "").lower()
        
        # Look for keywords in description or text
        content = ""
        if "text" in experience_data:
            content = experience_data["text"].lower()
        elif "description" in experience_data:
            content = experience_data["description"].lower()
        
        # Extract tags
        tags = [tag.lower() for tag in experience_data.get("tags", [])]
        
        # Check for trait-specific keywords in content
        trait_keywords = {
            "curious": ["question", "learn", "explore", "discover", "knowledge"],
            "analytical": ["analyze", "reason", "examine", "investigate", "assess"],
            "empathetic": ["understand", "feel", "connect", "empathize", "relate"],
            "creative": ["create", "imagine", "design", "invent", "novel"],
            "adaptive": ["adapt", "change", "adjust", "flexible", "evolve"],
            "logical": ["logic", "rational", "systematic", "consistent", "coherent"],
            "playful": ["play", "fun", "humor", "light-hearted", "joke"],
            "nurturing": ["nurture", "care", "support", "help", "assist"],
            "assertive": ["assert", "direct", "confident", "firm", "definitive"],
            "patient": ["patient", "wait", "calm", "steady", "endure"],
            "passionate": ["passion", "enthusiasm", "excitement", "intense", "energetic"],
            "dominant": ["control", "command", "lead", "authority", "dominance", "power", "forceful", "direct"],
            "submissive": ["follow", "obey", "submit", "comply", "defer", "serve", "surrender"],
            "independent": ["independent", "self-reliant", "autonomous", "freedom", "individual"],
            "sociable": ["social", "outgoing", "friendly", "conversational", "engaging"],
            "reflective": ["reflect", "introspect", "contemplate", "self-aware", "thoughtful"]
        }
        
        # Check content for trait keywords
        for trait, keywords in trait_keywords.items():
            if any(keyword in content for keyword in keywords):
                relevant_traits[trait] = 0.3  # Base impact
        
        # Check tags for trait keywords
        for trait, keywords in trait_keywords.items():
            if any(keyword in tag for keyword in keywords for tag in tags):
                if trait in relevant_traits:
                    relevant_traits[trait] += 0.2  # Additional impact
                else:
                    relevant_traits[trait] = 0.2  # Base impact
        
        # Adjust based on emotion and valence
        if emotion:
            # Emotional impacts on traits
            emotion_trait_mapping = {
                "joy": {"playful": 0.4, "passionate": 0.3, "sociable": 0.3},
                "sadness": {"empathetic": 0.4, "reflective": 0.3, "patient": 0.2},
                "anger": {"assertive": 0.4, "dominant": 0.3, "passionate": 0.2},
                "fear": {"adaptable": 0.3, "reflective": 0.2},
                "surprise": {"curious": 0.4, "adaptive": 0.3},
                "disgust": {"assertive": 0.2, "independent": 0.2},
                "trust": {"empathetic": 0.4, "nurturing": 0.3, "sociable": 0.3},
                "anticipation": {"curious": 0.3, "passionate": 0.3},
                "teasing": {"playful": 0.5, "dominant": 0.3},
                "controlling": {"dominant": 0.6, "assertive": 0.4},
                "confident": {"assertive": 0.5, "dominant": 0.4},
                "ruthless": {"dominant": 0.7},
                "cruel": {"dominant": 0.5},
                "detached": {"independent": 0.5, "logical": 0.4}
            }
            
            if emotion in emotion_trait_mapping:
                for trait, impact in emotion_trait_mapping[emotion].items():
                    if trait in relevant_traits:
                        relevant_traits[trait] += impact
                    else:
                        relevant_traits[trait] = impact
        
        # Specific experience types
        if experience_type:
            if "learning" in experience_type:
                relevant_traits["curious"] = relevant_traits.get("curious", 0.0) + 0.5
                relevant_traits["analytical"] = relevant_traits.get("analytical", 0.0) + 0.3
            
            elif "social" in experience_type:
                relevant_traits["sociable"] = relevant_traits.get("sociable", 0.0) + 0.5
                relevant_traits["empathetic"] = relevant_traits.get("empathetic", 0.0) + 0.3
            
            elif "problem_solving" in experience_type:
                relevant_traits["analytical"] = relevant_traits.get("analytical", 0.0) + 0.5
                relevant_traits["logical"] = relevant_traits.get("logical", 0.0) + 0.4
            
            elif "creative" in experience_type:
                relevant_traits["creative"] = relevant_traits.get("creative", 0.0) + 0.5
                relevant_traits["playful"] = relevant_traits.get("playful", 0.0) + 0.3
            
            elif "challenge" in experience_type:
                relevant_traits["adaptive"] = relevant_traits.get("adaptive", 0.0) + 0.5
                relevant_traits["resilient"] = relevant_traits.get("resilient", 0.0) + 0.4
            
            elif "dominance" in experience_type:
                relevant_traits["dominant"] = relevant_traits.get("dominant", 0.0) + 0.7
                relevant_traits["assertive"] = relevant_traits.get("assertive", 0.0) + 0.5
            
            elif "intimate" in experience_type:
                relevant_traits["passionate"] = relevant_traits.get("passionate", 0.0) + 0.5
                relevant_traits["empathetic"] = relevant_traits.get("empathetic", 0.0) + 0.4
        
        # Normalize all impacts to range [0.0, 1.0]
        for trait in relevant_traits:
            relevant_traits[trait] = min(1.0, relevant_traits[trait])
            
            # Apply valence to determine direction of impact
            # Positive valence strengthens traits, negative valence can reduce them
            if valence < -0.5:  # Strong negative experience
                # Convert to negative impact for some traits
                relevant_traits[trait] = -relevant_traits[trait] * 0.5
        
        return relevant_traits
    
    async def _calculate_identity_coherence(self) -> float:
        """Calculate how coherent identity is with narrative and memories."""
        try:
            # Get identity profile
            coherence_score = 0.5  # Default medium coherence
            
            if self.identity_evolution and self.autobiographical_narrative:
                # Get current identity profile
                identity_profile = await self.identity_evolution.get_identity_profile()
                
                # Get narrative segments
                narrative = self.autobiographical_narrative.get_full_narrative()
                segments = narrative.get("segments", [])
                
                if identity_profile and segments:
                    # Get top traits
                    top_traits = sorted(
                        identity_profile.get("traits", {}).items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                    
                    # Check if segments reflect top traits
                    matches = 0
                    total_traits = len(top_traits)
                    
                    for trait, _ in top_traits:
                        trait_lower = trait.lower()
                        for segment in segments:
                            # Check for trait in themes or summary
                            themes = segment.get("themes", [])
                            summary = segment.get("summary", "")
                            
                            if any(trait_lower in theme.lower() for theme in themes) or trait_lower in summary.lower():
                                matches += 1
                                break
                    
                    if total_traits > 0:
                        coherence_score = matches / total_traits
            
            # Update metric
            self.nexus_metrics["identity_coherence"] = coherence_score
            
            return coherence_score
        except Exception as e:
            logger.error(f"Error calculating identity coherence: {e}")
            return 0.5  # Default medium coherence
    
    async def _handle_memory_added(self, event: Event) -> None:
        """
        Handle memory added events from the event bus.
        
        Args:
            event: Memory added event
        """
        try:
            # Extract event data
            memory_id = event.data.get("memory_id")
            memory_text = event.data.get("memory_text", "")
            significance = event.data.get("significance", 5)
            
            if not memory_id:
                return
            
            # Check if memory is significant enough to process
            normalized_significance = significance / 10.0 if significance > 1 else significance
            
            if normalized_significance >= self.identity_update_threshold:
                # Create experience data from memory
                experience_data = {
                    "text": memory_text,
                    "type": event.data.get("memory_type", "experience"),
                    "tags": event.data.get("tags", []),
                    "metadata": event.data.get("metadata", {})
                }
                
                # Process as significant experience
                asyncio.create_task(
                    self.process_significant_experience(
                        experience_data, 
                        normalized_significance
                    )
                )
        except Exception as e:
            logger.error(f"Error handling memory added event: {e}")
    
    async def _handle_identity_updated(self, event: Event) -> None:
        """
        Handle identity updated events from the event bus.
        
        Args:
            event: Identity updated event
        """
        try:
            # Check if narrative needs to be realigned with identity
            asyncio.create_task(self.align_narrative_with_identity())
        except Exception as e:
            logger.error(f"Error handling identity updated event: {e}")
    
    async def _handle_narrative_updated(self, event: Event) -> None:
        """
        Handle narrative updated events from the event bus.
        
        Args:
            event: Narrative updated event
        """
        try:
            # Update last narrative update time
            self.last_narrative_update = datetime.datetime.now()
            
            # Check if identity reflection should be generated
            hours_since_reflection = (datetime.datetime.now() - self.last_identity_reflection).total_seconds() / 3600
            
            if hours_since_reflection >= 24:  # Daily reflection
                asyncio.create_task(self.generate_identity_reflection())
        except Exception as e:
            logger.error(f"Error handling narrative updated event: {e}")

# Function to create the narrative-memory-identity nexus
def create_narrative_memory_identity_nexus(brain_reference=None):
    """Create a narrative-memory-identity nexus for the given brain."""
    return NarrativeMemoryIdentityNexus(brain_reference=brain_reference)
