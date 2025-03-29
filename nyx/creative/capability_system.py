# nyx/creative/capability_system.py

import datetime
import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Set
import uuid

logger = logging.getLogger(__name__)

class Capability:
    """Represents a specific capability of the AI system."""
    
    def __init__(self, 
                name: str, 
                description: str,
                category: str,
                implementation_status: str = "not_implemented",
                confidence: float = 0.0,
                dependencies: List[str] = None,
                required_by: List[str] = None,
                examples: List[str] = None,
                metadata: Dict[str, Any] = None):
        """
        Initialize a capability.
        
        Args:
            name: Capability name
            description: Capability description
            category: Capability category
            implementation_status: Implementation status
            confidence: Confidence in ability to perform this capability (0.0-1.0)
            dependencies: List of capability names this depends on
            required_by: List of capability names that require this
            examples: List of example uses
            metadata: Additional metadata
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.category = category
        self.implementation_status = implementation_status
        self.confidence = confidence
        self.dependencies = dependencies or []
        self.required_by = required_by or []
        self.examples = examples or []
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "implementation_status": self.implementation_status,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
            "required_by": self.required_by,
            "examples": self.examples,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Capability':
        """Create from dictionary representation."""
        capability = cls(
            name=data["name"],
            description=data["description"],
            category=data["category"],
            implementation_status=data.get("implementation_status", "not_implemented"),
            confidence=data.get("confidence", 0.0),
            dependencies=data.get("dependencies", []),
            required_by=data.get("required_by", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {})
        )
        capability.id = data.get("id", capability.id)
        capability.created_at = data.get("created_at", capability.created_at)
        capability.updated_at = data.get("updated_at", capability.updated_at)
        return capability


class CapabilityModel:
    """Represents the AI's model of its capabilities and limitations."""
    
    def __init__(self, storage_path: str = "capability_model.json"):
        """
        Initialize the capability model.
        
        Args:
            storage_path: Path to store the capability model
        """
        self.storage_path = storage_path
        self.capabilities: Dict[str, Capability] = {}
        self.capability_categories: Set[str] = set()
        self.load_model()
    
    def load_model(self) -> None:
        """Load the capability model from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for capability_data in data.get("capabilities", []):
                    capability = Capability.from_dict(capability_data)
                    self.capabilities[capability.id] = capability
                    self.capability_categories.add(capability.category)
                
                logger.info(f"Loaded capability model with {len(self.capabilities)} capabilities")
            
            except Exception as e:
                logger.error(f"Error loading capability model: {e}")
        else:
            logger.info("No existing capability model found. Starting with empty model.")
            self.initialize_default_capabilities()
    
    def save_model(self) -> None:
        """Save the capability model to storage."""
        try:
            data = {
                "capabilities": [capability.to_dict() for capability in self.capabilities.values()],
                "updated_at": datetime.datetime.now().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved capability model with {len(self.capabilities)} capabilities")
        
        except Exception as e:
            logger.error(f"Error saving capability model: {e}")
    
    def initialize_default_capabilities(self) -> None:
        """Initialize the model with default capabilities."""
        default_categories = [
            "cognitive", "perception", "communication", "social", 
            "creative", "physical", "technical", "meta"
        ]
        
        default_capabilities = [
            # Cognitive capabilities
            Capability(
                name="reasoning",
                description="Ability to apply logical thinking to solve problems and make decisions",
                category="cognitive",
                implementation_status="implemented",
                confidence=0.8
            ),
            
            # Communication capabilities
            Capability(
                name="text_generation",
                description="Ability to generate coherent and contextually appropriate text",
                category="communication",
                implementation_status="implemented",
                confidence=0.9
            ),
            
            # Creative capabilities
            Capability(
                name="story_writing",
                description="Ability to create engaging fictional narratives",
                category="creative",
                implementation_status="implemented",
                confidence=0.7
            ),
            Capability(
                name="poetry_writing",
                description="Ability to create poetic compositions",
                category="creative",
                implementation_status="implemented",
                confidence=0.7
            ),
            
            # Technical capabilities
            Capability(
                name="code_generation",
                description="Ability to generate functional code in various programming languages",
                category="technical",
                implementation_status="implemented",
                confidence=0.8
            ),
            
            # Meta capabilities
            Capability(
                name="self_reflection",
                description="Ability to analyze and assess own performance and capabilities",
                category="meta",
                implementation_status="implemented",
                confidence=0.6
            )
        ]
        
        # Add all default capabilities
        for capability in default_capabilities:
            self.add_capability(capability)
        
        # Add all categories
        for category in default_categories:
            self.capability_categories.add(category)
        
        self.save_model()
    
    def add_capability(self, capability: Capability) -> str:
        """
        Add a new capability to the model.
        
        Args:
            capability: The capability to add
            
        Returns:
            ID of the added capability
        """
        # Add to capability list
        self.capabilities[capability.id] = capability
        
        # Add category if new
        self.capability_categories.add(capability.category)
        
        # Update dependencies
        for dep_id in capability.dependencies:
            if dep_id in self.capabilities:
                dep_capability = self.capabilities[dep_id]
                if capability.id not in dep_capability.required_by:
                    dep_capability.required_by.append(capability.id)
                    dep_capability.updated_at = datetime.datetime.now().isoformat()
        
        self.save_model()
        return capability.id
    
    def update_capability(self, capability_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing capability.
        
        Args:
            capability_id: ID of the capability to update
            updates: Dictionary of fields to update
            
        Returns:
            Whether the update was successful
        """
        if capability_id not in self.capabilities:
            return False
        
        capability = self.capabilities[capability_id]
        
        # Update fields
        for field, value in updates.items():
            if field in ["name", "description", "category", "implementation_status", "confidence"]:
                setattr(capability, field, value)
            elif field in ["dependencies", "required_by", "examples"]:
                if value is None:
                    setattr(capability, field, [])
                else:
                    setattr(capability, field, value)
            elif field == "metadata":
                if value is None:
                    capability.metadata = {}
                else:
                    capability.metadata.update(value)
        
        # Update timestamp
        capability.updated_at = datetime.datetime.now().isoformat()
        
        # Add new category if needed
        if "category" in updates:
            self.capability_categories.add(updates["category"])
        
        # Update dependency relationships
        if "dependencies" in updates:
            # First, remove this capability from required_by of old dependencies
            for cap in self.capabilities.values():
                if capability_id in cap.required_by and cap.id not in capability.dependencies:
                    cap.required_by.remove(capability_id)
                    cap.updated_at = datetime.datetime.now().isoformat()
            
            # Then, add this capability to required_by of new dependencies
            for dep_id in capability.dependencies:
                if dep_id in self.capabilities:
                    dep_capability = self.capabilities[dep_id]
                    if capability_id not in dep_capability.required_by:
                        dep_capability.required_by.append(capability_id)
                        dep_capability.updated_at = datetime.datetime.now().isoformat()
        
        self.save_model()
        return True
    
    def remove_capability(self, capability_id: str) -> bool:
        """
        Remove a capability from the model.
        
        Args:
            capability_id: ID of the capability to remove
            
        Returns:
            Whether the removal was successful
        """
        if capability_id not in self.capabilities:
            return False
        
        capability = self.capabilities[capability_id]
        
        # Update dependencies
        for dep_id in capability.dependencies:
            if dep_id in self.capabilities:
                dep_capability = self.capabilities[dep_id]
                if capability_id in dep_capability.required_by:
                    dep_capability.required_by.remove(capability_id)
                    dep_capability.updated_at = datetime.datetime.now().isoformat()
        
        # Update required_by
        for req_id in capability.required_by:
            if req_id in self.capabilities:
                req_capability = self.capabilities[req_id]
                if capability_id in req_capability.dependencies:
                    req_capability.dependencies.remove(capability_id)
                    req_capability.updated_at = datetime.datetime.now().isoformat()
        
        # Remove the capability
        del self.capabilities[capability_id]
        
        # Update categories
        self._update_categories()
        
        self.save_model()
        return True
    
    def _update_categories(self) -> None:
        """Update the set of capability categories."""
        self.capability_categories = set(cap.category for cap in self.capabilities.values())
    
    def get_capability(self, capability_id: str) -> Optional[Capability]:
        """
        Get a specific capability by ID.
        
        Args:
            capability_id: ID of the capability to get
            
        Returns:
            The capability, or None if not found
        """
        return self.capabilities.get(capability_id)
    
    def get_all_capabilities(self) -> List[Capability]:
        """Get all capabilities."""
        return list(self.capabilities.values())
    
    def get_capabilities_by_category(self, category: str) -> List[Capability]:
        """
        Get capabilities in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of capabilities in the category
        """
        return [cap for cap in self.capabilities.values() if cap.category == category]
    
    def get_capability_categories(self) -> List[str]:
        """Get all capability categories."""
        return sorted(list(self.capability_categories))
    
    def get_implemented_capabilities(self) -> List[Capability]:
        """Get all implemented capabilities."""
        return [cap for cap in self.capabilities.values() 
                if cap.implementation_status == "implemented"]
    
    def get_unimplemented_capabilities(self) -> List[Capability]:
        """Get all unimplemented capabilities."""
        return [cap for cap in self.capabilities.values() 
                if cap.implementation_status != "implemented"]
    
    def get_capability_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the capability dependency graph."""
        graph = {}
        for capability in self.capabilities.values():
            graph[capability.id] = {
                "name": capability.name,
                "dependencies": capability.dependencies,
                "required_by": capability.required_by
            }
        return graph


class CapabilityAssessmentSystem:
    """System for assessing AI capabilities and identifying gaps."""
    
    def __init__(self, creative_content_system=None, 
                capability_model_path: str = "capability_model.json"):
        """
        Initialize the capability assessment system.
        
        Args:
            creative_content_system: Optional system for storing assessment results
            capability_model_path: Path to the capability model file
        """
        self.creative_content_system = creative_content_system
        self.capability_model = CapabilityModel(capability_model_path)
        self.assessment_history = []
    
    async def assess_required_capabilities(self, goal: str) -> Dict[str, Any]:
        """
        Assess what capabilities would be required to achieve a goal.
        
        Args:
            goal: The goal to assess
            
        Returns:
            Assessment of required capabilities
        """
        # Get all implemented capabilities
        implemented_capabilities = self.capability_model.get_implemented_capabilities()
        
        # Simple keyword-based matching to find relevant capabilities
        goal_terms = set(goal.lower().split())
        
        relevant_capabilities = []
        for capability in implemented_capabilities:
            capability_terms = set(capability.name.lower().split() + 
                                capability.description.lower().split())
            
            # Calculate overlap
            overlap = goal_terms.intersection(capability_terms)
            if overlap:
                relevant_capabilities.append({
                    "capability": capability.to_dict(),
                    "relevance": len(overlap) / len(goal_terms)
                })
        
        # Sort by relevance
        relevant_capabilities.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Identify potential capability gaps
        potential_gaps = []
        
        # This is a simplified approach - in a real system you'd want more
        # sophisticated NLP to identify implied capabilities
        common_capability_terms = {
            "write": "text_generation",
            "generate": "text_generation",
            "create": "creative_content_generation",
            "analyze": "analytical_reasoning",
            "understand": "comprehension",
            "learn": "learning",
            "remember": "memory",
            "decide": "decision_making",
            "plan": "planning",
            "talk": "conversation",
            "speak": "speech_generation",
            "listen": "speech_recognition",
            "see": "image_recognition",
            "draw": "image_generation",
            "code": "code_generation",
            "program": "code_generation",
            "calculate": "mathematical_computation",
            "predict": "prediction",
            "explain": "explanation_generation",
            "summarize": "summarization",
            "translate": "translation"
        }
        
        for term in goal_terms:
            if term in common_capability_terms:
                capability_name = common_capability_terms[term]
                
                # Check if this capability is already represented
                if not any(cap["capability"]["name"] == capability_name 
                        for cap in relevant_capabilities):
                    potential_gaps.append({
                        "term": term,
                        "suggested_capability": capability_name,
                        "confidence": 0.7
                    })
        
        # Create assessment record
        assessment = {
            "goal": goal,
            "timestamp": datetime.datetime.now().isoformat(),
            "relevant_capabilities": relevant_capabilities,
            "potential_gaps": potential_gaps,
            "overall_feasibility": self._calculate_feasibility(
                relevant_capabilities, potential_gaps)
        }
        
        # Save assessment to history
        self.assessment_history.append(assessment)
        
        # Store as content if creative content system is available
        if self.creative_content_system:
            assessment_md = self._format_assessment_to_markdown(assessment)
            
            await self.creative_content_system.store_content(
                content_type="assessment",
                title=f"Capability Assessment: {goal[:50]}{'...' if len(goal) > 50 else ''}",
                content=assessment_md,
                metadata={
                    "goal": goal,
                    "assessment_type": "capability_assessment",
                    "timestamp": assessment["timestamp"]
                }
            )
        
        return assessment
    
    def _calculate_feasibility(self, 
                             relevant_capabilities: List[Dict[str, Any]], 
                             potential_gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall feasibility of a goal based on capabilities."""
        if not relevant_capabilities and not potential_gaps:
            return {
                "score": 0.1,
                "confidence": 0.3,
                "assessment": "No relevant capabilities identified"
            }
        
        # Calculate weighted score based on relevant capabilities
        total_score = 0
        total_weight = 0
        
        for rc in relevant_capabilities:
            relevance = rc["relevance"]
            capability = rc["capability"]
            confidence = capability["confidence"]
            
            total_score += relevance * confidence
            total_weight += relevance
        
        # Account for gaps
        gap_penalty = len(potential_gaps) * 0.1
        
        # Calculate final score
        if total_weight > 0:
            feasibility_score = total_score / total_weight
        else:
            feasibility_score = 0.5
        
        # Apply gap penalty, but don't go below 0.1
        feasibility_score = max(0.1, feasibility_score - gap_penalty)
        
        # Determine confidence in assessment
        if len(relevant_capabilities) > 3:
            confidence = 0.8
        elif len(relevant_capabilities) > 0:
            confidence = 0.6
        else:
            confidence = 0.4
        
        # Generate assessment text
        if feasibility_score > 0.8:
            assessment = "Highly feasible - all required capabilities available"
        elif feasibility_score > 0.6:
            assessment = "Feasible - most required capabilities available"
        elif feasibility_score > 0.4:
            assessment = "Moderately feasible - some capabilities may need enhancement"
        elif feasibility_score > 0.2:
            assessment = "Challenging - significant capability gaps exist"
        else:
            assessment = "Difficult - major capability enhancements required"
        
        return {
            "score": feasibility_score,
            "confidence": confidence,
            "assessment": assessment
        }
    
    def _format_assessment_to_markdown(self, assessment: Dict[str, Any]) -> str:
        """Format capability assessment as markdown."""
        md = f"# Capability Assessment\n\n"
        md += f"**Goal:** {assessment['goal']}\n\n"
        
        # Add feasibility section
        feasibility = assessment["overall_feasibility"]
        md += "## Overall Feasibility\n\n"
        md += f"**Score:** {feasibility['score']:.2f} / 1.0\n"
        md += f"**Confidence:** {feasibility['confidence']:.2f} / 1.0\n"
        md += f"**Assessment:** {feasibility['assessment']}\n\n"
        
        # Add relevant capabilities section
        md += "## Relevant Capabilities\n\n"
        if assessment["relevant_capabilities"]:
            for rc in assessment["relevant_capabilities"]:
                capability = rc["capability"]
                md += f"### {capability['name']} (Relevance: {rc['relevance']:.2f})\n\n"
                md += f"{capability['description']}\n\n"
                md += f"- **Category:** {capability['category']}\n"
                md += f"- **Implementation Status:** {capability['implementation_status']}\n"
                md += f"- **Confidence:** {capability['confidence']:.2f} / 1.0\n\n"
        else:
            md += "No directly relevant capabilities identified.\n\n"
        
        # Add capability gaps section
        md += "## Potential Capability Gaps\n\n"
        if assessment["potential_gaps"]:
            for gap in assessment["potential_gaps"]:
                md += f"- **Term '{gap['term']}'** suggests need for capability: "
                md += f"**{gap['suggested_capability']}** "
                md += f"(Confidence: {gap['confidence']:.2f})\n"
        else:
            md += "No significant capability gaps identified.\n\n"
        
        # Add recommendations section
        md += "## Recommendations\n\n"
        
        if not assessment["relevant_capabilities"] and not assessment["potential_gaps"]:
            md += "Insufficient information to provide specific recommendations.\n\n"
        else:
            if assessment["potential_gaps"]:
                md += "### Suggested New Capabilities\n\n"
                for gap in assessment["potential_gaps"]:
                    md += f"- Develop or enhance **{gap['suggested_capability']}** capability\n"
                md += "\n"
            
            if assessment["relevant_capabilities"]:
                low_confidence = [rc for rc in assessment["relevant_capabilities"] 
                                if rc["capability"]["confidence"] < 0.7]
                if low_confidence:
                    md += "### Capabilities to Strengthen\n\n"
                    for rc in low_confidence:
                        capability = rc["capability"]
                        md += f"- Improve confidence in **{capability['name']}** "
                        md += f"(current confidence: {capability['confidence']:.2f})\n"
        
        return md
    
    async def identify_capability_gaps(self) -> Dict[str, Any]:
        """
        Identify overall capability gaps based on model and dependencies.
        
        Returns:
            Gap analysis results
        """
        capabilities = self.capability_model.get_all_capabilities()
        
        # Identify gaps from dependencies
        dependency_gaps = []
        for capability in capabilities:
            for dep_id in capability.dependencies:
                if dep_id not in self.capability_model.capabilities:
                    dependency_gaps.append({
                        "capability_id": capability.id,
                        "capability_name": capability.name,
                        "missing_dependency_id": dep_id,
                        "gap_type": "missing_dependency"
                    })
        
        # Identify low-confidence capabilities
        low_confidence = [cap for cap in capabilities if cap.confidence < 0.5]
        
        # Identify capability category gaps
        category_counts = {}
        for capability in capabilities:
            category = capability.category
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        # Define expected minimum capabilities per category
        expected_minimums = {
            "cognitive": 3,
            "communication": 2,
            "creative": 3,
            "technical": 3,
            "meta": 2
        }
        
        category_gaps = []
        for category, expected in expected_minimums.items():
            count = category_counts.get(category, 0)
            if count < expected:
                category_gaps.append({
                    "category": category,
                    "current_count": count,
                    "expected_minimum": expected,
                    "gap": expected - count
                })
        
        # Create gap analysis
        gap_analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_capabilities": len(capabilities),
            "implemented_capabilities": len(self.capability_model.get_implemented_capabilities()),
            "dependency_gaps": dependency_gaps,
            "low_confidence_capabilities": [cap.to_dict() for cap in low_confidence],
            "category_gaps": category_gaps
        }
        
        # Store as content if creative content system is available
        if self.creative_content_system:
            gap_analysis_md = self._format_gap_analysis_to_markdown(gap_analysis)
            
            await self.creative_content_system.store_content(
                content_type="assessment",
                title=f"Capability Gap Analysis",
                content=gap_analysis_md,
                metadata={
                    "assessment_type": "capability_gap_analysis",
                    "timestamp": gap_analysis["timestamp"]
                }
            )
        
        return gap_analysis
    
    def _format_gap_analysis_to_markdown(self, gap_analysis: Dict[str, Any]) -> str:
        """Format capability gap analysis as markdown."""
        md = f"# Capability Gap Analysis\n\n"
        
        # Add summary section
        md += "## Summary\n\n"
        md += f"**Total Capabilities:** {gap_analysis['total_capabilities']}\n"
        md += f"**Implemented Capabilities:** {gap_analysis['implemented_capabilities']}\n"
        md += f"**Dependency Gaps:** {len(gap_analysis['dependency_gaps'])}\n"
        md += f"**Low Confidence Capabilities:** {len(gap_analysis['low_confidence_capabilities'])}\n"
        md += f"**Category Gaps:** {len(gap_analysis['category_gaps'])}\n\n"
        
        # Add dependency gaps section
        md += "## Dependency Gaps\n\n"
        if gap_analysis["dependency_gaps"]:
            for gap in gap_analysis["dependency_gaps"]:
                md += f"- **{gap['capability_name']}** depends on missing capability {gap['missing_dependency_id']}\n"
        else:
            md += "No dependency gaps identified.\n\n"
        
        # Add low confidence capabilities section
        md += "## Low Confidence Capabilities\n\n"
        if gap_analysis["low_confidence_capabilities"]:
            for cap in gap_analysis["low_confidence_capabilities"]:
                md += f"- **{cap['name']}** (Confidence: {cap['confidence']:.2f})\n"
                md += f"  *{cap['description']}*\n"
        else:
            md += "No low confidence capabilities identified.\n\n"
        
        # Add category gaps section
        md += "## Category Gaps\n\n"
        if gap_analysis["category_gaps"]:
            for gap in gap_analysis["category_gaps"]:
                md += f"- **{gap['category']}**: {gap['current_count']} capabilities "
                md += f"(minimum expected: {gap['expected_minimum']}, gap: {gap['gap']})\n"
        else:
            md += "No category gaps identified.\n\n"
        
        # Add recommendations section
        md += "## Recommendations\n\n"
        
        recommendations = []
        
        # Add dependency recommendations
        if gap_analysis["dependency_gaps"]:
            dependency_recs = set()
            for gap in gap_analysis["dependency_gaps"]:
                dependency_recs.add(f"Implement missing dependency: {gap['missing_dependency_id']}")
            for rec in dependency_recs:
                recommendations.append(rec)
        
        # Add low confidence recommendations
        if gap_analysis["low_confidence_capabilities"]:
            for cap in gap_analysis["low_confidence_capabilities"]:
                recommendations.append(f"Improve confidence in capability: {cap['name']}")
        
        # Add category recommendations
        if gap_analysis["category_gaps"]:
            for gap in gap_analysis["category_gaps"]:
                recommendations.append(f"Add {gap['gap']} more {gap['category']} capabilities")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                md += f"{i}. {rec}\n"
        else:
            md += "No significant gaps identified that require immediate attention.\n\n"
        
        return md
    
    async def add_desired_capability(self, 
                                 name: str, 
                                 description: str,
                                 category: str,
                                 examples: List[str] = None,
                                 dependencies: List[str] = None) -> Dict[str, Any]:
        """
        Add a desired capability to the model.
        
        Args:
            name: Capability name
            description: Capability description
            category: Capability category
            examples: Optional example uses
            dependencies: Optional dependency capability IDs
            
        Returns:
            Result of the operation
        """
        # Check if a similar capability already exists
        similar_capabilities = []
        for capability in self.capability_model.get_all_capabilities():
            if capability.name.lower() == name.lower():
                # Exact name match
                return {
                    "success": False,
                    "message": f"Capability with name '{name}' already exists",
                    "existing_capability": capability.to_dict()
                }
            
            # Check for similarity in name or description
            name_similarity = self._calculate_text_similarity(
                capability.name.lower(), name.lower())
            desc_similarity = self._calculate_text_similarity(
                capability.description.lower(), description.lower())
            
            similarity = max(name_similarity, desc_similarity)
            if similarity > 0.7:
                similar_capabilities.append({
                    "capability": capability.to_dict(),
                    "similarity": similarity
                })
        
        # If similar capabilities found, return them without adding
        if similar_capabilities:
            return {
                "success": False,
                "message": "Similar capabilities already exist",
                "similar_capabilities": similar_capabilities
            }
        
        # Create new capability
        new_capability = Capability(
            name=name,
            description=description,
            category=category,
            implementation_status="desired",
            confidence=0.0,
            examples=examples or [],
            dependencies=dependencies or []
        )
        
        # Add to model
        capability_id = self.capability_model.add_capability(new_capability)
        
        # Store as content if creative content system is available
        if self.creative_content_system:
            capability_md = self._format_capability_to_markdown(new_capability)
            
            await self.creative_content_system.store_content(
                content_type="assessment",
                title=f"New Desired Capability: {name}",
                content=capability_md,
                metadata={
                    "assessment_type": "desired_capability",
                    "capability_name": name,
                    "capability_id": capability_id,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        
        return {
            "success": True,
            "message": f"Added desired capability: {name}",
            "capability_id": capability_id,
            "capability": new_capability.to_dict()
        }
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple Jaccard similarity implementation
        if not text1 or not text2:
            return 0.0
        
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _format_capability_to_markdown(self, capability: Capability) -> str:
        """Format capability as markdown."""
        md = f"# Capability: {capability.name}\n\n"
        md += f"{capability.description}\n\n"
        
        md += f"**Category:** {capability.category}\n"
        md += f"**Status:** {capability.implementation_status}\n"
        md += f"**Confidence:** {capability.confidence:.2f} / 1.0\n\n"
        
        if capability.dependencies:
            md += "## Dependencies\n\n"
            for dep_id in capability.dependencies:
                md += f"- {dep_id}\n"
            md += "\n"
        
        if capability.examples:
            md += "## Examples\n\n"
            for example in capability.examples:
                md += f"- {example}\n"
            md += "\n"
        
        md += f"**Created:** {capability.created_at}\n"
        
        return md
