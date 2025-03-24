# nyx/core/procedural_memory/generalization.py

import datetime
import random
from typing import Dict, List, Any, Optional, Tuple
from .models import ActionTemplate, ChunkTemplate, ControlMapping, ProcedureTransferRecord

class ProceduralChunkLibrary:
    """Library of generalizable procedural chunks that can transfer across domains"""
    
    def __init__(self):
        self.chunk_templates = {}  # template_id -> ChunkTemplate
        self.action_templates = {}  # action_type -> ActionTemplate
        self.domain_chunks = {}  # domain -> [chunk_ids]
        self.control_mappings = []  # List of ControlMapping objects
        self.transfer_records = []  # List of ProcedureTransferRecord objects
        self.similarity_threshold = 0.7  # Minimum similarity for chunk matching
        
    def add_chunk_template(self, template: ChunkTemplate) -> str:
        """Add a chunk template to the library"""
        self.chunk_templates[template.id] = template
        
        # Update domain index
        for domain in template.domains:
            if domain not in self.domain_chunks:
                self.domain_chunks[domain] = []
            self.domain_chunks[domain].append(template.id)
        
        return template.id
    
    def add_action_template(self, template: ActionTemplate) -> str:
        """Add an action template to the library"""
        self.action_templates[template.action_type] = template
        return template.action_type
    
    def add_control_mapping(self, mapping: ControlMapping) -> None:
        """Add a control mapping between domains"""
        self.control_mappings.append(mapping)
    
    def record_transfer(self, record: ProcedureTransferRecord) -> None:
        """Record a procedure transfer between domains"""
        self.transfer_records.append(record)
    
    def find_matching_chunks(self, 
                           steps: List[Dict[str, Any]], 
                           source_domain: str,
                           target_domain: str) -> List[Dict[str, Any]]:
        """
        Find library chunks that match a sequence of steps
        
        Args:
            steps: List of step definitions from source procedure
            source_domain: Domain of the source procedure
            target_domain: Domain where we want to apply the chunk
            
        Returns:
            List of matching chunks with similarity scores
        """
        matches = []
        
        # Convert steps to action templates
        action_sequences = self._extract_action_sequence(steps, source_domain)
        
        # Skip if we couldn't extract actions
        if not action_sequences:
            return []
        
        # Find templates that match this action sequence
        for template_id, template in self.chunk_templates.items():
            # Skip if template doesn't support source domain
            if source_domain not in template.domains:
                continue
                
            # Calculate similarity between template and action sequence
            similarity = self._calculate_sequence_similarity(template.actions, action_sequences)
            
            if similarity >= self.similarity_threshold:
                # Check if template has been applied to target domain
                target_applicability = 0.5  # Default medium applicability
                
                if target_domain in template.domains:
                    # Template already used in target domain
                    target_applicability = 0.9
                    
                    # Adjust based on success rate if available
                    if target_domain in template.success_rate:
                        target_applicability *= template.success_rate[target_domain]
                
                matches.append({
                    "template_id": template_id,
                    "template_name": template.name,
                    "similarity": similarity,
                    "target_applicability": target_applicability,
                    "overall_score": similarity * 0.6 + target_applicability * 0.4,
                    "action_count": len(template.actions)
                })
        
        # Sort by overall score
        matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return matches
    
    def create_chunk_template_from_steps(self,
                                      chunk_id: str,
                                      name: str,
                                      steps: List[Dict[str, Any]],
                                      domain: str,
                                      success_rate: float = 0.9) -> Optional[ChunkTemplate]:
        """
        Create a generalizable chunk template from procedure steps
        
        Args:
            chunk_id: ID for the new chunk template
            name: Name for the template
            steps: Original procedure steps to generalize
            domain: Domain of the procedure
            success_rate: Initial success rate for this domain
            
        Returns:
            New chunk template or None if generalization failed
        """
        # Extract action templates
        action_sequence = self._extract_action_sequence(steps, domain)
        
        if not action_sequence:
            return None
        
        # Create template
        template = ChunkTemplate(
            id=chunk_id,
            name=name,
            description=f"Generalized chunk template for {name}",
            actions=action_sequence,
            domains=[domain],
            success_rate={domain: success_rate},
            execution_count={domain: 1}
        )
        
        # Add to library
        self.add_chunk_template(template)
        
        return template
    
    def map_chunk_to_new_domain(self, 
                              template_id: str, 
                              target_domain: str) -> List[Dict[str, Any]]:
        """
        Map a chunk template to a new domain
        
        Args:
            template_id: ID of the chunk template
            target_domain: Domain to map to
            
        Returns:
            List of mapped steps for the new domain
        """
        if template_id not in self.chunk_templates:
            return []
        
        template = self.chunk_templates[template_id]
        
        # Skip if already mapped to this domain
        if target_domain in template.domains:
            # Just return the existing mapping
            return self._generate_domain_steps(template, target_domain)
        
        # We need to create a new mapping
        mapped_steps = []
        
        # For each action in the template
        for i, action in enumerate(template.actions):
            # Check if we have a domain mapping
            if target_domain in action.domain_mappings:
                # Use existing mapping
                mapped_action = action.domain_mappings[target_domain]
            else:
                # Create new mapping
                mapped_action = self._map_action_to_domain(action, target_domain)
                
                if mapped_action:
                    # Save mapping for future use
                    action.domain_mappings[target_domain] = mapped_action
            
            if not mapped_action:
                # Couldn't map this action
                continue
                
            # Create step from mapped action
            step = {
                "id": f"step_{i+1}",
                "description": mapped_action.get("description", f"Step {i+1}"),
                "function": mapped_action.get("function"),
                "parameters": mapped_action.get("parameters", {})
            }
            
            mapped_steps.append(step)
        
        # Update template
        if mapped_steps:
            template.domains.append(target_domain)
            template.success_rate[target_domain] = 0.5  # Initial estimate
            template.execution_count[target_domain] = 0
            template.last_updated = datetime.datetime.now().isoformat()
        
        return mapped_steps
    
    def update_template_success(self, 
                              template_id: str, 
                              domain: str, 
                              success: bool) -> None:
        """Update success rate for a template in a specific domain"""
        if template_id not in self.chunk_templates:
            return
            
        template = self.chunk_templates[template_id]
        
        # Update execution count
        if domain not in template.execution_count:
            template.execution_count[domain] = 0
        template.execution_count[domain] += 1
        
        # Update success rate
        if domain not in template.success_rate:
            template.success_rate[domain] = 0.5  # Default
            
        # Use exponential moving average
        current_rate = template.success_rate[domain]
        success_value = 1.0 if success else 0.0
        
        # More weight to recent results but don't change too drastically
        template.success_rate[domain] = current_rate * 0.8 + success_value * 0.2
        
        # Update timestamp
        template.last_updated = datetime.datetime.now().isoformat()
    
    def get_control_mapping(self, 
                          source_domain: str, 
                          target_domain: str, 
                          action_type: str) -> Optional[ControlMapping]:
        """Get control mapping between domains for a specific action"""
        for mapping in self.control_mappings:
            if (mapping.source_domain == source_domain and
                mapping.target_domain == target_domain and
                mapping.action_type == action_type):
                return mapping
        
        return None
    
    def _extract_action_sequence(self, 
                              steps: List[Dict[str, Any]], 
                              domain: str) -> List[ActionTemplate]:
        """
        Extract generalized action sequence from procedure steps
        
        Args:
            steps: Procedure steps
            domain: Domain of the procedure
            
        Returns:
            List of action templates
        """
        action_sequence = []
        
        for i, step in enumerate(steps):
            # Skip if no function
            if "function" not in step:
                continue
            
            # Determine action type and intent
            action_type, intent = self._infer_action_type(step, domain)
            
            if not action_type:
                continue
                
            # Check if we already have a template for this action type
            if action_type in self.action_templates:
                # Use existing template
                template = self.action_templates[action_type]
                
                # Add domain-specific mapping if not present
                if domain not in template.domain_mappings:
                    template.domain_mappings[domain] = {
                        "function": step.get("function"),
                        "parameters": step.get("parameters", {}),
                        "description": step.get("description", f"Step {i+1}")
                    }
            else:
                # Create new template
                template = ActionTemplate(
                    action_type=action_type,
                    intent=intent,
                    parameters={},  # Generic parameters
                    domain_mappings={
                        domain: {
                            "function": step.get("function"),
                            "parameters": step.get("parameters", {}),
                            "description": step.get("description", f"Step {i+1}")
                        }
                    }
                )
                
                # Add to library
                self.add_action_template(template)
            
            # Add to sequence
            action_sequence.append(template)
        
        return action_sequence
    
    def _infer_action_type(self, step: Dict[str, Any], domain: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Infer the general action type and intent from a step
        
        Args:
            step: Procedure step
            domain: Domain of the procedure
            
        Returns:
            Tuple of (action_type, intent)
        """
        function = step.get("function", "")
        parameters = step.get("parameters", {})
        description = step.get("description", "").lower()
        
        # Try to infer from function name and description
        if isinstance(function, str):
            function_lower = function.lower()
            
            # Locomotion actions
            if any(word in function_lower or word in description for word in 
                ["move", "walk", "run", "navigate", "approach", "go", "sprint"]):
                return "locomotion", "navigation"
                
            # Interaction actions
            if any(word in function_lower or word in description for word in 
                ["press", "click", "push", "interact", "use", "activate"]):
                # Check for specific interaction types
                if "button" in parameters:
                    button = parameters["button"]
                    
                    # Common gaming controls
                    if domain == "gaming":
                        if button in ["R1", "R2", "RT", "RB"]:
                            return "primary_action", "interaction"
                        elif button in ["L1", "L2", "LT", "LB"]:
                            return "secondary_action", "targeting"
                        elif button in ["X", "A"]:
                            return "confirm", "interaction"
                        elif button in ["O", "B"]:
                            return "cancel", "navigation"
                
                return "interaction", "manipulation"
                
            # Target/select actions
            if any(word in function_lower or word in description for word in 
                ["select", "choose", "target", "aim", "focus", "look"]):
                return "selection", "targeting"
                
            # Acquisition actions
            if any(word in function_lower or word in description for word in 
                ["get", "pick", "grab", "take", "collect", "acquire"]):
                return "acquisition", "collection"
                
            # Communication actions
            if any(word in function_lower or word in description for word in 
                ["speak", "say", "ask", "tell", "communicate"]):
                return "communication", "information_exchange"
                
            # Cognitive actions
            if any(word in function_lower or word in description for word in 
                ["think", "decide", "analyze", "evaluate", "assess"]):
                return "cognition", "decision_making"
                
            # Creation actions
            if any(word in function_lower or word in description for word in 
                ["make", "build", "create", "construct", "compose"]):
                return "creation", "production"
        
        # Domain-specific inference
        if domain == "dbd":  # Dead by Daylight specific actions
            if "vault" in description:
                return "vault", "locomotion"
            if "generator" in description or "gen" in description:
                return "repair", "objective"
        
        # Default fallback - use a generic action type based on domain
        domain_action_types = {
            "cooking": "preparation",
            "driving": "operation",
            "gaming": "gameplay",
            "writing": "composition",
            "music": "performance",
            "programming": "coding",
            "sports": "movement"
        }
        
        return domain_action_types.get(domain, "generic_action"), "task_progress"
    
    def _calculate_sequence_similarity(self, 
                                    template_actions: List[ActionTemplate], 
                                    candidate_actions: List[ActionTemplate]) -> float:
        """
        Calculate similarity between two action sequences
        
        Args:
            template_actions: Actions from template
            candidate_actions: Actions extracted from steps
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Handle empty sequences
        if not template_actions or not candidate_actions:
            return 0.0
            
        # Dynamic programming approach for sequence alignment
        m = len(template_actions)
        n = len(candidate_actions)
        
        # Create DP table
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Fill DP table
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Calculate similarity between actions
                action_similarity = self._calculate_action_similarity(
                    template_actions[i-1], 
                    candidate_actions[j-1]
                )
                
                if action_similarity > 0.7:  # High similarity threshold
                    # These actions match, extend the sequence
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # Take max of skipping either action
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        # Length of longest common subsequence
        lcs = dp[m][n]
        
        # Calculate similarity based on sequence coverage
        similarity = (2 * lcs) / (m + n)  # Harmonic mean of coverage in both sequences
        
        return similarity
    
    def _calculate_action_similarity(self, action1: ActionTemplate, action2: ActionTemplate) -> float:
        """
        Calculate similarity between two actions
        
        Args:
            action1: First action
            action2: Second action
            
        Returns:
            Similarity score (0.0-1.0)
        """
        # Exact match on action type
        if action1.action_type == action2.action_type:
            return 1.0
            
        # Match on intent
        if action1.intent == action2.intent:
            return 0.8
            
        # Otherwise low similarity
        return 0.2
    
    def _map_action_to_domain(self, 
                            action: ActionTemplate, 
                            target_domain: str) -> Optional[Dict[str, Any]]:
        """
        Map an action to a specific domain
        
        Args:
            action: Action template to map
            target_domain: Target domain
            
        Returns:
            Mapped action parameters or None if mapping failed
        """
        # Check if we have existing mappings for other domains
        source_domains = [d for d in action.domain_mappings.keys()]
        
        if not source_domains:
            return None
            
        # Try to find control mappings for this action type
        for source_domain in source_domains:
            mapping = self.get_control_mapping(
                source_domain=source_domain,
                target_domain=target_domain,
                action_type=action.action_type
            )
            
            if mapping:
                # We found a mapping
                source_implementation = action.domain_mappings[source_domain]
                
                # Apply mapping to create target implementation
                target_impl = source_implementation.copy()
                
                # Update any mapped controls
                if "parameters" in target_impl:
                    params = target_impl["parameters"]
                    
                    for param_key, param_value in params.items():
                        # Check if this parameter is a control that needs mapping
                        if param_key in ["control", "button", "input_method"]:
                            if param_value == mapping.source_control:
                                params[param_key] = mapping.target_control
                
                return target_impl
        
        # No mapping found, use best guess
        best_source = source_domains[0]  # Just use first domain as best guess
        best_impl = action.domain_mappings[best_source].copy()
        
        # Mark as best guess
        if "parameters" not in best_impl:
            best_impl["parameters"] = {}
        best_impl["parameters"]["best_guess_mapping"] = True
        
        return best_impl
    
    def _generate_domain_steps(self, template: ChunkTemplate, domain: str) -> List[Dict[str, Any]]:
        """Generate concrete steps for a domain from a template"""
        steps = []
        
        # Create steps from actions
        for i, action in enumerate(template.actions):
            if domain in action.domain_mappings:
                impl = action.domain_mappings[domain]
                
                step = {
                    "id": f"step_{i+1}",
                    "description": impl.get("description", f"Step {i+1}"),
                    "function": impl.get("function"),
                    "parameters": impl.get("parameters", {})
                }
                
                steps.append(step)
        
        return steps
