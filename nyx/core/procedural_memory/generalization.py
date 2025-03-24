# nyx/core/procedural_memory/generalization.py

import datetime
import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from .models import ActionTemplate, ChunkTemplate, ControlMapping, ProcedureTransferRecord
import numpy as np
from functools import lru_cache
import concurrent.futures
from threading import Lock

logger = logging.getLogger(__name__)

class ProceduralChunkLibrary:
    """Library of generalizable procedural chunks that can transfer across domains"""
    
    def __init__(self):
        self.chunk_templates = {}  # template_id -> ChunkTemplate
        self.action_templates = {}  # action_type -> ActionTemplate
        self.domain_chunks = {}  # domain -> [chunk_ids]
        self.control_mappings = []  # List of ControlMapping objects
        self.transfer_records = []  # List of ProcedureTransferRecord objects
        self.similarity_threshold = 0.7  # Minimum similarity for chunk matching
        # Added fields for enhanced functionality
        self.template_cache = {}  # Cache for frequent template lookups
        self.similarity_cache = {}  # Cache for similarity calculations
        self.library_lock = Lock()  # Lock for thread safety
        self.domain_usage_stats = {}  # Domain usage statistics
        self.max_records = 100  # Maximum records to keep
        
    def add_chunk_template(self, template: ChunkTemplate) -> str:
        """Add a chunk template to the library"""
        with self.library_lock:
            self.chunk_templates[template.id] = template
            
            # Update domain index
            for domain in template.domains:
                if domain not in self.domain_chunks:
                    self.domain_chunks[domain] = []
                self.domain_chunks[domain].append(template.id)
                
                # Update domain usage statistics
                if domain not in self.domain_usage_stats:
                    self.domain_usage_stats[domain] = {"templates": 0, "usages": 0}
                self.domain_usage_stats[domain]["templates"] += 1
            
            # Clear relevant caches
            self._clear_related_caches(template.domains)
        
        return template.id
    
    def add_action_template(self, template: ActionTemplate) -> str:
        """Add an action template to the library"""
        with self.library_lock:
            self.action_templates[template.action_type] = template
        return template.action_type
    
    def add_control_mapping(self, mapping: ControlMapping) -> None:
        """Add a control mapping between domains"""
        with self.library_lock:
            # Check for existing mapping and update if found
            for i, existing in enumerate(self.control_mappings):
                if (existing.source_domain == mapping.source_domain and
                    existing.target_domain == mapping.target_domain and
                    existing.action_type == mapping.action_type and
                    existing.source_control == mapping.source_control):
                    # Update existing mapping
                    self.control_mappings[i] = mapping
                    return
                    
            # Add new mapping
            self.control_mappings.append(mapping)
            
            # Clear caches related to these domains
            self._clear_related_caches([mapping.source_domain, mapping.target_domain])
    
    def record_transfer(self, record: ProcedureTransferRecord) -> None:
        """Record a procedure transfer between domains"""
        with self.library_lock:
            self.transfer_records.append(record)
            
            # Limit number of records
            if len(self.transfer_records) > self.max_records:
                self.transfer_records = self.transfer_records[-self.max_records:]
            
            # Update domain usage statistics
            source_domain = record.source_domain
            target_domain = record.target_domain
            
            for domain in [source_domain, target_domain]:
                if domain not in self.domain_usage_stats:
                    self.domain_usage_stats[domain] = {"templates": 0, "usages": 0}
                self.domain_usage_stats[domain]["usages"] += 1
    
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
        # Generate cache key
        steps_hash = hash(str([(s.get("function"), str(s.get("parameters", {}))) for s in steps]))
        cache_key = f"{steps_hash}_{source_domain}_{target_domain}"
        
        # Check cache first
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
            
        # Convert steps to action templates
        action_sequences = self._extract_action_sequence(steps, source_domain)
        
        # Skip if we couldn't extract actions
        if not action_sequences:
            return []
        
        # Prepare list for multithreaded processing
        template_items = list(self.chunk_templates.items())
        
        # Use threading for better performance with many templates
        matches = []
        
        # Check if we should use threading based on number of templates
        if len(template_items) > 10:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit tasks
                future_to_template = {
                    executor.submit(
                        self._check_template_match, 
                        template_id, 
                        template, 
                        action_sequences,
                        source_domain,
                        target_domain
                    ): (template_id, template) 
                    for template_id, template in template_items
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_template):
                    result = future.result()
                    if result:
                        matches.append(result)
        else:
            # Simple sequential processing for small template sets
            for template_id, template in template_items:
                match = self._check_template_match(
                    template_id,
                    template,
                    action_sequences,
                    source_domain,
                    target_domain
                )
                if match:
                    matches.append(match)
        
        # Sort by overall score
        matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Cache the result
        self.template_cache[cache_key] = matches
        
        return matches
    
    def _check_template_match(self,
                          template_id: str,
                          template: ChunkTemplate,
                          action_sequences: List[ActionTemplate],
                          source_domain: str,
                          target_domain: str) -> Optional[Dict[str, Any]]:
        """Check if a template matches the action sequence (helper for concurrent processing)"""
        # Skip if template doesn't support source domain
        if source_domain not in template.domains:
            return None
            
        # Calculate similarity between template and action sequence
        # Use vectorized version for performance if available
        try:
            similarity = self._calculate_sequence_similarity_vectorized(template.actions, action_sequences)
        except Exception:
            # Fall back to regular version
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
            
            return {
                "template_id": template_id,
                "template_name": template.name,
                "similarity": similarity,
                "target_applicability": target_applicability,
                "overall_score": similarity * 0.6 + target_applicability * 0.4,
                "action_count": len(template.actions)
            }
        
        return None
    
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
        
        # Generate semantic tags for better searchability
        semantic_tags = set()
        for step in steps:
            # Extract tags from function name
            func_name = step.get("function", "")
            if isinstance(func_name, str):
                # Add function name itself as tag
                semantic_tags.add(func_name)
                
                # Add words from function name
                for word in func_name.split('_'):
                    if len(word) > 3:  # Only meaningful words
                        semantic_tags.add(word)
            
            # Extract tags from description
            description = step.get("description", "")
            if description:
                # Add key words from description
                for word in description.lower().split():
                    if len(word) > 3 and word not in ["with", "from", "this", "that", "step"]:
                        semantic_tags.add(word)
        
        # Create template with semantic information
        template = ChunkTemplate(
            id=chunk_id,
            name=name,
            description=f"Generalized chunk template for {name}",
            actions=action_sequence,
            domains=[domain],
            success_rate={domain: success_rate},
            execution_count={domain: 1},
            semantic_tags=list(semantic_tags)  # Convert set to list
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
        
        # Check for cached mappings
        cache_key = f"{template_id}_{target_domain}"
        if cache_key in self.template_cache:
            return self.template_cache[cache_key]
        
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
            with self.library_lock:
                template.domains.append(target_domain)
                template.success_rate[target_domain] = 0.5  # Initial estimate
                template.execution_count[target_domain] = 0
                template.last_updated = datetime.datetime.now().isoformat()
                
                # Update domain index
                if target_domain not in self.domain_chunks:
                    self.domain_chunks[target_domain] = []
                if template_id not in self.domain_chunks[target_domain]:
                    self.domain_chunks[target_domain].append(template_id)
                
                # Update domain usage stats
                if target_domain not in self.domain_usage_stats:
                    self.domain_usage_stats[target_domain] = {"templates": 0, "usages": 0}
                self.domain_usage_stats[target_domain]["templates"] += 1
                self.domain_usage_stats[target_domain]["usages"] += 1
        
        # Cache the result
        self.template_cache[cache_key] = mapped_steps
        
        return mapped_steps
    
    def update_template_success(self, 
                              template_id: str, 
                              domain: str, 
                              success: bool) -> None:
        """Update success rate for a template in a specific domain"""
        if template_id not in self.chunk_templates:
            return
            
        template = self.chunk_templates[template_id]
        
        with self.library_lock:
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
            
            # Update usage frequency
            template.usage_frequency += 1
            template.last_used = datetime.datetime.now().isoformat()
            
            # Clear caches that might be affected
            self._clear_related_caches([domain])
    
    def get_control_mapping(self, 
                          source_domain: str, 
                          target_domain: str, 
                          action_type: str) -> Optional[ControlMapping]:
        """Get control mapping between domains for a specific action"""
        # Check for direct mapping
        for mapping in self.control_mappings:
            if (mapping.source_domain == source_domain and
                mapping.target_domain == target_domain and
                mapping.action_type == action_type):
                return mapping
        
        # Check for bidirectional mapping
        for mapping in self.control_mappings:
            if (mapping.bidirectional and
                mapping.source_domain == target_domain and
                mapping.target_domain == source_domain and
                mapping.action_type == action_type):
                # Create reverse mapping
                return ControlMapping(
                    source_domain=source_domain,
                    target_domain=target_domain,
                    action_type=action_type,
                    source_control=mapping.target_control,
                    target_control=mapping.source_control,
                    confidence=mapping.confidence * 0.9,  # Slightly lower confidence when reversed
                    bidirectional=True
                )
        
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
        # Check cache first
        cache_key = (hash(tuple(a.action_type for a in template_actions)), 
                   hash(tuple(a.action_type for a in candidate_actions)))
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
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
        
        # Cache the result
        self.similarity_cache[cache_key] = similarity
        
        return similarity
    
    @lru_cache(maxsize=128)
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
    
    def _calculate_sequence_similarity_vectorized(self, 
                                             template_actions: List[ActionTemplate], 
                                             candidate_actions: List[ActionTemplate]) -> float:
        """Calculate similarity between action sequences using vectorized operations"""
        # Check cache first
        cache_key = (hash(tuple(a.action_type for a in template_actions)), 
                    hash(tuple(a.action_type for a in candidate_actions)))
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        # Convert action sequences to feature vectors
        template_features = np.array([self._action_to_feature_vector(action) for action in template_actions])
        candidate_features = np.array([self._action_to_feature_vector(action) for action in candidate_actions])
        
        # Use dynamic time warping or other sequence matching algorithms
        # For simple dot product similarity:
        if len(template_features) == 0 or len(candidate_features) == 0:
            return 0.0
            
        # Calculate cosine similarity between sequences
        similarity_matrix = np.zeros((len(template_features), len(candidate_features)))
        
        for i in range(len(template_features)):
            for j in range(len(candidate_features)):
                dot_product = np.dot(template_features[i], candidate_features[j])
                norm_a = np.linalg.norm(template_features[i])
                norm_b = np.linalg.norm(candidate_features[j])
                
                if norm_a > 0 and norm_b > 0:
                    similarity_matrix[i, j] = dot_product / (norm_a * norm_b)
        
        # Dynamic programming for sequence alignment
        dp = np.zeros((len(template_features) + 1, len(candidate_features) + 1))
        
        for i in range(1, len(template_features) + 1):
            for j in range(1, len(candidate_features) + 1):
                if similarity_matrix[i-1, j-1] > 0.7:  # High similarity threshold
                    dp[i, j] = dp[i-1, j-1] + 1
                else:
                    dp[i, j] = max(dp[i-1, j], dp[i, j-1])
        
        # Calculate similarity score
        lcs = dp[-1, -1]
        similarity = (2 * lcs) / (len(template_features) + len(candidate_features))
        
        # Cache the result
        self.similarity_cache[cache_key] = float(similarity)
        
        return float(similarity)
    
    @lru_cache(maxsize=128)
    def _action_to_feature_vector(self, action: ActionTemplate) -> np.ndarray:
        """Convert an action to a feature vector for similarity calculations"""
        # Create a feature vector based on action properties
        # This is a simplified example - expand as needed
        action_type_hash = hash(action.action_type) % 100
        intent_hash = hash(action.intent) % 100
        
        # Create a simple feature vector (expand as needed)
        features = np.zeros(200)
        features[action_type_hash] = 1.0
        features[intent_hash + 100] = 1.0
        
        return features
    
    def _clear_related_caches(self, domains: List[str]) -> None:
        """Clear caches related to specific domains"""
        # Clear similarity cache (it's simpler to just clear all)
        self.similarity_cache = {}
        
        # Clear template cache entries related to these domains
        keys_to_remove = []
        for key in self.template_cache:
            for domain in domains:
                if domain in key:
                    keys_to_remove.append(key)
                    break
                    
        for key in keys_to_remove:
            if key in self.template_cache:
                del self.template_cache[key]
    
    # New method for batch processing templates
    def process_templates_batch(self, templates: List[ChunkTemplate]) -> None:
        """Process multiple templates in a batch for efficiency"""
        with self.library_lock:
            for template in templates:
                self.chunk_templates[template.id] = template
                
                # Update domain index
                for domain in template.domains:
                    if domain not in self.domain_chunks:
                        self.domain_chunks[domain] = []
                    if template.id not in self.domain_chunks[domain]:
                        self.domain_chunks[domain].append(template.id)
            
            # Clear all caches since we added multiple templates
            self.template_cache = {}
            self.similarity_cache = {}
    
    # New method for template search by tags
    def search_templates_by_tags(self, tags: List[str], 
                                domain: Optional[str] = None) -> List[ChunkTemplate]:
        """Search for templates by semantic tags"""
        matching_templates = []
        
        for template_id, template in self.chunk_templates.items():
            # Skip if domain filter is applied and doesn't match
            if domain and domain not in template.domains:
                continue
                
            # Check for tag matches
            matches = 0
            for tag in tags:
                if tag in template.semantic_tags:
                    matches += 1
            
            # Add if any tags match
            if matches > 0:
                # Calculate match score
                match_score = matches / len(tags)
                
                matching_templates.append({
                    "template": template,
                    "match_score": match_score
                })
        
        # Sort by match score
        matching_templates.sort(key=lambda x: x["match_score"], reverse=True)
        
        # Return just the templates
        return [item["template"] for item in matching_templates]
    
    # New method to optimize memory usage
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage by cleaning up rarely used templates"""
        start_time = datetime.datetime.now()
        
        with self.library_lock:
            # Count templates before optimization
            templates_before = len(self.chunk_templates)
            
            # Find rarely used templates
            rarely_used = []
            for template_id, template in self.chunk_templates.items():
                # Check usage frequency
                if template.usage_frequency < 2:
                    # Check last used timestamp
                    if not template.last_used:
                        rarely_used.append(template_id)
                    else:
                        # Check if not used in last 30 days
                        last_used = datetime.datetime.fromisoformat(template.last_used)
                        days_since_use = (datetime.datetime.now() - last_used).days
                        
                        if days_since_use > 30:
                            rarely_used.append(template_id)
            
            # Don't remove all rarely used templates, keep some headroom
            templates_to_remove = rarely_used
            if len(templates_to_remove) > templates_before * 0.3:  # Don't remove more than 30%
                templates_to_remove = templates_to_remove[:int(templates_before * 0.3)]
            
            # Remove templates
            removed_count = 0
            for template_id in templates_to_remove:
                if template_id in self.chunk_templates:
                    template = self.chunk_templates[template_id]
                    
                    # Remove from domain index
                    for domain in template.domains:
                        if domain in self.domain_chunks and template_id in self.domain_chunks[domain]:
                            self.domain_chunks[domain].remove(template_id)
                    
                    # Remove template
                    del self.chunk_templates[template_id]
                    removed_count += 1
            
            # Clear caches
            self.template_cache = {}
            self.similarity_cache = {}
            
            # Calculate time taken
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            return {
                "templates_before": templates_before,
                "templates_removed": removed_count,
                "templates_after": len(self.chunk_templates),
                "execution_time": execution_time
            }
    
    # New method for getting library statistics
    def get_library_statistics(self) -> Dict[str, Any]:
        """Get statistics about the chunk library"""
        stats = {
            "templates_count": len(self.chunk_templates),
            "actions_count": len(self.action_templates),
            "domains_count": len(self.domain_chunks),
            "control_mappings_count": len(self.control_mappings),
            "transfer_records_count": len(self.transfer_records),
            "cache_sizes": {
                "template_cache": len(self.template_cache),
                "similarity_cache": len(self.similarity_cache)
            },
            "domain_usage": self.domain_usage_stats,
            "top_domains": [],
            "transfer_success_rate": 0.0
        }
        
        # Calculate top domains
        domain_templates = [(domain, len(templates)) for domain, templates in self.domain_chunks.items()]
        domain_templates.sort(key=lambda x: x[1], reverse=True)
        stats["top_domains"] = domain_templates[:5]  # Top 5 domains
        
        # Calculate transfer success rate
        if self.transfer_records:
            success_rates = [record.success_level for record in self.transfer_records]
            stats["transfer_success_rate"] = sum(success_rates) / len(success_rates)
        
        return stats
