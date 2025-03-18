# lore/lore_manager.py

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import asyncpg
import os
from datetime import datetime
import time

from db.connection import get_db_connection
from embedding.vector_store import generate_embedding, vector_similarity 
from utils.caching import LoreCache

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler

# Import the new cache manager
from .lore_cache_manager import LoreCacheManager

logger = logging.getLogger(__name__)

from .unified_validation import ValidationManager, LoreError, ErrorType
from .error_handler import ErrorHandler
from .monitoring import metrics
from .lore_agents import get_agents
from .base_manager import BaseManager
from .resource_manager import resource_manager

class LoreManager(BaseManager):
    """
    Manages all database operations for lore-related data with Nyx governance oversight.
    
    This class handles all database interactions for lore data, including:
    - World lore storage and retrieval
    - NPC knowledge management
    - Quest and narrative data
    - Location and environment descriptions
    
    Example:
        ```python
        lore_manager = LoreManager(user_id=123, conversation_id=456)
        await lore_manager.initialize()
        world_lore = await lore_manager.get_world_lore()
        ```
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        self.lore_data = {}
        self.resource_manager = resource_manager
        self.governor = None
        self.initialized = False
        self.directive_handler = None
        self.db_path = "lore_data"
        self._ensure_db_directory()
        self._component_counts = {}
        self.prohibited_actions = []
        self.state = {
            'initialized': False,
            'governance_initialized': False,
            'last_governance_check': None,
            'last_prohibition_update': None,
            'last_purge': None,
            'error_states': {},
            'recovery_strategies': {}
        }
        self.validation_manager = ValidationManager()
        self.error_handler = ErrorHandler()
        self._cleanup_task = None
        self.agents = None
        
        # Initialize the new cache manager
        self.cache_manager = LoreCacheManager(
            user_id=user_id,
            conversation_id=conversation_id,
            max_size_mb=500  # 500MB cache size
        )
        
    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
    
    def _get_component_path(self, component_id: str) -> str:
        """Get the file path for a component"""
        return os.path.join(self.db_path, f"{component_id}.json")
    
    async def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a lore component by ID"""
        try:
            # First try to get from cache
            cached_value = await self.cache_manager.get_lore('component', component_id)
            if cached_value is not None:
                return cached_value
            
            # If not in cache, try to get from file
            file_path = self._get_component_path(component_id)
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                component = json.load(f)
                
                # Cache the component for future use
                await self.cache_manager.set_lore('component', component_id, component)
                
                return component
        except Exception as e:
            logger.error(f"Error reading component {component_id}: {str(e)}")
            return None
    
    async def save_component(self, component_id: str, component: Dict[str, Any]) -> bool:
        """Save a lore component"""
        try:
            file_path = self._get_component_path(component_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(component, f, indent=2, ensure_ascii=False)
            
            # Update component counts
            component_type = component.get("type", "unknown")
            self._component_counts[component_type] = self._component_counts.get(component_type, 0) + 1
            
            # Update cache
            await self.cache_manager.set_lore('component', component_id, component)
            
            return True
        except Exception as e:
            logger.error(f"Error saving component {component_id}: {str(e)}")
            return False
    
    async def update_component(self, component_id: str, updates: Dict[str, Any]) -> bool:
        """Update a lore component"""
        try:
            component = await self.get_component(component_id)
            if not component:
                return False
            
            # Update the component
            component.update(updates)
            component["updated_at"] = datetime.utcnow().isoformat()
            
            # Save and update cache
            success = await self.save_component(component_id, component)
            if success:
                # Invalidate related components in cache
                await self.cache_manager.invalidate_lore('component', component_id)
            
            return success
        except Exception as e:
            logger.error(f"Error updating component {component_id}: {str(e)}")
            return False
    
    async def delete_component(self, component_id: str) -> bool:
        """Delete a lore component"""
        try:
            file_path = self._get_component_path(component_id)
            if not os.path.exists(file_path):
                return False
            
            # Get component type for updating counts
            component = await self.get_component(component_id)
            if component:
                component_type = component.get("type", "unknown")
                self._component_counts[component_type] = max(0, self._component_counts.get(component_type, 0) - 1)
            
            os.remove(file_path)
            
            # Invalidate cache
            await self.cache_manager.invalidate_lore('component', component_id)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting component {component_id}: {str(e)}")
            return False
    
    async def search_components(self, query: str, component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for lore components"""
        try:
            # Try to get from cache first
            cache_key = f"search:{query}:{component_type or 'all'}"
            cached_results = await self.cache_manager.get_lore('search', cache_key)
            if cached_results is not None:
                return cached_results
            
            results = []
            query = query.lower()
            
            for filename in os.listdir(self.db_path):
                if not filename.endswith('.json'):
                    continue
                
                component_id = filename[:-5]  # Remove .json extension
                component = await self.get_component(component_id)
                
                if not component:
                    continue
                
                # Filter by component type if specified
                if component_type and component.get("type") != component_type:
                    continue
                
                # Search in component fields
                if self._component_matches_query(component, query):
                    results.append(component)
            
            # Cache the search results
            await self.cache_manager.set_lore('search', cache_key, results)
            
            return results
        except Exception as e:
            logger.error(f"Error searching components: {str(e)}")
            return []
    
    def _component_matches_query(self, component: Dict[str, Any], query: str) -> bool:
        """Check if a component matches the search query"""
        searchable_fields = ["name", "description", "type"]
        
        for field in searchable_fields:
            if field in component and isinstance(component[field], str):
                if query in component[field].lower():
                    return True
        
        return False
    
    def get_component_counts(self) -> Dict[str, int]:
        """Get counts of components by type"""
        return self._component_counts.copy()
    
    async def get_all_components(self, component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all components, optionally filtered by type"""
        try:
            # Try to get from cache first
            cache_key = f"all:{component_type or 'all'}"
            cached_results = await self.cache_manager.get_lore('all', cache_key)
            if cached_results is not None:
                return cached_results
            
            components = []
            for filename in os.listdir(self.db_path):
                if not filename.endswith('.json'):
                    continue
                
                component_id = filename[:-5]  # Remove .json extension
                component = await self.get_component(component_id)
                
                if not component:
                    continue
                
                if component_type and component.get("type") != component_type:
                    continue
                
                components.append(component)
            
            # Cache the results
            await self.cache_manager.set_lore('all', cache_key, components)
            
            return components
        except Exception as e:
            logger.error(f"Error getting all components: {str(e)}")
            return []
    
    async def initialize(self) -> bool:
        """Initialize the LoreManager with governance and database connections."""
        try:
            # Initialize governance
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
            
            # Initialize directive handler
            self.directive_handler = DirectiveHandler(self.governor)
            
            # Check for restrictions
            restrictions = await self.governor.get_restrictions(
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="lore_manager"
            )
            
            if restrictions:
                self.prohibited_actions = restrictions.get("prohibited_actions", [])
            
            # Initialize database connection
            self.db = await get_db_connection()
            
            # Initialize component counts
            await self._update_component_counts()
            
            # Initialize agents
            self.agents = await get_agents(self)
            
            # Initialize cache manager
            await self.cache_manager.start()
            
            # Set state
            self.state['initialized'] = True
            self.state['governance_initialized'] = True
            self.state['last_governance_check'] = datetime.utcnow().isoformat()
            
            return True
        except Exception as e:
            logger.error(f"Error initializing LoreManager: {e}")
            return False
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Stop cache manager
            await self.cache_manager.stop()
            
            # Close database connection
            if hasattr(self, 'db'):
                await self.db.close()
            
            # Cancel cleanup task if it exists
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def _update_component_counts(self):
        """Update component counts from the database."""
        try:
            components = await self.get_all_components()
            self._component_counts = {}
            
            for component in components:
                component_type = component.get("type", "unknown")
                self._component_counts[component_type] = self._component_counts.get(component_type, 0) + 1
        except Exception as e:
            logger.error(f"Error updating component counts: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_cache_stats()

    async def _handle_action_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Handle action directives from the governance system."""
        try:
            directive_type = directive.get("type")
            action = directive.get("action")
            
            if directive_type == DirectiveType.ACTION:
                if action == "purge_lore":
                    return await self._purge_lore_category(directive.get("category"))
                elif action == "modify_lore":
                    return await self._modify_lore(directive.get("lore_id"), directive.get("modifications"))
                elif action == "validate_lore":
                    return await self._validate_lore(directive.get("lore_id"))
                elif action == "archive_lore":
                    return await self._archive_lore(directive.get("lore_id"))
            
            return {
                "status": "unknown_directive",
                "directive_type": directive_type,
                "action": action
            }
            
        except Exception as e:
            logger.error(f"Error handling action directive: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _handle_prohibition_directive(self, directive: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prohibition directives from the governance system."""
        try:
            prohibited_actions = directive.get("prohibited_actions", [])
            action_modifications = directive.get("action_modifications", {})
            
            # Update prohibited actions
            self.prohibited_actions = prohibited_actions
            
            # Update action modifications
            self.action_modifications = action_modifications
            
            # Update state
            self.state['last_prohibition_update'] = datetime.now()
            
            return {
                "status": "success",
                "prohibited_actions_updated": len(prohibited_actions),
                "modifications_updated": len(action_modifications)
            }
            
        except Exception as e:
            logger.error(f"Error handling prohibition directive: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _purge_lore_category(self, category: str) -> Dict[str, Any]:
        """Safely purge lore from a specific category."""
        try:
            # Check if category exists
            if category not in self._component_counts:
                return {
                    "status": "error",
                    "message": f"Category {category} not found"
                }
            
            # Get all components of this category
            query = """
                SELECT id, component_type
                FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = $3
            """
            components = await self.db.fetch(query, self.user_id, self.conversation_id, category)
            
            # Archive components before deletion
            for component in components:
                await self._archive_lore(component['id'])
            
            # Delete components
            delete_query = """
                DELETE FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = $3
            """
            await self.db.execute(delete_query, self.user_id, self.conversation_id, category)
            
            # Update component counts
            await self._update_component_counts()
            
            # Update state
            self.state['last_purge'] = datetime.now()
            
            return {
                "status": "success",
                "category": category,
                "components_purged": len(components)
            }
            
        except Exception as e:
            logger.error(f"Error purging lore category {category}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _modify_lore(
        self,
        lore_id: int,
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Modify existing lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'lore_id': lore_id,
                'modifications': modifications
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'lore_modification',
                task_context
            )
            
            # Get current lore
            query = """
                SELECT * FROM LoreComponents
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """
            lore = await self.db.fetchrow(query, lore_id, self.user_id, self.conversation_id)
            
            if not lore:
                return {
                    "status": "error",
                    "message": f"Lore with ID {lore_id} not found"
                }
            
            # Create task dependencies
            task_deps = {
                'validate_modifications': [],
                'check_conflicts': ['validate_modifications'],
                'apply_modifications': ['check_conflicts'],
                'validate_consistency': ['apply_modifications'],
                'update_metadata': ['validate_consistency'],
                'integrate': [
                    'validate_modifications',
                    'check_conflicts',
                    'apply_modifications',
                    'validate_consistency',
                    'update_metadata'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            modified_lore = await self._process_modification_results(results, lore)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_modified_lore(
                modified_lore,
                lore,
                modifications
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Update lore in database
            update_query = """
                UPDATE LoreComponents
                SET content = $1,
                    metadata = $2,
                    last_modified = NOW()
                WHERE id = $3
            """
            await self.db.execute(
                update_query,
                json.dumps(enhanced_lore['content']),
                json.dumps(enhanced_lore['metadata']),
                lore_id
            )
            
            # Cache the result
            cache_key = f"lore_{lore_id}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return {
                "status": "success",
                "lore_id": lore_id,
                "modified_fields": list(modifications.keys())
            }
            
        except Exception as e:
            logger.error(f"Error modifying lore {lore_id}: {e}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_modification_error(e, lore_id, modifications)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from modification error: {str(recovery_error)}")
            
            return {
                "status": "error",
                "error": str(e)
            }

    async def _integrate_lore(
        self,
        lore_parts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate different parts of lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'lore_parts': lore_parts
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'lore_integration',
                task_context
            )
            
            # Create task dependencies
            task_deps = {
                'analyze_components': [],
                'validate_consistency': ['analyze_components'],
                'resolve_conflicts': ['validate_consistency'],
                'merge_components': ['resolve_conflicts'],
                'validate_integration': ['merge_components'],
                'update_metadata': ['validate_integration'],
                'integrate': [
                    'analyze_components',
                    'validate_consistency',
                    'resolve_conflicts',
                    'merge_components',
                    'validate_integration',
                    'update_metadata'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            integrated_lore = await self._process_integration_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_integrated_lore(
                integrated_lore,
                lore_parts
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Store integrated lore
            await self.save_component(
                'integrated_lore',
                enhanced_lore
            )
            
            # Cache the result
            cache_key = f"integrated_lore_{hash(json.dumps(lore_parts))}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Error integrating lore: {e}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_integration_error(e, lore_parts)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from integration error: {str(recovery_error)}")
            
            return {}

    async def _validate_lore(self, lore_id: int) -> Dict[str, Any]:
        """Validate lore against governance rules and constraints."""
        try:
            # Get lore
            query = """
                SELECT * FROM LoreComponents
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """
            lore = await self.db.fetchrow(query, lore_id, self.user_id, self.conversation_id)
            
            if not lore:
                return {
                    "status": "error",
                    "message": f"Lore with ID {lore_id} not found"
                }
            
            # Check against prohibited content
            validation_results = {
                "prohibited_content": [],
                "constraint_violations": [],
                "metadata_issues": []
            }
            
            # Check content against prohibited terms
            content = lore['content']
            for term in self.prohibited_actions:
                if term.lower() in content.lower():
                    validation_results["prohibited_content"].append(term)
            
            # Check metadata constraints
            metadata = lore['metadata']
            if not metadata.get('version'):
                validation_results["metadata_issues"].append("Missing version")
            
            # Check content constraints
            if len(content) < 10:
                validation_results["constraint_violations"].append("Content too short")
            
            # Update validation status
            update_query = """
                UPDATE LoreComponents
                SET validation_status = $1,
                    validation_errors = $2,
                    last_validated = NOW()
                WHERE id = $3
            """
            await self.db.execute(
                update_query,
                "valid" if not any(validation_results.values()) else "invalid",
                json.dumps(validation_results),
                lore_id
            )
            
            return {
                "status": "success",
                "lore_id": lore_id,
                "validation_results": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error validating lore {lore_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _archive_lore(self, lore_id: int) -> Dict[str, Any]:
        """Archive lore for historical record keeping."""
        try:
            # Get lore
            query = """
                SELECT * FROM LoreComponents
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """
            lore = await self.db.fetchrow(query, lore_id, self.user_id, self.conversation_id)
            
            if not lore:
                return {
                    "status": "error",
                    "message": f"Lore with ID {lore_id} not found"
                }
            
            # Create archive record
            archive_query = """
                INSERT INTO LoreArchives (
                    user_id, conversation_id, original_id,
                    content, metadata, archived_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
            """
            await self.db.execute(
                archive_query,
                self.user_id,
                self.conversation_id,
                lore_id,
                lore['content'],
                lore['metadata']
            )
            
            return {
                "status": "success",
                "lore_id": lore_id,
                "archived": True
            }
            
        except Exception as e:
            logger.error(f"Error archiving lore {lore_id}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def get_world_lore(self) -> List[Dict[str, Any]]:
        """Get all world lore components."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'world_lore'
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting world lore: {e}")
            return []

    async def get_location_lore(self, location_id: int) -> List[Dict[str, Any]]:
        """Get lore specific to a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'location_lore'
                AND metadata->>'location_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting location lore: {e}")
            return []

    async def get_faction_lore(self) -> List[Dict[str, Any]]:
        """Get all faction-related lore."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'faction_lore'
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting faction lore: {e}")
            return []

    async def get_npc_relationships(self, npc_id: int) -> List[Dict[str, Any]]:
        """Get relationships for a specific NPC."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'npc_relationship'
                AND metadata->>'npc_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, npc_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting NPC relationships: {e}")
            return []

    async def get_npc_beliefs(self, npc_id: int) -> List[Dict[str, Any]]:
        """Get beliefs for a specific NPC."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'npc_belief'
                AND metadata->>'npc_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, npc_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting NPC beliefs: {e}")
            return []

    async def update_npc_knowledge(self, npc_id: int, knowledge_updates: List[Dict[str, Any]]) -> bool:
        """Update NPC knowledge with new information."""
        try:
            for update in knowledge_updates:
                query = """
                    INSERT INTO LoreComponents (
                        user_id, conversation_id, component_type,
                        content, metadata, created_at
                    ) VALUES ($1, $2, 'npc_knowledge', $3, $4, NOW())
                """
                await self.db.execute(
                    query,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(update['content']),
                    json.dumps({
                        'npc_id': npc_id,
                        'type': update['type'],
                        'relevance_score': update.get('relevance_score', 0.5)
                    })
                )
            return True
        except Exception as e:
            logger.error(f"Error updating NPC knowledge: {e}")
            return False

    async def update_npc_beliefs(self, npc_id: int, belief_updates: List[Dict[str, Any]]) -> bool:
        """Update NPC beliefs with new information."""
        try:
            for update in belief_updates:
                query = """
                    INSERT INTO LoreComponents (
                        user_id, conversation_id, component_type,
                        content, metadata, created_at
                    ) VALUES ($1, $2, 'npc_belief', $3, $4, NOW())
                """
                await self.db.execute(
                    query,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(update['content']),
                    json.dumps({
                        'npc_id': npc_id,
                        'belief_id': update['belief_id'],
                        'strengthened_themes': update.get('strengthened_themes', []),
                        'new_evidence': update.get('new_evidence', '')
                    })
                )
            return True
        except Exception as e:
            logger.error(f"Error updating NPC beliefs: {e}")
            return False

    async def update_npc_relationships(self, npc_id: int, relationship_updates: List[Dict[str, Any]]) -> bool:
        """Update NPC relationships with new information."""
        try:
            for update in relationship_updates:
                query = """
                    INSERT INTO LoreComponents (
                        user_id, conversation_id, component_type,
                        content, metadata, created_at
                    ) VALUES ($1, $2, 'npc_relationship', $3, $4, NOW())
                """
                await self.db.execute(
                    query,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(update['content']),
                    json.dumps({
                        'npc_id': npc_id,
                        'related_npc_id': update['npc_id'],
                        'type': update['type'],
                        'strength': update.get('strength', 0.5),
                        'context': update.get('context', '')
                    })
                )
            return True
        except Exception as e:
            logger.error(f"Error updating NPC relationships: {e}")
            return False

    async def add_npc_memories(self, npc_id: int, memories: List[Dict[str, Any]]) -> bool:
        """Add new memories for an NPC."""
        try:
            for memory in memories:
                query = """
                    INSERT INTO LoreComponents (
                        user_id, conversation_id, component_type,
                        content, metadata, created_at
                    ) VALUES ($1, $2, 'npc_memory', $3, $4, NOW())
                """
                await self.db.execute(
                    query,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(memory['content']),
                    json.dumps({
                        'npc_id': npc_id,
                        'type': memory['type'],
                        'emotional_impact': memory.get('emotional_impact', 0),
                        'timestamp': memory.get('timestamp', datetime.now().isoformat())
                    })
                )
            return True
        except Exception as e:
            logger.error(f"Error adding NPC memories: {e}")
            return False

    async def get_environment_state(self, location_id: int) -> Dict[str, Any]:
        """Get current environment state for a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'environment_state'
                AND metadata->>'location_id' = $3
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, self.user_id, self.conversation_id, location_id)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Error getting environment state: {e}")
            return {}

    async def get_environmental_conditions(self, location_id: int) -> Dict[str, Any]:
        """Get environmental conditions for a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'environmental_conditions'
                AND metadata->>'location_id' = $3
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, self.user_id, self.conversation_id, location_id)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Error getting environmental conditions: {e}")
            return {}

    async def get_location_events(self, location_id: int) -> List[Dict[str, Any]]:
        """Get active events for a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'location_event'
                AND metadata->>'location_id' = $3
                AND metadata->>'status' = 'active'
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting location events: {e}")
            return []

    async def get_upcoming_events(self, location_id: int) -> List[Dict[str, Any]]:
        """Get upcoming events for a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'location_event'
                AND metadata->>'location_id' = $3
                AND metadata->>'status' = 'scheduled'
                AND metadata->>'start_time' > NOW()
                ORDER BY metadata->>'start_time'
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting upcoming events: {e}")
            return []

    async def get_location_resources(self, location_id: int) -> List[Dict[str, Any]]:
        """Get available resources for a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'location_resource'
                AND metadata->>'location_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting location resources: {e}")
            return []

    async def get_resource_scarcity(self, location_id: int) -> Dict[str, float]:
        """Get resource scarcity levels for a location."""
        try:
            query = """
                SELECT metadata->>'resource_type' as type,
                       metadata->>'scarcity_level' as level
                FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'resource_scarcity'
                AND metadata->>'location_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return {row['type']: float(row['level']) for row in results}
        except Exception as e:
            logger.error(f"Error getting resource scarcity: {e}")
            return {}

    async def get_adjacent_locations(self, location_id: int) -> List[Dict[str, Any]]:
        """Get adjacent locations for a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'location_connection'
                AND metadata->>'source_location_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting adjacent locations: {e}")
            return []

    async def get_travel_routes(self, location_id: int) -> List[Dict[str, Any]]:
        """Get travel routes from a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'travel_route'
                AND metadata->>'source_location_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting travel routes: {e}")
            return []

    async def get_location_factions(self, location_id: int) -> List[Dict[str, Any]]:
        """Get factions present in a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'faction_presence'
                AND metadata->>'location_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting location factions: {e}")
            return []

    async def _get_current_time(self) -> Dict[str, Any]:
        """Get current game time."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'game_time'
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, self.user_id, self.conversation_id)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Error getting current time: {e}")
            return {}

    async def _get_nearby_npcs(self, location_id: int) -> List[Dict[str, Any]]:
        """Get NPCs currently in a location."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'npc_location'
                AND metadata->>'location_id' = $3
                AND metadata->>'status' = 'present'
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, location_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting nearby NPCs: {e}")
            return []

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and metrics."""
        try:
            return {
                'metrics': metrics.get_metrics_snapshot(),
                'errors': self.validation_manager.get_error_stats(),
                'component_counts': self._component_counts,
                'cache_stats': self.get_cache_stats(),
                'state': self.state
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}

    async def _validate_component(self, component: Dict[str, Any]) -> bool:
        """Validate a lore component before saving."""
        try:
            # Validate against schema
            self.validation_manager.validate_entity(
                component.get('type', 'unknown'),
                component
            )
            
            # Check against prohibited content
            content = component.get('content', '')
            for term in self.prohibited_actions:
                if term.lower() in content.lower():
                    raise LoreError(
                        f"Prohibited content found: {term}",
                        ErrorType.VALIDATION
                    )
            
            return True
        except Exception as e:
            self.error_handler.handle_error(e)
            return False

    async def get_quest_lore(self, quest_id: int) -> List[Dict[str, Any]]:
        """Get lore specific to a quest."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'quest_lore'
                AND metadata->>'quest_id' = $3
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, quest_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting quest lore: {e}")
            return []

    async def get_narrative_data(self, narrative_id: int) -> Dict[str, Any]:
        """Get narrative data including progression and stages."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'narrative_data'
                AND metadata->>'narrative_id' = $3
            """
            result = await self.db.fetchrow(query, self.user_id, self.conversation_id, narrative_id)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Error getting narrative data: {e}")
            return {}

    async def update_narrative_progression(self, narrative_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update narrative progression data."""
        try:
            query = """
                UPDATE LoreComponents
                SET content = $1,
                    metadata = jsonb_set(
                        metadata,
                        '{stages}',
                        $2::jsonb
                    ),
                    last_modified = NOW()
                WHERE user_id = $3 
                AND conversation_id = $4
                AND component_type = 'narrative_data'
                AND metadata->>'narrative_id' = $5
            """
            await self.db.execute(
                query,
                json.dumps(data),
                json.dumps({stage: data}),
                self.user_id,
                self.conversation_id,
                narrative_id
            )
            return True
        except Exception as e:
            logger.error(f"Error updating narrative progression: {e}")
            return False

    async def get_social_links(self, entity_id: int, entity_type: str) -> List[Dict[str, Any]]:
        """Get social links for an entity."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'social_link'
                AND metadata->>'entity_id' = $3
                AND metadata->>'entity_type' = $4
            """
            results = await self.db.fetch(query, self.user_id, self.conversation_id, entity_id, entity_type)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting social links: {e}")
            return []

    async def update_social_links(self, entity_id: int, entity_type: str, links: List[Dict[str, Any]]) -> bool:
        """Update social links for an entity."""
        try:
            for link in links:
                query = """
                    INSERT INTO LoreComponents (
                        user_id, conversation_id, component_type,
                        content, metadata, created_at
                    ) VALUES ($1, $2, 'social_link', $3, $4, NOW())
                """
                await self.db.execute(
                    query,
                    self.user_id,
                    self.conversation_id,
                    json.dumps(link['content']),
                    json.dumps({
                        'entity_id': entity_id,
                        'entity_type': entity_type,
                        'link_type': link['type'],
                        'strength': link.get('strength', 0.5)
                    })
                )
            return True
        except Exception as e:
            logger.error(f"Error updating social links: {e}")
            return False

    async def get_quest_progression(self, quest_id: int) -> Dict[str, Any]:
        """Get quest progression data."""
        try:
            query = """
                SELECT * FROM LoreComponents
                WHERE user_id = $1 
                AND conversation_id = $2
                AND component_type = 'quest_progression'
                AND metadata->>'quest_id' = $3
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = await self.db.fetchrow(query, self.user_id, self.conversation_id, quest_id)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Error getting quest progression: {e}")
            return {}

    async def update_quest_progression(self, quest_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update quest progression data."""
        try:
            query = """
                INSERT INTO LoreComponents (
                    user_id, conversation_id, component_type,
                    content, metadata, created_at
                ) VALUES ($1, $2, 'quest_progression', $3, $4, NOW())
            """
            await self.db.execute(
                query,
                self.user_id,
                self.conversation_id,
                json.dumps(data),
                json.dumps({
                    'quest_id': quest_id,
                    'stage': stage,
                    'timestamp': datetime.now().isoformat()
                })
            )
            return True
        except Exception as e:
            logger.error(f"Error updating quest progression: {e}")
            return False

    async def get_quest_context(self, quest_id: int) -> Dict[str, Any]:
        """Get comprehensive quest context using the QuestAgent."""
        if not self.agents:
            return {}
        quest_agent, _, _, _ = self.agents
        return await quest_agent.get_quest_context(quest_id)

    async def update_quest_stage(self, quest_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update quest stage using the QuestAgent."""
        if not self.agents:
            return False
        quest_agent, _, _, _ = self.agents
        return await quest_agent.update_quest_stage(quest_id, stage, data)

    async def get_narrative_context(self, narrative_id: int) -> Dict[str, Any]:
        """Get comprehensive narrative context using the NarrativeAgent."""
        if not self.agents:
            return {}
        _, narrative_agent, _, _ = self.agents
        return await narrative_agent.get_narrative_context(narrative_id)

    async def update_narrative_stage(self, narrative_id: int, stage: str, data: Dict[str, Any]) -> bool:
        """Update narrative stage using the NarrativeAgent."""
        if not self.agents:
            return False
        _, narrative_agent, _, _ = self.agents
        return await narrative_agent.update_narrative_stage(narrative_id, stage, data)

    async def get_social_context(self, entity_id: int, entity_type: str) -> Dict[str, Any]:
        """Get comprehensive social context using the SocialAgent."""
        if not self.agents:
            return {}
        _, _, social_agent, _ = self.agents
        return await social_agent.get_social_context(entity_id, entity_type)

    async def update_relationships(self, entity_id: int, entity_type: str, links: List[Dict[str, Any]]) -> bool:
        """Update social relationships using the SocialAgent."""
        if not self.agents:
            return False
        _, _, social_agent, _ = self.agents
        return await social_agent.update_relationships(entity_id, entity_type, links)

    async def get_environment_context(self, location_id: int) -> Dict[str, Any]:
        """Get comprehensive environment context using the EnvironmentAgent."""
        if not self.agents:
            return {}
        _, _, _, environment_agent = self.agents
        return await environment_agent.get_environment_context(location_id)

    async def update_environment_state(self, location_id: int, updates: Dict[str, Any]) -> bool:
        """Update environment state using the EnvironmentAgent."""
        if not self.agents:
            return False
        _, _, _, environment_agent = self.agents
        return await environment_agent.update_environment_state(location_id, updates)

    async def get_conflict_lore(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Get lore specific to a conflict."""
        _, _, _, _, conflict_agent, _, _ = await get_agents(self)
        return await conflict_agent.get_conflict_lore(conflict_id)

    async def update_conflict_state(self, conflict_id: int, state: str, data: Dict[str, Any]) -> bool:
        """Update conflict state and related data."""
        _, _, _, _, conflict_agent, _, _ = await get_agents(self)
        return await conflict_agent.update_conflict_state(conflict_id, state, data)

    async def get_artifact_lore(self, artifact_id: int) -> Dict[str, Any]:
        """Get lore specific to an artifact."""
        _, _, _, _, _, artifact_agent, _ = await get_agents(self)
        return await artifact_agent.get_artifact_lore(artifact_id)

    async def update_artifact_state(self, artifact_id: int, state: str, data: Dict[str, Any]) -> bool:
        """Update artifact state and related data."""
        _, _, _, _, _, artifact_agent, _ = await get_agents(self)
        return await artifact_agent.update_artifact_state(artifact_id, state, data)

    async def update_game_time(self, time_data: Dict[str, Any]) -> bool:
        """Update game time and related events."""
        _, _, _, environment_agent, _, _, _ = await get_agents(self)
        return await environment_agent.update_game_time(time_data)

    async def create_event(self, event_data: Dict[str, Any]) -> bool:
        """Create a new event in the game world."""
        _, _, _, _, _, _, event_agent = await get_agents(self)
        return await event_agent.create_event(event_data)

    async def update_event_status(self, event_id: int, status: str, data: Dict[str, Any]) -> bool:
        """Update event status and related data."""
        _, _, _, _, _, _, event_agent = await get_agents(self)
        return await event_agent.update_event_status(event_id, status, data)

    async def update_resource_levels(self, location_id: int, resources: Dict[str, float]) -> bool:
        """Update resource levels for a location."""
        _, _, _, environment_agent, _, _, _ = await get_agents(self)
        return await environment_agent.update_resource_levels(location_id, resources)

    async def start(self):
        """Start the lore manager and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the lore manager and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_lore_data(
        self,
        lore_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get lore data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('lore', lore_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting lore data: {e}")
            return None
    
    async def set_lore_data(
        self,
        lore_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set lore data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('lore', lore_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting lore data: {e}")
            return False
    
    async def invalidate_lore_data(
        self,
        lore_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate lore data cache."""
        try:
            await self.invalidate_cached_data('lore', lore_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating lore data: {e}")
    
    async def get_lore_metadata(
        self,
        lore_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get lore metadata from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('lore_metadata', lore_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting lore metadata: {e}")
            return None
    
    async def set_lore_metadata(
        self,
        lore_id: str,
        metadata: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set lore metadata in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('lore_metadata', lore_id, metadata, tags)
        except Exception as e:
            logger.error(f"Error setting lore metadata: {e}")
            return False
    
    async def invalidate_lore_metadata(
        self,
        lore_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate lore metadata cache."""
        try:
            await self.invalidate_cached_data('lore_metadata', lore_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating lore metadata: {e}")
    
    async def get_lore_relationships(
        self,
        lore_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get lore relationships from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('lore_relationships', lore_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting lore relationships: {e}")
            return None
    
    async def set_lore_relationships(
        self,
        lore_id: str,
        relationships: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set lore relationships in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('lore_relationships', lore_id, relationships, tags)
        except Exception as e:
            logger.error(f"Error setting lore relationships: {e}")
            return False
    
    async def invalidate_lore_relationships(
        self,
        lore_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate lore relationships cache."""
        try:
            await self.invalidate_cached_data('lore_relationships', lore_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating lore relationships: {e}")
    
    async def get_lore_history(
        self,
        lore_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get lore history from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('lore_history', lore_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting lore history: {e}")
            return None
    
    async def set_lore_history(
        self,
        lore_id: str,
        history: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set lore history in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('lore_history', lore_id, history, tags)
        except Exception as e:
            logger.error(f"Error setting lore history: {e}")
            return False
    
    async def invalidate_lore_history(
        self,
        lore_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate lore history cache."""
        try:
            await self.invalidate_cached_data('lore_history', lore_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating lore history: {e}")
    
    async def get_lore_validation(
        self,
        lore_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get lore validation from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('lore_validation', lore_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting lore validation: {e}")
            return None
    
    async def set_lore_validation(
        self,
        lore_id: str,
        validation: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set lore validation in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('lore_validation', lore_id, validation, tags)
        except Exception as e:
            logger.error(f"Error setting lore validation: {e}")
            return False
    
    async def invalidate_lore_validation(
        self,
        lore_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate lore validation cache."""
        try:
            await self.invalidate_cached_data('lore_validation', lore_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating lore validation: {e}")
    
    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        try:
            return await self.resource_manager.get_resource_stats()
        except Exception as e:
            logger.error(f"Error getting resource stats: {e}")
            return {}
    
    async def optimize_resources(self):
        """Optimize resource usage."""
        try:
            await self.resource_manager._optimize_resource_usage('memory')
        except Exception as e:
            logger.error(f"Error optimizing resources: {e}")
    
    async def cleanup_resources(self):
        """Clean up unused resources."""
        try:
            await self.resource_manager._cleanup_all_resources()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")

# Create a singleton instance for easy access
lore_manager = LoreManager()
