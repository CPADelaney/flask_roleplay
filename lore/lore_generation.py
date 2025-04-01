# lore/lore_generation.py

"""
Lore Generation System

This module provides comprehensive lore generation capabilities,
including dynamic generation, evolution, and connection management.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from .data_access_layer import LoreDataAccess, LocationDataAccess, NPCDataAccess
from .integration_layer import LoreIntegrator

# Import agent framework
from nyx.agent import Agent, AgentContext, RunContextWrapper, AgentRegistry
from nyx.agent_capabilities import AgentCapability

import json
import asyncio
import psutil
import time
from nyx.resource_pool import ResourcePool
from nyx.performance_monitor import PerformanceMonitor
from nyx.cache_manager import CacheManager
from nyx.version_control import VersionControl
from nyx.parallel_processor import ParallelProcessor
from nyx.distributed_executor import DistributedExecutor
from nyx.task_priority import TaskPriority
from nyx.cache_predictor import CachePredictor
from nyx.version_branch import VersionBranch
from .lore_cache_manager import LoreCacheManager
from .base_manager import BaseManager
from .resource_manager import resource_manager

logger = logging.getLogger(__name__)

class LoreGenerationAgent(BaseManager):
    """
    Specialized agent for handling lore generation tasks.
    Extends the base Agent class with generation-specific capabilities.
    """
    
    def __init__(
        self,
        user_id: int,
        conversation_id: int,
        max_size_mb: float = 100,
        redis_url: Optional[str] = None
    ):
        super().__init__(user_id, conversation_id, max_size_mb, redis_url)
        # Define core capabilities
        self.capabilities = {
            AgentCapability.LORE_MANAGEMENT,
            AgentCapability.CONTENT_GENERATION,
            AgentCapability.CONTEXT_ENHANCEMENT,
            AgentCapability.VALIDATION,
            AgentCapability.ERROR_HANDLING,
            AgentCapability.STATE_MANAGEMENT,
            AgentCapability.RESOURCE_MANAGEMENT,
            AgentCapability.PERFORMANCE_MONITORING,
            AgentCapability.PARALLEL_PROCESSING,
            AgentCapability.VERSION_CONTROL,
            AgentCapability.DISTRIBUTED_EXECUTION,
            AgentCapability.CACHE_MANAGEMENT
        }
        
        # Initialize data access layers
        self.lore_dal = LoreDataAccess(user_id, conversation_id)
        self.location_dal = LocationDataAccess(user_id, conversation_id)
        self.npc_dal = NPCDataAccess(user_id, conversation_id)
        self.lore_integrator = LoreIntegrator(user_id, conversation_id)
        
        # Initialize state management
        self.state = {
            'initialized': False,
            'initialization_time': None,
            'last_generation': None,
            'generation_count': 0,
            'validation_stats': {
                'successful': 0,
                'failed': 0,
                'fixed': 0
            },
            'performance_metrics': {
                'generation_times': [],
                'validation_times': [],
                'integration_times': [],
                'error_rates': {
                    'total': 0,
                    'recovered': 0,
                    'unrecovered': 0
                },
                'cache_hits': 0,
                'cache_misses': 0,
                'parallel_tasks': 0,
                'distributed_tasks': 0,
                'cache_predictions': {
                    'correct': 0,
                    'incorrect': 0
                }
            },
            'resource_usage': {
                'memory': 0,
                'cpu': 0,
                'network': 0
            },
            'active_tasks': [],
            'task_history': [],
            'error_states': {},
            'recovery_strategies': {},
            'version_history': [],
            'cache_stats': {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'predictions': 0
            },
            'task_priorities': {},
            'task_dependencies': {},
            'branch_states': {}
        }
        
        # Initialize resource pools
        self.resource_pools = {
            'generation': ResourcePool(max_workers=5, timeout=300),
            'validation': ResourcePool(max_workers=3, timeout=180),
            'integration': ResourcePool(max_workers=4, timeout=240),
            'cache': ResourcePool(max_workers=2, timeout=120)
        }
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Initialize cache management
        self.cache_manager = CacheManager(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Initialize version control
        self.version_control = VersionControl(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Initialize parallel processor
        self.parallel_processor = ParallelProcessor(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Initialize distributed executor
        self.distributed_executor = DistributedExecutor(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Initialize cache predictor
        self.cache_predictor = CachePredictor(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Initialize version branch manager
        self.version_branch = VersionBranch(
            user_id=user_id,
            conversation_id=conversation_id,
            system_name="lore_generation_agent"
        )
        
        # Start background monitoring
        self._start_background_monitoring()
        
        self.generation_data = {}
        self.resource_manager = resource_manager
        
    def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        asyncio.create_task(self._monitor_resource_usage())
        asyncio.create_task(self._monitor_performance())
        asyncio.create_task(self._cleanup_resources())
    
    async def _monitor_resource_usage(self):
        """Monitor system resource usage."""
        while True:
            try:
                process = psutil.Process()
                self.state['resource_usage'].update({
                    'memory': process.memory_info().rss,
                    'cpu': process.cpu_percent(),
                    'network': process.io_counters().read_bytes + process.io_counters().write_bytes
                })
                
                # Check for resource limits
                if self.state['resource_usage']['memory'] > 512 * 1024 * 1024:  # 512MB
                    await self._handle_resource_limit('memory')
                if self.state['resource_usage']['cpu'] > 70:  # 70% CPU
                    await self._handle_resource_limit('cpu')
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Error monitoring resource usage: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_performance(self):
        """Monitor system performance metrics."""
        while True:
            try:
                # Calculate performance metrics
                total_generations = self.state['generation_count']
                if total_generations > 0:
                    self.state['performance_metrics']['error_rates']['total'] = (
                        self.state['validation_stats']['failed'] / total_generations
                    )
                    self.state['performance_metrics']['error_rates']['recovered'] = (
                        self.state['validation_stats']['fixed'] / total_generations
                    )
                    self.state['performance_metrics']['error_rates']['unrecovered'] = (
                        (self.state['validation_stats']['failed'] - self.state['validation_stats']['fixed']) / 
                        total_generations
                    )
                
                # Check for performance issues
                if self.state['performance_metrics']['error_rates']['total'] > 0.1:  # 10% error rate
                    await self._handle_performance_issue('error_rate')
                if len(self.state['performance_metrics']['generation_times']) > 0:
                    avg_time = sum(self.state['performance_metrics']['generation_times'][-10:]) / 10
                    if avg_time > 10:  # 10 seconds average generation time
                        await self._handle_performance_issue('generation_time')
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logging.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(300)
    
    async def _cleanup_resources(self):
        """Clean up system resources."""
        while True:
            try:
                # Clear old performance metrics
                if len(self.state['performance_metrics']['generation_times']) > 1000:
                    self.state['performance_metrics']['generation_times'] = (
                        self.state['performance_metrics']['generation_times'][-1000:]
                    )
                if len(self.state['performance_metrics']['validation_times']) > 1000:
                    self.state['performance_metrics']['validation_times'] = (
                        self.state['performance_metrics']['validation_times'][-1000:]
                    )
                if len(self.state['performance_metrics']['integration_times']) > 1000:
                    self.state['performance_metrics']['integration_times'] = (
                        self.state['performance_metrics']['integration_times'][-1000:]
                    )
                
                # Clean up resource pools
                for pool_name, pool in self.resource_pools.items():
                    await pool.cleanup()
                
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logging.error(f"Error cleaning up resources: {e}")
                await asyncio.sleep(3600)
    
    async def _handle_resource_limit(self, resource_type: str):
        """Handle resource limit reached."""
        try:
            # Log the resource limit
            logging.warning(f"Resource limit reached for {resource_type}")
            
            # Implement recovery strategies
            if resource_type == 'memory':
                # Clear caches
                await self.cache_manager.clear_old_entries()
                # Reduce parallel tasks
                await self.parallel_processor.reduce_workers()
                # Clear unused resources
                await self._cleanup_unused_resources()
            elif resource_type == 'cpu':
                # Reduce parallel tasks
                await self.parallel_processor.reduce_workers()
                # Optimize processing
                await self._optimize_processing()
                # Adjust task priorities
                await self._adjust_task_priorities()
            elif resource_type == 'network':
                # Reduce distributed tasks
                await self.distributed_executor.reduce_tasks()
                # Optimize network usage
                await self._optimize_network_usage()
            
            # Report to governance
            await self.governor.report_resource_limit(
                resource_type=resource_type,
                current_usage=self.state['resource_usage'][resource_type]
            )
            
            # Update state
            self.state['resource_usage'][resource_type] = 0
            
        except Exception as e:
            logging.error(f"Error handling resource limit: {e}")
    
    async def _handle_performance_issue(self, issue_type: str):
        """Handle performance issues."""
        try:
            # Log the performance issue
            logging.warning(f"Performance issue detected: {issue_type}")
            
            # Implement recovery strategies
            if issue_type == 'error_rate':
                # Increase validation
                await self._increase_validation()
                # Reduce parallel tasks
                await self.parallel_processor.reduce_workers()
                # Add additional error handling
                await self._enhance_error_handling()
            elif issue_type == 'generation_time':
                # Optimize processing
                await self._optimize_processing()
                # Increase caching
                await self.cache_manager.increase_cache_size()
                # Optimize task scheduling
                await self._optimize_task_scheduling()
            elif issue_type == 'validation_time':
                # Optimize validation process
                await self._optimize_validation()
                # Adjust validation thresholds
                await self._adjust_validation_thresholds()
            
            # Report to governance
            await self.governor.report_performance_issue(
                issue_type=issue_type,
                metrics=self.state['performance_metrics']
            )
            
            # Update performance metrics
            self.state['performance_metrics']['last_optimization'] = time.time()
            
        except Exception as e:
            logging.error(f"Error handling performance issue: {e}")
    
    async def _optimize_processing(self):
        """Optimize system processing."""
        try:
            # Analyze current performance
            performance_data = await self.performance_monitor.get_performance_data()
            
            # Optimize based on performance data
            if performance_data['cache_hit_rate'] < 0.7:  # 70% cache hit rate
                await self.cache_manager.optimize_cache_strategy()
                await self.cache_predictor.update_prediction_model()
            
            if performance_data['parallel_efficiency'] < 0.8:  # 80% parallel efficiency
                await self.parallel_processor.optimize_worker_count()
                await self._optimize_task_distribution()
            
            if performance_data['distributed_efficiency'] < 0.8:  # 80% distributed efficiency
                await self.distributed_executor.optimize_task_distribution()
                await self._optimize_network_usage()
            
            # Update resource pools
            for pool_name, pool in self.resource_pools.items():
                await pool.optimize()
            
            # Optimize memory usage
            await self._optimize_memory_usage()
            
            # Update state
            self.state['performance_metrics']['last_optimization'] = time.time()
            
        except Exception as e:
            logging.error(f"Error optimizing processing: {e}")
    
    async def _increase_validation(self):
        """Increase validation rigor."""
        try:
            # Increase validation thresholds
            self.validation_thresholds = {
                'min_quality_score': 0.8,  # Increased from 0.7
                'max_inconsistency_score': 0.2,  # Decreased from 0.3
                'min_completeness_score': 0.9,  # Increased from 0.8
                'max_duplication_score': 0.1,  # Decreased from 0.2
                'min_coherence_score': 0.85,  # New threshold
                'max_contradiction_score': 0.15  # New threshold
            }
            
            # Add additional validation checks
            self.additional_validation_checks = [
                self._validate_lore_consistency,
                self._validate_lore_completeness,
                self._validate_lore_quality,
                self._validate_lore_coherence,
                self._validate_lore_contradictions
            ]
            
            # Increase validation frequency
            self.validation_frequency = {
                'pre_generation': True,
                'post_generation': True,
                'pre_integration': True,
                'post_integration': True,
                'periodic': True
            }
            
            # Report to governance
            await self.governor.report_validation_increase(
                new_thresholds=self.validation_thresholds
            )
            
            # Update state
            self.state['validation_stats']['increased_validation'] = True
            self.state['validation_stats']['last_increase'] = time.time()
            
        except Exception as e:
            logging.error(f"Error increasing validation: {e}")
    
    async def _warm_up_cache(self):
        """Warm up cache with frequently accessed lore."""
        try:
            # Get frequently accessed lore patterns
            patterns = await self.cache_predictor.get_frequent_patterns()
            
            # Pre-generate and cache lore for these patterns
            for pattern in patterns:
                try:
                    lore = await self._generate_for_pattern(pattern)
                    if lore:
                        await self.cache_manager.set(
                            f"pattern_{hash(json.dumps(pattern))}",
                            lore,
                            ttl=3600  # 1 hour TTL
                        )
                        self.state['cache_stats']['warmed_up'] += 1
                except Exception as e:
                    logger.error(f"Failed to warm up cache for pattern: {e}")
            
            # Update cache statistics
            self.state['cache_stats']['last_warmup'] = time.time()
            
        except Exception as e:
            logger.error(f"Failed to warm up cache: {e}")
    
    async def _generate_for_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate lore for a specific pattern."""
        try:
            # Create run context
            run_ctx = RunContextWrapper(context={})
            
            # Generate lore using pattern
            lore = await self._generate_lore_from_pattern(pattern, run_ctx)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_generated_lore(lore, pattern, run_ctx)
            
            # Validate against existing lore
            if enhanced_lore:
                validation_result = await self._validate_against_existing_lore(enhanced_lore)
                if not validation_result['is_valid']:
                    enhanced_lore = await self._fix_lore_inconsistencies(enhanced_lore, validation_result['issues'])
            
            # Update generation statistics
            self.state['generation_stats']['patterns_generated'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate lore for pattern: {e}")
            return None
    
    async def generate_dynamic_lore(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any],
        generation_params: Dict[str, Any] = None,
        context: Optional[AgentContext] = None
    ) -> Dict[str, Any]:
        """Generate dynamic lore with distributed execution and predictive caching."""
        try:
            # Verify agent is initialized
            if not self.state.get('initialized'):
                raise RuntimeError("Agent not initialized")
                
            # Create run context
            run_ctx = RunContextWrapper(context=context.context if context else {})
            
            # Check cache with prediction
            cache_key = f"lore_{hash(json.dumps(base_context))}_{hash(json.dumps(world_state))}"
            
            # Predict if content will be needed
            prediction = await self.cache_predictor.predict(cache_key)
            if prediction['will_be_needed']:
                self.state['cache_stats']['predictions'] += 1
                if prediction['should_prefetch']:
                    await self._prefetch_lore(cache_key, base_context, world_state)
            
            # Check cache
            cached_lore = await self.cache_manager.get(cache_key)
            if cached_lore:
                self.state['cache_stats']['hits'] += 1
                self.state['performance_metrics']['cache_hits'] += 1
                if prediction['will_be_needed']:
                    self.state['performance_metrics']['cache_predictions']['correct'] += 1
                return cached_lore
            
            self.state['cache_stats']['misses'] += 1
            self.state['performance_metrics']['cache_misses'] += 1
            if prediction['will_be_needed']:
                self.state['performance_metrics']['cache_predictions']['incorrect'] += 1
            
            # Update state
            self.state['generation_count'] += 1
            self.state['last_generation'] = datetime.now().isoformat()
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'lore_generation',
                base_context,
                world_state
            )
            
            # Create task dependencies
            task_deps = self._create_task_dependencies(base_context, world_state)
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            integrated_lore = await self._process_distributed_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_generated_lore(
                integrated_lore,
                base_context,
                run_ctx
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Cache the result
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate dynamic lore: {str(e)}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_generation_error(e, base_context, world_state)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from generation error: {str(recovery_error)}")
            
            raise
    
    def _create_task_dependencies(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Create task dependencies for distributed execution."""
        return {
            'analyze_timeline': [],
            'analyze_cultural': ['analyze_timeline'],
            'analyze_political': ['analyze_timeline'],
            'generate_historical': ['analyze_timeline'],
            'generate_cultural': ['analyze_cultural'],
            'generate_political': ['analyze_political'],
            'integrate': [
                'generate_historical',
                'generate_cultural',
                'generate_political'
            ]
        }
    
    async def _process_distributed_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process results from distributed execution."""
        try:
            # Extract results
            historical_lore = results['generate_historical']
            cultural_lore = results['generate_cultural']
            political_lore = results['generate_political']
            
            # Integrate results
            integrated_lore = await self._integrate_generated_lore(
                historical_lore,
                cultural_lore,
                political_lore,
                RunContextWrapper(context={})
            )
            
            return integrated_lore
            
        except Exception as e:
            logger.error(f"Failed to process distributed results: {e}")
            raise
    
    async def _prefetch_lore(
        self,
        cache_key: str,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any]
    ):
        """Prefetch lore based on prediction."""
        try:
            # Generate lore in background
            asyncio.create_task(
                self._generate_and_cache_lore(
                    cache_key,
                    base_context,
                    world_state
                )
            )
        except Exception as e:
            logger.error(f"Failed to prefetch lore: {e}")
    
    async def _generate_and_cache_lore(
        self,
        cache_key: str,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any]
    ):
        """Generate and cache lore in background."""
        try:
            lore = await self._generate_lore_from_pattern(
                {'base_context': base_context, 'world_state': world_state},
                RunContextWrapper(context={})
            )
            await self.cache_manager.set(cache_key, lore)
        except Exception as e:
            logger.error(f"Failed to generate and cache lore: {e}")
    
    async def _recover_from_generation_error(
        self,
        error: Exception,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Recover from generation errors with fallback strategies."""
        try:
            # Log error details
            logger.error(f"Generation error: {str(error)}")
            
            # Try to recover using cached data
            cache_key = f"generation_{hash(json.dumps(base_context))}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                logger.info("Recovered from cache")
                return cached_data
            
            # Try to recover using simplified generation
            try:
                simplified_context = {
                    'world': base_context.get('world', {}),
                    'time_period': base_context.get('time_period', 'present'),
                    'location': base_context.get('location', {})
                }
                
                simplified_lore = await self._generate_lore_from_pattern(
                    simplified_context,
                    RunContextWrapper()
                )
                
                if simplified_lore:
                    logger.info("Recovered using simplified generation")
                    return simplified_lore
                    
            except Exception as sim_error:
                logger.error(f"Simplified generation failed: {str(sim_error)}")
            
            # Try to recover using default templates
            try:
                default_template = self._get_default_template(base_context)
                if default_template:
                    logger.info("Recovered using default template")
                    return default_template
                    
            except Exception as template_error:
                logger.error(f"Template recovery failed: {str(template_error)}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error during recovery: {str(e)}")
            return None
    
    def _get_default_template(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get a default template based on context."""
        try:
            world_type = context.get('world', {}).get('type', 'fantasy')
            time_period = context.get('time_period', 'present')
            
            template = {
                'type': 'default_template',
                'world_type': world_type,
                'time_period': time_period,
                'content': {
                    'description': f"Default {world_type} world in {time_period}",
                    'elements': []
                },
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source': 'default_template',
                    'recovery_used': True
                }
            }
            
            return template
            
        except Exception as e:
            logger.error(f"Error getting default template: {str(e)}")
            return None
    
    async def cleanup(self) -> bool:
        """Clean up agent resources."""
        try:
            # Clean up data access layers
            await self.lore_dal.cleanup()
            await self.location_dal.cleanup()
            await self.npc_dal.cleanup()
            await self.lore_integrator.cleanup()
            
            # Clean up management systems
            await self.cache_manager.cleanup()
            await self.version_control.cleanup()
            await self.parallel_processor.cleanup()
            await self.distributed_executor.cleanup()
            await self.cache_predictor.cleanup()
            
            # Unregister from agent registry
            AgentRegistry.unregister_agent(self)
            
            # Clean up resource pools
            for pool in self.resource_pools.values():
                await pool.cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup LoreGenerationAgent: {str(e)}")
            return False

    async def _generate_lore_from_pattern(
        self,
        context: Dict[str, Any],
        run_ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """Generate lore based on a pattern from the context"""
        try:
            # Extract base context and world state
            base_context = context.get('base_context', {})
            world_state = context.get('world_state', {})
            
            # Determine generation pattern
            pattern = self._determine_generation_pattern(base_context, world_state)
            
            # Generate based on pattern
            if pattern == "world_foundation":
                return await self._generate_world_foundation(base_context, world_state, run_ctx)
            elif pattern == "faction_development":
                return await self._generate_faction_development(base_context, world_state, run_ctx)
            elif pattern == "cultural_elements":
                return await self._generate_cultural_elements(base_context, world_state, run_ctx)
            elif pattern == "historical_events":
                return await self._generate_historical_events(base_context, world_state, run_ctx)
            elif pattern == "location_details":
                return await self._generate_location_details(base_context, world_state, run_ctx)
            else:
                return await self._generate_generic_lore(base_context, world_state, run_ctx)
                
        except Exception as e:
            logging.error(f"Error generating lore from pattern: {e}")
            return {"error": str(e)}
            
    def _determine_generation_pattern(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> str:
        """Determine the appropriate generation pattern based on context"""
        # Check for specific triggers in base_context
        if "world_foundation" in base_context.get("tags", []):
            return "world_foundation"
        elif "faction" in base_context.get("tags", []):
            return "faction_development"
        elif "culture" in base_context.get("tags", []):
            return "cultural_elements"
        elif "history" in base_context.get("tags", []):
            return "historical_events"
        elif "location" in base_context.get("tags", []):
            return "location_details"
            
        # Check world state for patterns
        if "factions" in world_state and not world_state["factions"]:
            return "faction_development"
        elif "cultural_elements" in world_state and not world_state["cultural_elements"]:
            return "cultural_elements"
        elif "historical_events" in world_state and not world_state["historical_events"]:
            return "historical_events"
        elif "locations" in world_state and not world_state["locations"]:
            return "location_details"
            
        return "generic"
        
    async def _generate_world_foundation(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any],
        run_ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """Generate world foundation lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'base_context': base_context,
                'world_state': world_state,
                'run_ctx': run_ctx
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'world_foundation',
                task_context
            )
            
            # Create task dependencies
            task_deps = {
                'analyze_environment': [],
                'generate_cosmology': ['analyze_environment'],
                'generate_magic_system': ['analyze_environment'],
                'generate_history': ['analyze_environment'],
                'generate_social_structure': ['analyze_environment'],
                'integrate': [
                    'generate_cosmology',
                    'generate_magic_system',
                    'generate_history',
                    'generate_social_structure'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            foundation_lore = await self._process_foundation_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_foundation_lore(
                foundation_lore,
                base_context,
                run_ctx
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Cache the result
            cache_key = f"world_foundation_{hash(json.dumps(base_context))}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate world foundation: {str(e)}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_foundation_error(e, base_context, world_state)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from foundation error: {str(recovery_error)}")
            
            raise
        
    async def _generate_faction_development(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any],
        run_ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """Generate faction development lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'base_context': base_context,
                'world_state': world_state,
                'run_ctx': run_ctx
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'faction_development',
                task_context
            )
            
            # Create task dependencies
            task_deps = {
                'analyze_social_structure': [],
                'generate_faction_types': ['analyze_social_structure'],
                'generate_faction_relationships': ['generate_faction_types'],
                'generate_faction_goals': ['generate_faction_types'],
                'generate_faction_resources': ['generate_faction_types'],
                'integrate': [
                    'generate_faction_relationships',
                    'generate_faction_goals',
                    'generate_faction_resources'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            faction_lore = await self._process_faction_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_faction_lore(
                faction_lore,
                base_context,
                run_ctx
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Cache the result
            cache_key = f"faction_development_{hash(json.dumps(base_context))}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate faction development: {str(e)}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_faction_error(e, base_context, world_state)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from faction error: {str(recovery_error)}")
            
            raise
        
    async def _generate_cultural_elements(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any],
        run_ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """Generate cultural elements lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'base_context': base_context,
                'world_state': world_state,
                'run_ctx': run_ctx
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'cultural_elements',
                task_context
            )
            
            # Create task dependencies
            task_deps = {
                'analyze_society': [],
                'generate_traditions': ['analyze_society'],
                'generate_beliefs': ['analyze_society'],
                'generate_customs': ['analyze_society'],
                'generate_art': ['analyze_society'],
                'integrate': [
                    'generate_traditions',
                    'generate_beliefs',
                    'generate_customs',
                    'generate_art'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            cultural_lore = await self._process_cultural_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_cultural_lore(
                cultural_lore,
                base_context,
                run_ctx
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Cache the result
            cache_key = f"cultural_elements_{hash(json.dumps(base_context))}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate cultural elements: {str(e)}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_cultural_error(e, base_context, world_state)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from cultural error: {str(recovery_error)}")
            
            raise
        
    async def _generate_historical_events(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any],
        run_ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """Generate historical events lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'base_context': base_context,
                'world_state': world_state,
                'run_ctx': run_ctx
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'historical_events',
                task_context
            )
            
            # Create task dependencies
            task_deps = {
                'analyze_timeline': [],
                'generate_major_events': ['analyze_timeline'],
                'generate_minor_events': ['analyze_timeline'],
                'generate_consequences': ['generate_major_events', 'generate_minor_events'],
                'generate_connections': ['generate_consequences'],
                'integrate': [
                    'generate_major_events',
                    'generate_minor_events',
                    'generate_consequences',
                    'generate_connections'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            historical_lore = await self._process_historical_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_historical_lore(
                historical_lore,
                base_context,
                run_ctx
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Cache the result
            cache_key = f"historical_events_{hash(json.dumps(base_context))}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate historical events: {str(e)}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_historical_error(e, base_context, world_state)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from historical error: {str(recovery_error)}")
            
            raise
        
    async def _generate_location_details(
        self,
        base_context: Dict[str, Any],
        world_state: Dict[str, Any],
        run_ctx: RunContextWrapper
    ) -> Dict[str, Any]:
        """Generate location details lore with agentic framework integration."""
        try:
            # Create task context
            task_context = {
                'base_context': base_context,
                'world_state': world_state,
                'run_ctx': run_ctx
            }
            
            # Create version branch
            branch = await self.version_control.create_branch(
                'location_details',
                task_context
            )
            
            # Create task dependencies
            task_deps = {
                'analyze_geography': [],
                'generate_landmarks': ['analyze_geography'],
                'generate_settlements': ['analyze_geography'],
                'generate_resources': ['analyze_geography'],
                'generate_climate': ['analyze_geography'],
                'integrate': [
                    'generate_landmarks',
                    'generate_settlements',
                    'generate_resources',
                    'generate_climate'
                ]
            }
            
            # Execute tasks with distributed executor
            results = await self.distributed_executor.execute_tasks(
                task_deps,
                priority=TaskPriority.HIGH
            )
            
            # Process results
            location_lore = await self._process_location_results(results)
            
            # Validate and enhance
            enhanced_lore = await self._enhance_location_lore(
                location_lore,
                base_context,
                run_ctx
            )
            
            # Update version branch
            await branch.update(enhanced_lore, 'completed')
            
            # Cache the result
            cache_key = f"location_details_{hash(json.dumps(base_context))}"
            await self.cache_manager.set(cache_key, enhanced_lore)
            
            # Update validation stats
            self.state['validation_stats']['successful'] += 1
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Failed to generate location details: {str(e)}")
            self.state['validation_stats']['failed'] += 1
            
            # Try to recover from error
            try:
                recovered_lore = await self._recover_from_location_error(e, base_context, world_state)
                if recovered_lore:
                    return recovered_lore
            except Exception as recovery_error:
                logger.error(f"Failed to recover from location error: {str(recovery_error)}")
            
            raise

    async def start(self):
        """Start the generation agent and resource management."""
        await super().start()
        await self.resource_manager.start()
    
    async def stop(self):
        """Stop the generation agent and cleanup resources."""
        await super().stop()
        await self.resource_manager.stop()
    
    async def get_generation_data(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get generation data from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('generation', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting generation data: {e}")
            return None
    
    async def set_generation_data(
        self,
        generation_id: str,
        data: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set generation data in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('generation', generation_id, data, tags)
        except Exception as e:
            logger.error(f"Error setting generation data: {e}")
            return False
    
    async def invalidate_generation_data(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate generation data cache."""
        try:
            await self.invalidate_cached_data('generation', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating generation data: {e}")
    
    async def get_validation_stats(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get validation stats from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('validation_stats', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting validation stats: {e}")
            return None
    
    async def set_validation_stats(
        self,
        generation_id: str,
        stats: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set validation stats in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('validation_stats', generation_id, stats, tags)
        except Exception as e:
            logger.error(f"Error setting validation stats: {e}")
            return False
    
    async def invalidate_validation_stats(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate validation stats cache."""
        try:
            await self.invalidate_cached_data('validation_stats', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating validation stats: {e}")
    
    async def get_performance_metrics(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get performance metrics from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('performance_metrics', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return None
    
    async def set_performance_metrics(
        self,
        generation_id: str,
        metrics: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set performance metrics in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('performance_metrics', generation_id, metrics, tags)
        except Exception as e:
            logger.error(f"Error setting performance metrics: {e}")
            return False
    
    async def invalidate_performance_metrics(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate performance metrics cache."""
        try:
            await self.invalidate_cached_data('performance_metrics', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating performance metrics: {e}")
    
    async def get_generation_patterns(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get generation patterns from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('generation_patterns', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting generation patterns: {e}")
            return None
    
    async def set_generation_patterns(
        self,
        generation_id: str,
        patterns: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set generation patterns in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('generation_patterns', generation_id, patterns, tags)
        except Exception as e:
            logger.error(f"Error setting generation patterns: {e}")
            return False
    
    async def invalidate_generation_patterns(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate generation patterns cache."""
        try:
            await self.invalidate_cached_data('generation_patterns', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating generation patterns: {e}")
    
    async def get_resource_usage(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get resource usage from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('resource_usage', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting resource usage: {e}")
            return None
    
    async def set_resource_usage(
        self,
        generation_id: str,
        usage: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set resource usage in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('resource_usage', generation_id, usage, tags)
        except Exception as e:
            logger.error(f"Error setting resource usage: {e}")
            return False
    
    async def invalidate_resource_usage(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate resource usage cache."""
        try:
            await self.invalidate_cached_data('resource_usage', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating resource usage: {e}")
    
    async def get_active_tasks(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get active tasks from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('active_tasks', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting active tasks: {e}")
            return None
    
    async def set_active_tasks(
        self,
        generation_id: str,
        tasks: List[Dict[str, Any]],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set active tasks in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('active_tasks', generation_id, tasks, tags)
        except Exception as e:
            logger.error(f"Error setting active tasks: {e}")
            return False
    
    async def invalidate_active_tasks(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate active tasks cache."""
        try:
            await self.invalidate_cached_data('active_tasks', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating active tasks: {e}")
    
    async def get_error_states(
        self,
        generation_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """Get error states from cache or fetch if not available."""
        try:
            # Check resource availability before fetching
            if fetch_func:
                await self.resource_manager._check_resource_availability('memory')
            
            return await self.get_cached_data('error_states', generation_id, fetch_func)
        except Exception as e:
            logger.error(f"Error getting error states: {e}")
            return None
    
    async def set_error_states(
        self,
        generation_id: str,
        states: Dict[str, Any],
        tags: Optional[Set[str]] = None
    ) -> bool:
        """Set error states in cache."""
        try:
            # Check resource availability before setting
            await self.resource_manager._check_resource_availability('memory')
            
            return await self.set_cached_data('error_states', generation_id, states, tags)
        except Exception as e:
            logger.error(f"Error setting error states: {e}")
            return False
    
    async def invalidate_error_states(
        self,
        generation_id: Optional[str] = None,
        recursive: bool = True
    ) -> None:
        """Invalidate error states cache."""
        try:
            await self.invalidate_cached_data('error_states', generation_id, recursive)
        except Exception as e:
            logger.error(f"Error invalidating error states: {e}")
    
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

    async def _process_modification_results(
        self,
        results: Dict[str, Any],
        original_lore: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process results from modification tasks."""
        try:
            # Extract results from each task
            validation_result = results.get('validate_modifications', {})
            conflict_result = results.get('check_conflicts', {})
            modification_result = results.get('apply_modifications', {})
            consistency_result = results.get('validate_consistency', {})
            metadata_result = results.get('update_metadata', {})
            
            # Combine results
            modified_lore = original_lore.copy()
            
            # Apply validated modifications
            if validation_result.get('status') == 'success':
                modified_lore.update(modification_result.get('modifications', {}))
            
            # Apply conflict resolutions
            if conflict_result.get('status') == 'success':
                modified_lore.update(conflict_result.get('resolutions', {}))
            
            # Apply consistency fixes
            if consistency_result.get('status') == 'success':
                modified_lore.update(consistency_result.get('fixes', {}))
            
            # Update metadata
            if metadata_result.get('status') == 'success':
                modified_lore['metadata'] = {
                    **modified_lore.get('metadata', {}),
                    **metadata_result.get('metadata', {})
                }
            
            return modified_lore
            
        except Exception as e:
            logger.error(f"Error processing modification results: {e}")
            return original_lore

    async def _process_integration_results(
        self,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process results from integration tasks."""
        try:
            # Extract results from each task
            analysis_result = results.get('analyze_components', {})
            consistency_result = results.get('validate_consistency', {})
            conflict_result = results.get('resolve_conflicts', {})
            merge_result = results.get('merge_components', {})
            validation_result = results.get('validate_integration', {})
            metadata_result = results.get('update_metadata', {})
            
            # Combine results
            integrated_lore = {}
            
            # Apply component analysis
            if analysis_result.get('status') == 'success':
                integrated_lore.update(analysis_result.get('components', {}))
            
            # Apply consistency fixes
            if consistency_result.get('status') == 'success':
                integrated_lore.update(consistency_result.get('fixes', {}))
            
            # Apply conflict resolutions
            if conflict_result.get('status') == 'success':
                integrated_lore.update(conflict_result.get('resolutions', {}))
            
            # Apply merged components
            if merge_result.get('status') == 'success':
                integrated_lore.update(merge_result.get('merged', {}))
            
            # Apply validation fixes
            if validation_result.get('status') == 'success':
                integrated_lore.update(validation_result.get('fixes', {}))
            
            # Update metadata
            if metadata_result.get('status') == 'success':
                integrated_lore['metadata'] = metadata_result.get('metadata', {})
            
            return integrated_lore
            
        except Exception as e:
            logger.error(f"Error processing integration results: {e}")
            return {}

    async def _enhance_modified_lore(
        self,
        modified_lore: Dict[str, Any],
        original_lore: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance modified lore with additional context and relationships."""
        try:
            enhanced_lore = modified_lore.copy()
            
            # Add modification history
            enhanced_lore['metadata'] = {
                **enhanced_lore.get('metadata', {}),
                'modification_history': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'modifications': modifications,
                        'original_state': original_lore
                    }
                ]
            }
            
            # Update relationships
            if 'relationships' in enhanced_lore:
                enhanced_lore['relationships'] = await self._update_relationships(
                    enhanced_lore['relationships'],
                    modifications
                )
            
            # Update validation state
            enhanced_lore['validation_state'] = {
                'last_validated': datetime.now().isoformat(),
                'status': 'modified',
                'validation_required': True
            }
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Error enhancing modified lore: {e}")
            return modified_lore

    async def _enhance_integrated_lore(
        self,
        integrated_lore: Dict[str, Any],
        lore_parts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance integrated lore with additional context and relationships."""
        try:
            enhanced_lore = integrated_lore.copy()
            
            # Add integration history
            enhanced_lore['metadata'] = {
                **enhanced_lore.get('metadata', {}),
                'integration_history': [
                    {
                        'timestamp': datetime.now().isoformat(),
                        'integrated_parts': list(lore_parts.keys()),
                        'part_states': lore_parts
                    }
                ]
            }
            
            # Update relationships between integrated parts
            if 'relationships' in enhanced_lore:
                enhanced_lore['relationships'] = await self._update_integrated_relationships(
                    enhanced_lore['relationships'],
                    lore_parts
                )
            
            # Update validation state
            enhanced_lore['validation_state'] = {
                'last_validated': datetime.now().isoformat(),
                'status': 'integrated',
                'validation_required': True
            }
            
            return enhanced_lore
            
        except Exception as e:
            logger.error(f"Error enhancing integrated lore: {e}")
            return integrated_lore

    async def _update_relationships(
        self,
        relationships: Dict[str, Any],
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update relationships based on modifications."""
        try:
            updated_relationships = relationships.copy()
            
            # Update affected relationships
            for key, value in modifications.items():
                if key in updated_relationships:
                    updated_relationships[key] = {
                        **updated_relationships[key],
                        'last_modified': datetime.now().isoformat(),
                        'modification_history': [
                            {
                                'timestamp': datetime.now().isoformat(),
                                'changes': value
                            }
                        ]
                    }
            
            return updated_relationships
            
        except Exception as e:
            logger.error(f"Error updating relationships: {e}")
            return relationships

    async def _update_integrated_relationships(
        self,
        relationships: Dict[str, Any],
        lore_parts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update relationships between integrated lore parts."""
        try:
            updated_relationships = relationships.copy()
            
            # Add relationships between integrated parts
            for part_id, part_data in lore_parts.items():
                if part_id not in updated_relationships:
                    updated_relationships[part_id] = {
                        'type': 'integrated_component',
                        'created_at': datetime.now().isoformat(),
                        'related_parts': list(lore_parts.keys()),
                        'integration_history': [
                            {
                                'timestamp': datetime.now().isoformat(),
                                'action': 'integration',
                                'part_data': part_data
                            }
                        ]
                    }
            
            return updated_relationships
            
        except Exception as e:
            logger.error(f"Error updating integrated relationships: {e}")
            return relationships

    async def _apply_partial_modifications(
        self,
        lore_id: int,
        modifications: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply partial modifications to lore."""
        try:
            # Get current lore
            query = """
                SELECT * FROM LoreComponents
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """
            lore = await self.db.fetchrow(query, lore_id, self.user_id, self.conversation_id)
            
            if not lore:
                return None
            
            # Apply modifications
            modified_lore = lore.copy()
            for key, value in modifications.items():
                if key in modified_lore:
                    modified_lore[key] = value
            
            # Update metadata
            modified_lore['metadata'] = {
                **modified_lore.get('metadata', {}),
                'partial_modification': {
                    'timestamp': datetime.now().isoformat(),
                    'modified_fields': list(modifications.keys())
                }
            }
            
            # Update in database
            update_query = """
                UPDATE LoreComponents
                SET content = $1,
                    metadata = $2,
                    last_modified = NOW()
                WHERE id = $3
            """
            await self.db.execute(
                update_query,
                json.dumps(modified_lore['content']),
                json.dumps(modified_lore['metadata']),
                lore_id
            )
            
            return modified_lore
            
        except Exception as e:
            logger.error(f"Error applying partial modifications: {str(e)}")
            return None

    async def _integrate_partial_parts(
        self,
        lore_parts: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Integrate partial lore parts."""
        try:
            integrated_lore = {
                'content': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source': 'partial_integration',
                    'integrated_parts': list(lore_parts.keys())
                }
            }
            
            # Integrate parts
            for part_id, part_data in lore_parts.items():
                if isinstance(part_data, dict) and 'content' in part_data:
                    integrated_lore['content'][part_id] = part_data['content']
            
            # Add relationships
            integrated_lore['relationships'] = {
                part_id: {
                    'type': 'integrated_component',
                    'created_at': datetime.now().isoformat(),
                    'related_parts': list(lore_parts.keys())
                }
                for part_id in lore_parts
            }
            
            return integrated_lore
            
        except Exception as e:
            logger.error(f"Error integrating partial parts: {str(e)}")
            return None

    async def _integrate_simplified_parts(
        self,
        simplified_parts: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Integrate simplified lore parts."""
        try:
            integrated_lore = {
                'content': {},
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'source': 'simplified_integration',
                    'integrated_parts': list(simplified_parts.keys())
                }
            }
            
            # Integrate parts
            for part_id, part_data in simplified_parts.items():
                integrated_lore['content'][part_id] = part_data['content']
            
            # Add basic relationships
            integrated_lore['relationships'] = {
                part_id: {
                    'type': 'simplified_component',
                    'created_at': datetime.now().isoformat(),
                    'related_parts': list(simplified_parts.keys())
                }
                for part_id in simplified_parts
            }
            
            return integrated_lore
            
        except Exception as e:
            logger.error(f"Error integrating simplified parts: {str(e)}")
            return None

    async def _get_backup(self, lore_id: int) -> Optional[Dict[str, Any]]:
        """Get backup of lore."""
        try:
            # Try to get from version control first
            version = await self.version_control.get_latest_version(lore_id)
            if version:
                return version
            
            # Try to get from cache
            cache_key = f"lore_backup_{lore_id}"
            backup = await self.cache_manager.get(cache_key)
            if backup:
                return backup
            
            # Try to get from database
            query = """
                SELECT * FROM LoreComponents
                WHERE id = $1 AND user_id = $2 AND conversation_id = $3
            """
            lore = await self.db.fetchrow(query, lore_id, self.user_id, self.conversation_id)
            
            if lore:
                # Cache the backup
                await self.cache_manager.set(cache_key, lore)
                return lore
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting backup: {str(e)}")
            return None

# Create a singleton instance for easy access
generation_agent = LoreGenerationAgent() 
