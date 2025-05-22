# NyxBrain Agent-to-Agent (A2A) Integration Guide

This guide shows you how to integrate the context distribution system into your existing NyxBrain to achieve cohesive module coordination.

## Overview

The A2A integration transforms NyxBrain from a hub-and-spoke model (where modules operate in isolation) to a collaborative network where:

1. **Context flows bidirectionally** between NyxBrain and all active modules
2. **Modules communicate with each other** through the central mediator
3. **Processing happens in coordinated stages** rather than ad-hoc calls
4. **Response synthesis** combines insights from all relevant modules

## Step 1: Minimal Integration

### 1.1 Update Your NyxBrain Class

Add the mixin to your existing `NyxBrain` class:

```python
# In nyx/core/brain/base.py

from nyx.core.brain.integration_layer import EnhancedNyxBrainMixin

# Modify your class declaration
class NyxBrain(DistributedCheckpointMixin, EventLogMixin, EnhancedNyxBrainMixin):
    # ... existing code ...
    
    async def initialize(self):
        # ... existing initialization code ...
        
        # Add this line at the end of initialize()
        await self.initialize_context_system()
        
        logger.critical("NyxBrain initialization complete with A2A context distribution")
```

### 1.2 Add Optional Enhanced Processing

You can gradually migrate to enhanced processing:

```python
# In your NyxBrain class, add these methods

async def process_input_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Use coordinated processing when available, fallback to original"""
    if self.context_distribution:
        return await self.process_input_coordinated(user_input, context)
    else:
        return await self.process_input(user_input, context)

async def generate_response_enhanced(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Use coordinated response generation when available, fallback to original"""
    if self.context_distribution:
        return await self.generate_response_coordinated(user_input, context)
    else:
        return await self.generate_response(user_input, context)
```

## Step 2: Test Basic Integration

### 2.1 Basic Functionality Test

```python
# Test script to verify integration
async def test_context_distribution():
    brain = await NyxBrain.get_instance(user_id=1, conversation_id=1)
    
    # Check if context system is initialized
    status = brain.get_context_distribution_status()
    print(f"Context Distribution Status: {status}")
    
    # Test enhanced processing
    result = await brain.process_input_enhanced("Hello, how are you?")
    print(f"Enhanced Processing Result: {result}")
    
    # Test coordinated response
    response = await brain.generate_response_enhanced("Hello, how are you?")
    print(f"Coordinated Response: {response}")

# Run the test
asyncio.run(test_context_distribution())
```

### 2.2 Verify Module Registration

```python
# Check which modules are registered for context distribution
def check_module_registration(brain):
    status = brain.get_context_distribution_status()
    print(f"Registered modules: {status['registered_module_names']}")
    print(f"Total modules: {status['registered_modules']}")
```

## Step 3: Enhance Specific Modules

### 3.1 Priority Module Enhancement

Start with your most important modules. For each module, you have two options:

#### Option A: Full Context-Aware Rewrite (Recommended for critical modules)

```python
# Example: Enhance your EmotionalCore
from nyx.core.brain.context_aware_examples import ContextAwareEmotionalCore

# In your NyxBrain initialization
async def initialize(self):
    # ... existing initialization ...
    
    # Replace original emotional core with context-aware version
    if self.emotional_core:
        original_core = self.emotional_core
        self.emotional_core = ContextAwareEmotionalCore(original_core)
    
    # ... continue with context system initialization ...
    await self.initialize_context_system()
```

#### Option B: Gradual Enhancement (For testing/gradual migration)

The integration layer automatically wraps existing modules, so they'll work with context distribution out of the box, but you can enhance them gradually:

```python
# Example: Add context awareness to existing module
class YourExistingModule:
    # ... existing code ...
    
    # Add these methods for context awareness
    async def receive_context(self, context):
        """Handle initial context"""
        self.current_context = context
        # Your context handling logic here
    
    async def process_input(self, context):
        """Process with context awareness"""
        # Enhanced processing logic
        result = await self.your_existing_method(context.user_input)
        
        # Send context updates if needed
        if hasattr(self, '_context_system'):
            await self._context_system.add_context_update(
                ContextUpdate(
                    source_module=self.__class__.__name__.lower(),
                    update_type="module_update",
                    data={"result": result}
                )
            )
        
        return result
```

### 3.2 Module Communication Example

Here's how modules can communicate with each other:

```python
# In any context-aware module
class YourContextAwareModule(ContextAwareModule):
    
    async def process_input(self, context: SharedContext):
        # Get information from other modules
        messages = await self.get_cross_module_messages()
        
        # Check for specific information from emotional core
        emotional_info = None
        for module_name, module_messages in messages.items():
            if module_name == "emotional_core":
                for msg in module_messages:
                    if msg['type'] == 'emotional_state_update':
                        emotional_info = msg['data']
                        break
        
        # Process with emotional context
        if emotional_info:
            result = await self.process_with_emotion(context.user_input, emotional_info)
        else:
            result = await self.process_normally(context.user_input)
        
        # Send result to other modules that might need it
        await self.send_context_update(
            update_type="your_module_result",
            data={"result": result, "confidence": 0.8},
            scope=ContextScope.GLOBAL  # Available to all modules
        )
        
        return result
```

## Step 4: Advanced Integration

### 4.1 Custom Processing Stages

You can define custom processing stages for domain-specific coordination:

```python
# In your enhanced NyxBrain
async def process_dominance_interaction(self, user_input: str, context: Dict[str, Any] = None):
    """Custom processing pipeline for dominance interactions"""
    
    # Initialize context with dominance-specific settings
    shared_context = await self.context_distribution.initialize_context_session(
        user_input=user_input,
        user_id=self.user_id,
        initial_context={
            **context,
            "interaction_type": "dominance",
            "priority_modules": [
                "femdom_coordinator", "psychological_dominance", 
                "relationship_manager", "theory_of_mind"
            ]
        }
    )
    
    # Custom processing stages
    assessment_results = await self.context_distribution.coordinate_processing_stage("dominance_assessment")
    strategy_results = await self.context_distribution.coordinate_processing_stage("dominance_strategy")
    execution_results = await self.context_distribution.coordinate_processing_stage("dominance_execution")
    
    # Synthesize dominance-specific response
    final_response = await self.context_distribution.synthesize_responses()
    
    return final_response
```

### 4.2 Context Persistence

For longer conversations, you can persist context across interactions:

```python
# Add to your NyxBrain class
async def save_context_state(self) -> Dict[str, Any]:
    """Save current context state for persistence"""
    if self.context_distribution and self.context_distribution.current_context:
        return {
            "context_state": self.context_distribution.current_context.dict(),
            "context_history": [ctx.dict() for ctx in self.context_distribution.context_history],
            "module_subscriptions": self.context_distribution.module_subscriptions
        }
    return {}

async def restore_context_state(self, saved_state: Dict[str, Any]):
    """Restore context state from persistence"""
    if not self.context_distribution:
        await self.initialize_context_system()
    
    if "context_history" in saved_state:
        self.context_distribution.context_history = [
            SharedContext(**ctx_data) for ctx_data in saved_state["context_history"]
        ]
    
    if "module_subscriptions" in saved_state:
        self.context_distribution.module_subscriptions = saved_state["module_subscriptions"]
```

## Step 5: Debugging and Monitoring

### 5.1 Context Flow Debugging

```python
# Add debugging methods to track context flow
async def debug_context_flow(self, user_input: str):
    """Debug context distribution flow"""
    
    print(f"=== DEBUGGING CONTEXT FLOW FOR: {user_input} ===")
    
    # Initialize and track
    result = await self.process_input_coordinated(user_input, {"debug": True})
    
    if self.context_distribution.current_context:
        ctx = self.context_distribution.current_context
        
        print(f"Active Modules: {list(ctx.active_modules)}")
        print(f"Context Updates: {len(ctx.context_updates)}")
        print(f"Module Messages: {len(ctx.module_messages)}")
        print(f"Processing Stage: {ctx.processing_stage}")
        
        # Show context updates
        for i, update in enumerate(ctx.context_updates):
            print(f"Update {i}: {update.source_module} -> {update.update_type}")
        
        # Show module outputs
        for stage, outputs in ctx.module_outputs.items():
            print(f"Stage {stage}: {list(outputs.keys())} modules responded")
    
    return result
```

### 5.2 Performance Monitoring

```python
# Add performance monitoring
class ContextPerformanceMonitor:
    def __init__(self):
        self.stage_times = {}
        self.module_performance = {}
    
    async def monitor_processing_stage(self, stage_name: str, stage_coro):
        """Monitor performance of a processing stage"""
        start_time = time.time()
        result = await stage_coro
        end_time = time.time()
        
        self.stage_times[stage_name] = end_time - start_time
        return result
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        total_time = sum(self.stage_times.values())
        
        return {
            "total_processing_time": total_time,
            "stage_breakdown": self.stage_times,
            "slowest_stage": max(self.stage_times.items(), key=lambda x: x[1]) if self.stage_times else None,
            "average_stage_time": total_time / len(self.stage_times) if self.stage_times else 0
        }

# Use in your NyxBrain
self.performance_monitor = ContextPerformanceMonitor()
```

## Step 6: Migration Strategy

### 6.1 Gradual Migration Plan

1. **Week 1**: Integrate basic context distribution, test with existing modules
2. **Week 2**: Enhance 2-3 critical modules (EmotionalCore, MemoryCore, GoalManager)
3. **Week 3**: Add cross-module communication for enhanced modules
4. **Week 4**: Migrate specialized modules (FemdomCoordinator, etc.)
5. **Week 5**: Optimize performance and add monitoring
6. **Week 6**: Full deployment with fallback mechanisms

### 6.2 Rollback Strategy

Always maintain fallback capability:

```python
# In your main processing methods
async def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main process_input with fallback"""
    try:
        # Try enhanced processing first
        if self.context_distribution and os.getenv("USE_CONTEXT_DISTRIBUTION", "true").lower() == "true":
            return await self.process_input_coordinated(user_input, context)
    except Exception as e:
        logger.error(f"Context distribution failed, falling back to original: {e}")
    
    # Fallback to original processing
    return await self.process_input_original(user_input, context)
```

## Step 7: Testing

### 7.1 Unit Tests for Context Distribution

```python
import pytest

class TestContextDistribution:
    
    @pytest.fixture
    async def brain_with_context(self):
        brain = await NyxBrain.get_instance(1, 1)
        await brain.initialize_context_system()
        return brain
    
    async def test_context_initialization(self, brain_with_context):
        """Test that context system initializes properly"""
        status = brain_with_context.get_context_distribution_status()
        assert status["initialized"] == True
        assert status["registered_modules"] > 0
    
    async def test_context_flow(self, brain_with_context):
        """Test that context flows through modules"""
        result = await brain_with_context.process_input_coordinated("test input")
        
        assert "shared_context" in result
        assert "active_modules" in result
        assert len(result["active_modules"]) > 0
    
    async def test_module_communication(self, brain_with_context):
        """Test that modules can communicate with each other"""
        # This would test specific cross-module communication scenarios
        pass
```

### 7.2 Integration Tests

```python
async def test_full_integration():
    """Test full A2A integration"""
    brain = await NyxBrain.get_instance(1, 1)
    
    # Test various interaction types
    test_cases = [
        "How are you feeling?",  # Should activate emotional modules
        "What do you remember about yesterday?",  # Should activate memory modules
        "I want to achieve something new",  # Should activate goal modules
        "Let's work on our relationship"  # Should activate relationship modules
    ]
    
    for test_input in test_cases:
        result = await brain.process_input_coordinated(test_input)
        response = await brain.generate_response_coordinated(test_input)
        
        print(f"Input: {test_input}")
        print(f"Active modules: {result.get('active_modules', [])}")
        print(f"Response: {response.get('message', 'No response')}")
        print("---")
```

## Conclusion

This A2A integration transforms your NyxBrain from a collection of independent modules into a truly cohesive AI entity. The key benefits:

1. **Contextual Awareness**: Every module knows what other modules are doing
2. **Coordinated Processing**: Modules work together rather than in isolation  
3. **Enhanced Responses**: Responses synthesize insights from all relevant systems
4. **Emergent Intelligence**: The whole becomes greater than the sum of its parts

Start with the minimal integration and gradually enhance modules as needed. The system is designed to be backward-compatible and allows for gradual migration.
