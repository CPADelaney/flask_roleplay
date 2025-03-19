# nyx/eternal/service_provider.py

from typing import Dict, Any, Optional
import logging

from .meta_learning_system import MetaLearningSystem
from .dynamic_adaptation_system import DynamicAdaptationSystem
from .internal_feedback_system import InternalFeedbackSystem
from .adapters import MetaLearningAdapter, DynamicAdaptationAdapter, InternalFeedbackAdapter
from .context import OpenAINyxContext

logger = logging.getLogger(__name__)

class ServiceProvider:
    """Service provider for OpenAI Agents SDK systems"""
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, user_id: int, conversation_id: int) -> 'ServiceProvider':
        """Get or create service provider instance"""
        key = f"{user_id}:{conversation_id}"
        
        if key not in cls._instances:
            cls._instances[key] = ServiceProvider(user_id, conversation_id)
        
        return cls._instances[key]
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Create systems
        self.meta_learning = MetaLearningSystem()
        self.dynamic_adaptation = DynamicAdaptationSystem()
        self.internal_feedback = InternalFeedbackSystem()
        
        # Create adapters
        self.meta_learning_adapter = MetaLearningAdapter()
        self.dynamic_adaptation_adapter = DynamicAdaptationAdapter()
        self.internal_feedback_adapter = InternalFeedbackAdapter()
        
        # Create context
        self.context = OpenAINyxContext(user_id, conversation_id)
        self.context.meta_learning = self.meta_learning
        self.context.dynamic_adaptation = self.dynamic_adaptation
        self.context.internal_feedback = self.internal_feedback
        
        logger.info(f"Created service provider for user {user_id}, conversation {conversation_id}")
    
    def get_context(self) -> OpenAINyxContext:
        """Get context instance"""
        return self.context
    
    def get_meta_learning(self) -> MetaLearningSystem:
        """Get meta-learning system"""
        return self.meta_learning
    
    def get_dynamic_adaptation(self) -> DynamicAdaptationSystem:
        """Get dynamic adaptation system"""
        return self.dynamic_adaptation
    
    def get_internal_feedback(self) -> InternalFeedbackSystem:
        """Get internal feedback system"""
        return self.internal_feedback
    
    def get_meta_learning_adapter(self) -> MetaLearningAdapter:
        """Get meta-learning adapter"""
        return self.meta_learning_adapter
    
    def get_dynamic_adaptation_adapter(self) -> DynamicAdaptationAdapter:
        """Get dynamic adaptation adapter"""
        return self.dynamic_adaptation_adapter
    
    def get_internal_feedback_adapter(self) -> InternalFeedbackAdapter:
        """Get internal feedback adapter"""
        return self.internal_feedback_adapter
