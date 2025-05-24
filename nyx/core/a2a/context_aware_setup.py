# nyx/core/a2a/context_aware_setup.py

import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

async def setup_context_aware_creative_modules(nyx_brain=None) -> Dict[str, Any]:
    """
    Setup all context-aware creative and tool modules for A2A integration
    
    Args:
        nyx_brain: Optional reference to a NyxBrain instance
        
    Returns:
        Dictionary of initialized context-aware modules
    """
    modules = {}
    
    try:
        # ========================================================================================
        # CREATIVE SYSTEM
        # ========================================================================================
        
        # Import original creative system
        from nyx.creative.agentic_system import AgenticCreativitySystemV2
        from nyx.core.a2a.context_aware_creative_system import ContextAwareCreativeSystem
        
        # Initialize original creative system
        repo_root = getattr(nyx_brain, "creative_system_root", ".") if nyx_brain else "."
        original_creative = AgenticCreativitySystemV2(repo_root=repo_root)
        
        # Wrap with context-aware version
        context_aware_creative = ContextAwareCreativeSystem(original_creative)
        modules["creative_system"] = context_aware_creative
        
        logger.info("✓ Context-aware creative system initialized")
        
        # ========================================================================================
        # ANALYSIS SANDBOX
        # ========================================================================================
        
        # Import analysis components
        from nyx.creative.analysis_sandbox import CodeAnalyzer, SandboxExecutor
        from nyx.core.a2a.context_aware_analysis_sandbox import ContextAwareAnalysisSandbox
        
        # Initialize original components
        original_analyzer = CodeAnalyzer(creative_content_system=original_creative.storage)
        original_sandbox = SandboxExecutor(creative_content_system=original_creative.storage)
        
        # Wrap with context-aware version
        context_aware_analysis = ContextAwareAnalysisSandbox(
            original_code_analyzer=original_analyzer,
            original_sandbox_executor=original_sandbox
        )
        modules["analysis_sandbox"] = context_aware_analysis
        
        logger.info("✓ Context-aware analysis sandbox initialized")
        
        # ========================================================================================
        # CAPABILITY SYSTEM
        # ========================================================================================
        
        # Import capability system
        from nyx.creative.capability_system import CapabilityAssessmentSystem
        from nyx.core.a2a.context_aware_capability_system import ContextAwareCapabilitySystem
        
        # Initialize original system
        original_capability = CapabilityAssessmentSystem(
            creative_content_system=original_creative.storage
        )
        
        # Wrap with context-aware version
        context_aware_capability = ContextAwareCapabilitySystem(original_capability)
        modules["capability_system"] = context_aware_capability
        
        logger.info("✓ Context-aware capability system initialized")
        
        # ========================================================================================
        # CONTENT SYSTEM
        # ========================================================================================
        
        # Import content system
        from nyx.creative.content_system import CreativeContentSystem
        from nyx.core.a2a.context_aware_content_system import ContextAwareContentSystem
        
        # Initialize original system
        original_content = CreativeContentSystem()
        
        # Wrap with context-aware version
        context_aware_content = ContextAwareContentSystem(original_content)
        modules["content_system"] = context_aware_content
        
        logger.info("✓ Context-aware content system initialized")
        
        # ========================================================================================
        # LOGGING UTILS
        # ========================================================================================
        
        # Import logging utils
        from nyx.creative.logging_utils import NyxLogger
        from nyx.core.a2a.context_aware_logging import ContextAwareNyxLogger
        
        # Initialize original logger
        original_logger = NyxLogger(content_system=original_content)
        
        # Wrap with context-aware version
        context_aware_logger = ContextAwareNyxLogger(original_logger)
        modules["logging_system"] = context_aware_logger
        
        logger.info("✓ Context-aware logging system initialized")
        
        # ========================================================================================
        # COMPUTER USE AGENT
        # ========================================================================================
        
        # Import computer use agent
        from nyx.tools.computer_use_agent import ComputerUseAgent
        from nyx.core.a2a.context_aware_computer_use_agent import ContextAwareComputerUseAgent
        
        # Initialize original agent
        original_computer_agent = ComputerUseAgent(logger=original_logger)
        
        # Wrap with context-aware version
        context_aware_computer = ContextAwareComputerUseAgent(original_computer_agent)
        modules["computer_use_agent"] = context_aware_computer
        
        logger.info("✓ Context-aware computer use agent initialized")
        
        # ========================================================================================
        # EMAIL IDENTITY MANAGER
        # ========================================================================================
        
        # Import email identity manager
        from nyx.tools.email_identity_manager import EmailIdentityManager
        from nyx.core.a2a.context_aware_email_identity_manager import ContextAwareEmailIdentityManager
        
        # Initialize original manager with dependencies
        memory_core = getattr(nyx_brain, "memory_core", None) if nyx_brain else None
        original_email_manager = EmailIdentityManager(
            logger=original_logger,
            computer_user=original_computer_agent,
            memory_core=memory_core
        )
        
        # Wrap with context-aware version
        context_aware_email = ContextAwareEmailIdentityManager(original_email_manager)
        modules["email_identity_manager"] = context_aware_email
        
        logger.info("✓ Context-aware email identity manager initialized")
        
        # ========================================================================================
        # CLAIM VALIDATION
        # ========================================================================================
        
        # Import claim validation
        from nyx.tools.claim_validation import BLACKLISTED_SOURCES
        from nyx.core.a2a.context_aware_claim_validation import ContextAwareClaimValidation
        
        # Initialize context-aware claim validator
        context_aware_claim = ContextAwareClaimValidation(
            blacklisted_sources=BLACKLISTED_SOURCES,
            computer_use_agent=original_computer_agent
        )
        modules["claim_validation"] = context_aware_claim
        
        logger.info("✓ Context-aware claim validation initialized")
        
        # ========================================================================================
        # SOCIAL BROWSING
        # ========================================================================================
        
        # Import social browsing components
        from nyx.tools.social_browsing import (
            SentimentProfiler, ThreadTracker, ContextUnspooler,
            ProvocationEngine, PersonaMonitor, DesireRegistry
        )
        from nyx.core.a2a.context_aware_social_browsing import ContextAwareSocialBrowsing
        
        # Initialize all social browsing components
        sentiment_profiler = SentimentProfiler()
        thread_tracker = ThreadTracker()
        context_unspooler = ContextUnspooler(computer_user=original_computer_agent)
        provocation_engine = ProvocationEngine()
        persona_monitor = PersonaMonitor()
        desire_registry = DesireRegistry(
            memory_core=memory_core,
            logger=original_logger
        )
        
        # Create context-aware social browsing
        context_aware_social = ContextAwareSocialBrowsing(
            sentiment_profiler=sentiment_profiler,
            thread_tracker=thread_tracker,
            context_unspooler=context_unspooler,
            provocation_engine=provocation_engine,
            persona_monitor=persona_monitor,
            desire_registry=desire_registry,
            computer_use_agent=original_computer_agent,
            claim_validator=lambda self, text, source: context_aware_claim._validate_social_claim(text, source)
        )
        modules["social_browsing"] = context_aware_social
        
        logger.info("✓ Context-aware social browsing initialized")
        
        # ========================================================================================
        # UI INTERACTION
        # ========================================================================================
        
        # Import UI interaction
        from nyx.tools.ui_interaction import UIConversationManager
        from nyx.core.a2a.context_aware_ui_interaction import ContextAwareUIInteraction
        
        # Initialize original UI manager
        system_context = getattr(nyx_brain, "system_context", None) if nyx_brain else None
        original_ui_manager = UIConversationManager(system_context=system_context)
        
        # Wrap with context-aware version
        context_aware_ui = ContextAwareUIInteraction(original_ui_manager)
        modules["ui_interaction"] = context_aware_ui
        
        logger.info("✓ Context-aware UI interaction initialized")
        
        # ========================================================================================
        # INTEGRATION WITH NYXBRAIN
        # ========================================================================================
        
        if nyx_brain:
            # Run initial creative system analysis
            await original_creative.incremental_codebase_analysis()
            
            # Register all modules with NyxBrain if it has the context distribution system
            if hasattr(nyx_brain, 'context_distribution') and nyx_brain.context_distribution:
                for module_name, module in modules.items():
                    # Set the module as an attribute on NyxBrain
                    setattr(nyx_brain, module_name, module)
                    
                    # Register with context distribution if the module is context-aware
                    if hasattr(module, 'set_context_system'):
                        module.set_context_system(nyx_brain.context_distribution)
                        logger.info(f"  → Registered {module_name} with context distribution")
            
            # Store reference to creative content in memory if available
            if hasattr(nyx_brain, "memory_core") and nyx_brain.memory_core:
                await nyx_brain.memory_core.add_memory(
                    memory_text="Initialized context-aware creative and tool systems",
                    memory_type="system",
                    significance=5,
                    tags=["creativity", "tools", "initialization", "a2a"]
                )
            
            logger.info(f"✓ All creative modules integrated with NyxBrain instance")
        
        # ========================================================================================
        # SUMMARY
        # ========================================================================================
        
        logger.info(f"""
╔════════════════════════════════════════════════════════════════╗
║          Context-Aware Creative System Setup Complete          ║
╠════════════════════════════════════════════════════════════════╣
║ Modules Initialized:                                           ║
║   • Creative System (Code Analysis, Generation)                ║
║   • Analysis Sandbox (Code Review, Execution)                  ║
║   • Capability System (Self-Assessment)                        ║
║   • Content System (Storage, Retrieval)                        ║
║   • Logging System (Thoughts, Actions, Evolution)              ║
║   • Computer Use Agent (Web Browsing, Automation)              ║
║   • Email Identity Manager (Privacy, Multi-Identity)           ║
║   • Claim Validation (Fact Checking, Misinformation)           ║
║   • Social Browsing (Sentiment, Personas, Engagement)          ║
║   • UI Interaction (Conversations, Proactive Engagement)       ║
║                                                                ║
║ All modules are now context-aware and ready for A2A           ║
║ coordination through the context distribution system.          ║
╚════════════════════════════════════════════════════════════════╝
        """)
        
        return modules
        
    except Exception as e:
        logger.error(f"Error during context-aware creative system setup: {e}", exc_info=True)
        raise


async def integrate_creative_modules_with_brain(nyx_brain) -> bool:
    """
    Integrate all creative modules with an existing NyxBrain instance
    
    Args:
        nyx_brain: The NyxBrain instance to integrate with
        
    Returns:
        Success status
    """
    try:
        # Ensure NyxBrain has context distribution initialized
        if not hasattr(nyx_brain, 'context_distribution') or not nyx_brain.context_distribution:
            logger.warning("NyxBrain doesn't have context distribution initialized. Initializing now...")
            await nyx_brain.initialize_context_system()
        
        # Setup all context-aware modules
        modules = await setup_context_aware_creative_modules(nyx_brain)
        
        # The setup function already handles integration, but we can do additional setup here
        
        # Create cross-module connections if needed
        if "social_browsing" in modules and "email_identity_manager" in modules:
            # Social browsing can use email manager for identity creation
            modules["social_browsing"].email_manager = modules["email_identity_manager"]
            logger.info("✓ Connected social browsing to email identity manager")
        
        if "creative_system" in modules and "analysis_sandbox" in modules:
            # Creative system can use analysis sandbox for code validation
            modules["creative_system"].code_validator = modules["analysis_sandbox"]
            logger.info("✓ Connected creative system to analysis sandbox")
        
        logger.info("✓ Creative modules successfully integrated with NyxBrain")
        return True
        
    except Exception as e:
        logger.error(f"Failed to integrate creative modules: {e}", exc_info=True)
        return False


# Convenience function for standalone testing
async def test_creative_modules():
    """Test creative modules without NyxBrain integration"""
    logger.info("Testing context-aware creative modules in standalone mode...")
    
    modules = await setup_context_aware_creative_modules()
    
    # Run some basic tests
    logger.info("\nRunning basic functionality tests...")
    
    # Test creative system
    if "creative_system" in modules:
        creative = modules["creative_system"]
        # This would normally receive context through the A2A system
        logger.info("✓ Creative system ready for context distribution")
    
    # Test capability system
    if "capability_system" in modules:
        capability = modules["capability_system"]
        # Can still use original methods
        assessment = await capability.original_system.assess_required_capabilities(
            "I want to write poetry about the moon"
        )
        logger.info(f"✓ Capability assessment: Feasibility {assessment['overall_feasibility']['score']:.2f}")
    
    logger.info("\n✓ All tests completed successfully!")
    return modules


if __name__ == "__main__":
    # Run standalone test
    asyncio.run(test_creative_modules())
