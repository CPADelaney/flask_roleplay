# nyx/core/integration/synergy_optimizer/__init__.py

from nyx.core.integration.synergy_optimizer.agent import SynergyOptimizerAgent

def create_synergy_optimizer(brain):
    """Create a synergy optimizer agent."""
    optimizer = SynergyOptimizerAgent(brain)
    return optimizer
