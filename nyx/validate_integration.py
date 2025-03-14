# nyx/validate_integration.py
# New file for validating governance integration

import logging
import asyncio
import inspect
from typing import Dict, List, Any, Set

from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType

logger = logging.getLogger(__name__)

async def validate_agent_governance_integration(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Validate that all agents are properly integrated with governance.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        
    Returns:
        Validation results
    """
    # Get governance system
    governance = await get_central_governance(user_id, conversation_id)
    
    # Get all registered agents
    registered_agents = governance.registered_agents
    
    # Required integration components
    required_components = [
        "check_permission", 
        "report_action",
        "handle_directive"
    ]
    
    results = {}
    
    # Check each agent
    for agent_type, agents in registered_agents.items():
        if isinstance(agents, dict):
            for agent_id, agent in agents.items():
                agent_key = f"{agent_type}_{agent_id}"
                results[agent_key] = check_agent_integration(agent, required_components)
        else:
            # Single agent
            agent_key = agent_type
            results[agent_key] = check_agent_integration(agents, required_components)
    
    # Calculate summary metrics
    total_agents = len(results)
    fully_integrated = sum(1 for r in results.values() if r.get("fully_integrated", False))
    missing_components = {}
    
    for agent_key, result in results.items():
        missing = result.get("missing_components", [])
        for component in missing:
            if component not in missing_components:
                missing_components[component] = 0
            missing_components[component] += 1
    
    summary = {
        "total_agents": total_agents,
        "fully_integrated": fully_integrated,
        "integration_rate": fully_integrated / total_agents if total_agents > 0 else 0,
        "missing_components": missing_components
    }
    
    return {
        "summary": summary,
        "results": results
    }

def check_agent_integration(agent, required_components: List[str]) -> Dict[str, Any]:
    """
    Check if an agent has the required governance integration components.
    
    Args:
        agent: The agent instance
        required_components: List of required component names
        
    Returns:
        Integration check results
    """
    # Get all methods and attributes of the agent
    all_methods = {}
    for name, method in inspect.getmembers(agent):
        if inspect.ismethod(method) or inspect.isfunction(method):
            all_methods[name] = method
    
    # Check for each required component
    found_components = []
    missing_components = []
    
    for component in required_components:
        # Check direct method
        if component in all_methods:
            found_components.append(component)
            continue
            
        # Check for method with component in name
        found = False
        for name in all_methods:
            if component in name:
                found_components.append(component)
                found = True
                break
                
        # Check for decorated methods
        if not found:
            for name, method in all_methods.items():
                if hasattr(method, "__wrapped__"):
                    # Check if the method is decorated with a governance decorator
                    original = method.__wrapped__
                    if hasattr(original, "__name__") and component in original.__name__:
                        found_components.append(component)
                        found = True
                        break
        
        if not found:
            missing_components.append(component)
    
    return {
        "fully_integrated": len(missing_components) == 0,
        "found_components": found_components,
        "missing_components": missing_components,
        "integration_score": len(found_components) / len(required_components)
    }

async def run_integration_validation():
    """
    Command-line utility to run integration validation.
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m nyx.validate_integration <user_id> <conversation_id>")
        return
    
    user_id = int(sys.argv[1])
    conversation_id = int(sys.argv[2])
    
    results = await validate_agent_governance_integration(user_id, conversation_id)
    
    # Print summary
    summary = results["summary"]
    print(f"INTEGRATION VALIDATION SUMMARY")
    print(f"===========================")
    print(f"Total agents: {summary['total_agents']}")
    print(f"Fully integrated: {summary['fully_integrated']} ({summary['integration_rate']*100:.1f}%)")
    print(f"Missing components:")
    for component, count in summary.get("missing_components", {}).items():
        print(f"  - {component}: {count} agents")
    
    print("\nDETAILED RESULTS")
    print("===============")
    for agent_key, result in results["results"].items():
        status = "✅ FULLY INTEGRATED" if result["fully_integrated"] else f"❌ INCOMPLETE ({result['integration_score']*100:.0f}%)"
        print(f"{agent_key}: {status}")
        if not result["fully_integrated"]:
            print(f"  Missing: {', '.join(result['missing_components'])}")
    
    # Return exit code based on validation success
    if summary["fully_integrated"] == summary["total_agents"]:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_integration_validation())
