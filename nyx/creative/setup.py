# creative/setup.py
import asyncio
from agentic_system import integrate_with_existing_system

async def setup_agentic_creativity(nyx_brain=None):
    # Integration with existing system if available
    system = await integrate_with_existing_system(nyx_brain)
    print(f"Agentic creativity system initialized at: {system.content_system.base_directory}")
    return system

if __name__ == "__main__":
    asyncio.run(setup_agentic_creativity())
