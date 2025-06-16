import asyncio
import datetime
import re
from agents import Agent, Runner, RunConfig, ModelSettings

class ReflectionAgent(Agent):
    """Simple agent producing reflective insights from event logs."""
    
    def __init__(self):
        super().__init__(
            name="ReflectionAgent",
            instructions="""You are Nyx's inner critic. Analyze the provided event logs and provide concise insights.
            
            Format your response as:
            - A brief insight about patterns or notable events
            - Include "priority: [low/normal/high]" to indicate importance
            
            Keep insights under 100 words and focus on actionable observations.""",
            model="gpt-4o-mini",  # or whatever model you prefer
            model_settings=ModelSettings(temperature=0.7, max_tokens=200)
        )

async def run_reflection(events: list) -> str:
    """Run reflection on events and return insight"""
    # Create the reflection agent
    agent = ReflectionAgent()
    
    # Format events into a prompt
    event_text = "\n".join(f"{e.get('role', 'unknown')}:{e.get('content', '')}" for e in events[-100:])
    
    prompt = f"""Analyze these recent events and provide insights:

{event_text}

What patterns do you notice? What should Nyx pay attention to?"""
    
    # Run the agent
    run_config = RunConfig(
        workflow_name="Simple Reflection",
        trace_id=f"reflection-{datetime.datetime.utcnow().isoformat()}"
    )
    
    result = await Runner.run(
        agent,
        prompt,
        run_config=run_config
    )
    
    # Extract the insight text
    insight = str(result.final_output) if result.final_output else "No insights generated."
    
    # Extract priority
    priority = _extract_priority(insight)
    
    # Store in memory (if MemoryManager is available)
    try:
        from nyx.core.memory.memory_manager import MemoryManager
        await MemoryManager.add(insight, {"type": "insight", "priority": priority})
    except ImportError:
        pass  # MemoryManager not available
    
    # Apply strategy if high priority
    if priority == "high":
        try:
            from strategy import manager as strategy_manager
            asyncio.create_task(strategy_manager.apply(insight))
        except Exception:
            pass
    
    return insight

def _extract_priority(text: str) -> str:
    """Extract priority level from text"""
    m = re.search(r"priority\s*:\s*(\w+)", text, re.IGNORECASE)
    return m.group(1).lower() if m and m.group(1).lower() in ["low", "normal", "high"] else "normal"

# Global state for scheduling
log_buffer: list = []
last_reflection_time: datetime.datetime = datetime.datetime.utcnow()

async def schedule_reflection() -> None:
    """Run reflection when buffer is large or too much time has passed."""
    global last_reflection_time
    
    current_time = datetime.datetime.utcnow()
    time_since_last = current_time - last_reflection_time
    
    if len(log_buffer) >= 100 or time_since_last > datetime.timedelta(hours=2):
        # Run reflection
        if log_buffer:  # Only run if there are events
            asyncio.create_task(run_reflection(log_buffer.copy()))
            log_buffer.clear()
        last_reflection_time = current_time

# Optional: Add event to buffer
def add_event(role: str, content: str) -> None:
    """Add an event to the reflection buffer"""
    log_buffer.append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.utcnow().isoformat()
    })
    
    # Check if we should trigger reflection
    asyncio.create_task(schedule_reflection())
