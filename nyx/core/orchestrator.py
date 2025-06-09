from nyx.core.memory.memory_manager import MemoryManager

async def prepare_context(ctx: str, user_msg: str) -> str:
    """Prepend relevant memories to context.

    Parameters
    ----------
    ctx: str
        Existing context or system prompt.
    user_msg: str
        Latest user message used to fetch relevant memories.
    Returns
    -------
    str
        Augmented context including a KNOWLEDGE section and memory comments.
    """
    hits = await MemoryManager.fetch_relevant(user_msg, k=5)
    if not hits:
        return ctx

    bullet_lines = "\n".join(
        "- " + (h["text"][:300] + ("…" if len(h["text"]) > 300 else ""))
        for h in hits
    )
    knowledge = f"KNOWLEDGE:\n{bullet_lines}\n"
    comments = "".join(
        f"<!--MEM:{h.get('meta', {}).get('uid')},{h.get('score',0):.2f}-->"
        for h in hits
    )
    return f"{knowledge}{comments}\n{ctx}"
