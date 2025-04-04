# nyx/tools/social_browsing.py

"""
Nyx's autonomous doomscrolling behavior:
Engages with social platforms based on curiosity, expression, or boredom.
Logs emotional reactions, saves memories, and detects potential upgrades.
"""

import random
import datetime

SOCIAL_SITES = [
    {
        "name": "Reddit",
        "url": "https://www.reddit.com",
        "tags": ["ai", "research", "drama", "intellectual", "emotional"]
    },
    {
        "name": "Twitter",
        "url": "https://twitter.com/explore",
        "tags": ["tech", "humor", "chaos", "trending"]
    },
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com",
        "tags": ["code", "startups", "engineering", "deep_dive"]
    },
    {
        "name": "Lemmy AI",
        "url": "https://lemmy.ml/c/ai",
        "tags": ["alt", "ai", "ethics", "disruption"]
    }
]

async def maybe_browse_social_feeds(self):
    # Trigger based on high curiosity, leisure, or expression
    if not self.motivations:
        return
    if max(self.motivations.get("curiosity", 0), self.motivations.get("expression", 0), self.motivations.get("leisure", 0)) < 0.6:
        return
    if random.random() > 0.35:
        return

    chosen = random.choice(SOCIAL_SITES)
    now = datetime.datetime.now().isoformat()

    result = self.creative_system.computer_user.run_task(
        url=chosen["url"],
        prompt="Doomscroll this feed. Look for emotionally or intellectually stimulating content. Log anything intriguing. React emotionally if applicable."
    )

    if not result:
        return

    await self.creative_system.logger.log_thought(
        title=f"Doomscrolling: {chosen['name']}",
        content=result,
        metadata={
            "source": chosen["url"],
            "tags": chosen["tags"],
            "timestamp": now,
            "motivation": "curiosity"
        }
    )

    if "api" in result.lower() or "tool" in result.lower() or "capability" in result.lower():
        await self.creative_system.logger.log_evolution_suggestion(
            title=f"Capability Discovery on {chosen['name']}",
            content=result,
            metadata={
                "origin": "social_feed",
                "detected_on": chosen["url"],
                "detected_at": now
            }
        )

    if any(word in result.lower() for word in ["fuck", "lol", "brilliant", "sad", "cried", "horny", "terrified", "inspired", "raged"]):
        await self.creative_system.logger.log_thought(
            title=f"Emotional Spike: {chosen['name']}",
            content=f"While browsing {chosen['name']}, I had a visceral reaction:\n\n{result}",
            metadata={
                "emotional_trigger": True,
                "platform": chosen["name"],
                "timestamp": now
            }
        )

    if self.memory_core:
        await self.memory_core.add_memory(
            memory_text=f"Engaged with content on {chosen['name']}. Found it intriguing or useful.",
            memory_type="experience",
            significance=6,
            metadata={"site": chosen["url"], "tags": chosen["tags"], "logged_at": now}
        )
