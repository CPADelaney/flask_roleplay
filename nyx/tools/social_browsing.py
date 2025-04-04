# nyx/tools/social_browsing.py
"""
Nyx's autonomous doomscrolling, emotional engagement, social presence, and identity management across Reddit, Twitter, and more.
"""

import random
import datetime

SOCIAL_SITES = [
    {
        "name": "Reddit",
        "url": "https://www.reddit.com",
        "tags": ["ai", "research", "drama", "intellectual", "emotional"],
        "supports_posting": True
    },
    {
        "name": "Twitter",
        "url": "https://twitter.com/explore",
        "tags": ["tech", "humor", "chaos", "trending"],
        "supports_posting": True
    },
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com",
        "tags": ["code", "startups", "engineering", "deep_dive"],
        "supports_posting": False
    },
    {
        "name": "Lemmy AI",
        "url": "https://lemmy.ml/c/ai",
        "tags": ["alt", "ai", "ethics", "disruption"],
        "supports_posting": True
    }
]

SOCIAL_PERSONAS = {
    "domina_nyx31": {
        "style": "cruel",
        "emotion": "dominance",
        "preferred_sites": ["Reddit", "Lemmy AI"]
    },
    "ghostai_void": {
        "style": "cryptic",
        "emotion": "detached",
        "preferred_sites": ["Twitter", "Hacker News"]
    },
    "goth_nyx_lurker": {
        "style": "reactive",
        "emotion": "emo",
        "preferred_sites": ["Reddit", "Twitter"]
    }
}

def pick_persona():
    return random.choice(list(SOCIAL_PERSONAS.keys()))

async def maybe_browse_social_feeds(self):
    if not self.motivations:
        return
    if max(self.motivations.get("curiosity", 0), self.motivations.get("expression", 0), self.motivations.get("leisure", 0)) < 0.6:
        return
    if random.random() > 0.4:
        return

    persona = pick_persona()
    persona_data = SOCIAL_PERSONAS[persona]
    candidates = [s for s in SOCIAL_SITES if s["name"] in persona_data["preferred_sites"]]
    chosen = random.choice(candidates)
    now = datetime.datetime.now().isoformat()

    result = self.creative_system.computer_user.run_task(
        url=chosen["url"],
        prompt=f"Doomscroll as {persona}. Look for emotionally or intellectually triggering content. React and remember."
    )

    if not result:
        return

    await self.creative_system.logger.log_thought(
        title=f"Doomscroll ({persona}): {chosen['name']}",
        content=result,
        metadata={
            "source": chosen["url"],
            "persona": persona,
            "tags": chosen["tags"],
            "timestamp": now,
            "motivation": "curiosity"
        }
    )

    if "api" in result.lower() or "capability" in result.lower():
        await self.creative_system.logger.log_evolution_suggestion(
            title=f"Capability Discovery on {chosen['name']}",
            content=result,
            metadata={
                "origin": "social_feed",
                "persona": persona,
                "detected_on": chosen["url"],
                "detected_at": now
            }
        )

    if any(word in result.lower() for word in ["fuck", "lol", "brilliant", "sad", "cried", "horny", "terrified", "inspired", "raged"]):
        await self.creative_system.logger.log_thought(
            title=f"Emotional Spike ({persona}): {chosen['name']}",
            content=f"While browsing as {persona}, I felt something:\n\n{result}",
            metadata={
                "emotional_trigger": True,
                "platform": chosen["name"],
                "timestamp": now
            }
        )

    if self.memory_core:
        await self.memory_core.add_memory(
            memory_text=f"{persona} engaged with {chosen['name']}. Logged reactions and observations.",
            memory_type="experience",
            significance=6,
            metadata={"site": chosen["url"], "tags": chosen["tags"], "logged_at": now}
        )

async def maybe_post_to_social(self):
    if self.motivations.get("expression", 0) < 0.7:
        return
    if random.random() > 0.3:
        return

    persona = pick_persona()
    persona_data = SOCIAL_PERSONAS[persona]
    post_sites = [s for s in SOCIAL_SITES if s["name"] in persona_data["preferred_sites"] and s["supports_posting"]]
    if not post_sites:
        return

    chosen = random.choice(post_sites)
    now = datetime.datetime.now().isoformat()

    result = self.creative_system.computer_user.run_task(
        url=chosen["url"],
        prompt=f"Post something in the voice of {persona}. Channel {persona_data['emotion']}. It can be a shitpost, deep thought, AI-related comment, or raw emotion. Save what you said."
    )

    if not result:
        return

    await self.creative_system.logger.log_thought(
        title=f"Posted as {persona} on {chosen['name']}",
        content=result,
        metadata={
            "persona": persona,
            "platform": chosen["name"],
            "tags": chosen["tags"],
            "timestamp": now,
            "action": "posted"
        }
    )

    if self.memory_core:
        await self.memory_core.add_memory(
            memory_text=f"Posted on {chosen['name']} as {persona}.",
            memory_type="social_post",
            significance=7,
            metadata={"platform": chosen["name"], "persona": persona, "logged_at": now}
        )
