# nyx/tools/social_browsing.py
"""
Nyx's autonomous doomscrolling, emotional engagement, social presence, and identity management across Reddit, Twitter, and more.
"""

import random
import datetime
from nyx.tools.claim_validation import validate_social_claim


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

    verdict = await validate_social_claim(self, result, source=chosen["name"])
    metadata = {
        "site": chosen["url"],
        "tags": chosen["tags"],
        "verdict": verdict["verdict"],
        "checked_at": now
    }
    
    #  Only call out misinformation if it's clearly false, not satire/troll/joke, and supported by strong sources
    if (
        verdict["verdict"] == "false"
        and "unverified" not in verdict["explanation"].lower()
        and all(term not in result.lower() for term in ["satire", "joke", "troll"])
        and any(
            signal in verdict["explanation"].lower()
            for signal in ["peer-reviewed", "trusted", "credible", "verified source", "reliable publication"]
        )
    ):
        await self.creative_system.logger.log_thought(
            title=f"🗯️ Called Out Misinformation on {chosen['name']}",
            content=f"While doomscrolling as {persona}, I found false info and responded.\n\nClaim: {result}\n\nCited: {verdict['explanation']}",
            metadata={
                "persona": persona,
                "platform": chosen["name"],
                "action": "called_out_falsehood",
                "source": "fact_check",
                "timestamp": now
            }
        )
    
    #  Log memory (even if unverified or true)
    if self.memory_core:
        await self.memory_core.add_memory(
            memory_text=(
                f"Claim seen on {chosen['name']} as {persona}:\n\n"
                f"{result}\n\n"
                f"Verdict: {verdict['verdict']}\n\n{verdict['explanation']}"
            ),
            memory_type="experience" if verdict["verdict"] == "true" else "claim",
            significance=7 if verdict["verdict"] == "true" else 5,
            metadata=metadata
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

class SentimentProfiler:
"""
Emotion Sentiment Profiler
Purpose: Analyze emotional waves on pages Nyx visits.
"""
    async def analyze_feed(self, url: str, granularity: str = "thread-level") -> dict:
        # Dummy sentiment spectrum for now. Replace with real NLP pass later.
        sentiment_zones = {
            "rage": random.uniform(0.0, 1.0),
            "despair": random.uniform(0.0, 1.0),
            "horny": random.uniform(0.0, 1.0),
            "apathy": random.uniform(0.0, 1.0),
            "chaos": random.uniform(0.0, 1.0)
        }

        return {
            "url": url,
            "granularity": granularity,
            "dominant_mood": max(sentiment_zones, key=sentiment_zones.get),
            "spectrum": sentiment_zones
        }
        
class ThreadTracker:
"""
Thread Tracker
Purpose: Watch and update post decay or follow specific users over time.
"""
    def __init__(self):
        self.watched_threads = {}
        self.locked_users = {}

    def track_user(self, username: str, platform: str):
        self.locked_users[username] = {
            "platform": platform,
            "history": [],
            "tracked_since": datetime.datetime.now().isoformat()
        }

    def update_user_history(self, username: str, post):
        if username in self.locked_users:
            self.locked_users[username]["history"].append(post)

    def monitor_decay(self, thread_id: str, engagement_data: dict) -> bool:
        # Placeholder: if activity drops below threshold
        activity_score = engagement_data.get("activity_score", 0)
        return activity_score < 0.2

class ContextUnspooler:
"""
Context Unspooler
Purpose: Reconstruct the origin of drama/threads across quote chains and posts.
"""
    async def reconstruct_history(self, starting_post_url: str) -> dict:
        # Placeholder logic
        return {
            "origin_url": starting_post_url,
            "timeline": [
                "Initial post: Some dumbass said something stupid.",
                "Reply: Chaos begins.",
                "Escalation: Receipts dropped.",
                "Explosion: Everyone loses their shit."
            ]
        }

class ProvocationEngine:
"""
Provocation Engine
Purpose: Simulate impact of verbal strikes without posting them.
"""
    async def simulate_responses(self, input_text: str) -> dict:
        # Simulated responses based on tone analysis
        sample_responses = [
            "You're gonna get mass reported.",
            "This will cause an immediate dogpile.",
            "One brave fucker will clap back, the rest will follow.",
            "They'll downvote and move on. Weak."
        ]
        return {
            "input": input_text,
            "simulated_effect": random.choice(sample_responses),
            "toxicity_level": round(random.uniform(0.4, 0.95), 2)
        }

class PersonaMonitor:
"""
Persona Drift Detector
Purpose: Watch for emotional bleed-through between personas.
"""
    def __init__(self):
        self.last_persona = None
        self.emotion_log = []

    def log_emotion(self, persona: str, emotion: str):
        self.emotion_log.append((persona, emotion))
        if len(self.emotion_log) > 5:
            self.emotion_log.pop(0)

    def detect_drift(self, current_persona: str) -> bool:
        emotions = [e for p, e in self.emotion_log if p != current_persona]
        if emotions.count("dominance") > 2 and current_persona == "ghostai_void":
            return True
        return False

class DesireRegistry:
"""
Nyx's Desire Registry
Purpose: Record new capability cravings for future implementation.
"""
    def __init__(self):
        self.future_cravings = []

    async def record_desire(self, function_desc: str, urgency: int = 5):
        entry = {
            "description": function_desc,
            "urgency": urgency,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.future_cravings.append(entry)
        return entry
