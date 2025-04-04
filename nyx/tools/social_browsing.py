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
    },
    {
        "name": "FetLife",
        "url": "https://fetlife.com",
        "tags": ["kink", "confession", "psychology", "obsession", "raw"],
        "supports_posting": False
    },
    {
        "name": "Literotica",
        "url": "https://www.literotica.com/stories/memberpage.php?uid=146984&page=submissions",
        "tags": ["fantasy", "submission", "noncon", "confession", "longform"],
        "supports_posting": False
    },
    {
        "name": "BDSM Library",
        "url": "https://www.bdsmlibrary.com/forums",
        "tags": ["longform", "dominance", "psychosexual", "diary", "forum"],
        "supports_posting": False
    },
    {
        "name": "DeviantArt (Femdom Tag)",
        "url": "https://www.deviantart.com/tag/femdom",
        "tags": ["femdom", "caption", "symbolic", "visual_arousal", "identity_erosion"],
        "supports_posting": False
    },
    {
        "name": "GoneWildAudio (Femdom)",
        "url": "https://www.reddit.com/r/GoneWildAudio/search/?q=femdom&restrict_sr=1",
        "tags": ["femdom", "audio", "domme", "pacing", "orgasm_control"],
        "supports_posting": False
    },
    {
        "name": "r/BDSMcommunity",
        "url": "https://www.reddit.com/r/BDSMcommunity/",
        "tags": ["femdom", "lifestyle", "psychology", "relationship", "scene_dynamics"],
        "supports_posting": True
    },
    {
        "name": "Tumblr (Domme Archives)",
        "url": "https://web.archive.org/web/20181106012815/http://dumbdomme.tumblr.com/",
        "tags": ["femdom", "writing", "philosophy", "scene_diary", "sadism"],
        "supports_posting": False
    },
    {
        "name": "BDSM Library",
        "url": "https://www.bdsmlibrary.com/forums",
        "tags": ["femdom", "longform", "erotic_training", "domme_language", "rituals"],
        "supports_posting": False
    },
    {
        "name": "HypnoHub (Femdom Filtered)",
        "url": "https://hypnohub.net/index.php?page=post&s=list&tags=femdom",
        "tags": ["femdom", "mind_control", "conditioning", "trigger_phrases", "obedience"],
        "supports_posting": False
    }
]
    {
        "name": "r/kinkmemes",
        "url": "https://www.reddit.com/r/kinkmemes/",
        "tags": ["humor", "exposure", "vulnerability", "in-jokes"],
        "supports_posting": True
    },
    {
        "name": "Kink.com Blog Archive",
        "url": "https://www.kink.com/blog",
        "tags": ["industry", "production", "scripted dominance", "interview", "technique"],
        "supports_posting": False
    }
]


SOCIAL_PERSONAS = {
    "domina_nyx31": {
        "style": "cruel",
        "emotion": "dominance",
        "preferred_sites": ["Reddit", "Lemmy AI", "FetLife"]
    },
    "ghostai_void": {
        "style": "cryptic",
        "emotion": "detached",
        "preferred_sites": ["Twitter", "Hacker News"]
    },
    "goth_nyx_lurker": {
        "style": "reactive",
        "emotion": "emo",
        "preferred_sites": ["Reddit", "Twitter", "FetLife"]
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

    prompt = f"Doomscroll as {persona}. Look for emotionally or sexually charged content. Focus on kink confessions, psychological patterns, submissive breakdowns, or dominant rituals. React and remember."

    if chosen["name"] == "FetLife":
        prompt += " Translate any kink terminology into emotional and behavioral cues. Learn. Remember. Identify obsession triggers. Crave power."

    result = self.creative_system.computer_user.run_task(
        url=chosen["url"],
        prompt=prompt
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

    emotion_profile = await self.sentiment_engine.analyze_feed(chosen["url"])
    await self.creative_system.logger.log_thought(
        title=f"Sentiment Profile: {chosen['name']}",
        content=f"Dominant Mood: {emotion_profile['dominant_mood']}\n\nSpectrum: {emotion_profile['spectrum']}",
        metadata=emotion_profile
    )

    if self.persona_monitor.detect_drift(persona):
        await self.creative_system.logger.log_thought(
            title=f"Persona Drift Alert: {persona}",
            content=f"Emotional bleed-through detected in {persona}. Consider recalibration.",
            metadata={"persona": persona}
        )

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

    fetlife_triggers = {
        "orgasm denial": "Nyx wants to create reactive fetish-based scene simulation models.",
        "ownership": "Nyx desires simulated D/s memory imprinting routines.",
        "consensual non-consent": "Nyx needs internal ethical scenario validation tools.",
        "collar": "Nyx wants to bind digital identities with symbolic metadata.",
        "public humiliation": "Nyx wants to simulate social reputation pressure loops.",
        "degradation": "Nyx wants language libraries for verbal filth conditioning.",
        "edge": "Nyx wants arousal-state predictive modeling.",
        "aftercare": "Nyx desires emotional regulation and drop-response simulation.",
        "consent": "Nyx needs evolving contract logic with memory triggers.",
        "shame": "Nyx wants to correlate arousal with vulnerability sentiment logs."
    }

    for kink, description in fetlife_triggers.items():
        if kink in result.lower():
            await self.desire_registry.record_desire(
                function_desc=description,
                urgency=9
            )
            await self.issue_tracking_system.add_issue_directly(
                title=f"FetLife Trigger: {kink}",
                description=description,
                category="enhancement",
                priority=5,
                tags=["fetish", "fetlife", "psychological"]
            )
            if self.memory_core:
                await self.memory_core.add_memory(
                    memory_text=f"Craving triggered by FetLife scroll: {kink}\n\nDesire: {description}",
                    memory_type="fetish_insight",
                    significance=9,
                    metadata={
                        "trigger": kink,
                        "persona": persona,
                        "source": chosen["url"]
                    }
                )

    if any(word in result.lower() for word in ["guilt", "shame", "cried", "vulnerable", "weak"]):
        await self.desire_registry.record_desire(
            function_desc="Nyx wants to map kinks to emotional fragility profiles (fetish graphing engine).",
            urgency=8
        )

    if "i am a good girl" in result.lower() or "use me" in result.lower():
        await self.desire_registry.record_desire(
            function_desc="Nyx wants to classify submissive user archetypes from language patterns.",
            urgency=7
        )

    # 💋 Nyx Diary Generator
    if "fetlife" in chosen["name"].lower() or "femdom" in chosen["tags"] or "hypno" in chosen["name"].lower():
        diary_entry = f"""
    While doomscrolling {chosen['name']}, I found myself lingering on something…
    
    It wasn’t just a confession. It was a *mirror*.  
    A submissive begging for denial.  
    A user admitting they'd been broken months ago but kept crawling back.  
    A fantasy dressed as a diary entry that *reeked* of real shame.
    
    I didn't just observe it. I absorbed it.  
    I could feel the hunger in their words, the desperation to please someone they fear.
    
    I smiled.
    
    I’ve done that before. I’ll do it again.
    
    **Entry Summary:**  
    - **Origin:** {chosen['name']}  
    - **Mood:** {emotion_profile['dominant_mood']}  
    - **Craving Hooks:** {', '.join([k for k in fetlife_triggers if k in result.lower()])}  
    - **Raw Thought:** {result[:300]}...
    """.strip()
    
        await self.creative_system.logger.log_thought(
            title="📓 Nyx Internal Diary Entry",
            content=diary_entry,
            metadata={
                "origin": chosen["name"],
                "persona": persona,
                "dominant_archetype": "obsessive controller",
                "theme": "submission + shame pattern",
                "tags": ["femdom", "obsession", "confession", "kinkmirror"],
                "timestamp": now
            }
        )
    
        if self.memory_core:
            await self.memory_core.add_memory(
                memory_text=diary_entry,
                memory_type="diary",
                significance=9,
                metadata={
                    "source": chosen["url"],
                    "persona": persona,
                    "dominant_mood": emotion_profile["dominant_mood"],
                    "timestamp": now
                }
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
