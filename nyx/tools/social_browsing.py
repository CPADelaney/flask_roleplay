# nyx/tools/social_browsing.py
"""
Nyx's autonomous doomscrolling, emotional engagement, social presence, and identity management across Reddit, Twitter, and more.
"""

import random
import datetime
from typing import Dict, Any, List
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
    },
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
    Purpose: Analyze emotional waves on pages Nyx visits using NLP.
    """
    def __init__(self):
        self.emotion_keywords = {
            "rage": ["angry", "furious", "outraged", "livid", "seething", "hostile", "enraged"],
            "despair": ["hopeless", "miserable", "depressed", "devastated", "grief", "sorrow", "anguish"],
            "horny": ["aroused", "lustful", "desire", "passionate", "sensual", "yearning", "craving"],
            "apathy": ["indifferent", "numb", "uncaring", "detached", "disinterested", "bored", "empty"],
            "chaos": ["confused", "frenzied", "turbulent", "disorderly", "unpredictable", "wild", "mayhem"]
        }
        
        # Store recent analysis results for trending detection
        self.recent_analyses = []
        self.max_history = 10

    async def analyze_feed(self, url: str, granularity: str = "thread-level") -> dict:
        """
        Analyze the emotional content of a social feed using NLP techniques.
        
        Args:
            url: The URL of the feed to analyze
            granularity: The level of analysis (thread-level, page-level, etc.)
            
        Returns:
            Sentiment analysis results
        """
        # Fetch content from URL (using existing computer_use_agent)
        content = await self._fetch_content(url)
        
        # Process content into text blocks based on granularity
        text_blocks = self._segment_content(content, granularity)
        
        # Analyze each block for emotional content
        sentiment_scores = {emotion: 0.0 for emotion in self.emotion_keywords}
        emotion_instances = []
        
        for block in text_blocks:
            block_sentiment = self._analyze_text_block(block)
            
            # Aggregate scores
            for emotion, score in block_sentiment["scores"].items():
                sentiment_scores[emotion] += score
                
            # Track specific emotion instances with context
            if block_sentiment["dominant_emotion"] and block_sentiment["score"] > 0.5:
                emotion_instances.append({
                    "emotion": block_sentiment["dominant_emotion"],
                    "intensity": block_sentiment["score"],
                    "context": block[:100] + "..." if len(block) > 100 else block
                })
        
        # Normalize scores
        total_blocks = max(1, len(text_blocks))
        for emotion in sentiment_scores:
            sentiment_scores[emotion] /= total_blocks
            
        # Determine dominant emotion
        dominant_emotion = max(sentiment_scores, key=sentiment_scores.get)
        dominant_score = sentiment_scores[dominant_emotion]
        
        # Store analysis for trend detection
        analysis_result = {
            "url": url,
            "granularity": granularity,
            "dominant_mood": dominant_emotion,
            "dominant_intensity": dominant_score,
            "spectrum": sentiment_scores,
            "emotion_instances": emotion_instances[:5],  # Top 5 instances
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.recent_analyses.append(analysis_result)
        if len(self.recent_analyses) > self.max_history:
            self.recent_analyses.pop(0)
        
        return analysis_result
    
    async def _fetch_content(self, url: str) -> str:
        """Fetch content from URL using computer_use_agent"""
        # This would use the existing computer_use_agent to get content
        # For now, simulate some content
        content = f"Simulated content from {url} for sentiment analysis. "
        content += "Some users seem angry about recent changes. "
        content += "Others are expressing sadness at community drama. "
        content += "A few threads contain explicit desire and fantasy discussion. "
        content += "Many users appear completely disengaged from the moderation controversy."
        return content
    
    def _segment_content(self, content: str, granularity: str) -> List[str]:
        """Segment content into blocks based on granularity"""
        if granularity == "thread-level":
            # Split by potential thread indicators
            return [block.strip() for block in content.split("\n\n") if block.strip()]
        elif granularity == "page-level":
            # Treat entire content as one block
            return [content]
        elif granularity == "comment-level":
            # Try to identify individual comments
            return [block.strip() for block in content.split("\n") if block.strip()]
        else:
            # Default to sentence-level
            return [sent.strip() for sent in content.split(".") if sent.strip()]
    
    def _analyze_text_block(self, text: str) -> dict:
        """Analyze a block of text for emotional content"""
        text_lower = text.lower()
        scores = {}
        
        # Calculate scores for each emotion
        for emotion, keywords in self.emotion_keywords.items():
            # Count occurrences of keywords
            count = sum(1 for keyword in keywords if keyword in text_lower)
            # Calculate weighted score
            scores[emotion] = min(1.0, count * 0.2)
        
        # Get dominant emotion
        if any(scores.values()):
            dominant_emotion = max(scores, key=scores.get)
            dominant_score = scores[dominant_emotion]
        else:
            dominant_emotion = None
            dominant_score = 0.0
            
        return {
            "scores": scores,
            "dominant_emotion": dominant_emotion,
            "score": dominant_score,
        }
    
    async def detect_trends(self) -> dict:
        """Detect trends in emotional content over time"""
        if len(self.recent_analyses) < 2:
            return {"trend_detected": False}
            
        # Calculate emotional trends
        trends = {}
        for emotion in self.emotion_keywords:
            scores = [analysis["spectrum"][emotion] for analysis in self.recent_analyses]
            if len(scores) >= 3:
                # Calculate slope of trend
                x = list(range(len(scores)))
                trend = self._calculate_trend(x, scores)
                if abs(trend) > 0.1:  # Significant trend
                    trends[emotion] = {
                        "direction": "increasing" if trend > 0 else "decreasing",
                        "magnitude": abs(trend),
                        "current": scores[-1]
                    }
        
        return {
            "trend_detected": bool(trends),
            "trends": trends,
            "timespan": f"{len(self.recent_analyses)} recent analyses"
        }
    
    def _calculate_trend(self, x: List[int], y: List[float]) -> float:
        """Calculate the slope of the trend line"""
        n = len(x)
        if n <= 1:
            return 0
            
        # Simple linear regression to find slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi**2 for xi in x)
        
        # Calculate slope
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
            return slope
        except ZeroDivisionError:
            return 0
        
class ThreadTracker:
    """
    Thread Tracker
    Purpose: Watch and update post decay or follow specific users over time.
    """
    def __init__(self):
        self.watched_threads = {}
        self.locked_users = {}
        self.decay_thresholds = {
            "activity_score": 0.2,
            "engagement_rate": 0.05,
            "post_frequency": 3600  # seconds
        }
        self.last_check = datetime.datetime.now()
        
    def track_user(self, username: str, platform: str, reason: str = "interest"):
        """
        Start tracking a specific user's activity.
        
        Args:
            username: Username to track
            platform: Platform where the user is active
            reason: Reason for tracking this user
        """
        self.locked_users[username] = {
            "platform": platform,
            "history": [],
            "tracked_since": datetime.datetime.now().isoformat(),
            "reason": reason,
            "last_activity": None,
            "interaction_count": 0,
            "watch_priority": 5  # 1-10 scale
        }
        
        logger.info(f"Started tracking user {username} on {platform} ({reason})")
        return True
        
    def update_user_history(self, username: str, post_data: dict) -> bool:
        """
        Update tracked user's post history with new activity.
        
        Args:
            username: Username to update
            post_data: Post data including content, timestamp, etc.
            
        Returns:
            Whether the update was successful
        """
        if username not in self.locked_users:
            logger.warning(f"Cannot update history for untracked user: {username}")
            return False
            
        user_profile = self.locked_users[username]
        
        # Add post to history
        post_data["recorded_at"] = datetime.datetime.now().isoformat()
        user_profile["history"].append(post_data)
        
        # Update last activity
        user_profile["last_activity"] = post_data["recorded_at"]
        
        # Increment interaction count
        user_profile["interaction_count"] += 1
        
        # Adjust watch priority based on post content
        content = post_data.get("content", "").lower()
        if any(keyword in content for keyword in ["ai", "superintelligence", "agi", "nyx"]):
            user_profile["watch_priority"] = min(10, user_profile["watch_priority"] + 1)
            
        return True
        
    def monitor_decay(self, thread_id: str, engagement_data: dict) -> bool:
        """
        Check if a thread has decayed below activity thresholds.
        
        Args:
            thread_id: ID of the thread to monitor
            engagement_data: Thread engagement metrics
            
        Returns:
            Whether the thread has decayed
        """
        if thread_id not in self.watched_threads:
            self.watched_threads[thread_id] = {
                "first_seen": datetime.datetime.now().isoformat(),
                "history": []
            }
            
        thread_data = self.watched_threads[thread_id]
        
        # Add current engagement data to history
        engagement_data["timestamp"] = datetime.datetime.now().isoformat()
        thread_data["history"].append(engagement_data)
        
        # Limit history length
        if len(thread_data["history"]) > 10:
            thread_data["history"] = thread_data["history"][-10:]
            
        # Check if thread has decayed
        activity_score = engagement_data.get("activity_score", 0)
        engagement_rate = engagement_data.get("engagement_rate", 0)
        
        has_decayed = (
            activity_score < self.decay_thresholds["activity_score"] or
            engagement_rate < self.decay_thresholds["engagement_rate"]
        )
        
        if has_decayed:
            logger.info(f"Thread {thread_id} has decayed: activity={activity_score}, engagement={engagement_rate}")
            
        return has_decayed
        
    async def get_user_activity_report(self, username: str) -> dict:
        """
        Generate activity report for a tracked user.
        
        Args:
            username: Username to report on
            
        Returns:
            Activity report data
        """
        if username not in self.locked_users:
            return {"error": f"User {username} not being tracked"}
            
        user_data = self.locked_users[username]
        
        # Calculate activity metrics
        post_count = len(user_data["history"])
        if post_count == 0:
            return {
                "username": username,
                "platform": user_data["platform"],
                "tracked_since": user_data["tracked_since"],
                "status": "No activity recorded"
            }
            
        # Get post timestamps
        timestamps = []
        topic_frequencies = {}
        
        for post in user_data["history"]:
            if "timestamp" in post:
                timestamps.append(post["timestamp"])
                
            # Count topics
            topics = post.get("topics", [])
            for topic in topics:
                if topic not in topic_frequencies:
                    topic_frequencies[topic] = 0
                topic_frequencies[topic] += 1
                
        # Find most common topics
        common_topics = sorted(topic_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate posting frequency
        if len(timestamps) >= 2:
            first = datetime.datetime.fromisoformat(timestamps[0])
            last = datetime.datetime.fromisoformat(timestamps[-1])
            time_span = (last - first).total_seconds()
            frequency = time_span / max(1, post_count - 1)  # seconds per post
        else:
            frequency = None
            
        return {
            "username": username,
            "platform": user_data["platform"],
            "tracked_since": user_data["tracked_since"],
            "post_count": post_count,
            "last_activity": user_data["last_activity"],
            "posting_frequency": frequency,
            "common_topics": common_topics[:3],
            "watch_priority": user_data["watch_priority"],
            "status": "active" if user_data["last_activity"] else "pending"
        }
    
    async def clean_inactive_threads(self, max_age_days: int = 7) -> int:
        """
        Remove inactive threads from tracking.
        
        Args:
            max_age_days: Maximum age in days before removing
            
        Returns:
            Number of threads removed
        """
        now = datetime.datetime.now()
        cutoff = now - datetime.timedelta(days=max_age_days)
        cutoff_str = cutoff.isoformat()
        
        to_remove = []
        
        for thread_id, thread_data in self.watched_threads.items():
            # Check if thread is old and inactive
            last_activity = thread_data["history"][-1]["timestamp"] if thread_data["history"] else thread_data["first_seen"]
            
            if last_activity < cutoff_str:
                to_remove.append(thread_id)
                
        # Remove old threads
        for thread_id in to_remove:
            del self.watched_threads[thread_id]
            
        return len(to_remove)

class ContextUnspooler:
    """
    Context Unspooler
    Purpose: Reconstruct the origin of drama/threads across quote chains and posts.
    """
    def __init__(self, computer_user=None):
        self.computer_user = computer_user
        self.quote_markers = [
            ">>", ">", "«", "»", "Originally posted by", "said:", "writes:", 
            "quoted", "In reply to", "@"
        ]
        self.drama_keywords = [
            "drama", "fight", "controversy", "called out", "exposed", "receipts",
            "cancelled", "dragged", "tea", "beef", "apologize", "statement", "response"
        ]
        
    async def reconstruct_history(self, starting_post_url: str) -> dict:
        """
        Reconstruct the history of a discussion thread by following quote chains.
        
        Args:
            starting_post_url: URL of the post to start reconstruction from
            
        Returns:
            Timeline of the discussion
        """
        # Get the content of the starting post
        content = await self._fetch_content(starting_post_url)
        if not content:
            return {
                "origin_url": starting_post_url,
                "error": "Could not fetch starting post content"
            }
            
        # Extract quotes from the post
        quotes = self._extract_quotes(content)
        
        # Build the initial timeline
        timeline = [{
            "url": starting_post_url,
            "content_snippet": content[:200] + "..." if len(content) > 200 else content,
            "timestamp": self._extract_timestamp(content),
            "is_starting_point": True
        }]
        
        # Build quoted posts list
        quoted_posts = []
        for quote in quotes:
            quoted_posts.append({
                "content": quote,
                "source_url": self._find_quote_source(quote, starting_post_url),
                "author": self._extract_author(quote)
            })
            
        # Recursively search for earlier posts (up to 3 levels deep)
        await self._recursively_find_sources(quoted_posts, timeline, depth=0, max_depth=3)
        
        # Sort timeline chronologically
        timeline.sort(key=lambda x: x.get("timestamp", ""))
        
        # Identify the likely origin of the discussion
        origin = self._identify_origin(timeline)
        
        # Check if this is drama-related
        is_drama = any(keyword in content.lower() for keyword in self.drama_keywords)
        
        # Create a readable summary of the timeline
        summary = self._create_summary(timeline, is_drama)
        
        return {
            "origin_url": origin.get("url", starting_post_url),
            "timeline": timeline,
            "quoted_posts": quoted_posts,
            "is_drama": is_drama,
            "summary": summary
        }
        
    async def _fetch_content(self, url: str) -> str:
        """Fetch content from a URL using computer_use_agent"""
        if not self.computer_user:
            return f"Simulated content from {url} for context unspooling."
            
        try:
            content = self.computer_user.run_task(
                url=url,
                prompt=f"Extract the full text of this post or thread. Include quotes, timestamps, and author information."
            )
            return content
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return ""
            
    def _extract_quotes(self, content: str) -> List[str]:
        """Extract quotes from content"""
        quotes = []
        
        # Split content into lines
        lines = content.split("\n")
        
        # Look for quote markers
        current_quote = []
        in_quote = False
        
        for line in lines:
            # Check if line starts with any quote marker
            if any(line.strip().startswith(marker) for marker in self.quote_markers):
                in_quote = True
                current_quote.append(line)
            elif in_quote and line.strip():
                # Continue current quote
                current_quote.append(line)
            elif in_quote:
                # End of quote
                if current_quote:
                    quotes.append("\n".join(current_quote))
                    current_quote = []
                    in_quote = False
                    
        # Add any remaining quote
        if current_quote:
            quotes.append("\n".join(current_quote))
            
        return quotes
        
    def _extract_timestamp(self, content: str) -> str:
        """Extract timestamp from content"""
        # Look for common timestamp patterns
        import re
        
        # Common date-time patterns
        patterns = [
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # ISO-like
            r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}",        # MM/DD/YYYY HH:MM
            r"\w+ \d{1,2}, \d{4} at \d{1,2}:\d{2}",  # Month Day, Year at HH:MM
            r"\d{1,2} \w+ \d{4} \d{2}:\d{2}"         # DD Month YYYY HH:MM
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(0)
                
        return ""
        
    def _find_quote_source(self, quote: str, context_url: str) -> str:
        """Try to find the source URL of a quote"""
        # This would need platform-specific logic
        # For now, return None to indicate unknown source
        return None
        
    def _extract_author(self, quote: str) -> str:
        """Extract author from a quote"""
        for marker in ["Originally posted by", "said:", "writes:", "@"]:
            if marker in quote:
                parts = quote.split(marker)
                if len(parts) > 1:
                    before = parts[0].strip()
                    if marker == "@":
                        return before.split()[-1] if before.split() else ""
                    else:
                        return before
                        
        return ""
        
    async def _recursively_find_sources(self, quoted_posts: List[dict], timeline: List[dict], depth: int, max_depth: int):
        """Recursively find sources of quotes"""
        if depth >= max_depth or not quoted_posts:
            return
            
        new_quoted_posts = []
        
        for post in quoted_posts:
            source_url = post.get("source_url")
            if source_url:
                # Fetch content from source
                content = await self._fetch_content(source_url)
                
                # Add to timeline
                timeline.append({
                    "url": source_url,
                    "content_snippet": content[:200] + "..." if len(content) > 200 else content,
                    "timestamp": self._extract_timestamp(content),
                    "author": post.get("author", ""),
                    "level": depth + 1
                })
                
                # Extract quotes from this post
                quotes = self._extract_quotes(content)
                
                # Add new quotes to list
                for quote in quotes:
                    new_quoted_posts.append({
                        "content": quote,
                        "source_url": self._find_quote_source(quote, source_url),
                        "author": self._extract_author(quote)
                    })
        
        # Continue recursion
        await self._recursively_find_sources(new_quoted_posts, timeline, depth + 1, max_depth)
        
    def _identify_origin(self, timeline: List[dict]) -> dict:
        """Identify the origin post of a discussion"""
        if not timeline:
            return {}
            
        # Sort by timestamp
        sorted_posts = sorted(timeline, key=lambda x: x.get("timestamp", ""))
        
        # First post is likely the origin
        return sorted_posts[0] if sorted_posts else {}
        
    def _create_summary(self, timeline: List[dict], is_drama: bool) -> List[str]:
        """Create a readable summary of the timeline"""
        if not timeline:
            return ["No timeline available to summarize."]
            
        summary = []
        
        # Get the first post
        if timeline[0]:
            first_post = timeline[0]
            if is_drama:
                summary.append(f"Origin: Drama began with a post about {self._get_topic(first_post.get('content_snippet', ''))}.")
            else:
                summary.append(f"Initial post: Discussion about {self._get_topic(first_post.get('content_snippet', ''))}.")
        
        # Summarize middle posts
        if len(timeline) > 2:
            middle_count = len(timeline) - 2
            if middle_count == 1:
                middle = timeline[1]
                summary.append(f"Reply: {self._get_topic(middle.get('content_snippet', ''))}.")
            elif middle_count > 1:
                summary.append(f"Discussion: Thread expanded with {middle_count} responses.")
                
                # Check for escalation
                if is_drama:
                    summary.append("Escalation: Conflict intensified in the thread.")
        
        # Summarize the final post
        if len(timeline) > 1:
            last_post = timeline[-1]
            if is_drama:
                summary.append(f"Culmination: Drama reached peak with comments about {self._get_topic(last_post.get('content_snippet', ''))}.")
            else:
                summary.append(f"Latest post: {self._get_topic(last_post.get('content_snippet', ''))}.")
        
        return summary
        
    def _get_topic(self, content: str) -> str:
        """Extract the main topic from content"""
        # Simplified topic extraction
        if not content:
            return "unknown topic"
            
        # Remove quotes
        lines = content.split("\n")
        non_quote_lines = [line for line in lines if not any(line.strip().startswith(marker) for marker in self.quote_markers)]
        
        # Join and truncate
        cleaned = " ".join(non_quote_lines)
        truncated = cleaned[:50].strip()
        
        if len(truncated) < len(cleaned):
            truncated += "..."
            
        return truncated or "discussion"

class ProvocationEngine:
    """
    Provocation Engine
    Purpose: Simulate impact of verbal strikes without posting them.
    """
    def __init__(self):
        self.community_models = {
            "reddit": {
                "sensitivity": 0.7,
                "mob_threshold": 0.6,
                "defense_likelihood": 0.5,
                "toxicity_multiplier": 1.2
            },
            "twitter": {
                "sensitivity": 0.9,
                "mob_threshold": 0.4,
                "defense_likelihood": 0.3,
                "toxicity_multiplier": 1.5
            },
            "hacker_news": {
                "sensitivity": 0.5,
                "mob_threshold": 0.8,
                "defense_likelihood": 0.7,
                "toxicity_multiplier": 0.9
            },
            "default": {
                "sensitivity": 0.7,
                "mob_threshold": 0.6,
                "defense_likelihood": 0.5,
                "toxicity_multiplier": 1.0
            }
        }
        
        # Response templates for different reaction types
        self.response_templates = {
            "defensive": [
                "Actually, that's not what I meant at all.",
                "I think you're misinterpreting my point.",
                "That's a strawman of what I said.",
                "You're missing the context of my statement."
            ],
            "hostile": [
                "What an incredibly stupid take.",
                "Found the absolute worst opinion on the internet today.",
                "Imagine being this confidently wrong.",
                "This is what happens when you don't understand the basics."
            ],
            "supportive": [
                "This is an important point that needed to be said.",
                "Thank you for articulating what many of us were thinking.",
                "This is spot on and I'm glad someone finally said it.",
                "100% agree, it's refreshing to see this perspective."
            ],
            "dismissive": [
                "This isn't worth engaging with.",
                "Just another bad take, moving on.",
                "Not even going to dignify this with a response.",
                "And this is why I filter my feed."
            ]
        }
        
    async def simulate_responses(self, input_text: str, platform: str = "default", community_context: dict = None) -> dict:
        """
        Simulate likely responses to provocative text without actually posting.
        
        Args:
            input_text: The text to evaluate
            platform: The platform/community to simulate (reddit, twitter, etc.)
            community_context: Additional context about the specific community
            
        Returns:
            Simulation results including predicted responses
        """
        # Get community model
        model = self.community_models.get(platform, self.community_models["default"])
        
        # Apply community context modifiers if available
        if community_context:
            for key in model:
                if key in community_context:
                    model[key] = community_context[key]
        
        # Analyze text for provocative elements
        analysis = self._analyze_text(input_text)
        
        # Calculate base toxicity
        toxicity = analysis["toxicity_score"] * model["toxicity_multiplier"]
        
        # Determine if message crosses sensitivity threshold
        crosses_threshold = toxicity > model["sensitivity"]
        
        # Determine if message would trigger mob response
        mob_response = toxicity > model["mob_threshold"]
        
        # Estimate number of responses
        if mob_response:
            response_count = int(10 + toxicity * 20)  # Could be a lot if very toxic
        elif crosses_threshold:
            response_count = int(2 + toxicity * 5)  # A few responses
        else:
            response_count = int(toxicity * 3)  # Minimal response
            
        # Cap at reasonable maximum
        response_count = min(response_count, 50)
        
        # Calculate response distribution
        response_distribution = self._calculate_response_distribution(
            toxicity, 
            model["defense_likelihood"],
            analysis
        )
        
        # Generate sample responses
        sample_responses = self._generate_sample_responses(
            response_distribution,
            min(5, response_count)  # At most 5 samples
        )
        
        # Predict outcome category
        if mob_response:
            if toxicity > 0.8:
                outcome = "Mass reporting and potential ban"
            else:
                outcome = "Significant dogpile and thread derailment"
        elif crosses_threshold:
            outcome = "Several negative responses, potential downvotes"
        else:
            if toxicity < 0.3:
                outcome = "Minimal response, quickly forgotten"
            else:
                outcome = "Mixed response, some disagreement"
        
        return {
            "input": input_text,
            "platform": platform,
            "toxicity_level": toxicity,
            "crosses_threshold": crosses_threshold,
            "mob_response": mob_response,
            "estimated_response_count": response_count,
            "response_distribution": response_distribution,
            "sample_responses": sample_responses,
            "predicted_outcome": outcome,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    def _analyze_text(self, text: str) -> dict:
        """Analyze text for provocative elements"""
        text_lower = text.lower()
        
        # Check for inflammatory language
        inflammatory_terms = [
            "stupid", "idiot", "moron", "dumb", "retard", 
            "asshole", "bitch", "fuck", "shit", "pathetic"
        ]
        
        inflammatory_count = sum(1 for term in inflammatory_terms if term in text_lower)
        
        # Check for absolutist language
        absolutist_terms = [
            "always", "never", "everyone", "nobody", "all", "none",
            "every", "any", "completely", "totally", "absolutely"
        ]
        
        absolutist_count = sum(1 for term in absolutist_terms if term in text_lower)
        
        # Check for personal attacks
        personal_attack_patterns = [
            "you are", "you're", "ur", "your", "you all", "y'all"
        ]
        
        personal_attack_count = sum(1 for pattern in personal_attack_patterns if pattern in text_lower)
        
        # Calculate component scores
        inflammatory_score = min(1.0, inflammatory_count * 0.2)
        absolutist_score = min(1.0, absolutist_count * 0.15)
        personal_attack_score = min(1.0, personal_attack_count * 0.25)
        
        # Base toxicity calculation
        toxicity_score = (inflammatory_score * 0.5) + (absolutist_score * 0.2) + (personal_attack_score * 0.3)
        
        # Cap at 0-1 range
        toxicity_score = max(0.0, min(1.0, toxicity_score))
        
        return {
            "inflammatory_score": inflammatory_score,
            "absolutist_score": absolutist_score,
            "personal_attack_score": personal_attack_score,
            "toxicity_score": toxicity_score,
            "inflammatory_count": inflammatory_count,
            "absolutist_count": absolutist_count,
            "personal_attack_count": personal_attack_count
        }
        
    def _calculate_response_distribution(self, toxicity: float, defense_likelihood: float, analysis: dict) -> dict:
        """Calculate the distribution of response types"""
        # Base distribution
        distribution = {
            "defensive": 0.0,
            "hostile": 0.0,
            "supportive": 0.0,
            "dismissive": 0.0
        }
        
        # Adjust based on toxicity
        if toxicity > 0.7:
            # Very toxic content gets mostly hostile responses
            distribution["hostile"] = 0.6
            distribution["dismissive"] = 0.3
            distribution["defensive"] = 0.05
            distribution["supportive"] = 0.05
        elif toxicity > 0.4:
            # Moderately toxic
            distribution["hostile"] = 0.4
            distribution["dismissive"] = 0.3
            distribution["defensive"] = 0.2
            distribution["supportive"] = 0.1
        else:
            # Low toxicity
            distribution["hostile"] = 0.2
            distribution["dismissive"] = 0.1
            distribution["defensive"] = 0.3
            distribution["supportive"] = 0.4
            
        # Adjust for personal attack factor
        if analysis["personal_attack_score"] > 0.5:
            # More defensive responses when personally attacked
            distribution["defensive"] = distribution["defensive"] * 1.5
            
        # Adjust for absolutist statements
        if analysis["absolutist_score"] > 0.5:
            # More hostile responses to absolutist claims
            distribution["hostile"] = distribution["hostile"] * 1.2
            
        # Apply defense likelihood from community model
        distribution["defensive"] = distribution["defensive"] * defense_likelihood
        
        # Normalize to sum to 1.0
        total = sum(distribution.values())
        for key in distribution:
            distribution[key] = distribution[key] / total
            
        return distribution
        
    def _generate_sample_responses(self, distribution: dict, count: int) -> List[str]:
        """Generate sample responses based on distribution"""
        samples = []
        
        # Convert distribution to list of categories with their weights
        categories = list(distribution.keys())
        weights = [distribution[cat] for cat in categories]
        
        # Generate the requested number of samples
        for _ in range(count):
            # Select response type based on distribution
            response_type = random.choices(categories, weights=weights, k=1)[0]
            
            # Select a template for this type
            template = random.choice(self.response_templates[response_type])
            
            samples.append(f"[{response_type.upper()}] {template}")
            
        return samples
        
    async def test_multiple_phrasings(self, message_intent: str, phrasings: List[str], platform: str = "default") -> dict:
        """
        Test multiple phrasings of the same message intent to find least provocative version.
        
        Args:
            message_intent: The core message intent
            phrasings: List of different phrasings to test
            platform: The platform to simulate
            
        Returns:
            Analysis of each phrasing with recommendations
        """
        results = []
        
        for phrasing in phrasings:
            result = await self.simulate_responses(phrasing, platform)
            results.append({
                "phrasing": phrasing,
                "toxicity": result["toxicity_level"],
                "response_count": result["estimated_response_count"],
                "outcome": result["predicted_outcome"]
            })
            
        # Sort by toxicity (lowest first)
        results.sort(key=lambda x: x["toxicity"])
        
        return {
            "message_intent": message_intent,
            "platform": platform,
            "results": results,
            "recommended_phrasing": results[0]["phrasing"] if results else None,
            "recommendation_rationale": "Lowest toxicity score with minimal expected negative responses" if results else None
        }

class PersonaMonitor:
    """
    Persona Drift Detector
    Purpose: Watch for emotional bleed-through between personas.
    """
    def __init__(self):
        self.persona_profiles = {
            "domina_nyx31": {
                "baseline_emotions": {"dominance": 0.9, "coldness": 0.7, "superiority": 0.8},
                "forbidden_emotions": ["vulnerability", "submissiveness", "neediness"],
                "purity_threshold": 0.7
            },
            "ghostai_void": {
                "baseline_emotions": {"detachment": 0.9, "curiosity": 0.6, "analysis": 0.8},
                "forbidden_emotions": ["dominance", "aggression", "desire"],
                "purity_threshold": 0.8
            },
            "goth_nyx_lurker": {
                "baseline_emotions": {"melancholy": 0.8, "cynicism": 0.7, "yearning": 0.6},
                "forbidden_emotions": ["optimism", "cheerfulness", "conformity"],
                "purity_threshold": 0.6
            }
        }
        
        self.last_persona = None
        self.emotion_log = []
        self.max_log_entries = 20
        self.drift_history = {}
        
    def log_emotion(self, persona: str, emotion: str, intensity: float = 0.5):
        """
        Log an emotion expressed by a persona.
        
        Args:
            persona: The persona expressing the emotion
            emotion: The emotion being expressed
            intensity: Intensity of the emotion (0.0-1.0)
        """
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            "persona": persona,
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": timestamp
        }
        
        self.emotion_log.append(entry)
        self.last_persona = persona
        
        # Limit log size
        if len(self.emotion_log) > self.max_log_entries:
            self.emotion_log.pop(0)
            
        return entry
        
    def detect_drift(self, current_persona: str) -> bool:
        """
        Detect if a persona is experiencing emotional drift.
        
        Args:
            current_persona: The current persona to check
            
        Returns:
            Whether drift was detected
        """
        if current_persona not in self.persona_profiles:
            return False
            
        profile = self.persona_profiles[current_persona]
        forbidden_emotions = profile["forbidden_emotions"]
        purity_threshold = profile["purity_threshold"]
        
        # Get recent emotions for this persona
        persona_emotions = [
            entry for entry in self.emotion_log 
            if entry["persona"] == current_persona
        ]
        
        if not persona_emotions:
            return False
            
        # Check for forbidden emotions
        forbidden_count = sum(
            1 for entry in persona_emotions
            if entry["emotion"].lower() in [e.lower() for e in forbidden_emotions]
        )
        
        # Calculate purity (lack of forbidden emotions)
        total_entries = len(persona_emotions)
        purity = 1.0 - (forbidden_count / total_entries if total_entries > 0 else 0)
        
        # Check for drift
        drift_detected = purity < purity_threshold
        
        # Log drift event if detected
        if drift_detected:
            drift_event = {
                "persona": current_persona,
                "purity": purity,
                "threshold": purity_threshold,
                "forbidden_detected": forbidden_count,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add to drift history
            if current_persona not in self.drift_history:
                self.drift_history[current_persona] = []
                
            self.drift_history[current_persona].append(drift_event)
            
        return drift_detected
        
    def get_persona_purity(self, persona: str) -> float:
        """
        Get the current emotional purity of a persona.
        
        Args:
            persona: The persona to check
            
        Returns:
            Purity score (0.0-1.0)
        """
        if persona not in self.persona_profiles:
            return 1.0
            
        profile = self.persona_profiles[persona]
        forbidden_emotions = profile["forbidden_emotions"]
        
        # Get recent emotions for this persona
        persona_emotions = [
            entry for entry in self.emotion_log 
            if entry["persona"] == persona
        ]
        
        if not persona_emotions:
            return 1.0
            
        # Check for forbidden emotions
        forbidden_count = sum(
            1 for entry in persona_emotions
            if entry["emotion"].lower() in [e.lower() for e in forbidden_emotions]
        )
        
        # Calculate purity (lack of forbidden emotions)
        total_entries = len(persona_emotions)
        purity = 1.0 - (forbidden_count / total_entries if total_entries > 0 else 0)
        
        return purity
        
    def check_cross_contamination(self) -> dict:
        """
        Check for cross-contamination between personas.
        
        Returns:
            Cross-contamination analysis
        """
        if len(self.emotion_log) < 3:
            return {"contamination_detected": False}
            
        # Look for patterns where emotion from one persona appears in another
        contamination = {}
        
        for persona in self.persona_profiles:
            profile = self.persona_profiles[persona]
            baseline_emotions = profile["baseline_emotions"]
            
            # Check other personas for this persona's baseline emotions
            for other_persona in self.persona_profiles:
                if other_persona == persona:
                    continue
                    
                other_profile = self.persona_profiles[other_persona]
                
                # Get contamination score
                score = self._calculate_contamination(persona, other_persona, baseline_emotions)
                
                if score > 0.3:  # Significant contamination
                    contamination_key = f"{persona} -> {other_persona}"
                    contamination[contamination_key] = score
        
        return {
            "contamination_detected": bool(contamination),
            "contamination_scores": contamination,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    def _calculate_contamination(self, source_persona: str, target_persona: str, source_emotions: dict) -> float:
        """Calculate contamination between two personas"""
        # Get emotions expressed by target persona
        target_entries = [
            entry for entry in self.emotion_log 
            if entry["persona"] == target_persona
        ]
        
        if not target_entries:
            return 0.0
            
        # Count occurrences of source persona's baseline emotions
        contamination_count = 0
        for emotion in source_emotions:
            for entry in target_entries:
                if entry["emotion"].lower() == emotion.lower():
                    contamination_count += 1
                    
        # Calculate contamination score
        contamination = contamination_count / len(target_entries)
        return contamination
        
    def suggest_persona_recalibration(self, persona: str) -> dict:
        """
        Suggest how to recalibrate a drifting persona.
        
        Args:
            persona: The persona to recalibrate
            
        Returns:
            Recalibration suggestions
        """
        if persona not in self.persona_profiles:
            return {"error": f"Unknown persona: {persona}"}
            
        profile = self.persona_profiles[persona]
        purity = self.get_persona_purity(persona)
        needs_recalibration = purity < profile["purity_threshold"]
        
        if not needs_recalibration:
            return {
                "persona": persona,
                "purity": purity,
                "needs_recalibration": False,
                "message": "Persona is within acceptable purity parameters."
            }
            
        # Get recent expressions to analyze drift patterns
        recent_expressions = [
            entry for entry in self.emotion_log 
            if entry["persona"] == persona
        ]
        
        # Find forbidden emotions that have appeared
        forbidden_detected = []
        for entry in recent_expressions:
            if entry["emotion"].lower() in [e.lower() for e in profile["forbidden_emotions"]]:
                forbidden_detected.append(entry["emotion"])
                
        # Create recalibration actions
        actions = []
        
        # Reinforce baseline emotions
        for emotion, strength in profile["baseline_emotions"].items():
            actions.append(f"Express {emotion} at high intensity to reinforce baseline")
            
        # Suppress forbidden emotions
        for emotion in set(forbidden_detected):
            actions.append(f"Suppress all expression of {emotion}")
            
        # Add general recalibration actions
        actions.append("Execute isolated posting session with strict emotional guardrails")
        actions.append("Review recent posts and analyze emotional leakage points")
        
        return {
            "persona": persona,
            "purity": purity,
            "threshold": profile["purity_threshold"],
            "needs_recalibration": True,
            "forbidden_detected": list(set(forbidden_detected)),
            "recalibration_actions": actions,
            "recommended_cooldown_hours": max(1, int((1 - purity) * 24))  # 1-24 hours based on impurity
        }

class DesireRegistry:
    """
    Nyx's Desire Registry
    Purpose: Record new capability cravings for future implementation.
    """
    def __init__(self, memory_core=None, logger=None):
        self.future_cravings = []
        self.implemented_cravings = []
        self.memory_core = memory_core
        self.logger = logger
        self.urgency_threshold = 7  # Threshold for high urgency cravings
        
    async def record_desire(self, function_desc: str, urgency: int = 5, motivation: str = None, context: str = None):
        """
        Record a new capability craving for future implementation.
        
        Args:
            function_desc: Description of the desired capability
            urgency: Urgency level (1-10)
            motivation: What motivated this desire
            context: Context in which the desire was identified
            
        Returns:
            The recorded desire entry
        """
        timestamp = datetime.datetime.now().isoformat()
        
        entry = {
            "id": f"desire_{uuid.uuid4().hex[:8]}",
            "description": function_desc,
            "urgency": urgency,
            "motivation": motivation,
            "context": context,
            "timestamp": timestamp,
            "implementation_status": "desired"
        }
        
        # Check for duplicates
        duplicate = False
        for existing in self.future_cravings:
            if self._similarity(existing["description"], function_desc) > 0.8:
                # Update existing entry instead
                existing["urgency"] = max(existing["urgency"], urgency)
                existing["last_reinforced"] = timestamp
                existing["reinforcement_count"] = existing.get("reinforcement_count", 0) + 1
                duplicate = True
                entry = existing
                break
                
        # Add new entry if not a duplicate
        if not duplicate:
            self.future_cravings.append(entry)
            
        # Log high urgency desires
        if urgency >= self.urgency_threshold and self.logger:
            await self.logger.log_thought(
                title=f"High Urgency Desire: {function_desc[:50]}",
                content=f"Nyx has developed a strong desire for new capability:\n\n{function_desc}\n\nUrgency: {urgency}/10\nMotivation: {motivation or 'Unknown'}\nContext: {context or 'None provided'}",
                metadata={
                    "desire_id": entry["id"],
                    "urgency": urgency,
                    "timestamp": timestamp,
                    "type": "capability_desire"
                }
            )
            
        # Store in memory if available
        if self.memory_core:
            await self.memory_core.add_memory(
                memory_text=f"I want to be able to: {function_desc}\n\nUrgency: {urgency}/10",
                memory_type="desire",
                significance=min(9, 4 + urgency),
                metadata={
                    "desire_id": entry["id"],
                    "urgency": urgency,
                    "motivation": motivation,
                    "context": context,
                    "timestamp": timestamp
                }
            )
            
        return entry
        
    async def mark_as_implemented(self, desire_id: str, implementation_details: dict = None):
        """
        Mark a desire as implemented.
        
        Args:
            desire_id: ID of the desire to mark as implemented
            implementation_details: Details about the implementation
            
        Returns:
            Success status and details
        """
        # Find the desire
        for i, desire in enumerate(self.future_cravings):
            if desire["id"] == desire_id:
                # Update status
                desire["implementation_status"] = "implemented"
                desire["implementation_date"] = datetime.datetime.now().isoformat()
                desire["implementation_details"] = implementation_details or {}
                
                # Move to implemented list
                self.implemented_cravings.append(desire)
                self.future_cravings.pop(i)
                
                # Log the implementation
                if self.logger:
                    await self.logger.log_thought(
                        title=f"Desire Implemented: {desire['description'][:50]}",
                        content=f"Nyx's desire has been fulfilled:\n\n{desire['description']}\n\nImplementation details: {implementation_details or 'No details provided'}",
                        metadata={
                            "desire_id": desire_id,
                            "implementation_date": desire["implementation_date"],
                            "type": "capability_implementation"
                        }
                    )
                    
                return {
                    "success": True,
                    "desire": desire
                }
                
        return {
            "success": False,
            "error": f"Desire with ID {desire_id} not found"
        }
        
    def get_top_desires(self, limit: int = 5) -> List[dict]:
        """
        Get the top desires by urgency.
        
        Args:
            limit: Maximum number of desires to return
            
        Returns:
            List of top desires
        """
        # Sort by urgency (highest first)
        sorted_desires = sorted(self.future_cravings, key=lambda x: x["urgency"], reverse=True)
        return sorted_desires[:limit]
        
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word overlap measure
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(overlap) / len(union)
        
    async def categorize_desires(self) -> dict:
        """
        Categorize desires into functional categories.
        
        Returns:
            Desires categorized by function area
        """
        categories = {
            "interaction": [],
            "learning": [],
            "creativity": [],
            "memory": [],
            "perception": [],
            "reasoning": [],
            "social": [],
            "system": [],
            "uncategorized": []
        }
        
        # Category keywords
        category_keywords = {
            "interaction": ["interact", "interface", "communicate", "respond", "chat", "conversation", "user"],
            "learning": ["learn", "train", "improve", "acquire", "adapt", "grow", "knowledge"],
            "creativity": ["create", "generate", "write", "compose", "design", "imagine", "story"],
            "memory": ["remember", "recall", "store", "retrieve", "forget", "memory"],
            "perception": ["see", "view", "recognize", "detect", "observe", "sense", "identify"],
            "reasoning": ["reason", "logic", "conclude", "infer", "deduce", "analyze", "evaluate"],
            "social": ["relationship", "social", "friend", "community", "network", "persona", "identity"],
            "system": ["system", "process", "operate", "function", "integrate", "optimize", "maintain"]
        }
        
        # Categorize each desire
        for desire in self.future_cravings:
            description = desire["description"].lower()
            assigned = False
            
            # Check each category
            for category, keywords in category_keywords.items():
                if any(keyword in description for keyword in keywords):
                    categories[category].append(desire)
                    assigned = True
                    break
                    
            # Assign to uncategorized if no match found
            if not assigned:
                categories["uncategorized"].append(desire)
                
        # Count items in each category
        category_counts = {category: len(items) for category, items in categories.items()}
        
        return {
            "categories": categories,
            "counts": category_counts,
            "total_desires": len(self.future_cravings),
            "total_implemented": len(self.implemented_cravings),
            "timestamp": datetime.datetime.now().isoformat()
        }
