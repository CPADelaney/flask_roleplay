# nyx/streamer/cross_game_knowledge.py

import logging
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import deque, Counter
from pathlib import Path
import threading
import pickle
import numpy as np

logger = logging.getLogger("enhanced_cross_game")

class GameMechanic:
    """Model for game mechanics"""
    
    def __init__(self,
               mechanic_id: str,
               name: str,
               description: str,
               examples: List[str] = None,
               games: List[str] = None,
               similar_mechanics: List[str] = None,
               creation_time: float = None):
        """
        Initialize a game mechanic
        
        Args:
            mechanic_id: Unique identifier
            name: Mechanic name
            description: Mechanic description
            examples: Examples of the mechanic in action
            games: Games that use this mechanic
            similar_mechanics: Similar mechanics
            creation_time: Creation timestamp
        """
        self.id = mechanic_id
        self.name = name
        self.description = description
        self.examples = examples or []
        self.games = games or []
        self.similar_mechanics = similar_mechanics or []
        self.creation_time = creation_time or time.time()
        
        # Learning data
        self.usage_count = 0
        self.last_usage_time = None
        self.learning_progress = 0.0  # 0.0 to 1.0
    
    def update_learning(self, progress_delta: float = 0.1):
        """
        Update learning progress for this mechanic
        
        Args:
            progress_delta: Amount to increase learning (0.0 to 1.0)
        """
        self.usage_count += 1
        self.last_usage_time = time.time()
        self.learning_progress = min(1.0, self.learning_progress + progress_delta)
    
    def add_example(self, example: str):
        """Add an example of this mechanic"""
        if example not in self.examples:
            self.examples.append(example)
    
    def add_game(self, game_id: str):
        """Add a game that uses this mechanic"""
        if game_id not in self.games:
            self.games.append(game_id)
    
    def add_similar_mechanic(self, mechanic_id: str):
        """Add a similar mechanic"""
        if mechanic_id not in self.similar_mechanics:
            self.similar_mechanics.append(mechanic_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "examples": self.examples,
            "games": self.games,
            "similar_mechanics": self.similar_mechanics,
            "creation_time": self.creation_time,
            "usage_count": self.usage_count,
            "last_usage_time": self.last_usage_time,
            "learning_progress": self.learning_progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameMechanic":
        """Create from dictionary"""
        mechanic = cls(
            mechanic_id=data["id"],
            name=data["name"],
            description=data["description"],
            examples=data.get("examples", []),
            games=data.get("games", []),
            similar_mechanics=data.get("similar_mechanics", []),
            creation_time=data.get("creation_time")
        )
        
        # Add learning data
        mechanic.usage_count = data.get("usage_count", 0)
        mechanic.last_usage_time = data.get("last_usage_time")
        mechanic.learning_progress = data.get("learning_progress", 0.0)
        
        return mechanic

class CrossGameInsight:
    """Model for insights that can be transferred between games"""
    
    def __init__(self,
               insight_id: str,
               content: str,
               source_game: str,
               target_game: str,
               mechanic: str,
               relevance: float = 0.5,
               context: Optional[str] = None,
               creation_time: float = None):
        """
        Initialize a cross-game insight
        
        Args:
            insight_id: Unique identifier
            content: Insight content
            source_game: Source game ID
            target_game: Target game ID
            mechanic: Mechanic ID
            relevance: Relevance score (0.0 to 1.0)
            context: Optional context where the insight applies
            creation_time: Creation timestamp
        """
        self.id = insight_id
        self.content = content
        self.source_game = source_game
        self.target_game = target_game
        self.mechanic = mechanic
        self.relevance = max(0.0, min(1.0, relevance))
        self.context = context
        self.creation_time = creation_time or time.time()
        
        # Usage tracking
        self.usage_count = 0
        self.last_usage_time = None
        self.effectiveness_score = 0.0  # 0.0 to 1.0 (how effective this insight is)
    
    def record_usage(self, effectiveness: float = 0.5):
        """
        Record usage of this insight
        
        Args:
            effectiveness: How effective the insight was (0.0 to 1.0)
        """
        self.usage_count += 1
        self.last_usage_time = time.time()
        
        # Update effectiveness with exponential moving average
        alpha = 0.3  # Weight for new observation
        self.effectiveness_score = (1 - alpha) * self.effectiveness_score + alpha * effectiveness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "source_game": self.source_game,
            "target_game": self.target_game,
            "mechanic": self.mechanic,
            "relevance": self.relevance,
            "context": self.context,
            "creation_time": self.creation_time,
            "usage_count": self.usage_count,
            "last_usage_time": self.last_usage_time,
            "effectiveness_score": self.effectiveness_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrossGameInsight":
        """Create from dictionary"""
        insight = cls(
            insight_id=data["id"],
            content=data["content"],
            source_game=data["source_game"],
            target_game=data["target_game"],
            mechanic=data["mechanic"],
            relevance=data.get("relevance", 0.5),
            context=data.get("context"),
            creation_time=data.get("creation_time")
        )
        
        # Add usage data
        insight.usage_count = data.get("usage_count", 0)
        insight.last_usage_time = data.get("last_usage_time")
        insight.effectiveness_score = data.get("effectiveness_score", 0.0)
        
        return insight

class GameLearningPattern:
    """Model for learning patterns across games"""
    
    def __init__(self,
               pattern_id: str,
               pattern_type: str,
               description: str,
               games: List[str] = None,
               mechanics: List[str] = None,
               confidence: float = 0.5,
               creation_time: float = None):
        """
        Initialize a game learning pattern
        
        Args:
            pattern_id: Unique identifier
            pattern_type: Type of pattern
            description: Pattern description
            games: Games where this pattern was observed
            mechanics: Mechanics involved in this pattern
            confidence: Confidence in this pattern (0.0 to 1.0)
            creation_time: Creation timestamp
        """
        self.id = pattern_id
        self.pattern_type = pattern_type
        self.description = description
        self.games = games or []
        self.mechanics = mechanics or []
        self.confidence = max(0.0, min(1.0, confidence))
        self.creation_time = creation_time or time.time()
        
        # Learning data
        self.application_count = 0
        self.success_count = 0
        self.last_application_time = None
    
    def apply_pattern(self, success: bool = True):
        """
        Record an application of this pattern
        
        Args:
            success: Whether the application was successful
        """
        self.application_count += 1
        if success:
            self.success_count += 1
        self.last_application_time = time.time()
    
    def get_success_rate(self) -> float:
        """Get the success rate of this pattern"""
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "games": self.games,
            "mechanics": self.mechanics,
            "confidence": self.confidence,
            "creation_time": self.creation_time,
            "application_count": self.application_count,
            "success_count": self.success_count,
            "last_application_time": self.last_application_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameLearningPattern":
        """Create from dictionary"""
        pattern = cls(
            pattern_id=data["id"],
            pattern_type=data["pattern_type"],
            description=data["description"],
            games=data.get("games", []),
            mechanics=data.get("mechanics", []),
            confidence=data.get("confidence", 0.5),
            creation_time=data.get("creation_time")
        )
        
        # Add learning data
        pattern.application_count = data.get("application_count", 0)
        pattern.success_count = data.get("success_count", 0)
        pattern.last_application_time = data.get("last_application_time")
        
        return pattern

class CrossGameKnowledgeSystem:
    """
    Enhanced system for maintaining and transferring knowledge between games,
    with improved learning, memory consolidation, and efficient knowledge transfer.
    """
    
    def __init__(self, data_dir: str = "cross_game_data"):
        """
        Initialize the cross-game knowledge system
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Storage
        self.games = {}  # game_id -> game data
        self.mechanics = {}  # mechanic_id -> GameMechanic
        self.insights = {}  # insight_id -> CrossGameInsight
        self.learning_patterns = {}  # pattern_id -> GameLearningPattern
        
        # Similarity matrices
        self.game_similarity_matrix = {}  # game_id -> {other_game_id -> similarity}
        self.mechanic_similarity_matrix = {}  # mechanic_id -> {other_mechanic_id -> similarity}
        
        # Recent activity
        self.recent_insights = deque(maxlen=20)
        self.recent_learnings = deque(maxlen=20)
        
        # Current session
        self.current_game_id = None
        self.current_mechanics_used = set()
        
        # Learning settings
        self.learning_rate = 0.1
        self.transfer_threshold = 0.6
        
        # Memory management
        self.auto_save_enabled = True
        self.auto_save_interval = 300  # 5 minutes in seconds
        self.last_auto_save_time = time.time()
        
        # Load data
        self._load_data()
        
        # Start auto-save thread if enabled
        if self.auto_save_enabled:
            self._start_auto_save_thread()
        
        logger.info(f"CrossGameKnowledgeSystem initialized with {len(self.games)} games, "
                   f"{len(self.mechanics)} mechanics, {len(self.insights)} insights")
    
    def _load_data(self):
        """Load data from files"""
        try:
            # Load games
            games_file = self.data_dir / "games.json"
            if games_file.exists():
                with open(games_file, 'r') as f:
                    self.games = json.load(f)
            
            # Load mechanics
            mechanics_file = self.data_dir / "mechanics.json"
            if mechanics_file.exists():
                with open(mechanics_file, 'r') as f:
                    mechanics_data = json.load(f)
                    for mechanic_id, data in mechanics_data.items():
                        self.mechanics[mechanic_id] = GameMechanic.from_dict(data)
            
            # Load insights
            insights_file = self.data_dir / "insights.json"
            if insights_file.exists():
                with open(insights_file, 'r') as f:
                    insights_data = json.load(f)
                    for insight_id, data in insights_data.items():
                        self.insights[insight_id] = CrossGameInsight.from_dict(data)
            
            # Load learning patterns
            patterns_file = self.data_dir / "learning_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    for pattern_id, data in patterns_data.items():
                        self.learning_patterns[pattern_id] = GameLearningPattern.from_dict(data)
            
            # Load similarity matrices
            game_sim_file = self.data_dir / "game_similarity.json"
            if game_sim_file.exists():
                with open(game_sim_file, 'r') as f:
                    self.game_similarity_matrix = json.load(f)
            
            mechanic_sim_file = self.data_dir / "mechanic_similarity.json"
            if mechanic_sim_file.exists():
                with open(mechanic_sim_file, 'r') as f:
                    self.mechanic_similarity_matrix = json.load(f)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_data(self):
        """Save data to files"""
        try:
            # Save games
            with open(self.data_dir / "games.json", 'w') as f:
                json.dump(self.games, f, indent=2)
            
            # Save mechanics
            mechanics_data = {mechanic_id: mechanic.to_dict() for mechanic_id, mechanic in self.mechanics.items()}
            with open(self.data_dir / "mechanics.json", 'w') as f:
                json.dump(mechanics_data, f, indent=2)
            
            # Save insights
            insights_data = {insight_id: insight.to_dict() for insight_id, insight in self.insights.items()}
            with open(self.data_dir / "insights.json", 'w') as f:
                json.dump(insights_data, f, indent=2)
            
            # Save learning patterns
            patterns_data = {pattern_id: pattern.to_dict() for pattern_id, pattern in self.learning_patterns.items()}
            with open(self.data_dir / "learning_patterns.json", 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save similarity matrices
            with open(self.data_dir / "game_similarity.json", 'w') as f:
                json.dump(self.game_similarity_matrix, f, indent=2)
            
            with open(self.data_dir / "mechanic_similarity.json", 'w') as f:
                json.dump(self.mechanic_similarity_matrix, f, indent=2)
            
            self.last_auto_save_time = time.time()
            logger.info("Saved cross-game knowledge data")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def _start_auto_save_thread(self):
        """Start auto-save thread"""
        def auto_save_loop():
            while self.auto_save_enabled:
                try:
                    current_time = time.time()
                    if current_time - self.last_auto_save_time >= self.auto_save_interval:
                        self._save_data()
                except Exception as e:
                    logger.error(f"Error in auto-save thread: {e}")
                
                # Sleep for a bit
                time.sleep(60)  # Check every minute
        
        # Start thread
        save_thread = threading.Thread(target=auto_save_loop, daemon=True)
        save_thread.start()
        logger.info("Started auto-save thread")
    
    def add_game(self,
               game_id: str,
               game_name: str,
               genre: List[str],
               mechanics: List[str] = None,
               description: str = "",
               release_year: Optional[int] = None,
               publisher: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a game to the knowledge base
        
        Args:
            game_id: Unique identifier
            game_name: Game name
            genre: Game genres
            mechanics: Game mechanics
            description: Game description
            release_year: Release year
            publisher: Publisher
            
        Returns:
            Game data
        """
        game_data = {
            "id": game_id,
            "name": game_name,
            "genre": genre,
            "mechanics": mechanics or [],
            "description": description,
            "release_year": release_year,
            "publisher": publisher,
            "added_at": time.time(),
            "play_count": 0,
            "last_played": None
        }
        
        self.games[game_id] = game_data
        
        # Update mechanic references
        for mechanic_id in mechanics or []:
            if mechanic_id in self.mechanics:
                self.mechanics[mechanic_id].add_game(game_id)
        
        # Update game similarity matrix
        self._update_game_similarity(game_id)
        
        logger.info(f"Added game: {game_name} ({game_id})")
        return game_data
    
    def add_mechanic(self,
                   mechanic_id: str,
                   name: str,
                   description: str,
                   examples: List[str] = None,
                   games: List[str] = None) -> GameMechanic:
        """
        Add a game mechanic
        
        Args:
            mechanic_id: Unique identifier
            name: Mechanic name
            description: Mechanic description
            examples: Examples of the mechanic
            games: Games that use this mechanic
            
        Returns:
            The added mechanic
        """
        mechanic = GameMechanic(
            mechanic_id=mechanic_id,
            name=name,
            description=description,
            examples=examples,
            games=games
        )
        
        self.mechanics[mechanic_id] = mechanic
        
        # Update game references
        for game_id in games or []:
            if game_id in self.games:
                if "mechanics" not in self.games[game_id]:
                    self.games[game_id]["mechanics"] = []
                
                if mechanic_id not in self.games[game_id]["mechanics"]:
                    self.games[game_id]["mechanics"].append(mechanic_id)
        
        # Update mechanic similarity matrix
        self._update_mechanic_similarity(mechanic_id)
        
        logger.info(f"Added mechanic: {name} ({mechanic_id})")
        return mechanic
    
    def add_insight(self,
                  content: str,
                  source_game: str,
                  target_game: str,
                  mechanic: str,
                  relevance: float = 0.5,
                  context: Optional[str] = None) -> CrossGameInsight:
        """
        Add a cross-game insight
        
        Args:
            content: Insight content
            source_game: Source game ID
            target_game: Target game ID
            mechanic: Related mechanic ID
            relevance: Relevance score (0.0 to 1.0)
            context: Optional context
            
        Returns:
            The added insight
        """
        # Generate ID
        insight_id = f"insight_{int(time.time())}_{source_game}_{target_game}"
        
        insight = CrossGameInsight(
            insight_id=insight_id,
            content=content,
            source_game=source_game,
            target_game=target_game,
            mechanic=mechanic,
            relevance=relevance,
            context=context
        )
        
        self.insights[insight_id] = insight
        
        # Add to recent insights
        self.recent_insights.append(insight_id)
        
        logger.info(f"Added insight: {insight_id} ({source_game} -> {target_game})")
        return insight
    
    def add_learning_pattern(self,
                           pattern_type: str,
                           description: str,
                           games: List[str] = None,
                           mechanics: List[str] = None,
                           confidence: float = 0.5) -> GameLearningPattern:
        """
        Add a game learning pattern
        
        Args:
            pattern_type: Type of pattern
            description: Pattern description
            games: Games where this pattern was observed
            mechanics: Mechanics involved in this pattern
            confidence: Confidence in this pattern (0.0 to 1.0)
            
        Returns:
            The added pattern
        """
        # Generate ID
        pattern_id = f"pattern_{int(time.time())}_{pattern_type}"
        
        pattern = GameLearningPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            games=games,
            mechanics=mechanics,
            confidence=confidence
        )
        
        self.learning_patterns[pattern_id] = pattern
        
        # Add to recent learnings
        self.recent_learnings.append(pattern_id)
        
        logger.info(f"Added learning pattern: {pattern_id} ({pattern_type})")
        return pattern
    
    def _update_game_similarity(self, game_id: str):
        """
        Update similarity scores for a game
        
        Args:
            game_id: Game ID to update
        """
        if game_id not in self.games:
            return
        
        game_data = self.games[game_id]
        similarities = {}
        
        for other_id, other_data in self.games.items():
            if other_id == game_id:
                continue
            
            # Calculate genre similarity
            genre_sim = self._calculate_list_similarity(
                game_data.get("genre", []),
                other_data.get("genre", [])
            )
            
            # Calculate mechanic similarity
            mechanic_sim = self._calculate_list_similarity(
                game_data.get("mechanics", []),
                other_data.get("mechanics", [])
            )
            
            # Calculate description similarity if available
            desc_sim = 0.0
            if game_data.get("description") and other_data.get("description"):
                desc_sim = self._calculate_text_similarity(
                    game_data["description"],
                    other_data["description"]
                )
            
            # Weighted combination
            similarity = (
                genre_sim * 0.3 +
                mechanic_sim * 0.5 +
                desc_sim * 0.2
            )
            
            similarities[other_id] = similarity
        
        # Update similarity matrix
        self.game_similarity_matrix[game_id] = similarities
        
        # Update reverse similarities
        for other_id, similarity in similarities.items():
            if other_id not in self.game_similarity_matrix:
                self.game_similarity_matrix[other_id] = {}
            
            self.game_similarity_matrix[other_id][game_id] = similarity
    
    def _update_mechanic_similarity(self, mechanic_id: str):
        """
        Update similarity scores for a mechanic
        
        Args:
            mechanic_id: Mechanic ID to update
        """
        if mechanic_id not in self.mechanics:
            return
        
        mechanic = self.mechanics[mechanic_id]
        similarities = {}
        
        for other_id, other_mechanic in self.mechanics.items():
            if other_id == mechanic_id:
                continue
            
            # Calculate name similarity
            name_sim = self._calculate_text_similarity(
                mechanic.name,
                other_mechanic.name
            )
            
            # Calculate description similarity
            desc_sim = self._calculate_text_similarity(
                mechanic.description,
                other_mechanic.description
            )
            
            # Calculate game overlap similarity
            game_sim = self._calculate_list_similarity(
                mechanic.games,
                other_mechanic.games
            )
            
            # Calculate example similarity
            example_sim = 0.0
            if mechanic.examples and other_mechanic.examples:
                # Compare each example with each other example
                similarities_sum = 0.0
                comparison_count = 0
                
                for ex1 in mechanic.examples:
                    for ex2 in other_mechanic.examples:
                        similarities_sum += self._calculate_text_similarity(ex1, ex2)
                        comparison_count += 1
                
                if comparison_count > 0:
                    example_sim = similarities_sum / comparison_count
            
            # Weighted combination
            similarity = (
                name_sim * 0.2 +
                desc_sim * 0.4 +
                game_sim * 0.3 +
                example_sim * 0.1
            )
            
            similarities[other_id] = similarity
            
            # Update similar mechanics lists if high similarity
            if similarity >= 0.7:
                mechanic.add_similar_mechanic(other_id)
                other_mechanic.add_similar_mechanic(mechanic_id)
        
        # Update similarity matrix
        self.mechanic_similarity_matrix[mechanic_id] = similarities
    
    def _calculate_list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """
        Calculate similarity between two lists (Jaccard similarity)
        
        Args:
            list1: First list
            list2: Second list
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        set1 = set(list1)
        set2 = set(list2)
        
        if not set1 and not set2:
            return 1.0  # Both empty
        
        if not set1 or not set2:
            return 0.0  # One is empty
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        import difflib
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def set_current_game(self, game_id: str):
        """
        Set the current game being played
        
        Args:
            game_id: Game ID
        """
        if game_id not in self.games:
            logger.warning(f"Unknown game: {game_id}")
            return
        
        # Update play stats for current game before changing
        if self.current_game_id:
            if self.current_game_id in self.games:
                self.games[self.current_game_id]["play_count"] = self.games[self.current_game_id].get("play_count", 0) + 1
                self.games[self.current_game_id]["last_played"] = time.time()
        
        # Set new current game
        self.current_game_id = game_id
        self.current_mechanics_used.clear()
        
        # Update play stats
        self.games[game_id]["play_count"] = self.games[game_id].get("play_count", 0) + 1
        self.games[game_id]["last_played"] = time.time()
        
        logger.info(f"Set current game: {game_id}")
    
    def record_mechanic_usage(self, mechanic_id: str, success: bool = True, learning_amount: float = 0.1):
        """
        Record usage of a game mechanic
        
        Args:
            mechanic_id: Mechanic ID
            success: Whether the usage was successful
            learning_amount: Amount of learning progress (0.0 to 1.0)
            
        Returns:
            Updated mechanic data
        """
        if mechanic_id not in self.mechanics:
            logger.warning(f"Unknown mechanic: {mechanic_id}")
            return None
        
        mechanic = self.mechanics[mechanic_id]
        
        # Update learning progress
        mechanic.update_learning(learning_amount)
        
        # Add to current mechanics used
        self.current_mechanics_used.add(mechanic_id)
        
        # Apply relevant learning patterns
        self._apply_learning_patterns([mechanic_id], success)
        
        return mechanic.to_dict()
    
    def _apply_learning_patterns(self, mechanic_ids: List[str], success: bool = True):
        """
        Apply learning patterns for mechanics
        
        Args:
            mechanic_ids: List of mechanic IDs
            success: Whether the application was successful
        """
        # Find relevant patterns
        relevant_patterns = []
        
        for pattern_id, pattern in self.learning_patterns.items():
            # Check if pattern has any of these mechanics
            if any(m_id in pattern.mechanics for m_id in mechanic_ids):
                relevant_patterns.append(pattern)
        
        # Apply patterns
        for pattern in relevant_patterns:
            pattern.apply_pattern(success)
    
    def get_similar_games(self, game_id: str, min_similarity: float = 0.5, max_games: int = 5) -> List[Dict[str, Any]]:
        """
        Find games similar to the specified game
        
        Args:
            game_id: Game ID
            min_similarity: Minimum similarity threshold
            max_games: Maximum number of games to return
            
        Returns:
            List of similar games with similarity scores
        """
        if game_id not in self.games:
            logger.warning(f"Unknown game: {game_id}")
            return []
        
        # Check if game has similarities
        if game_id not in self.game_similarity_matrix:
            logger.warning(f"No similarity data for game: {game_id}")
            return []
        
        # Get similarities
        similarities = self.game_similarity_matrix[game_id]
        
        # Filter by threshold
        filtered = [(other_id, sim) for other_id, sim in similarities.items() if sim >= min_similarity]
        
        # Sort by similarity (highest first)
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to max_games
        filtered = filtered[:max_games]
        
        # Format results
        results = []
        for other_id, similarity in filtered:
            if other_id in self.games:
                game_data = self.games[other_id].copy()
                game_data["similarity"] = similarity
                
                # Add common mechanics
                game_mechanics = set(game_data.get("mechanics", []))
                source_mechanics = set(self.games[game_id].get("mechanics", []))
                common_mechanics = list(game_mechanics.intersection(source_mechanics))
                
                game_data["common_mechanics"] = common_mechanics
                results.append(game_data)
        
        return results
    
    def get_applicable_insights(self, 
                              target_game: str, 
                              context: Optional[str] = None,
                              min_relevance: float = 0.6,
                              max_insights: int = 5) -> List[Dict[str, Any]]:
        """
        Find insights applicable to the target game
        
        Args:
            target_game: Target game ID
            context: Optional context to match
            min_relevance: Minimum relevance threshold
            max_insights: Maximum number of insights to return
            
        Returns:
            List of applicable insights
        """
        if target_game not in self.games:
            logger.warning(f"Unknown game: {target_game}")
            return []
        
        # Find direct insights for this game
        direct_insights = []
        for insight_id, insight in self.insights.items():
            if insight.target_game == target_game and insight.relevance >= min_relevance:
                direct_insights.append(insight)
        
        # If context provided, adjust relevance based on context match
        if context and direct_insights:
            for insight in direct_insights:
                if insight.context:
                    # Calculate context similarity
                    context_similarity = self._calculate_text_similarity(context, insight.context)
                    
                    # Adjust relevance
                    adjusted_relevance = insight.relevance * (0.5 + 0.5 * context_similarity)
                    
                    # Update if falls below threshold
                    if adjusted_relevance < min_relevance:
                        direct_insights.remove(insight)
        
        # Get similar games
        similar_games = self.get_similar_games(target_game, min_similarity=0.6)
        
        # Find insights from similar games that might apply
        indirect_insights = []
        for similar_game in similar_games:
            similar_id = similar_game["id"]
            similarity = similar_game["similarity"]
            
            # Find insights where this game is the source
            for insight_id, insight in self.insights.items():
                if insight.source_game == similar_id and insight.target_game != target_game:
                    # Adjust relevance by game similarity
                    adjusted_relevance = insight.relevance * similarity
                    
                    if adjusted_relevance >= min_relevance:
                        # Create a modified version for the target game
                        adapted_insight = CrossGameInsight(
                            insight_id=f"adapted_{insight.id}",
                            content=insight.content,
                            source_game=insight.source_game,
                            target_game=target_game,  # Change target
                            mechanic=insight.mechanic,
                            relevance=adjusted_relevance,
                            context=insight.context
                        )
                        
                        indirect_insights.append(adapted_insight)
        
        # Combine insights
        all_insights = direct_insights + indirect_insights
        
        # Sort by relevance
        all_insights.sort(key=lambda x: x.relevance, reverse=True)
        
        # Limit to max_insights
        all_insights = all_insights[:max_insights]
        
        # Format results
        results = []
        for insight in all_insights:
            # Record usage
            self.insights.get(insight.id, insight).record_usage()
            
            # Format
            result = insight.to_dict()
            if insight.source_game in self.games:
                result["source_game_name"] = self.games[insight.source_game]["name"]
            if insight.target_game in self.games:
                result["target_game_name"] = self.games[insight.target_game]["name"]
            if insight.mechanic in self.mechanics:
                result["mechanic_name"] = self.mechanics[insight.mechanic].name
            
            results.append(result)
        
        return results
    
    def get_learning_opportunities(self, game_id: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Find learning opportunities for a game
        
        Args:
            game_id: Game ID
            max_results: Maximum number of opportunities to return
            
        Returns:
            List of learning opportunities
        """
        if game_id not in self.games:
            logger.warning(f"Unknown game: {game_id}")
            return []
        
        # Get game mechanics
        game_mechanics = self.games[game_id].get("mechanics", [])
        if not game_mechanics:
            return []
        
        opportunities = []
        
        # Check mechanics with low learning progress
        for mechanic_id in game_mechanics:
            if mechanic_id in self.mechanics:
                mechanic = self.mechanics[mechanic_id]
                
                # Check if learning progress is low
                if mechanic.learning_progress < 0.5:
                    opportunities.append({
                        "type": "mechanic_learning",
                        "mechanic_id": mechanic_id,
                        "mechanic_name": mechanic.name,
                        "current_progress": mechanic.learning_progress,
                        "description": f"Improve understanding of the '{mechanic.name}' mechanic"
                    })
        
        # Check for mechanics in similar games that aren't in this game
        similar_games = self.get_similar_games(game_id)
        for similar_game in similar_games:
            similar_id = similar_game["id"]
            similar_mechanics = set(self.games[similar_id].get("mechanics", []))
            
            # Find mechanics not in this game
            new_mechanics = similar_mechanics - set(game_mechanics)
            
            for mechanic_id in new_mechanics:
                if mechanic_id in self.mechanics:
                    mechanic = self.mechanics[mechanic_id]
                    
                    opportunities.append({
                        "type": "new_mechanic",
                        "mechanic_id": mechanic_id,
                        "mechanic_name": mechanic.name,
                        "source_game": similar_id,
                        "source_game_name": similar_game["name"],
                        "description": f"Learn the '{mechanic.name}' mechanic from {similar_game['name']}"
                    })
        
        # Check for applicable learning patterns
        for pattern_id, pattern in self.learning_patterns.items():
            # Check if pattern involves this game's mechanics
            if any(m_id in pattern.mechanics for m_id in game_mechanics):
                # Check if pattern has a good success rate
                if pattern.get_success_rate() >= 0.7:
                    opportunities.append({
                        "type": "apply_pattern",
                        "pattern_id": pattern_id,
                        "pattern_type": pattern.pattern_type,
                        "success_rate": pattern.get_success_rate(),
                        "description": pattern.description
                    })
        
        # Sort by type (prioritize mechanic learning)
        opportunities.sort(key=lambda x: 0 if x["type"] == "mechanic_learning" else 1)
        
        return opportunities[:max_results]
    
    def generate_insight(self, 
                       source_game: str, 
                       target_game: str,
                       mechanic_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Generate a new cross-game insight based on existing knowledge
        
        Args:
            source_game: Source game ID
            target_game: Target game ID
            mechanic_id: Optional specific mechanic ID
            
        Returns:
            Generated insight, or None if not possible
        """
        if source_game not in self.games or target_game not in self.games:
            logger.warning(f"Unknown game(s): {source_game} or {target_game}")
            return None
        
        # Find common mechanics if not specified
        if not mechanic_id:
            source_mechanics = set(self.games[source_game].get("mechanics", []))
            target_mechanics = set(self.games[target_game].get("mechanics", []))
            
            common_mechanics = source_mechanics.intersection(target_mechanics)
            
            if not common_mechanics:
                logger.warning(f"No common mechanics between {source_game} and {target_game}")
                return None
            
            # Choose mechanic with highest similarity
            mechanic_id = max(
                common_mechanics,
                key=lambda m_id: self.mechanics[m_id].learning_progress if m_id in self.mechanics else 0
            )
        
        # Check if mechanic exists
        if mechanic_id not in self.mechanics:
            logger.warning(f"Unknown mechanic: {mechanic_id}")
            return None
        
        mechanic = self.mechanics[mechanic_id]
        
        # Generate insight content
        # In a real implementation, this would use an LLM or template system
        # For now, use a simple template
        source_name = self.games[source_game]["name"]
        target_name = self.games[target_game]["name"]
        
        content = f"The {mechanic.name} mechanic in {source_name} is similar to {target_name}, but "
        
        # Add some variation based on mechanic
        if "combat" in mechanic.name.lower():
            content += f"the combat in {source_name} is more fluid, while in {target_name} it's more strategic."
        elif "crafting" in mechanic.name.lower():
            content += f"crafting in {source_name} has more components, while in {target_name} it's more streamlined."
        elif "exploration" in mechanic.name.lower():
            content += f"exploration in {source_name} is more open-ended, while in {target_name} it's more guided."
        else:
            content += f"it's implemented differently, with {source_name} focusing more on depth and {target_name} on accessibility."
        
        # Create the insight
        insight = self.add_insight(
            content=content,
            source_game=source_game,
            target_game=target_game,
            mechanic=mechanic_id,
            relevance=0.8
        )
        
        return insight.to_dict()
    
    def discover_patterns(self) -> List[Dict[str, Any]]:
        """
        Automatically discover new learning patterns
        
        Returns:
            List of discovered patterns
        """
        # In a real implementation, this would use sophisticated pattern recognition
        # For now, use a simple approach
        new_patterns = []
        
        # Look for mechanics that are commonly used together
        mechanic_pairs = {}
        
        for game_id, game_data in self.games.items():
            mechanics = game_data.get("mechanics", [])
            
            # Skip if fewer than 2 mechanics
            if len(mechanics) < 2:
                continue
            
            # Record all pairs
            for i, mech1 in enumerate(mechanics):
                for mech2 in mechanics[i+1:]:
                    pair = tuple(sorted([mech1, mech2]))
                    
                    if pair not in mechanic_pairs:
                        mechanic_pairs[pair] = 0
                    
                    mechanic_pairs[pair] += 1
        
        # Find common pairs (used in at least 2 games)
        common_pairs = [(pair, count) for pair, count in mechanic_pairs.items() if count >= 2]
        
        # Create patterns for common pairs
        for (mech1, mech2), count in common_pairs:
            # Check if mechanics exist
            if mech1 in self.mechanics and mech2 in self.mechanics:
                # Find games with both mechanics
                games_with_both = []
                
                for game_id, game_data in self.games.items():
                    if mech1 in game_data.get("mechanics", []) and mech2 in game_data.get("mechanics", []):
                        games_with_both.append(game_id)
                
                # Create pattern description
                mech1_name = self.mechanics[mech1].name
                mech2_name = self.mechanics[mech2].name
                
                description = f"The {mech1_name} and {mech2_name} mechanics are often used together."
                
                # Create the pattern
                pattern = self.add_learning_pattern(
                    pattern_type="mechanic_combination",
                    description=description,
                    games=games_with_both,
                    mechanics=[mech1, mech2],
                    confidence=min(1.0, count / 2)  # Scale confidence by occurrence count
                )
                
                new_patterns.append(pattern.to_dict())
        
        # Look for mechanics with similar implementation across games
        for mechanic_id, mechanic in self.mechanics.items():
            # Skip if fewer than 3 games
            if len(mechanic.games) < 3:
                continue
            
            # Calculate average learning progress
            progress_sum = mechanic.learning_progress
            count = 1
            
            for game_id in mechanic.games:
                for insight_id, insight in self.insights.items():
                    if insight.source_game == game_id and insight.mechanic == mechanic_id:
                        progress_sum += insight.effectiveness_score
                        count += 1
            
            avg_progress = progress_sum / count
            
            # If high progress, create a pattern
            if avg_progress >= 0.7:
                description = f"The {mechanic.name} mechanic has consistent implementation across games."
                
                pattern = self.add_learning_pattern(
                    pattern_type="consistent_mechanic",
                    description=description,
                    games=mechanic.games,
                    mechanics=[mechanic_id],
                    confidence=avg_progress
                )
                
                new_patterns.append(pattern.to_dict())
        
        return new_patterns
    
    def consolidate_knowledge(self) -> Dict[str, Any]:
        """
        Consolidate and optimize stored knowledge
        
        Returns:
            Consolidation results
        """
        results = {
            "removed_insights": 0,
            "removed_patterns": 0,
            "combined_insights": 0,
            "optimized_mechanics": 0
        }
        
        # Remove low-relevance insights that haven't been used
        insights_to_remove = []
        
        for insight_id, insight in self.insights.items():
            if insight.relevance < 0.4 and insight.usage_count == 0:
                insights_to_remove.append(insight_id)
        
        for insight_id in insights_to_remove:
            del self.insights[insight_id]
        
        results["removed_insights"] = len(insights_to_remove)
        
        # Remove unsuccessful patterns
        patterns_to_remove = []
        
        for pattern_id, pattern in self.learning_patterns.items():
            if pattern.application_count >= 5 and pattern.get_success_rate() < 0.3:
                patterns_to_remove.append(pattern_id)
        
        for pattern_id in patterns_to_remove:
            del self.learning_patterns[pattern_id]
        
        results["removed_patterns"] = len(patterns_to_remove)
        
        # Combine similar insights
        combined_count = 0
        processed_insights = set()
        
        for insight_id, insight in self.insights.items():
            if insight_id in processed_insights:
                continue
            
            similar_insights = []
            
            # Find similar insights
            for other_id, other in self.insights.items():
                if other_id == insight_id or other_id in processed_insights:
                    continue
                
                # Check if same source, target, and mechanic
                if (other.source_game == insight.source_game and
                    other.target_game == insight.target_game and
                    other.mechanic == insight.mechanic):
                    
                    # Check content similarity
                    similarity = self._calculate_text_similarity(insight.content, other.content)
                    
                    if similarity >= 0.6:
                        similar_insights.append(other_id)
            
            # If found similar insights, combine them
            if similar_insights:
                # Choose the highest relevance and effectiveness
                max_relevance = insight.relevance
                max_effectiveness = insight.effectiveness_score
                best_content = insight.content
                total_usage = insight.usage_count
                
                for other_id in similar_insights:
                    other = self.insights[other_id]
                    
                    max_relevance = max(max_relevance, other.relevance)
                    max_effectiveness = max(max_effectiveness, other.effectiveness_score)
                    total_usage += other.usage_count
                    
                    # Choose content from most used insight
                    if other.usage_count > insight.usage_count:
                        best_content = other.content
                
                # Update this insight
                insight.relevance = max_relevance
                insight.effectiveness_score = max_effectiveness
                insight.content = best_content
                insight.usage_count = total_usage
                
                # Mark as processed
                processed_insights.add(insight_id)
                processed_insights.update(similar_insights)
                
                # Remove the similar insights
                for other_id in similar_insights:
                    del self.insights[other_id]
                
                combined_count += len(similar_insights)
        
        results["combined_insights"] = combined_count
        
        # Optimize mechanics (merge similar mechanics with different names)
        optimized_count = 0
        
        for mechanic_id, similarities in self.mechanic_similarity_matrix.items():
            if mechanic_id not in self.mechanics:
                continue
            
            for other_id, similarity in similarities.items():
                if other_id not in self.mechanics or other_id == mechanic_id:
                    continue
                
                # If very similar, merge them
                if similarity >= 0.85:
                    # Keep the one with higher learning progress or more games
                    mech1 = self.mechanics[mechanic_id]
                    mech2 = self.mechanics[other_id]
                    
                    if (mech1.learning_progress > mech2.learning_progress or
                        len(mech1.games) > len(mech2.games)):
                        # Keep mech1, merge mech2 into it
                        keep_id, remove_id = mechanic_id, other_id
                    else:
                        # Keep mech2, merge mech1 into it
                        keep_id, remove_id = other_id, mechanic_id
                    
                    # Merge
                    keep = self.mechanics[keep_id]
                    remove = self.mechanics[remove_id]
                    
                    # Combine examples
                    for example in remove.examples:
                        keep.add_example(example)
                    
                    # Combine games
                    for game_id in remove.games:
                        keep.add_game(game_id)
                        
                        # Update game references
                        if game_id in self.games:
                            if "mechanics" in self.games[game_id]:
                                if remove_id in self.games[game_id]["mechanics"]:
                                    self.games[game_id]["mechanics"].remove(remove_id)
                                
                                if keep_id not in self.games[game_id]["mechanics"]:
                                    self.games[game_id]["mechanics"].append(keep_id)
                    
                    # Combine similar mechanics
                    for sim_id in remove.similar_mechanics:
                        if sim_id != keep_id:
                            keep.add_similar_mechanic(sim_id)
                    
                    # Update learning progress
                    keep.learning_progress = max(keep.learning_progress, remove.learning_progress)
                    keep.usage_count += remove.usage_count
                    
                    # Update insights
                    for insight_id, insight in self.insights.items():
                        if insight.mechanic == remove_id:
                            insight.mechanic = keep_id
                    
                    # Update patterns
                    for pattern_id, pattern in self.learning_patterns.items():
                        if remove_id in pattern.mechanics:
                            pattern.mechanics.remove(remove_id)
                            if keep_id not in pattern.mechanics:
                                pattern.mechanics.append(keep_id)
                    
                    # Remove the merged mechanic
                    del self.mechanics[remove_id]
                    
                    optimized_count += 1
                    
                    # Skip other mechanics, as mechanic_id might have been removed
                    break
        
        results["optimized_mechanics"] = optimized_count
        
        return results
    
    def save(self):
        """Save all data"""
        self._save_data()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge system statistics
        
        Returns:
            Statistics
        """
        return {
            "games_count": len(self.games),
            "mechanics_count": len(self.mechanics),
            "insights_count": len(self.insights),
            "patterns_count": len(self.learning_patterns),
            "current_game": self.current_game_id,
            "current_mechanics_used": len(self.current_mechanics_used),
            "average_mechanic_learning": sum(m.learning_progress for m in self.mechanics.values()) / max(1, len(self.mechanics)),
            "average_insight_effectiveness": sum(i.effectiveness_score for i in self.insights.values()) / max(1, len(self.insights)),
            "average_pattern_success": sum(p.get_success_rate() for p in self.learning_patterns.values()) / max(1, len(self.learning_patterns))
        }
    
    def seed_initial_knowledge(self):
        """Seed the system with initial knowledge"""
        # Add some games
        self.add_game(
            game_id="witcher3",
            game_name="The Witcher 3: Wild Hunt",
            genre=["RPG", "Open World", "Action"],
            mechanics=["combat", "crafting", "dialog_choices", "skill_tree", "open_world"],
            description="An action RPG in a fantasy world, following the adventures of monster hunter Geralt of Rivia."
        )
        
        self.add_game(
            game_id="skyrim",
            game_name="The Elder Scrolls V: Skyrim",
            genre=["RPG", "Open World", "Action"],
            mechanics=["combat", "crafting", "skill_tree", "open_world", "first_person"],
            description="An open world action RPG set in the province of Skyrim, where the player character can explore freely."
        )
        
        self.add_game(
            game_id="eldenring",
            game_name="Elden Ring",
            genre=["Action", "RPG", "Open World"],
            mechanics=["combat", "crafting", "skill_tree", "open_world", "souls_like"],
            description="An action RPG set in a fantasy world created by Hidetaka Miyazaki and George R. R. Martin."
        )
        
        # Add mechanics
        self.add_mechanic(
            mechanic_id="combat",
            name="Combat System",
            description="The system for engaging in combat with enemies",
            examples=["Melee attacks", "Ranged attacks", "Blocking", "Dodging"]
        )
        
        self.add_mechanic(
            mechanic_id="crafting",
            name="Crafting System",
            description="The system for creating items from components",
            examples=["Weapon crafting", "Armor crafting", "Alchemy", "Cooking"]
        )
        
        self.add_mechanic(
            mechanic_id="skill_tree",
            name="Skill Tree",
            description="A system for character progression through skill points",
            examples=["Ability unlocks", "Stat increases", "Specialization paths"]
        )
        
        self.add_mechanic(
            mechanic_id="open_world",
            name="Open World",
            description="A large, continuous game world that can be freely explored",
            examples=["Free exploration", "Open map", "Non-linear progression"]
        )
        
        self.add_mechanic(
            mechanic_id="dialog_choices",
            name="Dialog Choices",
            description="A system for selecting dialog options in conversations",
            examples=["Branching conversations", "Relationship impacts", "Story choices"]
        )
        
        self.add_mechanic(
            mechanic_id="first_person",
            name="First Person Perspective",
            description="Game played from the first-person viewpoint",
            examples=["First-person combat", "First-person exploration"]
        )
        
        self.add_mechanic(
            mechanic_id="souls_like",
            name="Souls-like Mechanics",
            description="Challenging gameplay mechanics inspired by Dark Souls",
            examples=["Difficult combat", "Death penalties", "Sparse checkpoints"]
        )
        
        # Add insights
        self.add_insight(
            content="Combat in Witcher 3 focuses on dodging and using signs, while Skyrim emphasizes blocking and power attacks.",
            source_game="witcher3",
            target_game="skyrim",
            mechanic="combat",
            relevance=0.8,
            context="combat"
        )
        
        self.add_insight(
            content="Elden Ring's combat is more punishing than Witcher 3, requiring more precise timing and pattern recognition.",
            source_game="eldenring",
            target_game="witcher3",
            mechanic="combat",
            relevance=0.9,
            context="combat"
        )
        
        self.add_insight(
            content="Crafting in Skyrim is more straightforward than in Witcher 3, which has more complex recipes and ingredients.",
            source_game="skyrim",
            target_game="witcher3",
            mechanic="crafting",
            relevance=0.7,
            context="crafting"
        )
        
        # Add learning patterns
        self.add_learning_pattern(
            pattern_type="combat_transfer",
            description="Skills in timing attacks and dodges transfer well between action RPGs.",
            games=["witcher3", "eldenring"],
            mechanics=["combat"],
            confidence=0.8
        )
        
        self.add_learning_pattern(
            pattern_type="exploration_synergy",
            description="Open world exploration and crafting systems often complement each other.",
            games=["witcher3", "skyrim", "eldenring"],
            mechanics=["open_world", "crafting"],
            confidence=0.7
        )
        
        logger.info("Seeded initial knowledge")
