# nyx/streamer/enhanced_game_vision.py

import logging
import time
import asyncio
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import deque
import pickle
import json
from pathlib import Path

logger = logging.getLogger("enhanced_game_vision")

class GameKnowledgeBase:
    """Knowledge base for game-specific information and recognition"""
    
    def __init__(self, data_dir="game_data"):
        """Initialize the knowledge base with a data directory"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Game information
        self.games = {}  # game_id -> game info
        self.game_elements = {}  # game_id -> elements
        self.ui_layouts = {}  # game_id -> UI layouts
        
        # Cross-game similarities
        self.similar_games = {}  # game_id -> similar games
        
        # Load existing data if available
        self._load_data()
        
        logger.info(f"GameKnowledgeBase initialized with {len(self.games)} games")
    
    def _load_data(self):
        """Load data from files"""
        # Load games
        games_file = self.data_dir / "games.json"
        if games_file.exists():
            with open(games_file, 'r') as f:
                self.games = json.load(f)
        
        # Load game elements
        elements_file = self.data_dir / "game_elements.json"
        if elements_file.exists():
            with open(elements_file, 'r') as f:
                self.game_elements = json.load(f)
        
        # Load UI layouts
        layouts_file = self.data_dir / "ui_layouts.json"
        if layouts_file.exists():
            with open(layouts_file, 'r') as f:
                self.ui_layouts = json.load(f)
        
        # Load similar games
        similar_file = self.data_dir / "similar_games.json"
        if similar_file.exists():
            with open(similar_file, 'r') as f:
                self.similar_games = json.load(f)
    
    def save_data(self):
        """Save data to files"""
        # Save games
        with open(self.data_dir / "games.json", 'w') as f:
            json.dump(self.games, f, indent=2)
        
        # Save game elements
        with open(self.data_dir / "game_elements.json", 'w') as f:
            json.dump(self.game_elements, f, indent=2)
        
        # Save UI layouts
        with open(self.data_dir / "ui_layouts.json", 'w') as f:
            json.dump(self.ui_layouts, f, indent=2)
        
        # Save similar games
        with open(self.data_dir / "similar_games.json", 'w') as f:
            json.dump(self.similar_games, f, indent=2)
        
        logger.info("Saved knowledge base data")
    
    def add_game(self, 
                game_id: str, 
                game_name: str, 
                game_genre: List[str], 
                description: str = "",
                features: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a game to the knowledge base"""
        game_info = {
            "id": game_id,
            "name": game_name,
            "genre": game_genre,
            "description": description,
            "features": features or {},
            "added_at": time.time()
        }
        
        self.games[game_id] = game_info
        
        # Initialize game elements and UI layouts if not exists
        if game_id not in self.game_elements:
            self.game_elements[game_id] = {}
        
        if game_id not in self.ui_layouts:
            self.ui_layouts[game_id] = {}
        
        # Save data
        self.save_data()
        
        logger.info(f"Added game: {game_name}")
        return game_info
    
    def add_game_element(self, 
                        game_id: str, 
                        element_id: str, 
                        element_name: str,
                        element_type: str,
                        visual_features: Dict[str, Any] = None,
                        description: str = "") -> Dict[str, Any]:
        """Add a game element for recognition"""
        if game_id not in self.games:
            logger.warning(f"Cannot add element for unknown game: {game_id}")
            return None
        
        element_info = {
            "id": element_id,
            "name": element_name,
            "type": element_type,
            "visual_features": visual_features or {},
            "description": description,
            "added_at": time.time()
        }
        
        # Initialize game elements if not exists
        if game_id not in self.game_elements:
            self.game_elements[game_id] = {}
        
        self.game_elements[game_id][element_id] = element_info
        
        # Save data
        self.save_data()
        
        logger.info(f"Added game element: {element_name} ({element_type}) for {game_id}")
        return element_info
    
    def add_ui_layout(self, 
                     game_id: str, 
                     layout_id: str, 
                     layout_name: str,
                     ui_regions: Dict[str, Dict[str, Any]],
                     description: str = "") -> Dict[str, Any]:
        """Add a UI layout for a game"""
        if game_id not in self.games:
            logger.warning(f"Cannot add UI layout for unknown game: {game_id}")
            return None
        
        layout_info = {
            "id": layout_id,
            "name": layout_name,
            "ui_regions": ui_regions,
            "description": description,
            "added_at": time.time()
        }
        
        # Initialize UI layouts if not exists
        if game_id not in self.ui_layouts:
            self.ui_layouts[game_id] = {}
        
        self.ui_layouts[game_id][layout_id] = layout_info
        
        # Save data
        self.save_data()
        
        logger.info(f"Added UI layout: {layout_name} for {game_id}")
        return layout_info
    
    def add_similar_games(self, 
                         game_id1: str, 
                         game_id2: str,
                         similarity_score: float,
                         similarity_aspects: List[str] = None) -> Dict[str, Any]:
        """Add similarity relationship between two games"""
        if game_id1 not in self.games or game_id2 not in self.games:
            logger.warning(f"Cannot add similarity for unknown games: {game_id1} or {game_id2}")
            return None
        
        similarity_info = {
            "games": [game_id1, game_id2],
            "score": similarity_score,
            "aspects": similarity_aspects or [],
            "added_at": time.time()
        }
        
        # Add to both games
        if game_id1 not in self.similar_games:
            self.similar_games[game_id1] = []
        
        if game_id2 not in self.similar_games:
            self.similar_games[game_id2] = []
        
        # Check if similarity already exists
        for idx, sim in enumerate(self.similar_games[game_id1]):
            if game_id2 in sim.get("games", []):
                # Update existing similarity
                self.similar_games[game_id1][idx] = similarity_info
                break
        else:
            # Add new similarity
            self.similar_games[game_id1].append(similarity_info)
        
        # Add to second game if not already exists
        for idx, sim in enumerate(self.similar_games[game_id2]):
            if game_id1 in sim.get("games", []):
                # Update existing similarity
                self.similar_games[game_id2][idx] = similarity_info
                break
        else:
            # Add new similarity
            self.similar_games[game_id2].append(similarity_info)
        
        # Save data
        self.save_data()
        
        logger.info(f"Added similarity between {game_id1} and {game_id2}: {similarity_score}")
        return similarity_info
    
    def get_game(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get game information by ID"""
        return self.games.get(game_id)
    
    def get_game_by_name(self, game_name: str) -> Optional[Dict[str, Any]]:
        """Get game information by name"""
        for game_id, game_info in self.games.items():
            if game_info.get("name") == game_name:
                return game_info
        return None
    
    def get_game_elements(self, game_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all elements for a game"""
        return self.game_elements.get(game_id, {})
    
    def get_ui_layouts(self, game_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all UI layouts for a game"""
        return self.ui_layouts.get(game_id, {})
    
    def get_similar_games(self, game_id: str, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Get games similar to the specified game"""
        if game_id not in self.similar_games:
            return []
        
        similar = []
        for sim in self.similar_games[game_id]:
            if sim.get("score", 0) >= min_score:
                # Get the other game ID
                other_game_id = next((g for g in sim.get("games", []) if g != game_id), None)
                if other_game_id and other_game_id in self.games:
                    similar.append({
                        "game_id": other_game_id,
                        "game_name": self.games[other_game_id].get("name", ""),
                        "similarity_score": sim.get("score", 0),
                        "similarity_aspects": sim.get("aspects", [])
                    })
        
        # Sort by similarity score (highest first)
        similar.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return similar

class EnhancedSpatialMemory:
    """Persistent spatial memory system for tracking game world and navigating"""
    
    def __init__(self, game_id: Optional[str] = None):
        """Initialize with optional game ID"""
        self.game_id = game_id
        
        # Location graph
        self.locations = {}  # location_id -> location data
        self.connections = {}  # location_id -> {connected location_id -> connection data}
        
        # Current state
        self.current_location_id = None
        self.visited_locations = set()
        self.location_history = deque(maxlen=20)
        
        # Visual landmarks for recognition
        self.location_landmarks = {}  # location_id -> list of landmarks
        
        # Path planning
        self.current_path = []  # Current planned path
        
        logger.info(f"EnhancedSpatialMemory initialized for game: {game_id}")
    
    def add_location(self, 
                    location_id: str, 
                    location_name: str,
                    location_type: str,
                    coordinates: Optional[Tuple[float, float, float]] = None,
                    landmarks: List[Dict[str, Any]] = None,
                    description: str = "") -> Dict[str, Any]:
        """Add a location to the spatial memory"""
        location_data = {
            "id": location_id,
            "name": location_name,
            "type": location_type,
            "coordinates": coordinates,
            "description": description,
            "added_at": time.time(),
            "visit_count": 0
        }
        
        self.locations[location_id] = location_data
        
        # Store landmarks if provided
        if landmarks:
            self.location_landmarks[location_id] = landmarks
        
        # Initialize connections
        if location_id not in self.connections:
            self.connections[location_id] = {}
        
        logger.info(f"Added location: {location_name} ({location_id})")
        return location_data
    
    def add_connection(self, 
                      location_id1: str, 
                      location_id2: str,
                      connection_type: str = "bidirectional",
                      distance: float = 1.0,
                      description: str = "") -> Dict[str, Any]:
        """Add a connection between locations"""
        if location_id1 not in self.locations or location_id2 not in self.locations:
            logger.warning(f"Cannot add connection between unknown locations: {location_id1} or {location_id2}")
            return None
        
        connection_data = {
            "locations": [location_id1, location_id2],
            "type": connection_type,
            "distance": distance,
            "description": description,
            "added_at": time.time()
        }
        
        # Add to first location
        if location_id1 not in self.connections:
            self.connections[location_id1] = {}
        
        self.connections[location_id1][location_id2] = connection_data
        
        # Add to second location if bidirectional
        if connection_type == "bidirectional":
            if location_id2 not in self.connections:
                self.connections[location_id2] = {}
            
            self.connections[location_id2][location_id1] = connection_data
        
        logger.info(f"Added {connection_type} connection between {location_id1} and {location_id2}")
        return connection_data
    
    def mark_location_visited(self, location_id: str):
        """Mark a location as visited"""
        if location_id not in self.locations:
            logger.warning(f"Cannot mark unknown location as visited: {location_id}")
            return
        
        # Update visit count
        self.locations[location_id]["visit_count"] = self.locations[location_id].get("visit_count", 0) + 1
        
        # Add to visited set
        self.visited_locations.add(location_id)
        
        # Add to history
        self.location_history.append(location_id)
        
        # Update current location
        self.current_location_id = location_id
        
        logger.info(f"Marked location visited: {location_id}")
    
    def get_current_location(self) -> Optional[Dict[str, Any]]:
        """Get the current location data"""
        if not self.current_location_id:
            return None
        
        return self.locations.get(self.current_location_id)
    
    def get_connected_locations(self, location_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get locations connected to the specified location (or current if None)"""
        if location_id is None:
            location_id = self.current_location_id
        
        if not location_id or location_id not in self.connections:
            return []
        
        connected = []
        for connected_id, connection in self.connections[location_id].items():
            if connected_id in self.locations:
                location_data = self.locations[connected_id].copy()
                location_data["connection"] = connection
                connected.append(location_data)
        
        return connected
    
    def find_path(self, 
                start_id: Optional[str] = None, 
                target_id: str = None) -> List[str]:
        """Find a path between locations (uses current location if start is None)"""
        if start_id is None:
            start_id = self.current_location_id
        
        if not start_id or not target_id:
            return []
        
        if start_id not in self.locations or target_id not in self.locations:
            return []
        
        # Simple breadth-first search
        visited = {start_id}
        queue = deque([(start_id, [start_id])])
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_id:
                self.current_path = path
                return path
            
            for neighbor in self.connections.get(current, {}):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def detect_loop(self) -> bool:
        """Detect if the agent is in a navigation loop"""
        if len(self.location_history) < 3:
            return False
        
        # Convert to list for easier manipulation
        history = list(self.location_history)
        
        # Check for simple loops (e.g., A->B->A->B)
        for pattern_len in range(1, 5):  # Check patterns up to length 5
            if len(history) >= pattern_len * 2:
                pattern = history[-pattern_len:]
                previous = history[-(pattern_len*2):-pattern_len]
                
                if pattern == previous:
                    return True
        
        return False
    
    def get_exploration_suggestion(self) -> Optional[Dict[str, Any]]:
        """Suggest a location to explore next"""
        if not self.current_location_id:
            return None
        
        # Get connected locations
        connected = self.get_connected_locations()
        
        if not connected:
            return None
        
        # Prioritize unvisited locations
        unvisited = [loc for loc in connected if loc["id"] not in self.visited_locations]
        
        if unvisited:
            return unvisited[0]
        
        # If all are visited, choose least visited
        least_visited = min(connected, key=lambda x: x.get("visit_count", 0))
        return least_visited
    
    def save_to_file(self, filepath: str):
        """Save spatial memory to a file"""
        data = {
            "game_id": self.game_id,
            "locations": self.locations,
            "connections": self.connections,
            "visited_locations": list(self.visited_locations),
            "current_location_id": self.current_location_id,
            "location_history": list(self.location_history),
            "location_landmarks": self.location_landmarks
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved spatial memory to {filepath}")
    
    def load_from_file(self, filepath: str) -> bool:
        """Load spatial memory from a file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.game_id = data.get("game_id")
            self.locations = data.get("locations", {})
            self.connections = data.get("connections", {})
            self.visited_locations = set(data.get("visited_locations", []))
            self.current_location_id = data.get("current_location_id")
            self.location_landmarks = data.get("location_landmarks", {})
            
            # Convert history to deque
            self.location_history = deque(
                data.get("location_history", []),
                maxlen=self.location_history.maxlen
            )
            
            logger.info(f"Loaded spatial memory from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading spatial memory: {e}")
            return False

class SceneGraphAnalyzer:
    """Analyzes scene graphs representing relationships between objects"""
    
    def __init__(self):
        """Initialize the scene graph analyzer"""
        # Current scene objects and relationships
        self.objects = {}  # object_id -> object data
        self.relationships = []  # list of relationship data
        
        # Scene history
        self.scene_history = deque(maxlen=10)
        
        logger.info("SceneGraphAnalyzer initialized")
    
    def add_object(self, 
                 object_id: str, 
                 object_type: str,
                 object_name: str,
                 bbox: Optional[Tuple[int, int, int, int]] = None,
                 attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add an object to the scene graph"""
        object_data = {
            "id": object_id,
            "type": object_type,
            "name": object_name,
            "bbox": bbox,  # x, y, width, height
            "attributes": attributes or {},
            "added_at": time.time()
        }
        
        self.objects[object_id] = object_data
        return object_data
    
    def add_relationship(self, 
                       subject_id: str, 
                       predicate: str,
                       object_id: str,
                       attributes: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a relationship between objects"""
        if subject_id not in self.objects or object_id not in self.objects:
            logger.warning(f"Cannot add relationship with unknown objects: {subject_id} or {object_id}")
            return None
        
        relationship_data = {
            "subject_id": subject_id,
            "predicate": predicate,
            "object_id": object_id,
            "attributes": attributes or {},
            "added_at": time.time()
        }
        
        self.relationships.append(relationship_data)
        return relationship_data
    
    def clear_scene(self):
        """Clear the current scene"""
        # Store current scene in history before clearing
        if self.objects:
            self.scene_history.append({
                "objects": self.objects.copy(),
                "relationships": self.relationships.copy(),
                "timestamp": time.time()
            })
        
        self.objects = {}
        self.relationships = []
    
    def build_from_detections(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a scene graph from object detections"""
        # Clear previous scene
        self.clear_scene()
        
        # Add detected objects
        for i, detection in enumerate(detections):
            object_id = f"obj_{i}"
            self.add_object(
                object_id=object_id,
                object_type=detection.get("class", "unknown"),
                object_name=detection.get("name", f"Object {i}"),
                bbox=detection.get("bbox"),
                attributes=detection.get("attributes", {})
            )
        
        # Add spatial relationships
        self._add_spatial_relationships()
        
        return {
            "object_count": len(self.objects),
            "relationship_count": len(self.relationships),
            "timestamp": time.time()
        }
    
    def _add_spatial_relationships(self):
        """Add spatial relationships based on object positions"""
        objects_with_bbox = [obj for obj in self.objects.values() if obj.get("bbox")]
        
        for obj1 in objects_with_bbox:
            for obj2 in objects_with_bbox:
                if obj1["id"] == obj2["id"]:
                    continue
                
                bbox1 = obj1["bbox"]
                bbox2 = obj2["bbox"]
                
                # Calculate centers
                center1 = (bbox1[0] + bbox1[2] // 2, bbox1[1] + bbox1[3] // 2)
                center2 = (bbox2[0] + bbox2[2] // 2, bbox2[1] + bbox2[3] // 2)
                
                # Add left/right relationship
                if center1[0] < center2[0]:
                    self.add_relationship(obj1["id"], "left_of", obj2["id"])
                else:
                    self.add_relationship(obj1["id"], "right_of", obj2["id"])
                
                # Add above/below relationship
                if center1[1] < center2[1]:
                    self.add_relationship(obj1["id"], "above", obj2["id"])
                else:
                    self.add_relationship(obj1["id"], "below", obj2["id"])
    
    def get_object_relationships(self, object_id: str) -> List[Dict[str, Any]]:
        """Get all relationships involving an object"""
        if object_id not in self.objects:
            return []
        
        return [
            rel for rel in self.relationships
            if rel["subject_id"] == object_id or rel["object_id"] == object_id
        ]
    
    def compare_with_previous(self) -> Dict[str, Any]:
        """Compare current scene with the previous scene"""
        if not self.scene_history:
            return {"changes": "no_previous_scene"}
        
        previous = self.scene_history[-1]
        
        # Find added objects
        added_objects = [
            obj for obj_id, obj in self.objects.items()
            if obj_id not in previous["objects"]
        ]
        
        # Find removed objects
        removed_objects = [
            obj for obj_id, obj in previous["objects"].items()
            if obj_id not in self.objects
        ]
        
        # Find changed objects
        changed_objects = []
        for obj_id, obj in self.objects.items():
            if obj_id in previous["objects"]:
                prev_obj = previous["objects"][obj_id]
                # Check if bbox changed
                if obj.get("bbox") != prev_obj.get("bbox"):
                    changed_objects.append({
                        "id": obj_id,
                        "name": obj.get("name"),
                        "change_type": "moved",
                        "old_bbox": prev_obj.get("bbox"),
                        "new_bbox": obj.get("bbox")
                    })
                # Check if attributes changed
                elif obj.get("attributes") != prev_obj.get("attributes"):
                    changed_objects.append({
                        "id": obj_id,
                        "name": obj.get("name"),
                        "change_type": "attributes_changed",
                        "old_attributes": prev_obj.get("attributes"),
                        "new_attributes": obj.get("attributes")
                    })
        
        return {
            "added_objects": added_objects,
            "removed_objects": removed_objects,
            "changed_objects": changed_objects,
            "timestamp": time.time()
        }
    
    def describe_scene(self) -> str:
        """Generate a natural language description of the current scene"""
        if not self.objects:
            return "Empty scene."
        
        # Group objects by type
        type_counts = {}
        for obj in self.objects.values():
            obj_type = obj.get("type", "unknown")
            if obj_type not in type_counts:
                type_counts[obj_type] = 0
            type_counts[obj_type] += 1
        
        # Generate description
        parts = []
        
        # Describe object types
        for obj_type, count in type_counts.items():
            if count == 1:
                parts.append(f"There is 1 {obj_type}")
            else:
                parts.append(f"There are {count} {obj_type}s")
        
        # Add some spatial relationships
        if len(self.relationships) > 0:
            # Just pick a few relationships
            sample_rels = self.relationships[:3]
            for rel in sample_rels:
                subject = self.objects.get(rel["subject_id"], {}).get("name", "something")
                predicate = rel["predicate"]
                obj = self.objects.get(rel["object_id"], {}).get("name", "something else")
                parts.append(f"{subject} is {predicate} {obj}")
        
        return " ".join(parts) + "."

class GameActionRecognition:
    """Recognizes and categorizes game actions and events"""
    
    def __init__(self):
        """Initialize action recognition"""
        # Action templates
        self.action_templates = {}  # action_id -> template data
        
        # Action history
        self.action_history = deque(maxlen=20)
        
        # Current action
        self.current_action_id = None
        self.current_action_start_time = None
        self.current_action_confidence = 0.0
        
        # Action transitions (for prediction)
        self.action_transitions = {}  # from_action -> {to_action -> count}
        
        logger.info("GameActionRecognition initialized")
    
    def add_action_template(self, 
                          action_id: str, 
                          action_name: str,
                          action_type: str,
                          required_objects: List[Dict[str, Any]] = None,
                          required_motion: Optional[str] = None,
                          duration_range: Optional[Tuple[float, float]] = None,
                          description: str = "") -> Dict[str, Any]:
        """Add an action template for recognition"""
        template = {
            "id": action_id,
            "name": action_name,
            "type": action_type,
            "required_objects": required_objects or [],
            "required_motion": required_motion,
            "duration_range": duration_range,
            "description": description,
            "added_at": time.time()
        }
        
        self.action_templates[action_id] = template
        
        # Initialize transitions
        if action_id not in self.action_transitions:
            self.action_transitions[action_id] = {}
        
        logger.info(f"Added action template: {action_name} ({action_id})")
        return template
    
    def detect_action(self, 
                     frame: np.ndarray, 
                     detected_objects: List[Dict[str, Any]],
                     scene_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect the current action based on frame and objects"""
        # Calculate match scores for all templates
        best_match = None
        best_score = 0.0
        
        for action_id, template in self.action_templates.items():
            # Calculate match score
            score = self._calculate_match_score(template, detected_objects, scene_info)
            
            if score > best_score and score >= 0.5:  # Threshold
                best_match = action_id
                best_score = score
        
        # Get current time
        current_time = time.time()
        
        if best_match:
            # Check if action changed
            if best_match != self.current_action_id:
                # Record transition
                if self.current_action_id:
                    self._record_transition(self.current_action_id, best_match)
                
                # Add previous action to history if it exists
                if self.current_action_id and self.current_action_start_time:
                    prev_action = {
                        "id": self.current_action_id,
                        "name": self.action_templates[self.current_action_id]["name"],
                        "type": self.action_templates[self.current_action_id]["type"],
                        "start_time": self.current_action_start_time,
                        "end_time": current_time,
                        "duration": current_time - self.current_action_start_time,
                        "confidence": self.current_action_confidence
                    }
                    self.action_history.append(prev_action)
                
                # Update current action
                self.current_action_id = best_match
                self.current_action_start_time = current_time
                self.current_action_confidence = best_score
            
            # Update confidence if same action
            elif best_score > self.current_action_confidence:
                self.current_action_confidence = best_score
            
            # Format result
            template = self.action_templates[best_match]
            duration = 0.0
            if self.current_action_start_time:
                duration = current_time - self.current_action_start_time
            
            result = {
                "id": best_match,
                "name": template["name"],
                "type": template["type"],
                "confidence": best_score,
                "duration": duration,
                "description": template.get("description", "")
            }
            
            return result
        
        # No action detected
        if self.current_action_id:
            # Add previous action to history
            prev_action = {
                "id": self.current_action_id,
                "name": self.action_templates[self.current_action_id]["name"],
                "type": self.action_templates[self.current_action_id]["type"],
                "start_time": self.current_action_start_time,
                "end_time": current_time,
                "duration": current_time - self.current_action_start_time,
                "confidence": self.current_action_confidence
            }
            self.action_history.append(prev_action)
            
            # Reset current action
            self.current_action_id = None
            self.current_action_start_time = None
            self.current_action_confidence = 0.0
        
        return {
            "id": "unknown",
            "name": "Unknown",
            "type": "unknown",
            "confidence": 0.0,
            "duration": 0.0
        }
    
    def _calculate_match_score(self, 
                             template: Dict[str, Any], 
                             detected_objects: List[Dict[str, Any]],
                             scene_info: Optional[Dict[str, Any]] = None) -> float:
        """Calculate how well objects match an action template"""
        # Required objects score
        req_objects = template.get("required_objects", [])
        object_score = 0.0
        
        if req_objects:
            matched = 0
            for req_obj in req_objects:
                req_type = req_obj.get("type")
                req_name = req_obj.get("name")
                
                for obj in detected_objects:
                    obj_type = obj.get("type", obj.get("class"))
                    obj_name = obj.get("name")
                    
                    if (req_type and obj_type and req_type == obj_type) or \
                       (req_name and obj_name and req_name in obj_name):
                        matched += 1
                        break
            
            object_score = matched / len(req_objects)
        else:
            object_score = 1.0  # No required objects
        
        # Motion score (if applicable)
        motion_score = 1.0
        req_motion = template.get("required_motion")
        
        if req_motion and scene_info and "motion" in scene_info:
            scene_motion = scene_info["motion"]
            motion_score = 1.0 if req_motion == scene_motion else 0.0
        
        # Duration score (if applicable)
        duration_score = 1.0
        duration_range = template.get("duration_range")
        
        if duration_range and self.current_action_start_time:
            min_duration, max_duration = duration_range
            current_duration = time.time() - self.current_action_start_time
            
            if current_duration < min_duration:
                # Not long enough yet
                duration_score = current_duration / min_duration
            elif max_duration and current_duration > max_duration:
                # Too long
                duration_score = max(0, 1.0 - (current_duration - max_duration) / max_duration)
        
        # Combine scores (weighted)
        return object_score * 0.6 + motion_score * 0.3 + duration_score * 0.1
    
    def _record_transition(self, from_action: str, to_action: str):
        """Record an action transition for prediction"""
        if from_action not in self.action_transitions:
            self.action_transitions[from_action] = {}
        
        if to_action not in self.action_transitions[from_action]:
            self.action_transitions[from_action][to_action] = 0
        
        self.action_transitions[from_action][to_action] += 1
    
    def predict_next_action(self) -> Optional[Dict[str, Any]]:
        """Predict the most likely next action based on history"""
        if not self.current_action_id or self.current_action_id not in self.action_transitions:
            return None
        
        transitions = self.action_transitions[self.current_action_id]
        if not transitions:
            return None
        
        # Find most frequent transition
        next_action_id = max(transitions.items(), key=lambda x: x[1])[0]
        transition_count = transitions[next_action_id]
        total_transitions = sum(transitions.values())
        
        if next_action_id in self.action_templates:
            template = self.action_templates[next_action_id]
            
            return {
                "id": next_action_id,
                "name": template["name"],
                "type": template["type"],
                "confidence": transition_count / total_transitions,
                "description": template.get("description", "")
            }
        
        return None
    
    def get_action_sequence(self, max_length: int = 5) -> List[Dict[str, Any]]:
        """Get the recent action sequence"""
        # Convert to list
        sequence = list(self.action_history)
        
        # Add current action if exists
        if self.current_action_id and self.current_action_start_time:
            current_action = {
                "id": self.current_action_id,
                "name": self.action_templates[self.current_action_id]["name"],
                "type": self.action_templates[self.current_action_id]["type"],
                "start_time": self.current_action_start_time,
                "end_time": time.time(),
                "duration": time.time() - self.current_action_start_time,
                "confidence": self.current_action_confidence,
                "is_current": True
            }
            sequence.append(current_action)
        
        # Return most recent actions (up to max_length)
        return sequence[-max_length:]
    
    def detect_pattern(self, max_length: int = 5) -> Optional[Dict[str, Any]]:
        """Detect if recent actions form a known pattern"""
        sequence = self.get_action_sequence(max_length * 2)  # Get enough history
        
        if len(sequence) < 3:
            return None
        
        # Extract IDs
        action_ids = [action["id"] for action in sequence]
        
        # Look for repeating patterns
        for pattern_len in range(2, min(len(action_ids) // 2 + 1, max_length) + 1):
            pattern = action_ids[-pattern_len:]
            previous = action_ids[-(pattern_len * 2):-pattern_len]
            
            if pattern == previous:
                return {
                    "type": "repetition",
                    "pattern": pattern,
                    "action_names": [self.action_templates[action_id]["name"] for action_id in pattern],
                    "confidence": 0.9
                }
        
        # Check for "combo" sequences in action-based games
        attack_actions = [action for action in sequence if "attack" in action.get("type", "")]
        if len(attack_actions) >= 3:
            return {
                "type": "combo",
                "action_count": len(attack_actions),
                "action_names": [action["name"] for action in attack_actions],
                "confidence": 0.8
            }
        
        return None

class EnhancedGameRecognitionSystem:
    """Advanced system for recognizing and understanding game content"""
    
    def __init__(self, knowledge_base: Optional[GameKnowledgeBase] = None):
        """Initialize the game recognition system"""
        self.knowledge_base = knowledge_base or GameKnowledgeBase()
        
        # Component systems
        self.spatial_memory = EnhancedSpatialMemory()
        self.scene_graph = SceneGraphAnalyzer()
        self.action_recognition = GameActionRecognition()
        
        # Current game info
        self.current_game_id = None
        self.current_game_info = None
        
        # Initialize models
        self._initialize_models()
        
        # Frame history
        self.frame_history = deque(maxlen=30)
        
        # Analysis cache (to avoid redundant processing)
        self.analysis_cache = {}
        self.cache_expiry = 0.2  # seconds
        
        logger.info("EnhancedGameRecognitionSystem initialized")
    
    def _initialize_models(self):
        """Initialize detection models"""
        # In a real implementation, this would load actual models
        # Here we just set up placeholders
        self.game_recognizer = None  # Game recognition model
        self.object_detector = None  # Object detection model
        self.text_recognizer = None  # OCR engine
        self.ui_detector = None  # UI element detector
        self.motion_detector = None  # Motion detection
        
        logger.info("Detection models initialized")
    
    async def identify_game(self, frame: np.ndarray) -> Dict[str, Any]:
        """Identify the game being played from a video frame"""
        # Check if we already have a current game
        if self.current_game_id and self.current_game_info:
            # Verify it's still the same game (reduced confidence threshold)
            result = self._identify_game_internal(frame, confidence_threshold=0.5)
            
            if result["game_id"] == self.current_game_id:
                return {
                    "game_id": self.current_game_id,
                    "game_name": self.current_game_info.get("name", "Unknown Game"),
                    "confidence": result["confidence"],
                    "is_current": True
                }
        
        # Identify game with normal threshold
        result = self._identify_game_internal(frame)
        
        if result["confidence"] >= 0.7:  # Good confidence
            # Update current game
            self.current_game_id = result["game_id"]
            self.current_game_info = self.knowledge_base.get_game(result["game_id"])
            
            # Update spatial memory
            self.spatial_memory.game_id = result["game_id"]
            
            return {
                "game_id": result["game_id"],
                "game_name": result["game_name"],
                "confidence": result["confidence"],
                "is_new": True
            }
        
        return result
    
    def _identify_game_internal(self, frame: np.ndarray, confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Internal game identification logic"""
        # This would use a real game recognition model in a real implementation
        # For now, just return a placeholder implementation
        
        # Simple game identification based on known games
        if not self.knowledge_base.games:
            return {
                "game_id": None,
                "game_name": "Unknown Game",
                "confidence": 0.0
            }
        
        # For demonstration, return the first game in the knowledge base
        first_game_id = next(iter(self.knowledge_base.games))
        first_game = self.knowledge_base.games[first_game_id]
        
        return {
            "game_id": first_game_id,
            "game_name": first_game.get("name", "Unknown Game"),
            "confidence": 0.9  # High confidence for demonstration
        }
    
    async def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the frame"""
        # Check cache
        cache_key = "objects"
        current_time = time.time()
        
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if current_time - cache_entry["timestamp"] < self.cache_expiry:
                return cache_entry["data"]
        
        # Placeholder object detection
        # This would use a real object detection model in a real implementation
        objects = [
            {
                "id": "obj_0",
                "class": "character",
                "name": "Player Character",
                "bbox": (100, 100, 50, 100),  # x, y, width, height
                "confidence": 0.95
            },
            {
                "id": "obj_1",
                "class": "item",
                "name": "Treasure Chest",
                "bbox": (200, 150, 30, 20),
                "confidence": 0.85
            },
            {
                "id": "obj_2",
                "class": "environment",
                "name": "Tree",
                "bbox": (300, 100, 40, 80),
                "confidence": 0.9
            }
        ]
        
        # Update cache
        self.analysis_cache[cache_key] = {
            "data": objects,
            "timestamp": current_time
        }
        
        return objects
    
    async def detect_text(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and recognize text in the frame"""
        # Check cache
        cache_key = "text"
        current_time = time.time()
        
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if current_time - cache_entry["timestamp"] < self.cache_expiry:
                return cache_entry["data"]
        
        # Placeholder text detection
        # This would use a real OCR engine in a real implementation
        text_regions = [
            {
                "id": "text_0",
                "text": "Health: 100/100",
                "bbox": (10, 10, 100, 20),
                "confidence": 0.92,
                "type": "status"
            },
            {
                "id": "text_1",
                "text": "Objective: Find the treasure",
                "bbox": (10, 40, 200, 20),
                "confidence": 0.88,
                "type": "objective"
            }
        ]
        
        # Update cache
        self.analysis_cache[cache_key] = {
            "data": text_regions,
            "timestamp": current_time
        }
        
        return text_regions
    
    async def detect_ui_elements(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements in the frame"""
        # Check cache
        cache_key = "ui"
        current_time = time.time()
        
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if current_time - cache_entry["timestamp"] < self.cache_expiry:
                return cache_entry["data"]
        
        # Placeholder UI detection
        # This would use a real UI detector in a real implementation
        ui_elements = [
            {
                "id": "ui_0",
                "type": "health_bar",
                "bbox": (10, 10, 100, 20),
                "confidence": 0.95,
                "value": 100,
                "max_value": 100
            },
            {
                "id": "ui_1",
                "type": "minimap",
                "bbox": (500, 10, 100, 100),
                "confidence": 0.9
            },
            {
                "id": "ui_2",
                "type": "inventory",
                "bbox": (10, 400, 400, 100),
                "confidence": 0.85,
                "items": 5
            }
        ]
        
        # Update cache
        self.analysis_cache[cache_key] = {
            "data": ui_elements,
            "timestamp": current_time
        }
        
        return ui_elements
    
    async def detect_events(self, 
                         current_frame: np.ndarray, 
                         previous_frame: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Detect events between frames"""
        if previous_frame is None:
            return []
        
        # This would use real event detection in a real implementation
        # For now, just return a placeholder implementation
        
        # Compare scene graphs if available
        events = []
        
        # Add to frame history
        self.frame_history.append(current_frame)
        
        return events
    
    async def recognize_location(self, frame: np.ndarray) -> Dict[str, Any]:
        """Recognize the current location in the game"""
        # Check if we have any locations in spatial memory
        if not self.spatial_memory.locations:
            # Create a new default location
            location_id = f"loc_{int(time.time())}"
            self.spatial_memory.add_location(
                location_id=location_id,
                location_name="Starting Area",
                location_type="area"
            )
            
            # Mark as visited
            self.spatial_memory.mark_location_visited(location_id)
            
            return {
                "id": location_id,
                "name": "Starting Area",
                "type": "area",
                "confidence": 0.8,
                "is_new": True
            }
        
        # Try to match with existing locations
        # This would use real location recognition in a real implementation
        
        # For now, just use the current location
        current = self.spatial_memory.get_current_location()
        
        if current:
            return {
                "id": current["id"],
                "name": current["name"],
                "type": current["type"],
                "confidence": 0.9,
                "visit_count": current.get("visit_count", 1)
            }
        
        # Create a new location if no current location
        location_id = f"loc_{int(time.time())}"
        self.spatial_memory.add_location(
            location_id=location_id,
            location_name=f"Area {location_id}",
            location_type="area"
        )
        
        # Mark as visited
        self.spatial_memory.mark_location_visited(location_id)
        
        return {
            "id": location_id,
            "name": f"Area {location_id}",
            "type": "area",
            "confidence": 0.7,
            "is_new": True
        }
    
    async def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive analysis of a frame"""
        start_time = time.time()
        
        # Get previous frame if available
        previous_frame = None
        if self.frame_history:
            previous_frame = self.frame_history[-1]
        
        # Run analyses in parallel for efficiency
        game_future = asyncio.create_task(self.identify_game(frame))
        objects_future = asyncio.create_task(self.detect_objects(frame))
        text_future = asyncio.create_task(self.detect_text(frame))
        ui_future = asyncio.create_task(self.detect_ui_elements(frame))
        events_future = asyncio.create_task(self.detect_events(frame, previous_frame))
        location_future = asyncio.create_task(self.recognize_location(frame))
        
        # Await all futures
        game_info = await game_future
        objects = await objects_future
        text_regions = await text_future
        ui_elements = await ui_future
        events = await events_future
        location = await location_future
        
        # Process scene graph
        scene_info = self.scene_graph.build_from_detections(objects)
        
        # Detect action
        action = self.action_recognition.detect_action(frame, objects, scene_info)
        
        # Build game status
        game_status = self._build_game_status(ui_elements, text_regions)
        
        # Create analysis result
        analysis = {
            "game": game_info,
            "objects": objects,
            "text_regions": text_regions,
            "ui_elements": ui_elements,
            "events": events,
            "location": location,
            "scene_graph": scene_info,
            "action": action,
            "game_status": game_status,
            "processing_time": time.time() - start_time,
            "timestamp": time.time()
        }
        
        # Add to frame history
        self.frame_history.append(frame)
        
        return analysis
    
    def _build_game_status(self, 
                         ui_elements: List[Dict[str, Any]],
                         text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build game status from UI elements and text"""
        # Extract player status
        player_status = {}
        
        # Extract from UI elements
        for ui in ui_elements:
            ui_type = ui.get("type", "").lower()
            
            # Health
            if "health" in ui_type:
                player_status["health"] = ui.get("value", 0)
                player_status["max_health"] = ui.get("max_value", 100)
            
            # Mana/energy
            elif "mana" in ui_type or "energy" in ui_type:
                player_status["mana"] = ui.get("value", 0)
                player_status["max_mana"] = ui.get("max_value", 100)
            
            # Experience/level
            elif "experience" in ui_type or "level" in ui_type:
                player_status["experience"] = ui.get("value", 0)
                player_status["max_experience"] = ui.get("max_value", 100)
                if "level" in ui:
                    player_status["level"] = ui.get("level", 1)
        
        # Extract from text
        objectives = []
        status_text = []
        
        for text in text_regions:
            text_type = text.get("type", "").lower()
            text_content = text.get("text", "")
            
            if "objective" in text_type:
                objectives.append(text_content)
            elif "status" in text_type:
                status_text.append(text_content)
            
            # Try to extract numeric values (e.g., "Health: 100/100")
            if ":" in text_content:
                label, value = text_content.split(":", 1)
                label = label.strip().lower()
                
                if "health" in label:
                    try:
                        if "/" in value:
                            current, maximum = value.strip().split("/")
                            player_status["health"] = float(current)
                            player_status["max_health"] = float(maximum)
                        else:
                            player_status["health"] = float(value.strip())
                    except ValueError:
                        pass
                
                elif "mana" in label or "energy" in label:
                    try:
                        if "/" in value:
                            current, maximum = value.strip().split("/")
                            player_status["mana"] = float(current)
                            player_status["max_mana"] = float(maximum)
                        else:
                            player_status["mana"] = float(value.strip())
                    except ValueError:
                        pass
        
        # Build game status
        game_status = {
            "player_status": player_status,
            "objectives": objectives,
            "status_text": status_text,
            "in_dialog": any("dialog" in ui.get("type", "").lower() for ui in ui_elements),
            "in_menu": any("menu" in ui.get("type", "").lower() for ui in ui_elements),
            "in_combat": self.action_recognition.current_action_id == "combat"
        }
        
        return game_status
    
    def seed_initial_knowledge(self):
        """Seed the system with initial game knowledge"""
        # Add some games
        self.knowledge_base.add_game(
            game_id="witcher3",
            game_name="The Witcher 3: Wild Hunt",
            game_genre=["RPG", "Open World", "Action"],
            description="An action RPG in a fantasy world, following the adventures of monster hunter Geralt of Rivia."
        )
        
        self.knowledge_base.add_game(
            game_id="skyrim",
            game_name="The Elder Scrolls V: Skyrim",
            game_genre=["RPG", "Open World", "Action"],
            description="An open world action RPG set in the province of Skyrim."
        )
        
        self.knowledge_base.add_game(
            game_id="eldenring",
            game_name="Elden Ring",
            game_genre=["Action", "RPG", "Open World"],
            description="An action RPG set in a fantasy world created by Hidetaka Miyazaki and George R. R. Martin."
        )
        
        # Add similarity relationships
        self.knowledge_base.add_similar_games(
            game_id1="witcher3",
            game_id2="skyrim",
            similarity_score=0.8,
            similarity_aspects=["Open World", "RPG", "Quest Structure"]
        )
        
        self.knowledge_base.add_similar_games(
            game_id1="witcher3",
            game_id2="eldenring",
            similarity_score=0.7,
            similarity_aspects=["Combat", "Fantasy Setting"]
        )
        
        # Add action templates
        self.action_recognition.add_action_template(
            action_id="combat",
            action_name="Combat",
            action_type="combat",
            required_objects=[{"type": "enemy"}, {"type": "weapon"}],
            required_motion="fast"
        )
        
        self.action_recognition.add_action_template(
            action_id="exploration",
            action_name="Exploration",
            action_type="exploration",
            required_objects=[{"type": "environment"}],
            required_motion="moderate"
        )
        
        self.action_recognition.add_action_template(
            action_id="dialog",
            action_name="Dialog",
            action_type="interaction",
            required_objects=[{"type": "npc"}],
            required_motion="none"
        )
        
        logger.info("Seeded initial knowledge")

class RealTimeGameProcessor:
    """
    Real-time processing system for game streams, handling frame capture,
    analysis, event generation, and optimized processing.
    """
    
    def __init__(self, 
               game_system: Optional[EnhancedGameRecognitionSystem] = None,
               input_source: Union[int, str] = 0,
               processing_fps: int = 30):
        """
        Initialize the processor
        
        Args:
            game_system: Game recognition system
            input_source: Video input source (camera index or file path)
            processing_fps: Target processing frame rate
        """
        self.game_system = game_system or EnhancedGameRecognitionSystem()
        self.input_source = input_source
        self.target_fps = processing_fps
        self.processing_interval = 1.0 / processing_fps
        
        # Video capture
        self.capture = None
        self.frame_width = 0
        self.frame_height = 0
        
        # Processing state
        self.is_running = False
        self.frame_count = 0
        self.last_frame_time = 0
        self.processing_times = deque(maxlen=100)
        
        # Analysis results
        self.last_analysis = None
        self.last_frame = None
        
        # Event handlers
        self.event_handlers = {
            "on_frame": [],
            "on_analysis": [],
            "on_game_change": [],
            "on_location_change": [],
            "on_action_change": [],
            "on_error": []
        }
        
        logger.info(f"RealTimeGameProcessor initialized with target {processing_fps} FPS")
    
    def start_processing(self):
        """Start real-time processing"""
        if self.is_running:
            logger.warning("Processing already running")
            return False
        
        # Initialize video capture
        try:
            self.capture = cv2.VideoCapture(self.input_source)
            
            if not self.capture.isOpened():
                logger.error(f"Could not open video source: {self.input_source}")
                return False
            
            # Get video properties
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Started video capture: {self.frame_width}x{self.frame_height}")
            
            # Set running flag
            self.is_running = True
            
            # Start processing thread
            import threading
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting processing: {e}")
            return False
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        logger.info("Stopped processing")
        return True
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            loop_start_time = time.time()
            
            try:
                # Capture frame
                ret, frame = self.capture.read()
                
                if not ret:
                    logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Store the frame
                self.last_frame = frame
                self.frame_count += 1
                
                # Notify frame handlers
                for handler in self.event_handlers["on_frame"]:
                    try:
                        handler(frame, self.frame_count)
                    except Exception as e:
                        logger.error(f"Error in frame handler: {e}")
                
                # Only process every N frames for efficiency
                if self.frame_count % 3 == 0:  # Process every 3rd frame
                    asyncio.run(self._process_frame(frame))
                
                # Calculate processing time and sleep to maintain target FPS
                process_time = time.time() - loop_start_time
                self.processing_times.append(process_time)
                
                sleep_time = max(0, self.processing_interval - process_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                self.last_frame_time = time.time()
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                for handler in self.event_handlers["on_error"]:
                    try:
                        handler(e)
                    except Exception as e2:
                        logger.error(f"Error in error handler: {e2}")
                
                time.sleep(0.1)
    
    async def _process_frame(self, frame: np.ndarray):
        """Process a single frame"""
        try:
            # Analyze frame
            analysis = await self.game_system.analyze_frame(frame)
            
            # Store result
            prev_analysis = self.last_analysis
            self.last_analysis = analysis
            
            # Notify analysis handlers
            for handler in self.event_handlers["on_analysis"]:
                try:
                    handler(analysis)
                except Exception as e:
                    logger.error(f"Error in analysis handler: {e}")
            
            # Check for game change
            if prev_analysis is None or prev_analysis["game"].get("game_id") != analysis["game"].get("game_id"):
                for handler in self.event_handlers["on_game_change"]:
                    try:
                        handler(analysis["game"])
                    except Exception as e:
                        logger.error(f"Error in game change handler: {e}")
            
            # Check for location change
            if prev_analysis is None or prev_analysis["location"].get("id") != analysis["location"].get("id"):
                for handler in self.event_handlers["on_location_change"]:
                    try:
                        handler(analysis["location"])
                    except Exception as e:
                        logger.error(f"Error in location change handler: {e}")
            
            # Check for action change
            if prev_analysis is None or prev_analysis["action"].get("id") != analysis["action"].get("id"):
                for handler in self.event_handlers["on_action_change"]:
                    try:
                        handler(analysis["action"])
                    except Exception as e:
                        logger.error(f"Error in action change handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            for handler in self.event_handlers["on_error"]:
                try:
                    handler(e)
                except Exception as e2:
                    logger.error(f"Error in error handler: {e2}")
    
    def register_handler(self, event_type: str, handler_function):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            logger.warning(f"Unknown event type: {event_type}")
            return False
        
        self.event_handlers[event_type].append(handler_function)
        return True
    
    def unregister_handler(self, event_type: str, handler_function):
        """Unregister an event handler"""
        if event_type not in self.event_handlers:
            logger.warning(f"Unknown event type: {event_type}")
            return False
        
        try:
            self.event_handlers[event_type].remove(handler_function)
            return True
        except ValueError:
            logger.warning(f"Handler not found for event type: {event_type}")
            return False
    
    def get_last_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis result"""
        return self.last_analysis
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.processing_times:
            avg_processing_time = 0
            current_fps = 0
        else:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            current_fps = 1.0 / max(avg_processing_time, 0.001)
        
        return {
            "frame_count": self.frame_count,
            "target_fps": self.target_fps,
            "current_fps": current_fps,
            "avg_processing_time": avg_processing_time,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "is_running": self.is_running
        }
