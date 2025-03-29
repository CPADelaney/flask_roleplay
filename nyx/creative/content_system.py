# nyx/creative/content_system.py

import os
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ContentType(Enum):
    STORY = "story"
    POEM = "poem"
    LYRICS = "lyrics"
    JOURNAL = "journal"
    CODE = "code"
    ANALYSIS = "analysis"
    ASSESSMENT = "assessment"

class CreativeContentSystem:
    """
    System for managing AI-generated creative content including stories,
    poetry, song lyrics, journal entries, code, and analyses.
    """
    
    def __init__(self, base_directory: str = "ai_creations"):
        """
        Initialize the creative content system.
        
        Args:
            base_directory: Base directory for storing AI creations
        """
        self.base_directory = base_directory
        self.content_index = {}
        self.initialize_storage()
    
    def initialize_storage(self):
        """Initialize the storage directory structure."""
        # Create base directory if it doesn't exist
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        
        # Create subdirectories for each content type
        for content_type in ContentType:
            type_dir = os.path.join(self.base_directory, content_type.value)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
        
        # Create an index file if it doesn't exist
        index_path = os.path.join(self.base_directory, "content_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    self.content_index = json.load(f)
            except Exception as e:
                logger.error(f"Error loading content index: {e}")
                self.content_index = {}
        else:
            with open(index_path, 'w') as f:
                json.dump({}, f)
            self.content_index = {}
            
        logger.info(f"Initialized creative content storage at {self.base_directory}")
    
    async def store_content(self, 
                       content_type: Union[ContentType, str], 
                       title: str,
                       content: str,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a piece of AI-generated content.
        
        Args:
            content_type: Type of content (story, poem, etc.)
            title: Title of the content
            content: The actual content text
            metadata: Additional metadata about the content
            
        Returns:
            Information about the stored content
        """
        # Convert string content type to enum if needed
        if isinstance(content_type, str):
            try:
                content_type = ContentType(content_type)
            except ValueError:
                logger.warning(f"Invalid content type: {content_type}. Using 'STORY' as default.")
                content_type = ContentType.STORY
        
        # Generate a unique ID for the content
        content_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        timestamp = datetime.datetime.now().isoformat()
        
        # Prepare content record
        content_record = {
            "id": content_id,
            "type": content_type.value,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
            "metadata": metadata
        }
        
        # Determine file path
        type_dir = os.path.join(self.base_directory, content_type.value)
        
        # Clean filename (remove characters that might be problematic for filenames)
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
        safe_title = safe_title.replace(' ', '_')
        
        # Ensure filename uniqueness by adding timestamp and ID
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_str}_{safe_title}_{content_id[:8]}"
        
        # Add appropriate extension based on content type
        if content_type == ContentType.CODE:
            # Use appropriate extension based on metadata
            lang = metadata.get("language", "python").lower()
            extension_map = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "java": ".java",
                "c": ".c",
                "cpp": ".cpp",
                "csharp": ".cs",
                "go": ".go",
                "ruby": ".rb",
                "php": ".php",
                "rust": ".rs",
                "swift": ".swift",
                "kotlin": ".kt",
                "html": ".html",
                "css": ".css"
            }
            extension = extension_map.get(lang, ".txt")
        elif content_type == ContentType.ANALYSIS or content_type == ContentType.ASSESSMENT:
            extension = ".md"  # Markdown for structured content
        else:
            extension = ".txt"  # Default for creative text
        
        filepath = os.path.join(type_dir, filename + extension)
        
        # Store the content
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Add to index
            content_record["filepath"] = filepath
            self.content_index[content_id] = content_record
            
            # Save updated index
            with open(os.path.join(self.base_directory, "content_index.json"), 'w') as f:
                json.dump(self.content_index, f, indent=2)
            
            logger.info(f"Stored new {content_type.value}: {title} (ID: {content_id})")
            
            return content_record
        
        except Exception as e:
            logger.error(f"Error storing content: {e}")
            return {"error": str(e)}
    
    async def retrieve_content(self, content_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific piece of content by ID.
        
        Args:
            content_id: ID of the content to retrieve
            
        Returns:
            Content information and text
        """
        if content_id not in self.content_index:
            return {"error": f"Content with ID {content_id} not found"}
        
        content_record = self.content_index[content_id]
        filepath = content_record["filepath"]
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content_text = f.read()
            
            result = content_record.copy()
            result["content"] = content_text
            
            return result
        
        except Exception as e:
            logger.error(f"Error retrieving content {content_id}: {e}")
            return {"error": str(e)}
    
    async def list_content(self, 
                      content_type: Optional[Union[ContentType, str]] = None,
                      limit: int = 100,
                      offset: int = 0) -> Dict[str, Any]:
        """
        List available content, optionally filtered by type.
        
        Args:
            content_type: Type of content to filter by (optional)
            limit: Maximum number of items to return
            offset: Offset for pagination
            
        Returns:
            List of content information
        """
        # Filter by content type if specified
        if content_type:
            if isinstance(content_type, str):
                type_str = content_type
            else:
                type_str = content_type.value
                
            filtered_contents = [
                record for record in self.content_index.values()
                if record["type"] == type_str
            ]
        else:
            filtered_contents = list(self.content_index.values())
        
        # Sort by creation date (newest first)
        sorted_contents = sorted(
            filtered_contents,
            key=lambda x: x["created_at"],
            reverse=True
        )
        
        # Apply pagination
        paginated = sorted_contents[offset:offset+limit]
        
        return {
            "total": len(filtered_contents),
            "limit": limit,
            "offset": offset,
            "items": paginated
        }
    
    async def get_recent_creations(self, days: int = 7) -> Dict[str, Any]:
        """
        Get recent AI creations within the specified timeframe.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Recent creations grouped by type
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        recent_items = {}
        
        # Group by content type
        for content_id, record in self.content_index.items():
            if record["created_at"] >= cutoff_str:
                content_type = record["type"]
                if content_type not in recent_items:
                    recent_items[content_type] = []
                
                recent_items[content_type].append(record)
        
        # Sort each group by creation date
        for content_type in recent_items:
            recent_items[content_type] = sorted(
                recent_items[content_type],
                key=lambda x: x["created_at"],
                reverse=True
            )
        
        # Calculate statistics
        stats = {
            "total_items": sum(len(items) for items in recent_items.values()),
            "by_type": {ctype: len(items) for ctype, items in recent_items.items()}
        }
        
        return {
            "timeframe_days": days,
            "stats": stats,
            "items": recent_items
        }
    
    async def search_content(self, query: str) -> List[Dict[str, Any]]:
        """
        Simple search for content based on title and metadata.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching content records
        """
        query = query.lower()
        results = []
        
        for content_id, record in self.content_index.items():
            # Search in title
            if query in record["title"].lower():
                results.append(record)
                continue
            
            # Search in metadata
            for key, value in record.get("metadata", {}).items():
                if isinstance(value, str) and query in value.lower():
                    results.append(record)
                    break
        
        return results
