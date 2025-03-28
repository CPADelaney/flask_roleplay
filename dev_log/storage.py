# dev_log/storage.py

import os
import json
import datetime
import asyncio
from typing import Dict, List, Any, Optional
import uuid
import logging

from dev_log.models import DevLogEntry, SynergyRecommendation, ModuleOptimization, SystemInsight

logger = logging.getLogger(__name__)

class DevLogStorage:
    """Storage for developer logs."""
    
    def __init__(self, storage_dir: str = None):
        """Initialize the storage."""
        self.storage_dir = storage_dir or os.path.join(os.getcwd(), "nyx_dev_logs")
        self.logs = {}
        self.loaded = False
        self._lock = asyncio.Lock()
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Log files
        self.log_file = os.path.join(self.storage_dir, "dev_logs.jsonl")
        self.recommendations_file = os.path.join(self.storage_dir, "synergy_recommendations.jsonl")
        self.optimizations_file = os.path.join(self.storage_dir, "module_optimizations.jsonl")
        self.insights_file = os.path.join(self.storage_dir, "system_insights.jsonl")
        
        logger.info(f"DevLogStorage initialized with directory: {self.storage_dir}")
    
    async def initialize(self):
        """Load logs from storage."""
        if self.loaded:
            return
            
        async with self._lock:
            try:
                # Load logs
                await self._load_logs()
                self.loaded = True
                logger.info(f"Loaded {len(self.logs)} dev logs")
                return True
            except Exception as e:
                logger.error(f"Error loading dev logs: {e}")
                return False
    
    async def _load_logs(self):
        """Load logs from storage files."""
        self.logs = {}
        
        # Load all log files
        for file_path in [self.log_file, self.recommendations_file, 
                         self.optimizations_file, self.insights_file]:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                entry = json.loads(line)
                                if "id" in entry:
                                    self.logs[entry["id"]] = entry
                except Exception as e:
                    logger.error(f"Error loading logs from {file_path}: {e}")
    
    async def add_log(self, log_entry: DevLogEntry) -> str:
        """
        Add a log entry.
        
        Args:
            log_entry: Log entry to add
            
        Returns:
            ID of the created log entry
        """
        if not self.loaded:
            await self.initialize()
            
        async with self._lock:
            # Convert to dict if needed
            if hasattr(log_entry, "model_dump"):
                entry_dict = log_entry.model_dump()
            else:
                entry_dict = dict(log_entry)
            
            # Ensure ID is set
            if "id" not in entry_dict or not entry_dict["id"]:
                entry_dict["id"] = f"log_{uuid.uuid4().hex[:8]}"
            
            # Ensure timestamp is set
            if "timestamp" not in entry_dict or not entry_dict["timestamp"]:
                entry_dict["timestamp"] = datetime.datetime.now().isoformat()
            elif isinstance(entry_dict["timestamp"], datetime.datetime):
                entry_dict["timestamp"] = entry_dict["timestamp"].isoformat()
            
            # Add to memory
            self.logs[entry_dict["id"]] = entry_dict
            
            # Save to appropriate file
            log_type = entry_dict.get("log_type", "general")
            
            if log_type == "synergy_recommendation":
                file_path = self.recommendations_file
            elif log_type == "module_optimization":
                file_path = self.optimizations_file
            elif log_type == "system_insight":
                file_path = self.insights_file
            else:
                file_path = self.log_file
            
            try:
                with open(file_path, 'a') as f:
                    f.write(json.dumps(entry_dict) + "\n")
            except Exception as e:
                logger.error(f"Error saving log entry to {file_path}: {e}")
            
            return entry_dict["id"]
    
    async def get_log(self, log_id: str) -> Optional[Dict[str, Any]]:
        """Get a log entry by ID."""
        if not self.loaded:
            await self.initialize()
            
        return self.logs.get(log_id)
    
    async def get_logs(self, 
                     log_type: Optional[str] = None, 
                     limit: int = 100, 
                     offset: int = 0,
                     tags: Optional[List[str]] = None,
                     source_module: Optional[str] = None,
                     severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get logs with optional filtering."""
        if not self.loaded:
            await self.initialize()
            
        async with self._lock:
            # Apply filters
            filtered = list(self.logs.values())
            
            if log_type:
                filtered = [log for log in filtered if log.get("log_type") == log_type]
            
            if tags:
                filtered = [log for log in filtered if any(tag in log.get("tags", []) for tag in tags)]
            
            if source_module:
                filtered = [log for log in filtered if log.get("source_module") == source_module]
            
            if severity:
                filtered = [log for log in filtered if log.get("severity") == severity]
            
            # Sort by timestamp (newest first)
            filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Apply pagination
            return filtered[offset:offset+limit]
    
    async def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get statistics about synergy recommendations."""
        if not self.loaded:
            await self.initialize()
            
        recommendations = await self.get_logs(log_type="synergy_recommendation")
        
        # Count by module
        by_module = {}
        for rec in recommendations:
            from_module = rec.get("from_module")
            to_module = rec.get("to_module")
            
            if from_module:
                if from_module not in by_module:
                    by_module[from_module] = 0
                by_module[from_module] += 1
            
            if to_module:
                if to_module not in by_module:
                    by_module[to_module] = 0
                by_module[to_module] += 1
        
        # Count by applied status
        applied_count = sum(1 for rec in recommendations if rec.get("applied", False))
        
        return {
            "total": len(recommendations),
            "applied": applied_count,
            "pending": len(recommendations) - applied_count,
            "by_module": by_module,
            "high_priority": sum(1 for rec in recommendations if rec.get("priority", 0) > 0.7)
        }
    
    async def update_recommendation(self, 
                                 recommendation_id: str, 
                                 applied: bool, 
                                 applied_by: str = "system",
                                 results: Optional[Dict[str, Any]] = None) -> bool:
        """Update a recommendation with application results."""
        if not self.loaded:
            await self.initialize()
            
        async with self._lock:
            if recommendation_id not in self.logs:
                return False
            
            rec = self.logs[recommendation_id]
            
            # Update fields
            rec["applied"] = applied
            rec["applied_timestamp"] = datetime.datetime.now().isoformat()
            rec["applied_by"] = applied_by
            
            if results:
                rec["results"] = results
            
            # Save changes
            try:
                # Rewrite the entire file (inefficient but simple)
                recommendations = await self.get_logs(log_type="synergy_recommendation")
                
                with open(self.recommendations_file, 'w') as f:
                    for r in recommendations:
                        f.write(json.dumps(r) + "\n")
                        
                return True
            except Exception as e:
                logger.error(f"Error updating recommendation {recommendation_id}: {e}")
                return False

# Initialize singleton instance
_instance = None

def get_dev_log_storage() -> DevLogStorage:
    """Get the singleton dev log storage instance."""
    global _instance
    if _instance is None:
        _instance = DevLogStorage()
    return _instance
