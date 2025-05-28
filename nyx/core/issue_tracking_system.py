# nyx/core/issue_tracking_system.py

import asyncio
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass, field, asdict
import uuid
import re

from agents import (
    Agent,
    Runner,
    ModelSettings,
    function_tool,
    trace,
    set_default_openai_key
)
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenAI API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    set_default_openai_key(api_key)

# ============== Data Models ==============

class IssueCategory(BaseModel):
    """Category for categorizing issues and ideas"""
    name: str
    description: str

class Issue(BaseModel):
    """Model for tracking issues, ideas, and requests"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    category: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    status: str = "open"
    priority: int = 1  # 1-5, with 5 being highest
    tags: List[str] = Field(default_factory=list)
    updates: List[Dict[str, Any]] = Field(default_factory=list)
    related_issues: List[str] = Field(default_factory=list)
    implementation_ideas: List[str] = Field(default_factory=list)
    context: Optional[str] = None
    
    def update_description(self, new_details: str) -> None:
        """Update the description with new details"""
        # Add an update to the updates list
        self.updates.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "previous_description": self.description,
            "update": new_details
        })
        
        # Update the description and updated_at timestamp
        self.description = f"{self.description}\n\nUpdate ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}): {new_details}"
        self.updated_at = datetime.datetime.now().isoformat()
    
    def add_implementation_idea(self, idea: str) -> None:
        """Add an implementation idea"""
        if idea not in self.implementation_ideas:
            self.implementation_ideas.append(idea)
            self.updated_at = datetime.datetime.now().isoformat()

class IssueSearchResult(BaseModel):
    """Search result for issues"""
    issue: Issue
    similarity_score: float

class IssueActionResult(BaseModel):
    """Result of an action on an issue"""
    success: bool
    message: str
    issue_id: Optional[str] = None
    issue: Optional[Issue] = None

class IssueDBStats(BaseModel):
    """Statistics about the issue database"""
    total_issues: int
    open_issues: int
    by_category: Dict[str, int]
    by_priority: Dict[str, int]
    recently_updated: List[Dict[str, Any]]
    
# ============== Issue Database ==============

class IssueDatabase:
    """Database for storing and retrieving issues"""
    
    def __init__(self, db_path: str = "issues_db.json"):
        self.db_path = db_path
        self.issues: Dict[str, Issue] = {}
        self.categories: Dict[str, IssueCategory] = {
            "bug": IssueCategory(name="bug", description="Problems and errors encountered"),
            "efficiency": IssueCategory(name="efficiency", description="Ideas for improving efficiency"),
            "enhancement": IssueCategory(name="enhancement", description="New functionality requests"),
            "refactoring": IssueCategory(name="refactoring", description="Code that needs restructuring"),
            "usability": IssueCategory(name="usability", description="User experience improvements"),
            "documentation": IssueCategory(name="documentation", description="Documentation needs")
        }
        
        # Load existing database if it exists
        self.load_db()
    
    def load_db(self) -> None:
        """Load the database from file"""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load categories
                    if "categories" in data:
                        self.categories = {k: IssueCategory(**v) if isinstance(v, dict) else v 
                                         for k, v in data["categories"].items()}
                    
                    # Load issues
                    if "issues" in data:
                        self.issues = {k: Issue(**v) if isinstance(v, dict) else v 
                                     for k, v in data["issues"].items()}
                
                logger.info(f"Loaded {len(self.issues)} issues from database")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
    
    def save_db(self) -> None:
        """Save the database to file"""
        try:
            data = {
                "categories": {k: asdict(v) if hasattr(v, "__dict__") else v.dict() 
                              for k, v in self.categories.items()},
                "issues": {k: asdict(v) if hasattr(v, "__dict__") else v.dict() 
                          for k, v in self.issues.items()}
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.issues)} issues to database")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def add_issue(self, issue: Issue) -> IssueActionResult:
        """Add a new issue to the database"""
        # Check for duplicates before adding
        similar_issues = self.find_similar_issues(issue.title, issue.description, threshold=0.7)
        
        if similar_issues:
            most_similar = similar_issues[0]
            
            # If very similar, update the existing issue instead
            if most_similar.similarity_score > 0.85:
                existing_issue = most_similar.issue
                existing_issue.update_description(issue.description)
                self.issues[existing_issue.id] = existing_issue
                self.save_db()
                
                return IssueActionResult(
                    success=True,
                    message=f"Updated existing similar issue: {existing_issue.title}",
                    issue_id=existing_issue.id,
                    issue=existing_issue
                )
            else:
                # Add as new but link to similar issues
                issue.related_issues = [s.issue.id for s in similar_issues]
                self.issues[issue.id] = issue
                self.save_db()
                
                return IssueActionResult(
                    success=True,
                    message=f"Added new issue with {len(issue.related_issues)} related issues",
                    issue_id=issue.id,
                    issue=issue
                )
        else:
            # Add as completely new issue
            self.issues[issue.id] = issue
            self.save_db()
            
            return IssueActionResult(
                success=True,
                message="Added new issue",
                issue_id=issue.id,
                issue=issue
            )
    
    def update_issue(self, issue_id: str, update_data: Dict[str, Any]) -> IssueActionResult:
        """Update an existing issue"""
        if issue_id not in self.issues:
            return IssueActionResult(
                success=False,
                message=f"Issue with ID {issue_id} not found"
            )
        
        issue = self.issues[issue_id]
        
        # Handle special case for appending to description
        if "append_description" in update_data:
            issue.update_description(update_data["append_description"])
            del update_data["append_description"]
        
        # Handle adding implementation ideas
        if "add_implementation_idea" in update_data:
            issue.add_implementation_idea(update_data["add_implementation_idea"])
            del update_data["add_implementation_idea"]
        
        # Update other fields
        for key, value in update_data.items():
            if hasattr(issue, key):
                setattr(issue, key, value)
        
        # Update the updated_at timestamp
        issue.updated_at = datetime.datetime.now().isoformat()
        
        # Save to database
        self.issues[issue_id] = issue
        self.save_db()
        
        return IssueActionResult(
            success=True,
            message=f"Updated issue: {issue.title}",
            issue_id=issue_id,
            issue=issue
        )
    
    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get an issue by ID"""
        return self.issues.get(issue_id)
    
    def get_all_issues(self) -> List[Issue]:
        """Get all issues"""
        return list(self.issues.values())
    
    def get_issues_by_category(self, category: str) -> List[Issue]:
        """Get issues by category"""
        return [issue for issue in self.issues.values() if issue.category == category]
    
    def get_issues_by_status(self, status: str) -> List[Issue]:
        """Get issues by status"""
        return [issue for issue in self.issues.values() if issue.status == status]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using a simple approach"""
        # This is a simple approach - in a real implementation you'd use
        # embeddings or a more sophisticated NLP approach
        
        # Normalize and tokenize the texts
        def normalize(text):
            # Convert to lowercase and remove punctuation
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            # Split into words
            return set(text.split())
        
        tokens1 = normalize(text1)
        tokens2 = normalize(text2)
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def find_similar_issues(self, title: str, description: str, threshold: float = 0.5) -> List[IssueSearchResult]:
        """Find issues similar to the given title and description"""
        results = []
        
        for issue_id, issue in self.issues.items():
            # Calculate similarity based on title and description
            title_similarity = self.calculate_text_similarity(title, issue.title)
            desc_similarity = self.calculate_text_similarity(description, issue.description)
            
            # Weighted similarity (title is more important)
            similarity = (title_similarity * 0.6) + (desc_similarity * 0.4)
            
            if similarity >= threshold:
                results.append(IssueSearchResult(
                    issue=issue,
                    similarity_score=similarity
                ))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def search_issues(self, query: str) -> List[Issue]:
        """Search issues by query"""
        results = []
        
        for issue in self.issues.values():
            if (query.lower() in issue.title.lower() or 
                query.lower() in issue.description.lower() or
                any(query.lower() in tag.lower() for tag in issue.tags)):
                results.append(issue)
        
        return results
    
    def get_stats(self) -> IssueDBStats:
        """Get statistics about the issue database"""
        total = len(self.issues)
        open_issues = len([i for i in self.issues.values() if i.status == "open"])
        
        # Count by category
        by_category = {}
        for issue in self.issues.values():
            cat = issue.category
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # Count by priority
        by_priority = {}
        for issue in self.issues.values():
            priority = str(issue.priority)
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        # Get recently updated issues
        sorted_issues = sorted(
            self.issues.values(), 
            key=lambda x: x.updated_at, 
            reverse=True
        )
        recently_updated = [
            {
                "id": issue.id,
                "title": issue.title,
                "updated_at": issue.updated_at,
                "category": issue.category,
                "priority": issue.priority
            }
            for issue in sorted_issues[:5]  # Top 5 most recently updated
        ]
        
        return IssueDBStats(
            total_issues=total,
            open_issues=open_issues,
            by_category=by_category,
            by_priority=by_priority,
            recently_updated=recently_updated
        )

# ============== Issue Tracker Agent ==============

class IssueTrackingSystem:
    """System for tracking issues using OpenAI Agents SDK"""
    
    def __init__(self, db_path: str = "issues_db.json"):
        """Initialize the issue tracking system"""
        self.db = IssueDatabase(db_path)
        
        # Initialize tools first
        self.add_issue_tool = function_tool(self._add_issue)
        # explicitly supply a JSON-schema for the update_issue tool
        self.update_issue_tool = function_tool(
            self._update_issue,
            name_override="update_issue",
            description_override="Update fields on an existing issue",
            parameters={
                "type": "object",
                "properties": {
                    "issue_id": {
                        "type": "string",
                        "description": "The ID of the issue to update"
                    },
                    "update_data": {
                        "type": "object",
                        "description": "A dict of field names to new values",
                        "additionalProperties": True
                    }
                },
                "required": ["issue_id", "update_data"]
            }
        )
        self.find_similar_issues_tool = function_tool(self._find_similar_issues)
        self.get_issue_tool = function_tool(self._get_issue)
        self.search_issues_tool = function_tool(self._search_issues)
        self.get_stats_tool = function_tool(self._get_stats)
        
        # Then create agents that use those tools
        self.issue_analyzer_agent = self._create_issue_analyzer_agent()
        self.issue_manager_agent = self._create_issue_manager_agent()
    
    def _create_issue_analyzer_agent(self) -> Agent:
        """Create an agent specialized in analyzing issues and ideas"""
        return Agent(
            name="Issue Analyzer",
            instructions="""
            You are the issue analyzer for an AI bot's self-improvement system.
            
            Your role is to:
            1. Analyze problems, efficiency ideas, and feature requests that the AI bot encounters
            2. Extract key information from the bot's experiences
            3. Identify patterns and recurring issues
            4. Assess priority and categorize issues appropriately
            5. Check for similar existing issues to avoid duplication
            
            When analyzing issues:
            - Look for specific, actionable information
            - Categorize accurately (bug, efficiency, enhancement, refactoring, usability, documentation)
            - Assign appropriate priority (1-5, with 5 being highest)
            - Add relevant tags
            - Suggest implementation ideas when possible
            
            Be thorough in your analysis, but focus on extracting practical, implementable insights.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.2,  # Keep it focused on accurate analysis
                top_p=0.9,
                max_tokens=1024
            )
        )
    
    def _create_issue_manager_agent(self) -> Agent:
        """Create an agent specialized in managing issues"""
        return Agent(
            name="Issue Manager",
            instructions="""
            You are the issue manager for an AI bot's self-improvement system.
            
            Your role is to:
            1. Process and store issues, efficiency ideas, and feature requests
            2. Identify duplicate or similar issues and consolidate information
            3. Track the status and progress of issues
            4. Provide summaries and reports of outstanding issues
            5. Help prioritize which issues to address first
            
            When managing issues:
            - Avoid creating duplicates by carefully checking existing issues
            - Append new details to existing issues when appropriate 
            - Maintain a clean, well-organized issue database
            - Provide helpful summaries and insights about tracked issues
            
            Your goal is to create a reliable system of record for all issues and improvement ideas
            that the development team can use to enhance the AI bot.
            """,
            model="gpt-4.1-nano",
            model_settings=ModelSettings(
                temperature=0.1,  # Keep it very factual and organized
                top_p=0.9,
                max_tokens=1024
            ),
            tools=[
                self.add_issue_tool,
                self.update_issue_tool,
                self.find_similar_issues_tool,
                self.get_issue_tool,
                self.search_issues_tool,
                self.get_stats_tool
            ]
        )
    
    async def process_observation(self, observation: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an observation from the AI bot to identify potential issues
        
        Args:
            observation: The observation text from the AI bot
            context: Optional context about what the bot was doing
            
        Returns:
            Processing results with issue information
        """
        with trace(workflow_name="process_issue_observation"):
            # Prepare the prompt for the analyzer
            # Avoid using backslash in f-string by constructing context part separately
            context_part = ""
            if context:
                context_part = f"CONTEXT:\n{context}"
            
            prompt = f"""
            Analyze this observation from the AI bot to identify potential issues, efficiency improvements, 
            or enhancement requests:
            
            OBSERVATION:
            {observation}
            
            {context_part}
            
            Determine if this contains an actionable issue, efficiency idea, or feature request.
            If yes, extract the relevant details and categorize appropriately.
            If no, briefly explain why no action is needed.
            """
            
            # Run the analyzer agent
            analyzer_result = await Runner.run(self.issue_analyzer_agent, prompt)
            analysis = analyzer_result.final_output
            
            # Prepare prompt for the manager agent to process the analysis
            manager_prompt = f"""
            An observation has been analyzed and potential issues identified.
            Please process this analysis to track any issues or improvement ideas.
            
            ORIGINAL OBSERVATION:
            {observation}
            
            ANALYSIS:
            {analysis}
            
            Based on the analysis, determine if a new issue should be created or if
            an existing issue should be updated. Check for similar existing issues to avoid duplication.
            """
            
            # Run the manager agent to process the issue
            manager_result = await Runner.run(self.issue_manager_agent, manager_prompt)
            processing_result = manager_result.final_output
            
            # Return the results
            return {
                "observation": observation,
                "analysis": analysis,
                "processing_result": processing_result
            }
    
    async def add_issue_directly(self, title: str, description: str, category: str, 
                             priority: int = 3, tags: List[str] = None) -> IssueActionResult:
        """
        Add a new issue directly to the system
        
        Args:
            title: Issue title
            description: Issue description
            category: Issue category
            priority: Issue priority (1-5)
            tags: Optional tags
            
        Returns:
            Result of the add operation
        """
        # Create the issue object
        new_issue = Issue(
            title=title,
            description=description,
            category=category,
            priority=priority,
            tags=tags or []
        )
        
        # Add to database
        result = self.db.add_issue(new_issue)
        return result
    
    async def get_issue_summary(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get a summary of all issues
        
        Args:
            detailed: Whether to include full issue details
            
        Returns:
            Summary information
        """
        with trace(workflow_name="get_issue_summary"):
            # Get database stats
            stats = self.db.get_stats()
            
            # Prepare prompt for the manager agent
            prompt = f"""
            Generate a summary of the current issues and improvement ideas in the system.
            
            STATS:
            Total Issues: {stats.total_issues}
            Open Issues: {stats.open_issues}
            
            Categories:
            {json.dumps(stats.by_category, indent=2)}
            
            Priorities:
            {json.dumps(stats.by_priority, indent=2)}
            
            Recently Updated:
            {json.dumps(stats.recently_updated, indent=2)}
            
            {"Include detailed information about each issue." if detailed else "Provide a concise summary only."}
            """
            
            # Run the manager agent to generate the summary
            result = await Runner.run(self.issue_manager_agent, prompt)
            summary = result.final_output
            
            # Return the results
            return {
                "stats": stats.dict(),
                "summary": summary,
                "issues": [issue.dict() for issue in self.db.get_all_issues()] if detailed else None
            }
    
    # Tool functions
    
    async def _add_issue(self, title: str, description: str, category: str,
                      priority: int = 3, tags: List[str] = None) -> Dict[str, Any]:
        """
        Add a new issue to the database (used as a tool for agents)
        """
        new_issue = Issue(
            title=title,
            description=description,
            category=category,
            priority=priority,
            tags=tags or []
        )
        
        result = self.db.add_issue(new_issue)
        return result.dict()
    
    async def _update_issue(self, issue_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing issue (used as a tool for agents)
        """
        result = self.db.update_issue(issue_id, update_data)
        return result.dict()
    
    async def _find_similar_issues(self, title: str, description: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find similar issues (used as a tool for agents)
        """
        results = self.db.find_similar_issues(title, description, threshold)
        return [{"issue": r.issue.dict(), "similarity_score": r.similarity_score} for r in results]
    
    async def _get_issue(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an issue by ID (used as a tool for agents)
        """
        issue = self.db.get_issue(issue_id)
        return issue.dict() if issue else None
    
    async def _search_issues(self, query: str) -> List[Dict[str, Any]]:
        """
        Search issues by query (used as a tool for agents)
        """
        issues = self.db.search_issues(query)
        return [issue.dict() for issue in issues]
    
    async def _get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics (used as a tool for agents)
        """
        stats = self.db.get_stats()
        return stats.dict()
