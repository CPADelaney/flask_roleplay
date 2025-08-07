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
    priority: int = Field(default=1, ge=1, le=5) # Added ge/le for consistency
    tags: List[str] = Field(default_factory=list)
    updates: List[Dict[str, Any]] = Field(default_factory=list)
    related_issues: List[str] = Field(default_factory=list)
    implementation_ideas: List[str] = Field(default_factory=list)
    context: Optional[str] = None
    
    def update_description(self, new_details: str) -> None:
        """Update the description with new details"""
        self.updates.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "previous_description": self.description,
            "update": new_details
        })
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

# NEW: Define a Pydantic model for the update payload
class IssueUpdatePayload(BaseModel):
    """
    Payload for updating an issue. Only provided fields will be updated.
    """
    title: Optional[str] = Field(default=None, description="New title for the issue.")
    # description: Optional[str] = Field(default=None, description="New full description for the issue. Note: `append_description` is often preferred.")
    category: Optional[str] = Field(default=None, description="New category for the issue (e.g., bug, enhancement).")
    status: Optional[str] = Field(default=None, description="New status for the issue (e.g., open, in_progress, resolved).")
    priority: Optional[int] = Field(default=None, ge=1, le=5, description="New priority for the issue (1-5, 5 is highest).")
    tags: Optional[List[str]] = Field(default=None, description="New list of tags for the issue. This will replace existing tags.")
    context: Optional[str] = Field(default=None, description="Updated context for the issue.")
    
    append_description: Optional[str] = Field(default=None, description="Text to append to the existing issue description.")
    add_implementation_idea: Optional[str] = Field(default=None, description="An implementation idea to add to the issue.")
    
    # If you need to allow completely arbitrary key-value updates matching Issue model attributes,
    # you might need a more dynamic approach or list all Issue attributes as Optional here.
    # For now, this covers common and special fields.
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
                "categories": {k: v.dict() for k, v in self.categories.items()},
                "issues": {k: v.dict() for k, v in self.issues.items()}
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.issues)} issues to database")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            
    def add_issue(self, issue: Issue) -> IssueActionResult:
        similar_issues = self.find_similar_issues(issue.title, issue.description, threshold=0.7)
        if similar_issues:
            most_similar = similar_issues[0]
            if most_similar.similarity_score > 0.85:
                existing_issue = most_similar.issue
                existing_issue.update_description(f"Related new information: {issue.description}") # Adjusted to be more specific
                self.issues[existing_issue.id] = existing_issue
                self.save_db()
                return IssueActionResult(
                    success=True,
                    message=f"Updated existing similar issue: {existing_issue.title} (ID: {existing_issue.id})",
                    issue_id=existing_issue.id,
                    issue=existing_issue
                )
            else:
                issue.related_issues = [s.issue.id for s in similar_issues]
                self.issues[issue.id] = issue
                self.save_db()
                return IssueActionResult(
                    success=True,
                    message=f"Added new issue (ID: {issue.id}) with {len(issue.related_issues)} related issues.",
                    issue_id=issue.id,
                    issue=issue
                )
        else:
            self.issues[issue.id] = issue
            self.save_db()
            return IssueActionResult(
                success=True,
                message=f"Added new issue (ID: {issue.id}).",
                issue_id=issue.id,
                issue=issue
            )
    
    
    def update_issue(self, issue_id: str, update_data: Dict[str, Any]) -> IssueActionResult:
        if issue_id not in self.issues:
            return IssueActionResult(
                success=False,
                message=f"Issue with ID {issue_id} not found"
            )
        
        issue = self.issues[issue_id]
        updated_fields = []

        if "append_description" in update_data:
            issue.update_description(update_data["append_description"])
            updated_fields.append("description (appended)")
            del update_data["append_description"] 
        
        if "add_implementation_idea" in update_data:
            issue.add_implementation_idea(update_data["add_implementation_idea"])
            updated_fields.append("implementation_ideas (added)")
            del update_data["add_implementation_idea"]
        
        for key, value in update_data.items():
            if hasattr(issue, key):
                # Prevent direct update of 'description' if 'append_description' is preferred path
                if key == "description": 
                    logger.warning("Direct update to 'description' field; consider using 'append_description'. Updating directly.")
                setattr(issue, key, value)
                updated_fields.append(key)
            else:
                logger.warning(f"Field '{key}' not found on Issue model, cannot update.")

        if not updated_fields:
             return IssueActionResult(
                success=False, # Or True if "no effective change" is not an error
                message=f"No valid fields to update provided for issue: {issue.title}",
                issue_id=issue_id,
                issue=issue
            )

        issue.updated_at = datetime.datetime.now().isoformat()
        self.issues[issue_id] = issue
        self.save_db()
        
        return IssueActionResult(
            success=True,
            message=f"Updated issue: {issue.title}. Fields changed: {', '.join(updated_fields)}",
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
        def normalize(text):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return set(text.split())
        
        tokens1 = normalize(text1)
        tokens2 = normalize(text2)
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union != 0 else 0.0
    
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
        self.db = IssueDatabase(db_path)
        self.add_issue_tool = function_tool(self._add_issue, name_override="add_issue", description_override="Add a new issue to the database. Checks for similarity to avoid duplicates or links related issues.")
        self.update_issue_tool = function_tool(self._update_issue, name_override="update_issue", description_override="Update fields on an existing issue. Use this to change status, priority, add details, etc.") # Keep description concise
        self.find_similar_issues_tool = function_tool(self._find_similar_issues, name_override="find_similar_issues", description_override="Find issues similar to a given title and description, helpful for checking duplicates before adding a new one.")
        self.get_issue_tool = function_tool(self._get_issue, name_override="get_issue", description_override="Retrieve a single issue by its ID.")
        self.search_issues_tool = function_tool(self._search_issues, name_override="search_issues", description_override="Search issues by a query string across titles, descriptions, and tags.")
        self.get_stats_tool = function_tool(self._get_stats, name_override="get_stats", description_override="Get summary statistics of the issue database (total issues, open issues, counts by category/priority).")
        
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
            model="gpt-5-nano",
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
            model="gpt-5-nano",
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
                "processing_result": processing_result,
                "tool_calls": manager_result.tool_calls, # For debugging
                "tool_outputs": manager_result.tool_outputs # For debugging
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
        with trace(workflow_name="get_issue_summary"):
            stats = self.db.get_stats()
            
            # Enhanced prompt for the manager agent
            detailed_info_request = "Include detailed information about each open issue (up to 10 most recent or highest priority)." if detailed else "Provide a concise summary of key statistics and trends only."
            prompt = f"""
            Generate a summary of the current issues and improvement ideas in the system.
            
            CURRENT STATS:
            {json.dumps(stats.dict(), indent=2)}
            
            {detailed_info_request}
            
            If providing details, focus on open issues. Mention any notable trends or urgent matters.
            """
            
            result = await Runner.run(self.issue_manager_agent, prompt)
            summary_text = result.final_output
            
            return_data = {
                "stats": stats.dict(),
                "summary": summary_text,
            }
            if detailed:
                # Fetch some issues if detailed view is requested, let agent decide which ones based on prompt
                 open_issues = self.db.get_issues_by_status("open")
                 # Sort by priority (desc) then updated_at (desc)
                 sorted_open_issues = sorted(open_issues, key=lambda i: (i.priority, i.updated_at), reverse=True)
                 return_data["detailed_issues_sample"] = [issue.dict() for issue in sorted_open_issues[:10]]

            return return_data
    
    # Tool functions
    
    async def _add_issue(self, title: str, description: str, category: str,
                      priority: int = 3, tags: Optional[List[str]] = None) -> Dict[str, Any]: # Added Optional
        new_issue = Issue(
            title=title, description=description, category=category,
            priority=priority, tags=tags or []
        )
        result = self.db.add_issue(new_issue)
        return result.dict()
    
    # MODIFIED: _update_issue uses IssueUpdatePayload
    async def _update_issue(self, issue_id: str, update_payload: IssueUpdatePayload) -> Dict[str, Any]:
        """
        Update an existing issue using specific fields.
        Only fields provided in `update_payload` will be changed.
        For example, to change status: `update_payload={"status": "in_progress"}`.
        To append to description: `update_payload={"append_description": "New details..."}`.
        """
        # Convert Pydantic model to dict, excluding fields that were not set by the agent
        update_data_dict = update_payload.dict(exclude_unset=True)
        
        if not update_data_dict:
            # Return a specific response if no actual update values are provided.
            # The agent should ideally not call this with an empty payload.
            existing_issue = self.db.get_issue(issue_id)
            if existing_issue:
                return IssueActionResult(
                    success=False, # Or True, depending on if "no change" is an error
                    message="No update fields provided. Issue remains unchanged.",
                    issue_id=issue_id,
                    issue=existing_issue
                ).dict()
            else:
                return IssueActionResult(
                    success=False,
                    message=f"No update fields provided and issue {issue_id} not found."
                ).dict()

        result = self.db.update_issue(issue_id, update_data_dict)
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
