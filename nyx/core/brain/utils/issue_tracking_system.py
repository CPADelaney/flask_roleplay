# nyx/core/brain/utils/issue_tracking_system.py
import logging
import asyncio
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)

class IssueCategory(Enum):
    """Categories for issues and limitations"""
    MEMORY = auto()
    REASONING = auto()
    EMOTIONAL = auto()
    IDENTITY = auto()
    PERFORMANCE = auto()
    TECHNICAL = auto()
    INTEGRATION = auto()
    SECURITY = auto()
    USER_EXPERIENCE = auto()
    OTHER = auto()

class IssueSeverity(Enum):
    """Severity levels for issues"""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    INFO = auto()

class IssueStatus(Enum):
    """Status options for issues"""
    OPEN = auto()
    IN_PROGRESS = auto()
    RESOLVED = auto()
    WONTFIX = auto()
    DUPLICATE = auto()

class IssueTrackingSystem:
    """
    System for tracking, classifying, and handling issues and limitations
    in the Nyx brain system
    """
    
    def __init__(self, brain=None, storage_path: str = None):
        """
        Initialize the issue tracking system
        
        Args:
            brain: Reference to the NyxBrain instance
            storage_path: Optional path for persistent storage
        """
        self.brain = brain
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".nyx", "issues"
        )
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Issues dictionary: id -> issue data
        self.issues = {}
        
        # Issue templates for common scenarios
        self.issue_templates = self._initialize_templates()
        
        # Handlers for different issue categories
        self.issue_handlers = {}
        
        # Load existing issues from storage
        self._load_issues()
        
        # Statistics and patterns
        self.issue_stats = {
            "by_category": {},
            "by_severity": {},
            "by_status": {},
            "recurrence_patterns": {},
            "resolution_times": []
        }
        
        # Auto-resolution strategies
        self.auto_resolution_strategies = {}
        
        logger.info("Issue tracking system initialized")
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for common issue types"""
        return {
            "memory_retrieval_failure": {
                "title": "Memory retrieval failure",
                "category": IssueCategory.MEMORY,
                "severity": IssueSeverity.MEDIUM,
                "description": "Failed to retrieve relevant memories for user query",
                "recommended_action": "Check memory indexing and try alternative retrieval methods"
            },
            "emotional_regulation_issue": {
                "title": "Emotional regulation issue",
                "category": IssueCategory.EMOTIONAL,
                "severity": IssueSeverity.MEDIUM,
                "description": "Inappropriate or excessive emotional response detected",
                "recommended_action": "Adjust emotional decay rates and expression thresholds"
            },
            "identity_contradiction": {
                "title": "Identity contradiction detected",
                "category": IssueCategory.IDENTITY,
                "severity": IssueSeverity.HIGH,
                "description": "Contradictory identity traits or preferences detected",
                "recommended_action": "Run identity consolidation and resolve contradictions"
            },
            "performance_degradation": {
                "title": "Performance degradation",
                "category": IssueCategory.PERFORMANCE,
                "severity": IssueSeverity.HIGH,
                "description": "Significant degradation in response time or quality",
                "recommended_action": "Check resource usage and consider optimization or scaling"
            },
            "integration_failure": {
                "title": "Component integration failure",
                "category": IssueCategory.INTEGRATION,
                "severity": IssueSeverity.HIGH,
                "description": "Failure in component integration or communication",
                "recommended_action": "Check component dependencies and communication channels"
            }
        }
    
    def _load_issues(self) -> None:
        """Load issues from storage"""
        try:
            issues_file = os.path.join(self.storage_path, "issues.json")
            if os.path.exists(issues_file):
                with open(issues_file, "r") as f:
                    issues_data = json.load(f)
                
                # Convert string enum keys back to enum objects
                for issue_id, issue in issues_data.items():
                    if "category" in issue and isinstance(issue["category"], str):
                        issue["category"] = getattr(IssueCategory, issue["category"])
                    if "severity" in issue and isinstance(issue["severity"], str):
                        issue["severity"] = getattr(IssueSeverity, issue["severity"])
                    if "status" in issue and isinstance(issue["status"], str):
                        issue["status"] = getattr(IssueStatus, issue["status"])
                
                self.issues = issues_data
                logger.info(f"Loaded {len(self.issues)} issues from storage")
        except Exception as e:
            logger.error(f"Error loading issues: {str(e)}")
    
    def _save_issues(self) -> None:
        """Save issues to storage"""
        try:
            # Convert enum objects to strings for JSON serialization
            serializable_issues = {}
            for issue_id, issue in self.issues.items():
                serializable_issue = issue.copy()
                if "category" in serializable_issue and isinstance(serializable_issue["category"], IssueCategory):
                    serializable_issue["category"] = serializable_issue["category"].name
                if "severity" in serializable_issue and isinstance(serializable_issue["severity"], IssueSeverity):
                    serializable_issue["severity"] = serializable_issue["severity"].name
                if "status" in serializable_issue and isinstance(serializable_issue["status"], IssueStatus):
                    serializable_issue["status"] = serializable_issue["status"].name
                
                serializable_issues[issue_id] = serializable_issue
            
            issues_file = os.path.join(self.storage_path, "issues.json")
            with open(issues_file, "w") as f:
                json.dump(serializable_issues, f, indent=2)
            
            logger.info(f"Saved {len(self.issues)} issues to storage")
        except Exception as e:
            logger.error(f"Error saving issues: {str(e)}")
    
    async def register_issue(self, 
                         issue_data: Dict[str, Any], 
                         auto_attempt_resolution: bool = True) -> Dict[str, Any]:
        """
        Register a new issue
        
        Args:
            issue_data: Information about the issue
            auto_attempt_resolution: Whether to automatically attempt resolution
            
        Returns:
            Registered issue with ID
        """
        # Generate ID if not provided
        if "id" not in issue_data:
            issue_data["id"] = f"ISSUE-{len(self.issues) + 1}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Add timestamp if not provided
        if "timestamp" not in issue_data:
            issue_data["timestamp"] = datetime.datetime.now().isoformat()
        
        # Set default status if not provided
        if "status" not in issue_data:
            issue_data["status"] = IssueStatus.OPEN
        
        # Add to issues dictionary
        self.issues[issue_data["id"]] = issue_data
        
        # Update statistics
        self._update_statistics(issue_data)
        
        # Check for similar issues and patterns
        similar_issues = self._find_similar_issues(issue_data)
        if similar_issues:
            issue_data["similar_issues"] = [issue["id"] for issue in similar_issues[:5]]
            
            # Check for recurring patterns
            if len(similar_issues) >= 3:
                pattern = self._identify_pattern(issue_data, similar_issues)
                if pattern:
                    issue_data["identified_pattern"] = pattern
        
        # Attempt automatic resolution if enabled
        resolution_result = None
        if auto_attempt_resolution:
            resolution_result = await self._attempt_auto_resolution(issue_data)
            if resolution_result and resolution_result.get("success"):
                issue_data["auto_resolution"] = resolution_result
                issue_data["status"] = IssueStatus.RESOLVED
                issue_data["resolution_timestamp"] = datetime.datetime.now().isoformat()
        
        # Save issues to storage
        self._save_issues()
        
        # Create return object
        result = {
            "issue": issue_data,
            "similar_issues": similar_issues[:5] if similar_issues else [],
            "auto_resolution_attempted": auto_attempt_resolution,
            "auto_resolution_result": resolution_result
        }
        
        return result
    
    def register_issue_from_template(self, 
                                   template_key: str, 
                                   context_data: Dict[str, Any],
                                   auto_attempt_resolution: bool = True) -> Dict[str, Any]:
        """
        Register an issue using a predefined template
        
        Args:
            template_key: Key of the template to use
            context_data: Context information for the issue
            auto_attempt_resolution: Whether to automatically attempt resolution
            
        Returns:
            Registered issue with ID
        """
        if template_key not in self.issue_templates:
            raise ValueError(f"Template '{template_key}' not found")
        
        # Create issue from template
        template = self.issue_templates[template_key]
        issue_data = template.copy()
        
        # Add context information
        issue_data.update({
            "context": context_data,
            "component": context_data.get("component", "unknown"),
            "source": context_data.get("source", "system")
        })
        
        # Register the issue
        return asyncio.run(self.register_issue(issue_data, auto_attempt_resolution))
    
    def update_issue(self, issue_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing issue
        
        Args:
            issue_id: ID of the issue to update
            updates: Updates to apply
            
        Returns:
            Updated issue
        """
        if issue_id not in self.issues:
            raise ValueError(f"Issue with ID '{issue_id}' not found")
        
        # Update the issue
        issue = self.issues[issue_id]
        
        # Add update to history if not present
        if "update_history" not in issue:
            issue["update_history"] = []
        
        # Add current state to history before updating
        history_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "previous_state": {k: v for k, v in issue.items() if k != "update_history"}
        }
        issue["update_history"].append(history_entry)
        
        # Apply updates
        issue.update(updates)
        
        # If resolved, add resolution timestamp
        if "status" in updates and updates["status"] == IssueStatus.RESOLVED:
            if "resolution_timestamp" not in issue:
                issue["resolution_timestamp"] = datetime.datetime.now().isoformat()
        
        # Save issues
        self._save_issues()
        
        return issue
    
    def get_issue(self, issue_id: str) -> Dict[str, Any]:
        """
        Get an issue by ID
        
        Args:
            issue_id: ID of the issue
            
        Returns:
            Issue data
        """
        if issue_id not in self.issues:
            raise ValueError(f"Issue with ID '{issue_id}' not found")
        
        return self.issues[issue_id]
    
    def get_issues(self, 
                 filters: Dict[str, Any] = None, 
                 limit: int = 50,
                 offset: int = 0,
                 sort_by: str = "timestamp",
                 sort_order: str = "desc") -> List[Dict[str, Any]]:
        """
        Get issues with optional filtering
        
        Args:
            filters: Optional filters to apply
            limit: Maximum number of issues to return
            offset: Offset for pagination
            sort_by: Field to sort by
            sort_order: Sort order ("asc" or "desc")
            
        Returns:
            List of issues
        """
        # Apply filters
        filtered_issues = self._filter_issues(self.issues.values(), filters)
        
        # Sort issues
        sorted_issues = sorted(
            filtered_issues,
            key=lambda x: x.get(sort_by, ""),
            reverse=(sort_order.lower() == "desc")
        )
        
        # Apply pagination
        paginated_issues = sorted_issues[offset:offset+limit]
        
        return paginated_issues
    
    def _filter_issues(self, 
                     issues: List[Dict[str, Any]], 
                     filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Filter issues based on criteria"""
        if not filters:
            return list(issues)
        
        result = []
        for issue in issues:
            matches = True
            for key, value in filters.items():
                if key == "status" and isinstance(value, str):
                    # Handle string status filters
                    try:
                        value = getattr(IssueStatus, value)
                    except AttributeError:
                        pass
                
                if key == "category" and isinstance(value, str):
                    # Handle string category filters
                    try:
                        value = getattr(IssueCategory, value)
                    except AttributeError:
                        pass
                
                if key == "severity" and isinstance(value, str):
                    # Handle string severity filters
                    try:
                        value = getattr(IssueSeverity, value)
                    except AttributeError:
                        pass
                
                if key == "date_range":
                    # Special handling for date range
                    start_date, end_date = value
                    issue_date = datetime.datetime.fromisoformat(issue.get("timestamp", ""))
                    if not (start_date <= issue_date <= end_date):
                        matches = False
                        break
                elif key not in issue or issue[key] != value:
                    matches = False
                    break
            
            if matches:
                result.append(issue)
        
        return result
    
    def _update_statistics(self, issue: Dict[str, Any]) -> None:
        """Update statistics with a new issue"""
        # Update category stats
        category = issue.get("category")
        if category:
            category_name = category.name if isinstance(category, IssueCategory) else str(category)
            if category_name not in self.issue_stats["by_category"]:
                self.issue_stats["by_category"][category_name] = 0
            self.issue_stats["by_category"][category_name] += 1
        
        # Update severity stats
        severity = issue.get("severity")
        if severity:
            severity_name = severity.name if isinstance(severity, IssueSeverity) else str(severity)
            if severity_name not in self.issue_stats["by_severity"]:
                self.issue_stats["by_severity"][severity_name] = 0
            self.issue_stats["by_severity"][severity_name] += 1
        
        # Update status stats
        status = issue.get("status")
        if status:
            status_name = status.name if isinstance(status, IssueStatus) else str(status)
            if status_name not in self.issue_stats["by_status"]:
                self.issue_stats["by_status"][status_name] = 0
            self.issue_stats["by_status"][status_name] += 1
        
        # Update resolution time if resolved
        if status == IssueStatus.RESOLVED and "resolution_timestamp" in issue and "timestamp" in issue:
            try:
                created = datetime.datetime.fromisoformat(issue["timestamp"])
                resolved = datetime.datetime.fromisoformat(issue["resolution_timestamp"])
                resolution_time = (resolved - created).total_seconds()
                self.issue_stats["resolution_times"].append(resolution_time)
            except (ValueError, TypeError):
                pass
    
    def _find_similar_issues(self, issue: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar issues to the given one"""
        similar_issues = []
        
        # Get category and component for more specific matching
        category = issue.get("category")
        component = issue.get("component")
        
        # First pass: exact category and component match
        for existing_id, existing in self.issues.items():
            if existing_id == issue.get("id"):
                continue  # Skip the current issue
                
            # Check for exact category and component match
            if (existing.get("category") == category and 
                existing.get("component") == component):
                similar_issues.append(existing)
        
        # If we didn't find enough, look for just category matches
        if len(similar_issues) < 5:
            for existing_id, existing in self.issues.items():
                if existing_id == issue.get("id") or existing in similar_issues:
                    continue
                    
                # Check for category match
                if existing.get("category") == category:
                    similar_issues.append(existing)
        
        # Sort by similarity score (more advanced scoring could be implemented)
        return similar_issues[:10]  # Limit to top 10
    
    def _identify_pattern(self, issue: Dict[str, Any], similar_issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identify potential patterns from similar issues"""
        # Check for time-based patterns
        time_pattern = self._check_time_based_pattern(similar_issues)
        if time_pattern:
            return {
                "type": "time_based",
                "details": time_pattern
            }
        
        # Check for context-based patterns
        context_pattern = self._check_context_based_pattern(issue, similar_issues)
        if context_pattern:
            return {
                "type": "context_based",
                "details": context_pattern
            }
        
        # No clear pattern identified
        return None
    
    def _check_time_based_pattern(self, issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check for time-based patterns in issues"""
        if len(issues) < 3:
            return None
            
        # Extract timestamps
        timestamps = []
        for issue in issues:
            if "timestamp" in issue:
                try:
                    timestamp = datetime.datetime.fromisoformat(issue["timestamp"])
                    timestamps.append(timestamp)
                except (ValueError, TypeError):
                    pass
        
        if len(timestamps) < 3:
            return None
            
        # Sort timestamps
        timestamps.sort()
        
        # Check for regular intervals
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        average_interval = sum(intervals) / len(intervals)
        variance = sum((i - average_interval) ** 2 for i in intervals) / len(intervals)
        
        # If variance is small relative to the average, we might have a pattern
        if variance < (average_interval * 0.2) ** 2:
            return {
                "interval_seconds": average_interval,
                "variance": variance,
                "regularity": "high" if variance < (average_interval * 0.1) ** 2 else "medium"
            }
        
        # Check for time-of-day patterns
        hours = [ts.hour for ts in timestamps]
        hour_counts = {}
        for hour in hours:
            if hour not in hour_counts:
                hour_counts[hour] = 0
            hour_counts[hour] += 1
        
        most_common_hour = max(hour_counts.items(), key=lambda x: x[1])
        if most_common_hour[1] >= len(hours) * 0.6:  # At least 60% at same hour
            return {
                "time_of_day": most_common_hour[0],
                "frequency": most_common_hour[1] / len(hours)
            }
        
        return None
    
    def _check_context_based_pattern(self, issue: Dict[str, Any], similar_issues: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check for context-based patterns in issues"""
        # Extract contexts
        contexts = []
        for existing in similar_issues:
            if "context" in existing and isinstance(existing["context"], dict):
                contexts.append(existing["context"])
        
        if len(contexts) < 3:
            return None
        
        # Check for common context keys/values
        common_keys = {}
        for context in contexts:
            for key, value in context.items():
                if key not in common_keys:
                    common_keys[key] = {}
                
                str_value = str(value)
                if str_value not in common_keys[key]:
                    common_keys[key][str_value] = 0
                common_keys[key][str_value] += 1
        
        # Find keys with consistent values
        patterns = {}
        for key, values in common_keys.items():
            most_common = max(values.items(), key=lambda x: x[1])
            if most_common[1] >= len(contexts) * 0.7:  # At least 70% have same value
                patterns[key] = {
                    "value": most_common[0],
                    "frequency": most_common[1] / len(contexts)
                }
        
        if patterns:
            return {
                "common_context_patterns": patterns
            }
        
        return None
    
    async def _attempt_auto_resolution(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to automatically resolve an issue"""
        # Get category and severity
        category = issue.get("category")
        severity = issue.get("severity")
        
        # Skip auto-resolution for critical issues
        if severity == IssueSeverity.CRITICAL:
            return {
                "success": False,
                "reason": "Auto-resolution not attempted for critical issues"
            }
        
        # Check if we have a registered handler for this category
        if category in self.issue_handlers:
            try:
                handler = self.issue_handlers[category]
                result = await handler(issue)
                return result
            except Exception as e:
                logger.error(f"Error in auto-resolution handler: {str(e)}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        # Check for generic resolution strategies
        if "recommended_action" in issue:
            action = issue.get("recommended_action")
            if hasattr(self, f"_resolve_{action.lower().replace(' ', '_')}"):
                try:
                    resolve_method = getattr(self, f"_resolve_{action.lower().replace(' ', '_')}")
                    result = await resolve_method(issue)
                    return result
                except Exception as e:
                    logger.error(f"Error in auto-resolution method: {str(e)}")
                    return {
                        "success": False,
                        "error": str(e)
                    }
        
        # No matching handler found
        return {
            "success": False,
            "reason": "No suitable auto-resolution handler found"
        }
    
    def register_issue_handler(self, category: IssueCategory, handler: callable) -> None:
        """
        Register a handler for a specific issue category
        
        Args:
            category: Issue category to handle
            handler: Async function to handle issues of this category
        """
        self.issue_handlers[category] = handler
        logger.info(f"Registered handler for {category.name} issues")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about tracked issues
        
        Returns:
            Statistics data
        """
        # Calculate resolution time statistics
        resolution_time_stats = {}
        if self.issue_stats["resolution_times"]:
            resolution_times = self.issue_stats["resolution_times"]
            resolution_time_stats = {
                "avg_seconds": sum(resolution_times) / len(resolution_times),
                "min_seconds": min(resolution_times),
                "max_seconds": max(resolution_times),
                "count": len(resolution_times)
            }
        
        # Calculate open vs. closed ratios
        total_issues = len(self.issues)
        open_issues = sum(1 for issue in self.issues.values() 
                           if issue.get("status") == IssueStatus.OPEN)
        resolved_issues = sum(1 for issue in self.issues.values() 
                               if issue.get("status") == IssueStatus.RESOLVED)
        
        open_ratio = open_issues / total_issues if total_issues > 0 else 0
        resolved_ratio = resolved_issues / total_issues if total_issues > 0 else 0
        
        return {
            "total_issues": total_issues,
            "open_issues": open_issues,
            "resolved_issues": resolved_issues,
            "open_ratio": open_ratio,
            "resolved_ratio": resolved_ratio,
            "by_category": self.issue_stats["by_category"],
            "by_severity": self.issue_stats["by_severity"],
            "by_status": self.issue_stats["by_status"],
            "resolution_times": resolution_time_stats,
            "patterns": {
                "time_based": sum(1 for issue in self.issues.values() 
                                 if issue.get("identified_pattern", {}).get("type") == "time_based"),
                "context_based": sum(1 for issue in self.issues.values() 
                                   if issue.get("identified_pattern", {}).get("type") == "context_based")
            }
        }
    
    async def analyze_issues(self) -> Dict[str, Any]:
        """
        Perform deeper analysis of issues and patterns
        
        Returns:
            Analysis results
        """
        # Group issues by various dimensions
        issues_by_component = {}
        issues_by_day = {}
        recurring_components = {}
        
        # Process all issues
        for issue in self.issues.values():
            # Group by component
            component = issue.get("component", "unknown")
            if component not in issues_by_component:
                issues_by_component[component] = []
            issues_by_component[component].append(issue)
            
            # Group by day
            if "timestamp" in issue:
                try:
                    date_str = datetime.datetime.fromisoformat(issue["timestamp"]).date().isoformat()
                    if date_str not in issues_by_day:
                        issues_by_day[date_str] = []
                    issues_by_day[date_str].append(issue)
                except (ValueError, TypeError):
                    pass
        
        # Identify components with recurring issues
        for component, issues in issues_by_component.items():
            if len(issues) >= 3:
                # Check if these issues have patterns
                patterns = [i for i in issues if "identified_pattern" in i]
                if patterns:
                    recurring_components[component] = {
                        "total_issues": len(issues),
                        "patterned_issues": len(patterns),
                        "ratio": len(patterns) / len(issues)
                    }
        
        # Calculate daily statistics
        daily_stats = {}
        for date, issues in issues_by_day.items():
            daily_stats[date] = {
                "count": len(issues),
                "categories": {},
                "severities": {}
            }
            
            # Count by category and severity
            for issue in issues:
                category = issue.get("category")
                if category:
                    category_name = category.name if isinstance(category, IssueCategory) else str(category)
                    if category_name not in daily_stats[date]["categories"]:
                        daily_stats[date]["categories"][category_name] = 0
                    daily_stats[date]["categories"][category_name] += 1
                
                severity = issue.get("severity")
                if severity:
                    severity_name = severity.name if isinstance(severity, IssueSeverity) else str(severity)
                    if severity_name not in daily_stats[date]["severities"]:
                        daily_stats[date]["severities"][severity_name] = 0
                    daily_stats[date]["severities"][severity_name] += 1
        
        # Identify trends
        trend_data = self._analyze_trends(daily_stats)
        
        return {
            "components": {
                "all": list(issues_by_component.keys()),
                "by_issue_count": sorted(
                    [(c, len(i)) for c, i in issues_by_component.items()],
                    key=lambda x: x[1],
                    reverse=True
                ),
                "recurring_issues": recurring_components
            },
            "daily_statistics": daily_stats,
            "trends": trend_data
        }
    
    def _analyze_trends(self, daily_stats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in the daily statistics"""
        if not daily_stats:
            return {"trend_detected": False}
        
        # Get dates in order
        dates = sorted(daily_stats.keys())
        if len(dates) < 5:  # Need at least 5 days for meaningful trends
            return {"trend_detected": False, "reason": "Not enough daily data"}
        
        # Calculate issue count trends
        counts = [daily_stats[date]["count"] for date in dates]
        count_trend = self._calculate_trend(counts)
        
        # Calculate category trends
        category_trends = {}
        all_categories = set()
        for date in dates:
            all_categories.update(daily_stats[date]["categories"].keys())
        
        for category in all_categories:
            category_counts = []
            for date in dates:
                category_counts.append(daily_stats[date]["categories"].get(category, 0))
            
            if any(c > 0 for c in category_counts):  # Only analyze non-zero data
                category_trends[category] = self._calculate_trend(category_counts)
        
        # Calculate severity trends
        severity_trends = {}
        all_severities = set()
        for date in dates:
            all_severities.update(daily_stats[date]["severities"].keys())
        
        for severity in all_severities:
            severity_counts = []
            for date in dates:
                severity_counts.append(daily_stats[date]["severities"].get(severity, 0))
            
            if any(c > 0 for c in severity_counts):  # Only analyze non-zero data
                severity_trends[severity] = self._calculate_trend(severity_counts)
        
        return {
            "trend_detected": True,
            "overall_count_trend": count_trend,
            "category_trends": category_trends,
            "severity_trends": severity_trends
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a series of values"""
        if not values or len(values) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(values)
        sum_x2 = sum(xi**2 for xi in x)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2) if (n * sum_x2 - sum_x**2) != 0 else 0
        
        # Determine trend direction and strength
        if abs(slope) < 0.1:
            trend = "stable"
            strength = "none"
        else:
            trend = "increasing" if slope > 0 else "decreasing"
            if abs(slope) < 0.5:
                strength = "mild"
            elif abs(slope) < 1.0:
                strength = "moderate"
            else:
                strength = "strong"
        
        return {
            "trend": trend,
            "strength": strength,
            "slope": slope,
            "values": values
        }
    
    async def get_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations based on issue analysis
        
        Returns:
            Recommendation data
        """
        # Analyze issues first
        analysis = await self.analyze_issues()
        statistics = await self.get_statistics()
        
        recommendations = []
        
        # Check for components with recurring issues
        if analysis["components"]["recurring_issues"]:
            for component, data in analysis["components"]["recurring_issues"].items():
                if data["ratio"] > 0.7:  # High recurrence rate
                    recommendations.append({
                        "type": "component_review",
                        "priority": "high",
                        "component": component,
                        "reason": f"Component has a high rate of recurring issues ({data['ratio']*100:.1f}%)",
                        "suggestion": "Conduct comprehensive review of component architecture and integration"
                    })
        
        # Check for concerning trends
        if analysis.get("trends", {}).get("trend_detected", False):
            # Check overall trend
            count_trend = analysis["trends"]["overall_count_trend"]
            if count_trend["trend"] == "increasing" and count_trend["strength"] in ["moderate", "strong"]:
                recommendations.append({
                    "type": "system_stability",
                    "priority": "high",
                    "reason": f"Strong increasing trend in overall issue count",
                    "suggestion": "Review recent system changes and consider stability improvements"
                })
            
            # Check category trends
            for category, trend in analysis["trends"].get("category_trends", {}).items():
                if trend["trend"] == "increasing" and trend["strength"] in ["moderate", "strong"]:
                    recommendations.append({
                        "type": "category_focus",
                        "priority": "medium",
                        "category": category,
                        "reason": f"Strong increasing trend in {category} issues",
                        "suggestion": f"Focus on improving {category} handling and robustness"
                    })
            
            # Check severity trends
            for severity, trend in analysis["trends"].get("severity_trends", {}).items():
                if severity in ["CRITICAL", "HIGH"] and trend["trend"] == "increasing":
                    recommendations.append({
                        "type": "severity_concern",
                        "priority": "high",
                        "severity": severity,
                        "reason": f"Increasing trend in {severity} severity issues",
                        "suggestion": "Urgently address high-severity issues and their root causes"
                    })
        
        # Check resolution time
        if statistics.get("resolution_times", {}).get("avg_seconds", 0) > 86400:  # More than 24 hours
            recommendations.append({
                "type": "resolution_time",
                "priority": "medium",
                "reason": "Average resolution time exceeds 24 hours",
                "suggestion": "Improve issue handling processes and consider automation for common issues"
            })
        
        # Check open ratio
        if statistics.get("open_ratio", 0) > 0.7:  # More than 70% open
            recommendations.append({
                "type": "open_issues",
                "priority": "medium",
                "reason": f"High ratio of open issues ({statistics['open_ratio']*100:.1f}%)",
                "suggestion": "Allocate additional resources to issue resolution"
            })
        
        return {
            "recommendations": recommendations,
            "based_on": {
                "total_issues": statistics.get("total_issues", 0),
                "components_analyzed": len(analysis["components"]["all"]),
                "days_analyzed": len(analysis.get("daily_statistics", {}))
            }
        }
    
    # Some example resolution methods for common issues
    
    async def _resolve_check_memory_indexing(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to resolve memory indexing issues"""
        if not self.brain or not hasattr(self.brain, "memory_core"):
            return {"success": False, "reason": "Memory core not available"}
            
        try:
            # Check if reindexing method exists
            if hasattr(self.brain.memory_core, "reindex_memories"):
                await self.brain.memory_core.reindex_memories()
                return {
                    "success": True,
                    "action": "reindexed_memories",
                    "message": "Memory reindexing completed successfully"
                }
            else:
                return {"success": False, "reason": "Reindexing method not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _resolve_adjust_emotional_decay_rates(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to resolve emotional regulation issues"""
        if not self.brain or not hasattr(self.brain, "emotional_core"):
            return {"success": False, "reason": "Emotional core not available"}
            
        try:
            # Check if adjustment method exists
            if hasattr(self.brain.emotional_core, "adjust_decay_rates"):
                await self.brain.emotional_core.adjust_decay_rates(increase_factor=1.2)
                return {
                    "success": True,
                    "action": "adjusted_decay_rates",
                    "message": "Emotional decay rates adjusted"
                }
            elif hasattr(self.brain.emotional_core, "reset_emotional_state"):
                await self.brain.emotional_core.reset_emotional_state()
                return {
                    "success": True,
                    "action": "reset_emotional_state",
                    "message": "Emotional state reset to baseline"
                }
            else:
                return {"success": False, "reason": "Adjustment methods not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _resolve_run_identity_consolidation(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to resolve identity contradiction issues"""
        if not self.brain or not hasattr(self.brain, "identity_evolution"):
            return {"success": False, "reason": "Identity evolution not available"}
            
        try:
            # Check if consolidation method exists
            if hasattr(self.brain.identity_evolution, "consolidate_identity"):
                result = await self.brain.identity_evolution.consolidate_identity()
                return {
                    "success": True,
                    "action": "consolidated_identity",
                    "contradictions_resolved": result.get("contradictions_resolved", 0),
                    "message": "Identity consolidation completed"
                }
            else:
                return {"success": False, "reason": "Consolidation method not available"}
        except Exception as e:
            return {"success": False, "error": str(e)}
