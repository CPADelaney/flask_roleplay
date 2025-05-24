# nyx/core/a2a/context_aware_analysis_sandbox.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from nyx.core.brain.integration_layer import ContextAwareModule
from nyx.core.brain.context_distribution import SharedContext, ContextUpdate, ContextScope, ContextPriority

logger = logging.getLogger(__name__)

class ContextAwareAnalysisSandbox(ContextAwareModule):
    """
    Advanced Analysis and Sandbox System with full context distribution capabilities
    """
    
    def __init__(self, original_code_analyzer, original_sandbox_executor=None):
        super().__init__("analysis_sandbox")
        self.code_analyzer = original_code_analyzer
        self.sandbox_executor = original_sandbox_executor
        self.context_subscriptions = [
            "code_analysis_request", "sandbox_execution_request", "goal_context_available",
            "safety_check_request", "capability_assessment_request", "code_generated",
            "analysis_needed", "execution_environment_update"
        ]
        self.current_analysis_context = {}
    
    async def on_context_received(self, context: SharedContext):
        """Initialize analysis processing for this context"""
        logger.debug(f"AnalysisSandbox received context for user: {context.user_id}")
        
        # Check if input contains code or requests analysis
        analysis_needed = await self._check_analysis_needed(context)
        
        if analysis_needed:
            await self.send_context_update(
                update_type="analysis_capability_ready",
                data={
                    "analysis_types": ["code_review", "module_analysis", "codebase_analysis"],
                    "sandbox_available": self.sandbox_executor is not None,
                    "languages_supported": ["python"],  # Can be extended
                    "safety_features": ["timeout", "memory_limit", "isolated_execution"]
                },
                priority=ContextPriority.NORMAL
            )
    
    async def on_context_update(self, update: ContextUpdate):
        """Handle updates from other modules"""
        
        if update.update_type == "code_generated":
            # Automatically analyze generated code
            code_data = update.data
            code = code_data.get("code", "")
            language = code_data.get("language", "python")
            
            if code:
                analysis = await self.code_analyzer.review_code(code, language)
                
                await self.send_context_update(
                    update_type="code_analysis_complete",
                    data={
                        "analysis": analysis,
                        "code_quality_score": self._calculate_quality_score(analysis),
                        "safety_verified": self._verify_code_safety(analysis)
                    },
                    priority=ContextPriority.HIGH
                )
        
        elif update.update_type == "sandbox_execution_request":
            # Execute code in sandbox
            execution_data = update.data
            code = execution_data.get("code", "")
            language = execution_data.get("language", "python")
            
            if code and self.sandbox_executor:
                result = await self._execute_safely(code, language, execution_data)
                
                await self.send_context_update(
                    update_type="sandbox_execution_complete",
                    data=result,
                    target_modules=[update.source_module],
                    scope=ContextScope.TARGETED
                )
        
        elif update.update_type == "goal_context_available":
            # Check if any goals require code analysis
            goal_data = update.data
            analysis_goals = self._identify_analysis_goals(goal_data)
            
            if analysis_goals:
                await self.send_context_update(
                    update_type="analysis_goals_identified",
                    data={
                        "analysis_goals": analysis_goals,
                        "ready_to_assist": True
                    }
                )
    
    async def process_input(self, context: SharedContext) -> Dict[str, Any]:
        """Process input for analysis needs"""
        user_input = context.user_input
        
        # Detect code blocks in input
        code_blocks = self._extract_code_blocks(user_input)
        
        analysis_results = []
        for block in code_blocks:
            # Analyze each code block
            analysis = await self.code_analyzer.review_code(
                block["code"],
                block.get("language", "python")
            )
            
            analysis_results.append({
                "code": block["code"],
                "language": block["language"],
                "analysis": analysis
            })
        
        # Check if user is requesting analysis of existing code
        if self._is_analysis_request(user_input):
            target = self._extract_analysis_target(user_input)
            
            if target["type"] == "module":
                analysis = await self.code_analyzer.analyze_module(target["path"])
                analysis_results.append({
                    "type": "module_analysis",
                    "target": target["path"],
                    "analysis": analysis
                })
            elif target["type"] == "codebase":
                analysis = await self.code_analyzer.analyze_codebase(
                    target["path"],
                    target.get("extensions")
                )
                analysis_results.append({
                    "type": "codebase_analysis",
                    "target": target["path"],
                    "analysis": analysis
                })
        
        # Send analysis updates if any
        if analysis_results:
            await self.send_context_update(
                update_type="input_analysis_complete",
                data={
                    "analysis_results": analysis_results,
                    "total_issues": sum(len(r["analysis"].get("issues", [])) for r in analysis_results),
                    "quality_metrics": self._aggregate_quality_metrics(analysis_results)
                },
                priority=ContextPriority.HIGH
            )
        
        # Get cross-module messages
        messages = await self.get_cross_module_messages()
        
        return {
            "code_blocks_found": len(code_blocks),
            "analysis_performed": len(analysis_results),
            "analysis_results": analysis_results,
            "cross_module_context": len(messages) > 0
        }
    
    async def process_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Perform deeper analysis based on context"""
        messages = await self.get_cross_module_messages()
        
        # Analyze code quality trends if we have history
        quality_trends = await self._analyze_quality_trends(context)
        
        # Check for security implications
        security_analysis = await self._perform_security_analysis(context)
        
        # Analyze dependencies and complexity
        complexity_analysis = await self._analyze_complexity(context)
        
        # Consider goal alignment
        goal_alignment = await self._analyze_goal_alignment(context, messages)
        
        return {
            "quality_trends": quality_trends,
            "security_analysis": security_analysis,
            "complexity_analysis": complexity_analysis,
            "goal_alignment": goal_alignment,
            "deep_analysis_complete": True
        }
    
    async def process_synthesis(self, context: SharedContext) -> Dict[str, Any]:
        """Synthesize analysis insights for response"""
        messages = await self.get_cross_module_messages()
        
        # Generate recommendations based on all analyses
        recommendations = await self._generate_recommendations(context, messages)
        
        # Prepare synthesis
        synthesis = {
            "code_quality_summary": await self._summarize_code_quality(context),
            "improvement_suggestions": recommendations["improvements"],
            "safety_considerations": recommendations["safety"],
            "next_steps": recommendations["next_steps"],
            "learning_opportunities": self._identify_learning_opportunities(context)
        }
        
        # Check if we should suggest sandbox execution
        if self._should_suggest_execution(context):
            synthesis["suggest_execution"] = True
            synthesis["execution_parameters"] = self._suggest_execution_params(context)
        
        return synthesis
    
    # ========================================================================================
    # HELPER METHODS
    # ========================================================================================
    
    async def _check_analysis_needed(self, context: SharedContext) -> bool:
        """Check if analysis is needed based on context"""
        analysis_keywords = ["analyze", "review", "check", "evaluate", "assess", "code", "implementation"]
        return any(keyword in context.user_input.lower() for keyword in analysis_keywords)
    
    async def _execute_safely(self, code: str, language: str, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code safely in sandbox with context awareness"""
        if not self.sandbox_executor:
            return {
                "error": "Sandbox executor not available",
                "success": False
            }
        
        # Add safety checks based on context
        if not self._verify_safe_for_execution(code):
            return {
                "error": "Code failed safety verification",
                "success": False,
                "safety_issues": self._get_safety_issues(code)
            }
        
        # Execute with monitoring
        try:
            result = await self.sandbox_executor.execute_code(
                code=code,
                language=language,
                save_output=execution_data.get("save_output", True)
            )
            
            # Add context-aware metadata
            result["execution_context"] = {
                "requested_by": execution_data.get("source_module"),
                "purpose": execution_data.get("purpose"),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall code quality score from analysis"""
        if "error" in analysis:
            return 0.0
        
        issues = analysis.get("issues", [])
        suggestions = analysis.get("suggestions", [])
        strengths = analysis.get("strengths", [])
        
        # Base score
        score = 0.7
        
        # Deduct for issues
        score -= len(issues) * 0.05
        
        # Add for strengths
        score += len(strengths) * 0.1
        
        # Consider suggestion severity
        score -= len(suggestions) * 0.02
        
        return max(0.0, min(1.0, score))
    
    def _verify_code_safety(self, analysis: Dict[str, Any]) -> bool:
        """Verify code is safe based on analysis"""
        issues = analysis.get("issues", [])
        
        # Check for dangerous patterns
        dangerous_types = ["bare_except", "eval_usage", "exec_usage", "system_call"]
        
        for issue in issues:
            if issue.get("type") in dangerous_types:
                return False
        
        return True
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks from text"""
        blocks = []
        
        # Simple extraction - look for triple backticks
        import re
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            language = match[0] or "python"
            code = match[1].strip()
            blocks.append({
                "language": language,
                "code": code
            })
        
        return blocks
    
    def _is_analysis_request(self, text: str) -> bool:
        """Check if text is requesting code analysis"""
        request_patterns = [
            "analyze this",
            "review this code",
            "check my code",
            "analyze the",
            "review the"
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in request_patterns)
    
    def _extract_analysis_target(self, text: str) -> Dict[str, Any]:
        """Extract what should be analyzed from text"""
        text_lower = text.lower()
        
        # Check for module path
        if "module" in text_lower or ".py" in text:
            # Extract potential file path
            import re
            path_pattern = r'[\w/\\]+\.py'
            match = re.search(path_pattern, text)
            if match:
                return {
                    "type": "module",
                    "path": match.group(0)
                }
        
        # Check for codebase analysis
        if "codebase" in text_lower or "directory" in text_lower:
            # Extract potential directory path
            import re
            dir_pattern = r'[\w/\\]+'
            match = re.search(dir_pattern, text)
            return {
                "type": "codebase",
                "path": match.group(0) if match else ".",
                "extensions": [".py"]  # Default to Python
            }
        
        return {"type": "unknown", "path": None}
    
    def _identify_analysis_goals(self, goal_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify goals that involve code analysis"""
        active_goals = goal_data.get("active_goals", [])
        analysis_goals = []
        
        analysis_keywords = ["analyze", "review", "improve", "optimize", "refactor", "debug"]
        
        for goal in active_goals:
            description = goal.get("description", "").lower()
            if any(keyword in description for keyword in analysis_keywords):
                analysis_goals.append(goal)
        
        return analysis_goals
    
    def _aggregate_quality_metrics(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate quality metrics from multiple analyses"""
        total_issues = 0
        total_suggestions = 0
        total_strengths = 0
        
        for result in analysis_results:
            analysis = result.get("analysis", {})
            total_issues += len(analysis.get("issues", []))
            total_suggestions += len(analysis.get("suggestions", []))
            total_strengths += len(analysis.get("strengths", []))
        
        return {
            "total_issues": total_issues,
            "total_suggestions": total_suggestions,
            "total_strengths": total_strengths,
            "overall_quality": "good" if total_issues < 3 else "needs_improvement"
        }
    
    async def _analyze_quality_trends(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze code quality trends over time"""
        # This would analyze historical data if available
        return {
            "trend": "improving",
            "recent_improvements": ["better_error_handling", "cleaner_structure"],
            "persistent_issues": ["line_length", "complex_functions"]
        }
    
    async def _perform_security_analysis(self, context: SharedContext) -> Dict[str, Any]:
        """Perform security analysis on code"""
        security_concerns = []
        
        # Check for common security issues
        if "eval" in context.user_input or "exec" in context.user_input:
            security_concerns.append("potential_code_injection")
        
        if "password" in context.user_input.lower() or "token" in context.user_input.lower():
            security_concerns.append("possible_credential_exposure")
        
        return {
            "security_level": "safe" if not security_concerns else "needs_review",
            "concerns": security_concerns,
            "recommendations": ["use_ast_literal_eval", "store_credentials_securely"] if security_concerns else []
        }
    
    async def _analyze_complexity(self, context: SharedContext) -> Dict[str, Any]:
        """Analyze code complexity"""
        # Simplified complexity analysis
        return {
            "cyclomatic_complexity": "moderate",
            "cognitive_complexity": "low",
            "suggestions": ["consider_breaking_down_large_functions"]
        }
    
    async def _analyze_goal_alignment(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze how code aligns with active goals"""
        goal_messages = messages.get("goal_manager", [])
        
        alignment_score = 0.7  # Default moderate alignment
        
        # Check if code supports active goals
        for msg in goal_messages:
            if msg.get("type") == "goal_context_available":
                goals = msg.get("data", {}).get("active_goals", [])
                if goals:
                    alignment_score = 0.9  # High alignment if supporting active goals
        
        return {
            "alignment_score": alignment_score,
            "supports_goals": alignment_score > 0.7,
            "recommendation": "code aligns well with objectives" if alignment_score > 0.7 else "consider goal alignment"
        }
    
    async def _generate_recommendations(self, context: SharedContext, messages: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate context-aware recommendations"""
        recommendations = {
            "improvements": [],
            "safety": [],
            "next_steps": []
        }
        
        # Base recommendations on context
        if self.current_analysis_context.get("total_issues", 0) > 5:
            recommendations["improvements"].append("Focus on addressing high-priority issues first")
        
        if self.current_analysis_context.get("security_concerns"):
            recommendations["safety"].append("Review and address security concerns before deployment")
        
        # Goal-based recommendations
        goal_messages = messages.get("goal_manager", [])
        if goal_messages:
            recommendations["next_steps"].append("Align code improvements with active goals")
        
        return recommendations
    
    async def _summarize_code_quality(self, context: SharedContext) -> str:
        """Summarize overall code quality"""
        if not self.current_analysis_context:
            return "No code analysis performed yet"
        
        issues = self.current_analysis_context.get("total_issues", 0)
        
        if issues == 0:
            return "Code quality is excellent with no issues detected"
        elif issues < 5:
            return "Code quality is good with minor issues to address"
        else:
            return "Code quality needs improvement with several issues identified"
    
    def _identify_learning_opportunities(self, context: SharedContext) -> List[str]:
        """Identify learning opportunities from analysis"""
        opportunities = []
        
        if "mutable_default" in str(self.current_analysis_context):
            opportunities.append("Learn about Python mutable default arguments")
        
        if "bare_except" in str(self.current_analysis_context):
            opportunities.append("Explore specific exception handling patterns")
        
        return opportunities
    
    def _should_suggest_execution(self, context: SharedContext) -> bool:
        """Determine if we should suggest executing code"""
        # Suggest execution if code is safe and user might benefit from seeing results
        return (
            self.sandbox_executor is not None and
            "execute" not in context.user_input.lower() and  # Not already requested
            self.current_analysis_context.get("safety_verified", False)
        )
    
    def _suggest_execution_params(self, context: SharedContext) -> Dict[str, Any]:
        """Suggest parameters for code execution"""
        return {
            "save_output": True,
            "timeout": 10,
            "purpose": "demonstrate_functionality"
        }
    
    def _verify_safe_for_execution(self, code: str) -> bool:
        """Verify code is safe for sandbox execution"""
        dangerous_patterns = [
            "subprocess", "os.system", "__import__", "eval", "exec",
            "compile", "open(", "file("
        ]
        
        code_lower = code.lower()
        return not any(pattern in code_lower for pattern in dangerous_patterns)
    
    def _get_safety_issues(self, code: str) -> List[str]:
        """Get specific safety issues in code"""
        issues = []
        
        if "subprocess" in code or "os.system" in code:
            issues.append("System command execution detected")
        
        if "eval" in code or "exec" in code:
            issues.append("Dynamic code execution detected")
        
        if "__import__" in code:
            issues.append("Dynamic import detected")
        
        return issues
    
    # Delegate methods to original analyzer
    def __getattr__(self, name):
        """Delegate missing methods to original analyzer"""
        return getattr(self.code_analyzer, name)
