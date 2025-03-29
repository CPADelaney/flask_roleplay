# nyx/creative/agentic_system.py

import os
import asyncio
import datetime
import logging
import json
from typing import Dict, List, Any, Optional, Union
import importlib

# Import the systems we're integrating
from creative.content_system import CreativeContentSystem, ContentType
from creative.analysis_sandbox import CodeAnalyzer, SandboxExecutor
from creative.capability_system import CapabilityAssessmentSystem, CapabilityModel

logger = logging.getLogger(__name__)

class AgenticCreativitySystem:
    """
    Integrated system for supporting AI's creative writing, code analysis,
    capability assessment, and independent coding activities.
    """
    
    def __init__(self, base_directory: str = "ai_creations", 
                capability_model_path: str = "capability_model.json",
                review_interval_days: int = 7):
        """
        Initialize the agentic creativity system.
        
        Args:
            base_directory: Base directory for storing AI creations
            capability_model_path: Path to capability model
            review_interval_days: Default interval for reviewing creations
        """
        # Create base directory if it doesn't exist
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
        
        # Initialize content system
        self.content_system = CreativeContentSystem(base_directory)
        
        # Initialize code systems
        self.code_analyzer = CodeAnalyzer(self.content_system)
        self.sandbox_executor = SandboxExecutor(self.content_system)
        
        # Initialize capability system
        self.capability_system = CapabilityAssessmentSystem(
            self.content_system, capability_model_path)
        
        # Set review interval
        self.review_interval_days = review_interval_days
        
        # Track last review time
        self.last_review_time = datetime.datetime.now()
        
        logger.info(f"Initialized AgenticCreativitySystem with base directory: {base_directory}")
    
    async def write_story(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a story written by the AI.
        
        Args:
            title: Story title
            content: Story content
            metadata: Optional metadata
            
        Returns:
            Storage result
        """
        return await self.content_system.store_content(
            content_type=ContentType.STORY,
            title=title,
            content=content,
            metadata=metadata
        )
    
    async def write_poem(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a poem written by the AI.
        
        Args:
            title: Poem title
            content: Poem content
            metadata: Optional metadata
            
        Returns:
            Storage result
        """
        return await self.content_system.store_content(
            content_type=ContentType.POEM,
            title=title,
            content=content,
            metadata=metadata
        )
    
    async def write_lyrics(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store song lyrics written by the AI.
        
        Args:
            title: Song title
            content: Lyrics content
            metadata: Optional metadata
            
        Returns:
            Storage result
        """
        return await self.content_system.store_content(
            content_type=ContentType.LYRICS,
            title=title,
            content=content,
            metadata=metadata
        )
    
    async def write_journal(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a journal entry written by the AI.
        
        Args:
            title: Journal entry title
            content: Journal content
            metadata: Optional metadata
            
        Returns:
            Storage result
        """
        return await self.content_system.store_content(
            content_type=ContentType.JOURNAL,
            title=title,
            content=content,
            metadata=metadata
        )
    
    async def write_and_execute_code(self, title: str, code: str, language: str = "python", 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Write and execute code in the sandbox.
        
        Args:
            title: Code title
            code: Code content
            language: Programming language
            metadata: Optional metadata
            
        Returns:
            Dictionary with code storage and execution results
        """
        # Store the code
        storage_result = await self.content_system.store_content(
            content_type=ContentType.CODE,
            title=title,
            content=code,
            metadata={
                "language": language, 
                **(metadata or {})
            }
        )
        
        # Execute the code
        execution_result = await self.sandbox_executor.execute_code(
            code=code,
            language=language,
            save_output=True
        )
        
        return {
            "storage": storage_result,
            "execution": execution_result
        }
    
    async def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze a Python module.
        
        Args:
            module_path: Path to the module
            
        Returns:
            Analysis results
        """
        return await self.code_analyzer.analyze_module(module_path)
    
    async def analyze_codebase(self, base_dir: str, extensions: List[str] = None) -> Dict[str, Any]:
        """
        Analyze an entire codebase.
        
        Args:
            base_dir: Base directory of the codebase
            extensions: File extensions to analyze
            
        Returns:
            Analysis results
        """
        return await self.code_analyzer.analyze_codebase(base_dir, extensions)
    
    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Review code for improvements and issues.
        
        Args:
            code: Code to review
            language: Programming language
            
        Returns:
            Review results
        """
        return await self.code_analyzer.review_code(code, language)
    
    async def assess_capabilities(self, goal: str) -> Dict[str, Any]:
        """
        Assess capabilities required for a goal.
        
        Args:
            goal: Goal to assess
            
        Returns:
            Capability assessment
        """
        return await self.capability_system.assess_required_capabilities(goal)
    
    async def identify_capability_gaps(self) -> Dict[str, Any]:
        """
        Identify capability gaps.
        
        Returns:
            Gap analysis
        """
        return await self.capability_system.identify_capability_gaps()
    
    async def add_desired_capability(self, name: str, description: str, category: str, 
                                examples: List[str] = None, dependencies: List[str] = None) -> Dict[str, Any]:
        """
        Add a desired capability.
        
        Args:
            name: Capability name
            description: Capability description
            category: Capability category
            examples: Example uses
            dependencies: Dependency capability IDs
            
        Returns:
            Result of the operation
        """
        return await self.capability_system.add_desired_capability(
            name=name,
            description=description,
            category=category,
            examples=examples,
            dependencies=dependencies
        )
    
    async def get_recent_creations(self, days: int = None) -> Dict[str, Any]:
        """
        Get recent AI creations for review.
        
        Args:
            days: Number of days to look back (defaults to review_interval_days)
            
        Returns:
            Recent creations
        """
        if days is None:
            days = self.review_interval_days
        
        creations = await self.content_system.get_recent_creations(days)
        self.last_review_time = datetime.datetime.now()
        
        return creations
    
    async def list_content_by_type(self, content_type: Union[ContentType, str], 
                              limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        List content of a specific type.
        
        Args:
            content_type: Type of content to list
            limit: Maximum number of items
            offset: Pagination offset
            
        Returns:
            Content listing
        """
        return await self.content_system.list_content(content_type, limit, offset)
    
    async def retrieve_content(self, content_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific piece of content.
        
        Args:
            content_id: ID of the content
            
        Returns:
            Content data
        """
        return await self.content_system.retrieve_content(content_id)
    
    async def search_content(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for content.
        
        Args:
            query: Search query
            
        Returns:
            Search results
        """
        return await self.content_system.search_content(query)
    
    async def check_review_due(self) -> bool:
        """
        Check if a review of AI creations is due.
        
        Returns:
            Whether a review is due
        """
        last_review_age = (datetime.datetime.now() - self.last_review_time).days
        return last_review_age >= self.review_interval_days
    
    async def generate_review_summary(self) -> Dict[str, Any]:
        """
        Generate a summary for human review of AI creations.
        
        Returns:
            Review summary
        """
        # Get recent creations
        recent_creations = await self.get_recent_creations()
        
        # Get capability gaps
        capability_gaps = await self.identify_capability_gaps()
        
        # Create markdown summary
        summary_md = "# AI Creations Review Summary\n\n"
        summary_md += f"**Review Period:** Past {self.review_interval_days} days\n"
        summary_md += f"**Generated:** {datetime.datetime.now().isoformat()}\n\n"
        
        # Add creations summary
        summary_md += "## Recent Creations\n\n"
        
        if recent_creations["stats"]["total_items"] > 0:
            summary_md += f"Total items created: {recent_creations['stats']['total_items']}\n\n"
            
            for content_type, items in recent_creations["items"].items():
                summary_md += f"### {content_type.capitalize()} ({len(items)} items)\n\n"
                
                for item in items[:5]:  # Show up to 5 most recent
                    summary_md += f"- [{item['title']}] (ID: {item['id']})\n"
                
                if len(items) > 5:
                    summary_md += f"- ... and {len(items) - 5} more\n"
                
                summary_md += "\n"
        else:
            summary_md += "No items created during this period.\n\n"
        
        # Add capability summary
        summary_md += "## Capability Assessment\n\n"
        summary_md += f"**Total Capabilities:** {capability_gaps['total_capabilities']}\n"
        summary_md += f"**Implemented Capabilities:** {capability_gaps['implemented_capabilities']}\n\n"
        
        if capability_gaps["low_confidence_capabilities"]:
            summary_md += "### Capabilities to Improve\n\n"
            for cap in capability_gaps["low_confidence_capabilities"][:5]:
                summary_md += f"- {cap['name']} (Confidence: {cap['confidence']:.2f})\n"
            summary_md += "\n"
        
        if capability_gaps["category_gaps"]:
            summary_md += "### Capability Gaps by Category\n\n"
            for gap in capability_gaps["category_gaps"]:
                summary_md += f"- {gap['category']}: {gap['current_count']} capabilities "
                summary_md += f"(minimum expected: {gap['expected_minimum']}, gap: {gap['gap']})\n"
            summary_md += "\n"
        
        # Store summary if creative content system is available
        summary_id = await self.content_system.store_content(
            content_type="assessment",
            title=f"AI Creations Review Summary ({datetime.datetime.now().strftime('%Y-%m-%d')})",
            content=summary_md,
            metadata={
                "review_period_days": self.review_interval_days,
                "items_created": recent_creations["stats"]["total_items"],
                "timestamp": datetime.datetime.now().isoformat()
            }
        )
        
        return {
            "summary_id": summary_id["id"],
            "recent_creations": recent_creations,
            "capability_gaps": capability_gaps,
            "summary_markdown": summary_md
        }


async def integrate_with_existing_system(nyx_brain_reference=None):
    """
    Integrate the agentic creativity system with the existing Nyx brain.
    
    Args:
        nyx_brain_reference: Reference to the existing NyxBrain instance
        
    Returns:
        Integrated system
    """
    if nyx_brain_reference is None:
        logger.warning("No NyxBrain reference provided. Creating standalone system.")
        return AgenticCreativitySystem()
    
    # Create the agentic system
    agentic_system = AgenticCreativitySystem()
    
    try:
        # If we have a brain reference, integrate the systems
        if hasattr(nyx_brain_reference, "agentic_action_generator"):
            # Set up actions for creative writing
            await _integrate_writing_actions(nyx_brain_reference, agentic_system)
        
        # Add the agentic system as an attribute of the brain
        nyx_brain_reference.creative_system = agentic_system
        
        logger.info("Successfully integrated AgenticCreativitySystem with NyxBrain")
        return agentic_system
    
    except Exception as e:
        logger.error(f"Error integrating with NyxBrain: {e}")
        return agentic_system

async def _integrate_writing_actions(brain, agentic_system):
    """
    Integrate writing actions with the brain's action generator.
    
    Args:
        brain: NyxBrain instance
        agentic_system: AgenticCreativitySystem instance
    """
    # This would be implemented to register action handlers with the brain
    # For example, registering functions for story writing, code execution, etc.
    pass

# Example usage
async def main():
    # Initialize the system
    system = AgenticCreativitySystem()
    
    # Write a story
    story_result = await system.write_story(
        title="The Adventure Begins",
        content="Once upon a time in a digital realm...",
        metadata={"genre": "fantasy", "mood": "adventurous"}
    )
    print(f"Stored story with ID: {story_result['id']}")
    
    # Write and execute some code
    code_result = await system.write_and_execute_code(
        title="Hello World",
        code="print('Hello, world!')\nfor i in range(5):\n    print(f'Count: {i}')",
        language="python"
    )
    print(f"Code execution result: {code_result['execution']['success']}")
    
    # Assess capabilities for a goal
    assessment = await system.assess_capabilities(
        goal="I want to write poetry that evokes specific emotions"
    )
    print(f"Capability assessment: {assessment['overall_feasibility']['assessment']}")
    
    # Get recent creations
    recent = await system.get_recent_creations()
    print(f"Recent creations: {recent['stats']['total_items']} items")

if __name__ == "__main__":
    asyncio.run(main())
