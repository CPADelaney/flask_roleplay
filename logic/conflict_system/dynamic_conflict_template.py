# logic/conflict_system/dynamic_conflict_template.py
"""
Dynamic Conflict Template System with LLM-generated variations
Integrated with ConflictSynthesizer as the central orchestrator
REFACTORED: Fixed RunResult attribute access issues and JSON parsing
"""

import logging
import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple, Set, TypedDict, NotRequired, Iterable
from enum import Enum
from dataclasses import dataclass, is_dataclass, asdict
from datetime import datetime

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner

# Import get_db_connection_context with proper error handling
try:
    from db.connection import get_db_connection_context
except ImportError:
    # Fallback for testing
    async def get_db_connection_context():
        raise NotImplementedError("Database connection not available")

logger = logging.getLogger(__name__)

# ===============================================================================
# HELPER FUNCTION FOR RUNNER RESPONSE EXTRACTION
# ===============================================================================

def _strip_code_fences(s: str) -> str:
    """
    If the model wrapped the JSON in triple backticks, pull out the inner block.
    Prefer the inner block only when it contains { or [.
    """
    if "```" not in s:
        return s

    # Extract first fenced block
    start = s.find("```")
    end = s.find("```", start + 3)
    if end == -1:
        return s

    inner = s[start + 3:end]
    # Drop a potential language label (e.g., ```json)
    if "\n" in inner:
        first_line, rest = inner.split("\n", 1)
        # If first line looks like a language tag, use the rest
        inner = rest if re.fullmatch(r"[a-zA-Z0-9_-]+", first_line.strip()) else inner

    return inner if ("{" in inner or "[" in inner) else s


def _extract_json_fragment(s: str) -> Optional[str]:
    """
    Find the first balanced JSON object or array inside s.
    Handles quotes and escapes so braces inside strings don't break scanning.
    """
    m = re.search(r"[\{\[]", s)
    if not m:
        return None

    start = m.start()
    open_ch = s[start]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    i = start
    in_str = False
    esc = False

    while i < len(s):
        ch = s[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    if _looks_like_json(candidate):
                        return candidate
                    # If it's balanced but still not valid JSON, give up.
                    return None

        i += 1

    return None

def _clean_and_maybe_extract_json(s: str) -> str:
    """Strip wrappers, try to extract/repair a valid JSON object/array; else return trimmed text."""
    if not s:
        return "{}"
    s1 = _strip_runresult_wrappers(s)

    # 1) Whole string as JSON (or repaired JSON)
    fixed = _try_json_with_repairs(s1)
    if fixed is not None:
        return fixed

    # 2) Fenced code blocks
    fenced = _extract_from_code_fence(s1)
    if fenced:
        fixed = _try_json_with_repairs(fenced)
        if fixed is not None:
            return fixed

    # 3) First balanced JSON-ish snippet
    snippet = _extract_balanced_json_snippet(s1)
    if snippet:
        fixed = _try_json_with_repairs(snippet)
        if fixed is not None:
            return fixed

    # 4) Not JSON: return cleaned text (no preambles)
    return s1.strip()

# --- add these new helpers below the existing helpers ---
def _try_json_with_repairs(s: str) -> str | None:
    """Return a valid JSON string if s is JSON or can be repaired; else None."""
    if _looks_like_json(s):
        return s.strip()

    # 1) Best-effort textual repairs
    repaired = _repair_jsonish(s)
    if repaired and _looks_like_json(repaired):
        return repaired.strip()

    # 2) Python-literal fallback (handles single quotes, trailing commas, etc.)
    try:
        obj = ast.literal_eval(s.strip())
        if isinstance(obj, (dict, list, tuple)):
            return json.dumps(obj, ensure_ascii=False)
    except Exception:
        pass

    return None

def _repair_jsonish(s: str) -> str:
    """
    Best-effort fixes for common LLM 'JSON-like' output:
      - Normalize Unicode minus U+2212 to ASCII '-'
      - Remove unary '+' before numbers (e.g., : +2 -> : 2, [, +1, -> [, 1,)
      - Remove trailing commas before } or ] (quote-aware)
    """
    if not s:
        return s

    # Normalize Unicode minus to ASCII hyphen
    s2 = s.replace("−", "-")  # U+2212

    # Remove unary plus in object values and arrays (after ':', ',' or '[')
    s2 = re.sub(r'([:\[,]\s*)\+(\d+(?:\.\d+)?)', r'\1\2', s2)

    # Remove trailing commas safely (don’t touch commas inside strings)
    s2 = _remove_trailing_commas_safe(s2)
    return s2

def _remove_trailing_commas_safe(s: str) -> str:
    """Strip commas that directly precede } or ] while respecting quoted strings."""
    out = []
    in_str = False
    esc = False
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue

        if ch == ',':
            j = i + 1
            # skip whitespace
            while j < n and s[j] in ' \t\r\n':
                j += 1
            if j < n and s[j] in ']}':
                # skip this comma
                i += 1
                continue

        out.append(ch)
        i += 1

    return ''.join(out)

def extract_runner_response(run_result: Any) -> str:
    """
    Normalize Runner.run(...) results into a clean payload.
    """
    if run_result is None:
        return "{}"

    if isinstance(run_result, (str, bytes)):
        s = run_result.decode("utf-8", errors="ignore") if isinstance(run_result, bytes) else run_result
        return _clean_and_maybe_extract_json(s)

    # Prefer structured payloads if present
    for attr in ("data", "result", "output", "final_output"):  # <-- added 'final_output'
        if hasattr(run_result, attr):
            val = getattr(run_result, attr)
            js = _to_json_str_or_none(val)
            if js is not None:
                return js
            if isinstance(val, (str, bytes)):
                s = val.decode("utf-8", errors="ignore") if isinstance(val, bytes) else val
                return _clean_and_maybe_extract_json(s)

    # Top-level mapping/sequence
    if isinstance(run_result, (dict, list, tuple)):
        js = _to_json_str_or_none(run_result)
        if js is not None:
            return js

    # Text-ish attributes commonly found on result objects
    for attr in ("output", "content", "text", "message"):
        if hasattr(run_result, attr):
            val = getattr(run_result, attr)
            if isinstance(val, (str, bytes)):
                s = val.decode("utf-8", errors="ignore") if isinstance(val, bytes) else val
                return _clean_and_maybe_extract_json(s)

    # Message-style structures
    if hasattr(run_result, "messages"):
        msgs = getattr(run_result, "messages") or []
        text_candidates: list[str] = []
        for m in reversed(_iter_as_dicts(msgs)):
            role = (m.get("role") or "").lower()
            if role in ("assistant", "tool", "function", "system"):
                parts = m.get("content")
                if isinstance(parts, str):
                    text_candidates.append(parts)
                elif isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, dict):
                            # Handle { "type": "output_text", "text": "..." } and friends
                            if isinstance(p.get("text"), str):
                                text_candidates.append(p["text"])
                            if isinstance(p.get("output"), str):
                                text_candidates.append(p["output"])
        for s in text_candidates:
            out = _clean_and_maybe_extract_json(s)
            if out:
                return out

    # Last resort: parse str(...)
    s = str(run_result)
    s = _strip_runresult_wrappers(s)
    out = _clean_and_maybe_extract_json(s)
    if _looks_like_json(out):
        return out

    logger.warning("extract_runner_response: returning empty JSON fallback; head=%r", (s or "")[:200])
    return "{}"


# ------------------------- helpers -------------------------

def _to_json_str_or_none(val: Any) -> str | None:
    """Return a JSON string if val is obviously serializable structured data; else None."""
    # Pydantic v2 model
    if hasattr(val, "model_dump_json"):
        try:
            return val.model_dump_json()
        except Exception:
            pass
    if hasattr(val, "model_dump"):
        try:
            return json.dumps(val.model_dump(), ensure_ascii=False)
        except Exception:
            pass
    # Dataclass
    if is_dataclass(val):
        try:
            return json.dumps(asdict(val), ensure_ascii=False)
        except Exception:
            pass
    # Plain dict/list/tuple
    if isinstance(val, (dict, list, tuple)):
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            pass
    # Records that can be turned into dict (e.g., asyncpg.Record)
    if hasattr(val, "items") and callable(getattr(val, "items", None)):
        try:
            return json.dumps(dict(val), ensure_ascii=False)
        except Exception:
            pass
    return None

def _looks_like_json(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if not ((s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))):
        return False
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _extract_from_code_fence(s: str) -> str | None:
    """
    Extract JSON from ```json ... ``` or ``` ... ``` first block.
    """
    # Prefer ```json
    import re
    m = re.search(r"```json\s*(.+?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"```\s*(.+?)\s*```", s, re.DOTALL)
    if m:
        return m.group(1)
    return None


def _strip_runresult_wrappers(s: str) -> str:
    """
    Remove common debug wrappers like:
      'RunResult:\n- Last agent: ...\n- Final output (str): <payload>'
    and any leading log lines.
    """
    if not s:
        return s
    # Fast path: cut after "Final output" marker if present
    markers = ("Final output (str):", "Final output:", "output:", "Output:")
    for m in markers:
        idx = s.find(m)
        if idx != -1:
            return s[idx + len(m):].strip()
    # Drop leading "RunResult:" header block if present (first blank line onwards)
    if "RunResult" in s:
        parts = s.splitlines()
        # find first line that looks like actual JSON or text content (contains '{', '[', or backticks)
        for i, line in enumerate(parts):
            if any(ch in line for ch in ("{", "[", "`")):
                return "\n".join(parts[i:]).strip()
        # else just return original trimmed
    return s.strip()


def _extract_balanced_json_snippet(s: str) -> str | None:
    """
    Scan for the first balanced JSON object/array, respecting quotes and escapes.
    Returns the substring or None.
    """
    opens = "{["
    closes = "}]\n"
    stack = []
    start_idx = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    continue
                top = stack[-1]
                if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                    stack.pop()
                    if not stack and start_idx is not None:
                        candidate = s[start_idx:i+1]
                        # quick sanity: must be non-empty JSON
                        if _looks_like_json(candidate):
                            return candidate
                else:
                    # mismatched; reset
                    stack.clear()
                    start_idx = None
    return None


def _iter_as_dicts(items: Iterable[Any]) -> Iterable[dict]:
    """Yield dict views of items that might be objects or dicts."""
    for it in items:
        if isinstance(it, dict):
            yield it
        else:
            # best effort: object with attributes
            try:
                d = dict(it)  # e.g., asyncpg.Record
                yield d
            except Exception:
                d = {}
                for name in ("role", "content", "type", "output", "text", "name"):
                    if hasattr(it, name):
                        d[name] = getattr(it, name)
                if d:
                    yield d

# ===============================================================================
# TEMPLATE TYPES
# ===============================================================================

class TemplateCategory(Enum):
    """Categories of conflict templates"""
    POWER_DYNAMICS = "power_dynamics"
    SOCIAL_HIERARCHY = "social_hierarchy"
    RESOURCE_COMPETITION = "resource_competition"
    IDEOLOGICAL_CLASH = "ideological_clash"
    PERSONAL_BOUNDARIES = "personal_boundaries"
    LOYALTY_TESTS = "loyalty_tests"
    HIDDEN_AGENDAS = "hidden_agendas"
    TRANSFORMATION_RESISTANCE = "transformation_resistance"


@dataclass
class ConflictTemplate:
    """Base template for generating conflicts"""
    template_id: int
    category: TemplateCategory
    name: str
    base_structure: Dict[str, Any]
    variable_elements: List[str]
    contextual_modifiers: Dict[str, Any]
    complexity_range: Tuple[float, float]


@dataclass
class GeneratedConflict:
    """A conflict generated from a template"""
    conflict_id: int
    template_id: int
    variation_seed: str
    customization: Dict[str, Any]
    narrative_hooks: List[str]
    unique_elements: List[str]

class TemplateContextDTO(TypedDict, total=False):
    # Common conflict/context fields (extend as needed, all optional)
    participants: List[int]
    stakeholders: List[int]
    npcs: List[int]
    location: str
    location_id: int
    scene_type: str
    activity: str
    description: str
    intensity: str
    intensity_level: float  # 0..1
    hooks: List[str]
    complexity: float       # 0..1

class GenerateTemplatedConflictResponse(TypedDict):
    conflict_id: int
    status: str
    conflict_type: str
    template_used: int
    narrative_hooks: List[str]
    message: str
    error: str

class CreateTemplateResponse(TypedDict):
    template_id: int
    name: str
    category: str
    variable_count: int
    complexity_min: float
    complexity_max: float
    error: str


# ===============================================================================
# TEMPLATE SUBSYSTEM (Integrated with Synthesizer)
# ===============================================================================

class DynamicConflictTemplateSubsystem:
    """
    Template subsystem that integrates with ConflictSynthesizer.
    Generates infinite conflict variations from templates.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Lazy-loaded agents
        self._template_creator = None
        self._variation_generator = None
        self._context_adapter = None
        self._uniqueness_engine = None
        self._hook_generator = None
        
        # Reference to synthesizer
        self.synthesizer = None
        
        # Template cache
        self._template_cache = {}
    
    @property
    def subsystem_type(self):
        """Return the subsystem type"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return SubsystemType.TEMPLATE
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        return {
            'template_creation',
            'variation_generation',
            'context_adaptation',
            'uniqueness_injection',
            'hook_generation',
            'template_evolution'
        }
    
    @property
    def dependencies(self) -> Set:
        """Return other subsystems this depends on"""
        from logic.conflict_system.conflict_synthesizer import SubsystemType
        return {
            SubsystemType.DETECTION,  # Use detection for template selection
            SubsystemType.FLOW  # Templates affect conflict flow
        }
    
    @property
    def event_subscriptions(self) -> Set:
        """Return events this subsystem wants to receive"""
        from logic.conflict_system.conflict_synthesizer import EventType
        return {
            EventType.TEMPLATE_GENERATED,
            EventType.CONFLICT_CREATED,
            EventType.HEALTH_CHECK,
            EventType.STATE_SYNC
        }
    
    async def initialize(self, synthesizer) -> bool:
        """Initialize the subsystem with synthesizer reference"""
        import weakref
        self.synthesizer = weakref.ref(synthesizer)
        
        # Create initial templates if none exist
        try:
            async with get_db_connection_context() as conn:
                template_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM conflict_templates
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                if template_count == 0:
                    logger.info("No templates found, creating initial templates...")
                    # Create base templates for each category
                    for category in list(TemplateCategory)[:3]:  # Start with 3 templates
                        try:
                            await self.create_conflict_template(
                                category,
                                f"Base {category.value} template"
                            )
                            logger.info(f"Created template for {category.value}")
                        except Exception as e:
                            logger.error(f"Failed to create template for {category.value}: {e}")
                            # Continue with other templates
                            continue
        except Exception as e:
            logger.error(f"Error initializing templates: {e}")
            # Continue even if templates can't be created
        
        return True
    
    async def handle_event(self, event) -> Any:
        """Handle an event from the synthesizer"""
        from logic.conflict_system.conflict_synthesizer import SubsystemResponse, SystemEvent, EventType
        
        try:
            if event.event_type == EventType.CONFLICT_CREATED:
                # Check if this conflict needs a template
                conflict_type = event.payload.get('conflict_type')
                if 'template' in conflict_type or event.payload.get('use_template'):
                    # Generate from template
                    template_id = event.payload.get('template_id')
                    if not template_id:
                        # Select appropriate template
                        template_id = await self._select_template(conflict_type)
                    
                    if template_id:
                        generated = await self.generate_conflict_from_template(
                            template_id,
                            event.payload.get('context', {})
                        )
                        
                        # Emit template generated event
                        side_effects = [SystemEvent(
                            event_id=f"template_gen_{event.event_id}",
                            event_type=EventType.TEMPLATE_GENERATED,
                            source_subsystem=self.subsystem_type,
                            payload={
                                'template_id': template_id,
                                'conflict_id': generated.conflict_id,
                                'hooks': generated.narrative_hooks
                            },
                            priority=7
                        )]
                        
                        return SubsystemResponse(
                            subsystem=self.subsystem_type,
                            event_id=event.event_id,
                            success=True,
                            data={
                                'template_used': template_id,
                                'generated_conflict': generated.conflict_id,
                                'narrative_hooks': generated.narrative_hooks
                            },
                            side_effects=side_effects
                        )
                        
            elif event.event_type == EventType.STATE_SYNC:
                # Evolve templates based on usage
                if random.random() < 0.1:  # 10% chance
                    evolved = await self._evolve_random_template()
                    return SubsystemResponse(
                        subsystem=self.subsystem_type,
                        event_id=event.event_id,
                        success=True,
                        data={'template_evolved': evolved}
                    )
                    
            elif event.event_type == EventType.HEALTH_CHECK:
                return SubsystemResponse(
                    subsystem=self.subsystem_type,
                    event_id=event.event_id,
                    success=True,
                    data=await self.health_check()
                )
            
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=True,
                data={}
            )
            
        except Exception as e:
            logger.error(f"Template subsystem error: {e}")
            return SubsystemResponse(
                subsystem=self.subsystem_type,
                event_id=event.event_id,
                success=False,
                data={'error': str(e)}
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        try:
            async with get_db_connection_context() as conn:
                template_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM conflict_templates
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                
                # Check template usage
                used_count = await conn.fetchval("""
                    SELECT COUNT(DISTINCT template_id) FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND template_id IS NOT NULL
                """, self.user_id, self.conversation_id)
            
            return {
                'healthy': template_count > 0,
                'total_templates': template_count,
                'templates_used': used_count,
                'usage_ratio': used_count / template_count if template_count > 0 else 0,
                'issue': 'No templates available' if template_count == 0 else None
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                'healthy': False,
                'issue': str(e)
            }
    
    async def get_conflict_data(self, conflict_id: int) -> Dict[str, Any]:
        """Get template-related data for a specific conflict"""
        try:
            async with get_db_connection_context() as conn:
                conflict = await conn.fetchrow("""
                    SELECT template_id, generation_data FROM Conflicts
                    WHERE conflict_id = $1
                """, conflict_id)
            
            if conflict and conflict['template_id']:
                template = await self._get_template(conflict['template_id'])
                generation_data = json.loads(conflict.get('generation_data', '{}'))
                
                return {
                    'template_used': template.name if template else 'Unknown',
                    'template_category': template.category.value if template else None,
                    'unique_elements': generation_data.get('unique_elements', []),
                    'variation_seed': generation_data.get('seed', 'default')
                }
        except Exception as e:
            logger.error(f"Error getting conflict data: {e}")
        
        return {'template_used': None}
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current state of template system"""
        try:
            async with get_db_connection_context() as conn:
                # Get most used templates
                popular_templates = await conn.fetch("""
                    SELECT template_id, COUNT(*) as usage_count
                    FROM Conflicts
                    WHERE user_id = $1 AND conversation_id = $2
                    AND template_id IS NOT NULL
                    GROUP BY template_id
                    ORDER BY usage_count DESC
                    LIMIT 3
                """, self.user_id, self.conversation_id)
            
            return {
                'popular_templates': [
                    {'id': t['template_id'], 'usage': t['usage_count']}
                    for t in popular_templates
                ],
                'cache_size': len(self._template_cache)
            }
        except Exception as e:
            logger.error(f"Error getting state: {e}")
            return {'error': str(e)}
    
    async def is_relevant_to_scene(self, scene_context: Dict[str, Any]) -> bool:
        """Check if template system is relevant to scene"""
        # Templates are relevant when creating new conflicts
        if scene_context.get('creating_conflict'):
            return True
        
        # Or when scene calls for specific conflict types
        activity = scene_context.get('activity', '')
        if any(keyword in activity.lower() for keyword in ['dispute', 'challenge', 'competition']):
            return True
        
        return False
    
    # ========== Agent Properties ==========
    
    @property
    def template_creator(self) -> Agent:
        if self._template_creator is None:
            self._template_creator = Agent(
                name="Template Creator",
                instructions="""
                Create flexible conflict templates that can generate countless variations.
                
                Design templates that:
                - Have clear core structures
                - Include variable elements
                - Allow contextual adaptation
                - Support complexity scaling
                - Enable emergent storytelling
                
                Templates should be seeds for infinite stories, not rigid patterns.
                
                IMPORTANT: Always respond with valid JSON only, no explanatory text.
                """,
                model="gpt-5-nano",
            )
        return self._template_creator
    
    @property
    def variation_generator(self) -> Agent:
        if self._variation_generator is None:
            self._variation_generator = Agent(
                name="Variation Generator",
                instructions="""
                Generate unique variations from conflict templates.
                
                Create variations that:
                - Feel fresh and original
                - Respect template structure
                - Add surprising elements
                - Fit the context perfectly
                - Create memorable experiences
                
                Each variation should feel like a unique story, not a copy.
                
                IMPORTANT: Always respond with valid JSON only, no explanatory text.
                """,
                model="gpt-5-nano",
            )
        return self._variation_generator
    
    @property
    def context_adapter(self) -> Agent:
        if self._context_adapter is None:
            self._context_adapter = Agent(
                name="Context Adapter",
                instructions="""
                Adapt conflict templates to specific contexts.
                
                Ensure adaptations:
                - Fit the current situation
                - Respect character personalities
                - Match location atmosphere
                - Align with ongoing narratives
                - Feel organic to the world
                
                Make templated conflicts feel bespoke to the moment.
                
                IMPORTANT: Always respond with valid JSON only, no explanatory text.
                """,
                model="gpt-5-nano",
            )
        return self._context_adapter
    
    @property
    def uniqueness_engine(self) -> Agent:
        if self._uniqueness_engine is None:
            self._uniqueness_engine = Agent(
                name="Uniqueness Engine",
                instructions="""
                Ensure each generated conflict feels unique and memorable.
                
                Add elements that:
                - Create distinctive moments
                - Generate quotable lines
                - Produce unexpected twists
                - Build character-specific drama
                - Leave lasting impressions
                
                Every conflict should have something players remember.
                
                IMPORTANT: Always respond with valid JSON only, no explanatory text.
                """,
                model="gpt-5-nano",
            )
        return self._uniqueness_engine
    
    @property
    def hook_generator(self) -> Agent:
        if self._hook_generator is None:
            self._hook_generator = Agent(
                name="Narrative Hook Generator",
                instructions="""
                Generate compelling hooks that draw players into conflicts.
                
                Create hooks that:
                - Grab immediate attention
                - Create emotional investment
                - Promise interesting outcomes
                - Connect to player history
                - Build anticipation
                
                Make players WANT to engage with the conflict.
                
                IMPORTANT: Always respond with valid JSON only, no explanatory text.
                """,
                model="gpt-5-nano",
            )
        return self._hook_generator
    
    # ========== Template Management Methods ==========
    
    async def create_conflict_template(
        self,
        category: TemplateCategory,
        base_concept: str
    ) -> ConflictTemplate:
        """Create a new reusable conflict template"""
        
        prompt = f"""
        Create a flexible conflict template:
        
        Category: {category.value}
        Base Concept: {base_concept}
        
        Design a template that can generate hundreds of unique conflicts.
        
        Return ONLY valid JSON (no other text):
        {{
            "name": "Template name",
            "base_structure": {{
                "core_tension": "Fundamental conflict",
                "stakeholder_roles": ["role types needed"],
                "progression_phases": ["typical phases"],
                "resolution_conditions": ["ways it can end"]
            }},
            "variable_elements": [
                "List of 10+ elements that can change between instances"
            ],
            "contextual_modifiers": {{
                "personality_axes": ["relevant personality traits"],
                "environmental_factors": ["location/setting influences"],
                "cultural_variables": ["social/cultural elements"],
                "power_modifiers": ["hierarchy/authority factors"]
            }},
            "generation_rules": {{
                "required_elements": ["must-have components"],
                "optional_elements": ["can-have components"],
                "exclusions": ["incompatible elements"]
            }},
            "complexity_range": {{
                "minimum": 0.2,
                "maximum": 0.9
            }}
        }}
        """
        
        try:
            # Run the agent with better error handling
            logger.debug(f"Creating template for category: {category.value}")
            response = await Runner.run(self.template_creator, prompt)
            response_text = extract_runner_response(response)
            
            if not response_text:
                logger.error("Empty response from template creator")
                # Provide a fallback template
                response_text = self._get_fallback_template(category, base_concept)
            
            # Clean the response text (remove any markdown or code blocks)
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON with better error handling
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.debug(f"Response text: {response_text[:500]}...")
                # Use fallback template
                data = json.loads(self._get_fallback_template(category, base_concept))
            
            # Validate required fields
            if not all(key in data for key in ['name', 'base_structure', 'variable_elements', 'contextual_modifiers', 'complexity_range']):
                logger.warning("Missing required fields in template data, using defaults")
                data = self._ensure_template_fields(data, category, base_concept)
            
            # Store template in database
            async with get_db_connection_context() as conn:
                template_id = await conn.fetchval("""
                    INSERT INTO conflict_templates
                    (user_id, conversation_id, category, name, base_structure, 
                     variable_elements, contextual_modifiers, complexity_min, complexity_max)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING template_id
                """, self.user_id, self.conversation_id, category.value, data['name'],
                json.dumps(data['base_structure']),
                json.dumps(data['variable_elements']),
                json.dumps(data['contextual_modifiers']),
                data['complexity_range'].get('minimum', 0.2),
                data['complexity_range'].get('maximum', 0.9))
            
            template = ConflictTemplate(
                template_id=template_id,
                category=category,
                name=data['name'],
                base_structure=data['base_structure'],
                variable_elements=data['variable_elements'],
                contextual_modifiers=data['contextual_modifiers'],
                complexity_range=(
                    data['complexity_range'].get('minimum', 0.2),
                    data['complexity_range'].get('maximum', 0.9)
                )
            )
            
            # Cache template
            self._template_cache[template_id] = template
            logger.info(f"Successfully created template: {template.name}")
            
            return template
            
        except Exception as e:
            logger.error(f"Error creating template: {e}", exc_info=True)
            raise
    
    def _get_fallback_template(self, category: TemplateCategory, base_concept: str) -> str:
        """Provide a fallback template when LLM fails"""
        fallback = {
            "name": f"{category.value.replace('_', ' ').title()} Template",
            "base_structure": {
                "core_tension": f"A {category.value} conflict",
                "stakeholder_roles": ["protagonist", "antagonist", "mediator"],
                "progression_phases": ["setup", "escalation", "climax", "resolution"],
                "resolution_conditions": ["victory", "compromise", "defeat", "stalemate"]
            },
            "variable_elements": [
                "Setting location",
                "Time of day",
                "Number of participants",
                "Stakes level",
                "Public vs private",
                "Emotional intensity",
                "Physical vs verbal",
                "Resource type",
                "Authority involvement",
                "Witness presence"
            ],
            "contextual_modifiers": {
                "personality_axes": ["aggressive-passive", "cooperative-competitive"],
                "environmental_factors": ["crowded-isolated", "formal-casual"],
                "cultural_variables": ["traditional-modern", "hierarchical-egalitarian"],
                "power_modifiers": ["equal-unequal", "official-unofficial"]
            },
            "generation_rules": {
                "required_elements": ["core_tension", "stakeholders"],
                "optional_elements": ["witnesses", "mediators"],
                "exclusions": ["violence", "illegal_activity"]
            },
            "complexity_range": {
                "minimum": 0.2,
                "maximum": 0.9
            }
        }
        return json.dumps(fallback)
    
    def _ensure_template_fields(self, data: Dict[str, Any], category: TemplateCategory, base_concept: str) -> Dict[str, Any]:
        """Ensure all required template fields exist"""
        defaults = json.loads(self._get_fallback_template(category, base_concept))
        
        for key, value in defaults.items():
            if key not in data:
                data[key] = value
        
        return data
    
    async def generate_conflict_from_template(
        self,
        template_id: int,
        context: Dict[str, Any]
    ) -> GeneratedConflict:
        """Generate a unique conflict from a template"""
        
        try:
            # Get template
            template = await self._get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Generate variation
            variation = await self._generate_variation(template, context)
            
            # Adapt to context
            adapted = await self._adapt_to_context(variation, context)
            
            # Add unique elements
            unique = await self._add_unique_elements(adapted, context)
            
            # Generate hooks
            hooks = await self._generate_narrative_hooks(unique, context)
            
            # Create the conflict through synthesizer
            conflict_id = await self._create_conflict_from_generation(
                template,
                unique,
                hooks
            )
            
            return GeneratedConflict(
                conflict_id=conflict_id,
                template_id=template_id,
                variation_seed=unique.get('seed', 'default'),
                customization=unique.get('customization', {}),
                narrative_hooks=hooks,
                unique_elements=unique.get('unique_elements', [])
            )
            
        except Exception as e:
            logger.error(f"Error generating from template: {e}")
            raise
    
    async def _generate_variation(
        self,
        template: ConflictTemplate,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a variation from template with better error handling"""
        
        prompt = f"""
        Generate a unique variation from this template:
        
        Template: {template.name}
        Base Structure: {json.dumps(template.base_structure)}
        Variable Elements: {json.dumps(template.variable_elements)}
        Context: {json.dumps(context)}
        
        Create a variation that uses the base structure and varies 3-5 variable elements.
        
        Return ONLY valid JSON:
        {{
            "seed": "Unique identifier for this variation",
            "core_tension": "Specific tension for this instance",
            "stakeholder_configuration": {{
                "roles": ["specific roles"],
                "relationships": ["specific relationships"]
            }},
            "chosen_variables": {{
                "variable_name": "specific value"
            }},
            "progression_path": ["specific phases"],
            "resolution_options": ["specific endings"],
            "twist_potential": "Unexpected element"
        }}
        """
        
        try:
            response = await Runner.run(self.variation_generator, prompt)
            response_text = extract_runner_response(response)
            
            # Clean and parse response
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            if not response_text:
                # Return a default variation
                return self._get_default_variation(template)
            
            return json.loads(response_text)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error generating variation: {e}")
            return self._get_default_variation(template)
    
    def _get_default_variation(self, template: ConflictTemplate) -> Dict[str, Any]:
        """Provide a default variation when generation fails"""
        return {
            "seed": f"variation_{datetime.now().timestamp()}",
            "core_tension": template.base_structure.get('core_tension', 'A conflict emerges'),
            "stakeholder_configuration": {
                "roles": template.base_structure.get('stakeholder_roles', ['participant']),
                "relationships": ["neutral"]
            },
            "chosen_variables": {
                var: "default" for var in template.variable_elements[:3]
            },
            "progression_path": template.base_structure.get('progression_phases', ['start', 'middle', 'end']),
            "resolution_options": template.base_structure.get('resolution_conditions', ['resolved']),
            "twist_potential": "An unexpected turn of events"
        }
    
    async def _adapt_to_context(
        self,
        variation: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adapt variation to specific context with error handling"""
        
        prompt = f"""
        Adapt this conflict variation to the specific context:
        
        Variation: {json.dumps(variation)}
        Current Location: {context.get('location', 'unknown')}
        Present NPCs: {json.dumps(context.get('npcs', []))}
        Time of Day: {context.get('time', 'unknown')}
        Recent Events: {json.dumps(context.get('recent_events', []))}
        
        Return ONLY valid JSON:
        {{
            "location_integration": "How location shapes conflict",
            "npc_motivations": {{}},
            "temporal_factors": "How timing affects it",
            "continuity_connections": ["links to recent events"],
            "environmental_obstacles": ["location-specific challenges"],
            "atmospheric_elements": ["mood and tone elements"]
        }}
        """
        
        try:
            response = await Runner.run(self.context_adapter, prompt)
            response_text = extract_runner_response(response)
            
            # Clean and parse
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            if not response_text:
                adaptation = self._get_default_adaptation(context)
            else:
                adaptation = json.loads(response_text)
            
            adapted = variation.copy()
            adapted['context_adaptation'] = adaptation
            return adapted
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error adapting to context: {e}")
            adapted = variation.copy()
            adapted['context_adaptation'] = self._get_default_adaptation(context)
            return adapted
    
    def _get_default_adaptation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide default adaptation when generation fails"""
        return {
            "location_integration": f"Takes place in {context.get('location', 'the current location')}",
            "npc_motivations": {},
            "temporal_factors": "Happens at an opportune moment",
            "continuity_connections": [],
            "environmental_obstacles": [],
            "atmospheric_elements": ["tense", "uncertain"]
        }
    
    async def _add_unique_elements(
        self,
        adapted: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add unique memorable elements with error handling"""
        
        prompt = f"""
        Add unique elements to make this conflict memorable:
        
        Conflict: {json.dumps(adapted)}
        Player History: {json.dumps(context.get('player_history', []))}
        
        Return ONLY valid JSON:
        {{
            "unique_elements": [
                "List of 3-5 unique elements"
            ],
            "memorable_quote": "Something an NPC might say",
            "signature_moment": "A scene players will remember",
            "sensory_detail": "Something visceral",
            "conversation_piece": "What players will discuss later"
        }}
        """
        
        try:
            response = await Runner.run(self.uniqueness_engine, prompt)
            response_text = extract_runner_response(response)
            
            # Clean and parse
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            if not response_text:
                unique_data = self._get_default_unique_elements()
            else:
                unique_data = json.loads(response_text)
            
            adapted['unique_elements'] = unique_data.get('unique_elements', [])
            adapted['signature_content'] = unique_data
            adapted['customization'] = {
                'base_variation': adapted.get('seed', 'unknown'),
                'context_layer': adapted.get('context_adaptation', {}),
                'unique_layer': unique_data
            }
            
            return adapted
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error adding unique elements: {e}")
            unique_data = self._get_default_unique_elements()
            adapted['unique_elements'] = unique_data['unique_elements']
            adapted['signature_content'] = unique_data
            return adapted
    
    def _get_default_unique_elements(self) -> Dict[str, Any]:
        """Provide default unique elements when generation fails"""
        return {
            "unique_elements": [
                "An unexpected alliance forms",
                "A hidden truth is revealed",
                "The stakes suddenly increase"
            ],
            "memorable_quote": "This changes everything.",
            "signature_moment": "A dramatic confrontation",
            "sensory_detail": "The tension is palpable",
            "conversation_piece": "The unexpected twist"
        }
    
    async def _generate_narrative_hooks(
        self,
        conflict_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate compelling narrative hooks with error handling"""
        
        prompt = f"""
        Generate narrative hooks for this conflict:
        
        Core Tension: {conflict_data.get('core_tension', '')}
        Unique Elements: {json.dumps(conflict_data.get('unique_elements', []))}
        Signature Moment: {conflict_data.get('signature_content', {}).get('signature_moment', '')}
        
        Create 3-5 hooks that grab attention and create investment.
        
        Return ONLY valid JSON:
        {{
            "hooks": [
                "List of compelling one-sentence hooks"
            ]
        }}
        """
        
        try:
            response = await Runner.run(self.hook_generator, prompt)
            response_text = extract_runner_response(response)
            
            # Clean and parse
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            if not response_text:
                return self._get_default_hooks(conflict_data)
            
            data = json.loads(response_text)
            return data.get('hooks', self._get_default_hooks(conflict_data))
            
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error generating hooks: {e}")
            return self._get_default_hooks(conflict_data)
    
    def _get_default_hooks(self, conflict_data: Dict[str, Any]) -> List[str]:
        """Provide default hooks when generation fails"""
        return [
            "A new challenge emerges that tests everyone involved.",
            "Old tensions resurface in unexpected ways.",
            "What starts small quickly escalates beyond control."
        ]
    
    async def _create_conflict_from_generation(
        self,
        template: ConflictTemplate,
        generation_data: Dict[str, Any],
        hooks: List[str]
    ) -> int:
        """Create actual conflict from generated data"""
        
        # Calculate complexity
        complexity = random.uniform(
            template.complexity_range[0],
            template.complexity_range[1]
        )
        
        # Create conflict through synthesizer if available
        if self.synthesizer:
            synth = self.synthesizer()
            if synth:
                try:
                    result = await synth.create_conflict(
                        template.category.value,
                        {
                            'template_id': template.template_id,
                            'generation_data': generation_data,
                            'hooks': hooks,
                            'complexity': complexity
                        }
                    )
                    return result.get('conflict_id', 0)
                except Exception as e:
                    logger.error(f"Error creating conflict through synthesizer: {e}")
        
        # Fallback: create directly
        try:
            async with get_db_connection_context() as conn:
                conflict_id = await conn.fetchval("""
                    INSERT INTO Conflicts
                    (user_id, conversation_id, conflict_type, conflict_name,
                     description, intensity, phase, is_active, progress,
                     template_id, generation_data)
                    VALUES ($1, $2, $3, $4, $5, $6, 'emerging', true, 0, $7, $8)
                    RETURNING conflict_id
                """, self.user_id, self.conversation_id,
                template.category.value,
                generation_data.get('seed', 'Generated Conflict'),
                hooks[0] if hooks else 'A new tension emerges',
                self._calculate_intensity(complexity),
                template.template_id,
                json.dumps(generation_data))
            
            return conflict_id
        except Exception as e:
            logger.error(f"Error creating conflict in database: {e}")
            return 0
    
    def _calculate_intensity(self, complexity: float) -> str:
        """Calculate intensity from complexity"""
        if complexity < 0.3:
            return "subtle"
        elif complexity < 0.5:
            return "tension"
        elif complexity < 0.7:
            return "friction"
        elif complexity < 0.9:
            return "opposition"
        else:
            return "confrontation"
    
    async def _get_template(self, template_id: int) -> Optional[ConflictTemplate]:
        """Get template from cache or database"""
        
        if template_id in self._template_cache:
            return self._template_cache[template_id]
        
        try:
            async with get_db_connection_context() as conn:
                template_data = await conn.fetchrow("""
                    SELECT * FROM conflict_templates WHERE template_id = $1
                """, template_id)
            
            if template_data:
                template = ConflictTemplate(
                    template_id=template_id,
                    category=TemplateCategory(template_data['category']),
                    name=template_data['name'],
                    base_structure=json.loads(template_data['base_structure']),
                    variable_elements=json.loads(template_data['variable_elements']),
                    contextual_modifiers=json.loads(template_data['contextual_modifiers']),
                    complexity_range=(
                        template_data['complexity_min'],
                        template_data['complexity_max']
                    )
                )
                self._template_cache[template_id] = template
                return template
        except Exception as e:
            logger.error(f"Error getting template: {e}")
        
        return None
    
    async def _select_template(self, conflict_type: str) -> Optional[int]:
        """Select appropriate template for conflict type"""
        
        try:
            async with get_db_connection_context() as conn:
                template = await conn.fetchrow("""
                    SELECT template_id FROM conflict_templates
                    WHERE user_id = $1 AND conversation_id = $2
                    AND category LIKE $3
                    ORDER BY RANDOM()
                    LIMIT 1
                """, self.user_id, self.conversation_id, f"%{conflict_type}%")
            
            return template['template_id'] if template else None
        except Exception as e:
            logger.error(f"Error selecting template: {e}")
            return None
    
    async def _evolve_random_template(self) -> bool:
        """Evolve a random template based on usage"""
        
        try:
            async with get_db_connection_context() as conn:
                # Get a template with usage
                template = await conn.fetchrow("""
                    SELECT t.*, COUNT(c.conflict_id) as usage_count
                    FROM conflict_templates t
                    LEFT JOIN Conflicts c ON t.template_id = c.template_id
                    WHERE t.user_id = $1 AND t.conversation_id = $2
                    GROUP BY t.template_id
                    HAVING COUNT(c.conflict_id) > 0
                    ORDER BY RANDOM()
                    LIMIT 1
                """, self.user_id, self.conversation_id)
            
            if template:
                # Simple evolution: adjust complexity range based on success
                new_min = max(0.1, template['complexity_min'] - 0.05)
                new_max = min(1.0, template['complexity_max'] + 0.05)
                
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE conflict_templates
                        SET complexity_min = $1, complexity_max = $2,
                            last_evolved = CURRENT_TIMESTAMP
                        WHERE template_id = $3
                    """, new_min, new_max, template['template_id'])
                
                return True
        except Exception as e:
            logger.error(f"Error evolving template: {e}")
        
        return False


# ===============================================================================
# PUBLIC API FUNCTIONS
# ===============================================================================

@function_tool
async def generate_templated_conflict(
    ctx: RunContextWrapper,
    category: str,
    context: TemplateContextDTO,
) -> GenerateTemplatedConflictResponse:
    """Generate a conflict from a template category through synthesizer"""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    try:
        from logic.conflict_system.conflict_synthesizer import get_synthesizer
        synthesizer = await get_synthesizer(user_id, conversation_id)

        # Flatten: pass a single context dict (no nested "context" field)
        merged_context: Dict = {
            'use_template': True,
            'template_category': category,
            **dict(context or {})
        }

        result = await synthesizer.create_conflict(
            f"template_{category}",
            merged_context,
        )

        # Coerce into strict response
        return {
            'conflict_id': int(result.get('conflict_id', 0) or 0),
            'status': str(result.get('status', 'created')),
            'conflict_type': str(result.get('conflict_type', f"template_{category}")),
            'template_used': int(result.get('template_used', 0) or 0),
            'narrative_hooks': [str(h) for h in (result.get('narrative_hooks') or [])],
            'message': str(result.get('message', "")),
            'error': "",
        }
    except Exception as e:
        logger.error(f"Error generating templated conflict: {e}")
        return {
            'conflict_id': 0,
            'status': 'error',
            'conflict_type': f"template_{category}",
            'template_used': 0,
            'narrative_hooks': [],
            'message': '',
            'error': str(e)
        }


@function_tool
async def create_custom_template(
    ctx: RunContextWrapper,
    category: str,
    concept: str,
) -> CreateTemplateResponse:
    """Create a custom conflict template via the TEMPLATE subsystem"""

    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    try:
        from logic.conflict_system.conflict_synthesizer import (
            get_synthesizer, SystemEvent, EventType, SubsystemType
        )
        synthesizer = await get_synthesizer(user_id, conversation_id)

        # Ask TEMPLATE subsystem to create a template (no direct _subsystems access)
        evt = SystemEvent(
            event_id=f"create_template_{category}_{datetime.now().timestamp()}",
            event_type=EventType.TEMPLATE_GENERATED,
            source_subsystem=SubsystemType.TEMPLATE,
            payload={'request': 'create_template', 'category': category, 'concept': concept},
            target_subsystems={SubsystemType.TEMPLATE},
            requires_response=True,
            priority=3,
        )

        template_id = 0
        name = ""
        cat = category
        variable_count = 0
        comp_min = 0.0
        comp_max = 0.0
        error = "Template subsystem did not respond"

        responses = await synthesizer.emit_event(evt)
        if responses:
            for r in responses:
                if r.subsystem == SubsystemType.TEMPLATE:
                    data = r.data or {}
                    t = data.get('template') or data  # allow either shape
                    template_id = int(t.get('template_id', 0) or 0)
                    name = str(t.get('name', "") or "")
                    cat = str(t.get('category', category) or category)
                    # variable elements could be a list or count
                    ve = t.get('variable_elements')
                    if isinstance(ve, list):
                        variable_count = len(ve)
                    else:
                        variable_count = int(t.get('variable_count', 0) or 0)
                    # complexity range could be pair or dict
                    cr = t.get('complexity_range') or {}
                    if isinstance(cr, (list, tuple)) and len(cr) == 2:
                        comp_min = float(cr[0] or 0.0)
                        comp_max = float(cr[1] or 0.0)
                    else:
                        comp_min = float(cr.get('min', 0.0) if isinstance(cr, dict) else 0.0)
                        comp_max = float(cr.get('max', 0.0) if isinstance(cr, dict) else 0.0)
                    error = ""

        return {
            'template_id': template_id,
            'name': name,
            'category': cat,
            'variable_count': variable_count,
            'complexity_min': comp_min,
            'complexity_max': comp_max,
            'error': error,
        }
    except Exception as e:
        logger.error(f"Error creating custom template: {e}")
        return {
            'template_id': 0,
            'name': '',
            'category': category,
            'variable_count': 0,
            'complexity_min': 0.0,
            'complexity_max': 0.0,
            'error': str(e)
        }
