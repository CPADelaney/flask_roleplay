"""Compatibility layer for legacy ConflictSystemIntegration imports.

This module provides a thin wrapper around :class:`ConflictSystemInterface`
so that existing call sites (Celery tasks, NPC systems, etc.) can continue to
use the familiar ``ConflictSystemIntegration`` name while the new
``ConflictSystemInterface`` hosts the real implementation.

Only the minimal lifecycle helpers that older code expects (``get_instance``,
``initialize`` and ``generate_conflict``) are implemented explicitly.  All
other attribute access is proxied to the underlying interface so consumers can
progressively migrate to the new API surface without breaking imports.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
from weakref import WeakValueDictionary

from agents.run_context import RunContextWrapper

from logic.conflict_system.integration import ConflictSystemInterface, IntegrationMode

logger = logging.getLogger(__name__)


class ConflictSystemIntegration:
    """Backwards-compatible wrapper around :class:`ConflictSystemInterface`."""

    _instances: "WeakValueDictionary[Tuple[int, int], ConflictSystemIntegration]" = WeakValueDictionary()
    _instance_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, user_id: int, conversation_id: int, mode: IntegrationMode | str = IntegrationMode.EMERGENT):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.mode = self._normalize_mode(mode)
        self._interface = ConflictSystemInterface(user_id, conversation_id)
        self._synthesizer = None  # cached synthesizer instance once resolved

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    @classmethod
    async def get_instance(cls, user_id: int, conversation_id: int) -> "ConflictSystemIntegration":
        """Return (and cache) a wrapper for the given conversation."""

        key = (user_id, conversation_id)
        async with cls._instance_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = cls(user_id, conversation_id)
                cls._instances[key] = instance
        return instance

    async def initialize(self, mode: IntegrationMode | str | None = None) -> Dict[str, Any]:
        """Initialize the underlying conflict system."""

        if mode is not None:
            self.mode = self._normalize_mode(mode)
        return await self._interface.initialize_system(self.mode)

    @staticmethod
    def _normalize_intensity(intensity: Any) -> float:
        """Convert intensity to numeric value for database storage."""
        if isinstance(intensity, (int, float)):
            return float(intensity)
        
        # Map string values to numeric scale (0.0 - 1.0)
        intensity_map = {
            'low': 0.3,
            'subtle': 0.2,
            'medium': 0.5,
            'moderate': 0.5,
            'high': 0.8,
            'critical': 0.9,
            'extreme': 1.0,
        }
        
        intensity_str = str(intensity).lower().strip()
        return intensity_map.get(intensity_str, 0.5)  # default to medium

    async def generate_conflict(
        self,
        ctx: Optional[RunContextWrapper],
        conflict_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate a conflict via the synthesizer, matching legacy semantics."""
        conflict_params = conflict_params or {}
        conflict_type = conflict_params.get("conflict_type", "standard")
        synthesizer = await self._get_synthesizer()
        try:
            await synthesizer.initialize_all_subsystems()
        except Exception:  # pragma: no cover - synthesizer handles idempotency
            logger.debug("Synthesizer subsystem initialization failed", exc_info=True)
        
        context_payload: Dict[str, Any] = {}
        if ctx is not None:
            if hasattr(ctx, "context") and isinstance(ctx.context, dict):
                context_payload.update(ctx.context)
            if hasattr(ctx, "data") and isinstance(ctx.data, dict):
                context_payload.update(ctx.data)
        
        context_payload.setdefault("user_id", self.user_id)
        context_payload.setdefault("conversation_id", self.conversation_id)
        
        # Merge additional parameters (intensity, player involvement, etc.)
        extra_context = {k: v for k, v in conflict_params.items() if k != "conflict_type"}
        context_payload.update(extra_context)
        
        result = await self._interface.create_conflict(conflict_type, context_payload)
        
        if not isinstance(result, dict):
            return None
        
        status = str(result.get("status", "")).lower()
        success = status not in {"failed", "error"}
        
        # Check if we need to create the database record
        conflict_id = result.get("conflict_id")
        conflict_name = result.get("conflict_name")
        
        if success and conflict_id is None:
            # Generate a meaningful conflict name
            if not conflict_name:
                conflict_name = await self._generate_conflict_name(conflict_type, context_payload)
            
            # Create the database record
            try:
                from db.connection import get_db_connection_context
                
                async with get_db_connection_context() as conn:
                    conflict_id = await conn.fetchval("""
                        INSERT INTO conflicts (
                            user_id, conversation_id,
                            conflict_name, conflict_type,
                            description, status, 
                            intensity, player_involvement,
                            created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        RETURNING conflict_id
                    """,
                        self.user_id,
                        self.conversation_id,
                        conflict_name,
                        conflict_type,
                        conflict_params.get('description', f'A {conflict_type} conflict has emerged'),
                        'active',
                        self._normalize_intensity(conflict_params.get('intensity', 'medium')),  # Convert to float
                        conflict_params.get('player_involvement', 'indirect')
                    )
                    
                    # Update result with the new conflict_id and name
                    result['conflict_id'] = conflict_id
                    result['conflict_name'] = conflict_name
                    
                    logger.info(f"Created conflict record {conflict_id}: {conflict_name}")
                    
            except Exception as e:
                logger.error(f"Error creating conflict database record: {e}", exc_info=True)
                return {
                    "success": False,
                    "conflict_id": None,
                    "conflict_type": conflict_type,
                    "message": f"Failed to create conflict record: {e}",
                    "conflict_details": None,
                    "raw_result": result,
                }
        
        return {
            "success": success,
            "conflict_id": conflict_id,
            "conflict_type": result.get("conflict_type", conflict_type),
            "conflict_name": conflict_name,
            "message": result.get("message", ""),
            "conflict_details": result.get("conflict_details"),
            "raw_result": result,
        }
    
    async def _generate_conflict_name(self, conflict_type: str, context: Dict[str, Any]) -> str:
        """Generate a unique conflict name using LLM based on type and context."""
        from logic.chatgpt_integration import EmptyLLMOutputError, generate_text_completion
        
        # Build context information for the LLM
        context_details = []
        if intensity := context.get('intensity'):
            context_details.append(f"intensity: {intensity}")
        if involvement := context.get('player_involvement'):
            context_details.append(f"player involvement: {involvement}")
        if description := context.get('description'):
            context_details.append(f"description: {description}")
        if participants := context.get('participants'):
            context_details.append(f"participants: {len(participants)} NPCs")
        
        context_str = ", ".join(context_details) if context_details else "standard conflict"
        
        system_prompt = """You are a creative narrative designer crafting evocative conflict names for a story-driven game.
    Your names should be:
    - Dramatic and memorable
    - 2-4 words maximum
    - Evocative of the conflict's nature
    - Appropriate for the intensity level
    - Unique and creative (avoid generic phrases like "Rising Tensions" or "Power Struggle")
    
    Return ONLY the conflict name, nothing else. No explanations, no quotes, no extra text."""
    
        user_prompt = f"""Create a compelling conflict name for this situation:
    
    Conflict Type: {conflict_type}
    Context: {context_str}
    
    Examples of good conflict names (for inspiration, don't copy):
    - The Velvet Ultimatum
    - Thorns of Devotion
    - Shattered Sanctuary
    - Whispers of Betrayal
    - Eclipse of Trust
    
    IMPORTANT: Respond with ONLY the conflict name (2-4 words). No quotes, no punctuation, no explanation.
    
    Your conflict name:"""
    
        try:
            # Generate name via LLM
            response = await generate_text_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=75,  # Increased from 20
                task_type="decision",
                model="gpt-5-nano"
            )
            
            # **DETAILED LOGGING**
            logger.info(f"LLM conflict name request - type: {conflict_type}, context: {context_str}")
            logger.info(f"Raw LLM response: {repr(response)}")
            logger.info(f"Response type: {type(response)}, is None: {response is None}, is empty: {not response if response else 'N/A'}")
            
            if getattr(response, "is_refusal", False):
                logger.warning("LLM refused to generate conflict name; using fallback template")
                return self._generate_conflict_name_fallback(conflict_type, context)

            if response:
                # Clean up the response more aggressively
                conflict_name = response.strip()
                
                # Remove common wrapper patterns
                conflict_name = conflict_name.strip('"').strip("'").strip('`')
                conflict_name = conflict_name.replace('**', '').replace('*', '')  # Remove markdown bold
                
                # If it contains multiple lines, take the first non-empty line
                lines = [line.strip() for line in conflict_name.split('\n') if line.strip()]
                if lines:
                    conflict_name = lines[0]
                
                # Remove any "Conflict name:" or similar prefixes
                for prefix in ['conflict name:', 'name:', 'title:']:
                    if conflict_name.lower().startswith(prefix):
                        conflict_name = conflict_name[len(prefix):].strip()
                
                logger.info(f"Cleaned conflict name: {repr(conflict_name)}")
                logger.info(f"Length: {len(conflict_name)}, Words: {len(conflict_name.split())}")
                
                # Validate - be more lenient
                word_count = len(conflict_name.split())
                if 1 <= word_count <= 6 and 3 <= len(conflict_name) <= 60:
                    # Apply intensity modifier if needed
                    intensity = context.get('intensity', '')
                    
                    # Only add modifier if name is short enough
                    if intensity == 'high' and word_count <= 3 and not any(word in conflict_name.lower() for word in ['critical', 'final', 'ultimate', 'breaking']):
                        conflict_name = f"Critical {conflict_name}"
                    elif intensity == 'low' and word_count <= 3 and not any(word in conflict_name.lower() for word in ['subtle', 'quiet', 'whispered', 'faint']):
                        conflict_name = f"Subtle {conflict_name}"
                    
                    logger.info(f"✓ Generated conflict name via LLM: {conflict_name}")
                    return conflict_name
                else:
                    logger.warning(f"✗ LLM conflict name failed validation - words: {word_count} (need 1-6), length: {len(conflict_name)} (need 3-60)")
            else:
                logger.warning("✗ LLM returned None or empty response for conflict name")
            
        except EmptyLLMOutputError as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            if diagnostics:
                logger.warning("✗ LLM returned empty response for conflict name; diagnostics: %s", diagnostics)
            else:
                logger.warning("✗ LLM returned empty response for conflict name; no diagnostics provided")
            logger.warning("Using fallback template for conflict name")
            return self._generate_conflict_name_fallback(conflict_type, context)
        except Exception as e:
            logger.error(f"✗ Exception generating conflict name via LLM: {e}", exc_info=True)

        # Fallback to template-based generation
        logger.warning("Using fallback template for conflict name")
        return self._generate_conflict_name_fallback(conflict_type, context)
    
    
    def _generate_conflict_name_fallback(self, conflict_type: str, context: Dict[str, Any]) -> str:
        """Fallback method using templates if LLM generation fails."""
        import random
        
        templates = {
            "major": [
                "Power Struggle",
                "Rising Tensions", 
                "Clash of Wills",
                "Inevitable Confrontation",
                "Critical Impasse"
            ],
            "minor": [
                "Minor Dispute",
                "Small Disagreement",
                "Tension Point",
                "Brief Friction"
            ],
            "social": [
                "Social Friction",
                "Relationship Strain",
                "Interpersonal Conflict",
                "Social Divide"
            ],
            "power_dynamics": [
                "Dominance Challenge",
                "Authority Questioned",
                "Control Contest",
                "Hierarchy Shift"
            ],
            "standard": [
                "Emerging Conflict",
                "Brewing Storm",
                "Tension Rising"
            ]
        }
        
        options = templates.get(conflict_type, templates["standard"])
        base_name = random.choice(options)
        
        # Add context-specific details if available
        intensity = context.get('intensity', '')
        if intensity == 'high':
            base_name = f"Critical {base_name}"
        elif intensity == 'low':
            base_name = f"Subtle {base_name}"
        
        return base_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _get_synthesizer(self):
        if self._synthesizer is None:
            self._synthesizer = await self._interface._get_synthesizer()
        return self._synthesizer

    @staticmethod
    def _normalize_mode(mode: IntegrationMode | str) -> IntegrationMode:
        if isinstance(mode, IntegrationMode):
            return mode
        try:
            return IntegrationMode(mode)
        except ValueError:
            try:
                return IntegrationMode[mode.upper()]
            except Exception:
                logger.warning("Unknown conflict integration mode '%s', defaulting to EMERGENT", mode)
                return IntegrationMode.EMERGENT

    # ------------------------------------------------------------------
    # Attribute proxying
    # ------------------------------------------------------------------
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - simple delegation
        if hasattr(self._interface, item):
            return getattr(self._interface, item)
        raise AttributeError(item)


__all__ = ["ConflictSystemIntegration"]
