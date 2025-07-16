# npcs/dynamic_templates.py
"""
Dynamic, environment‑aware template helpers for NPC generation.

This file centralises **all** the little hard‑coded tables and fallback text that used to live in
*npcs/new_npc_creation.py*.
Each helper below makes a **single GPT call** (cached) so data is:
  • **tailored to the current campaign setting** (environment description, culture, genre,…)
  • lightweight – only queried the first time it's needed thanks to async-aware caching
  • resilient – each has a static fallback in case GPT is unavailable.

────────────────────────────────────────────────────────────────────────────
What changed in this revision
────────────────────────────────────────────────────────────────────────────
• **Fixed async caching** - Replaced functools.lru_cache with async_result_cache that properly caches results, not coroutines
• **Enhanced cache robustness** - Using hash-based keys, hit/miss stats, and defensive programming
• **Prompt injection hardening** - All user-provided text is wrapped in triple quotes via qblock helper
• **Added missing imports** - Added Optional to imports
• **Mask‑slippage triggers** now return *criteria* (no memory text). Each row gives a cue the **player
  might notice** once the relevant stat crosses its threshold.
• **Relationship ladders** include *progression* **and** *regression* blurbs, acknowledging that bonds
  can strengthen or weaken over time.
• Added `get_calendar_day_names` for custom day names per‑setting.
• Helpers for beliefs, semantic topics, flashback triggers unchanged but kept for context.

────────────────────────────────────────────────────────────────────────────
Cache Inspection
────────────────────────────────────────────────────────────────────────────
Each cached function exposes cache management:
  • function.cache_info() - Returns cache statistics (hits, misses, size, maxsize)
  • function.cache_clear() - Clears the cache and resets statistics

Example:
  get_mask_slippage_triggers.cache_info()  # "CacheInfo(hits=5, misses=2, size=2, maxsize=64)"
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
from typing import Any, Dict, List, TypedDict, Optional
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────────────────

def qblock(txt: str) -> str:
    """Safely wrap text in triple quotes, escaping any embedded triple quotes."""
    return '"""\n' + (txt or "").replace('"""', r'\"\"\"') + '\n"""'

# ───────────────────────────────────────────────────────────────────────────
# Async-aware caching decorator
# ───────────────────────────────────────────────────────────────────────────

def async_result_cache(maxsize: int = 128):
    """
    Decorator that properly caches async function results (not coroutines).
    Uses a simple LRU eviction strategy with hit/miss statistics.
    """
    def decorator(fn):
        cache: dict[str, Any] = {}
        order: list[str] = []
        hits = misses = 0
        lock = asyncio.Lock()

        def _make_key(args, kwargs) -> str:
            """Create a hashable key from args and kwargs."""
            try:
                payload = {"args": args, "kwargs": kwargs}
                blob = json.dumps(payload, sort_keys=True, default=str)
            except Exception:
                blob = repr((args, kwargs))
            return hashlib.sha1(blob.encode("utf-8")).hexdigest()

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            nonlocal hits, misses
            key = _make_key(args, kwargs)
            async with lock:
                if key in cache:
                    hits += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Cache HIT %s.%s key=%s", fn.__module__, fn.__name__, key[:8])
                    try:
                        order.remove(key)
                    except ValueError:
                        pass
                    order.append(key)
                    return cache[key]
                # cache miss
                misses += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Cache MISS %s.%s key=%s", fn.__module__, fn.__name__, key[:8])
            result = await fn(*args, **kwargs)
            async with lock:
                cache[key] = result
                order.append(key)
                if len(order) > maxsize:
                    oldest = order.pop(0)
                    cache.pop(oldest, None)
            return result

        def _cache_clear():
            nonlocal hits, misses
            hits = misses = 0
            cache.clear()
            order.clear()

        wrapper.cache_clear = _cache_clear
        wrapper.cache_info = lambda: f"CacheInfo(hits={hits}, misses={misses}, size={len(cache)}, maxsize={maxsize})"
        return wrapper
    return decorator
        
# ───────────────────────────────────────────────────────────────────────────
# GPT wrapper
# ───────────────────────────────────────────────────────────────────────────

_client: AsyncOpenAI | None = None

def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client


def _strip_json_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if present.
    Returns stripped text.
    """
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```json") and t.endswith("```"):
        return t[7:-3].strip()
    if t.startswith("```") and t.endswith("```"):
        return t[3:-3].strip()
    return t


async def _gpt_json(
    system: str,
    user: str,
    *,
    model: str = "gpt-4.1-nano",
    max_output_tokens: int = 640,
) -> Any:
    """
    Call the OpenAI Responses API and return parsed JSON.

    We *hint* JSON format via instructions; the Responses API currently
    has no `response_format=` kwarg (unlike legacy chat.completions),
    so we enforce in prompt + parse defensively.

    Retries up to 3x on error (network / parse / API). Raises RuntimeError if all fail.
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OpenAI client unavailable – falling back")

    # Strengthen the system guidance to produce valid JSON.
    # (We append the instruction if caller didn't already mention JSON.)
    if "json" not in system.lower():
        system = (
            system.rstrip()
            + "\n\nYou MUST respond with a single valid JSON value (object or array). No prose."
        )
    if "json" not in user.lower():
        user = user.rstrip() + "\nReturn ONLY valid JSON (object or array)."

    instructions = system
    input_text = user

    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = await client.responses.create(
                model=model,
                instructions=instructions,
                input=input_text,
                max_output_tokens=max_output_tokens,
            )

            raw_text = getattr(resp, "output_text", None)
            if raw_text is None:
                raise ValueError("No output_text in response.")

            # First parse try
            try:
                return json.loads(raw_text)
            except Exception:
                pass

            # Strip code fences and retry
            stripped = _strip_json_fences(raw_text)
            try:
                return json.loads(stripped)
            except Exception as parse_err:
                raise ValueError(
                    f"Model output not valid JSON (after strip). Text starts: {stripped[:120]!r}"
                ) from parse_err

        except Exception as e:  # catch all; we retry
            last_err = e
            logger.warning("OpenAI JSON call failed (%s/3): %s", attempt + 1, e)
            await asyncio.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"OpenAI JSON call failed after 3 attempts: {last_err}")

# ───────────────────────────────────────────────────────────────────────────
# 1. Mask‑slippage criteria (viewer‑aware)
# ───────────────────────────────────────────────────────────────────────────

class SlippageRow(TypedDict):
    threshold: int        # stat value at which cue becomes active
    cue: str              # snake_case identifier
    description: str      # what the player can notice

@async_result_cache(maxsize=64)
async def get_mask_slippage_triggers(stat: str, environment_desc: str = "") -> List[SlippageRow]:
    """Return 4 progressive rows describing how the NPC's façade begins cracking *to the player*."""
    try:
        env_block = qblock(environment_desc or "Generic tavern-and-swords fantasy")
        stat_block = qblock(stat)
        data = await _gpt_json(
            "You output structured design data for mask-slippage.",
            f"""
Setting:
{env_block}

Create FOUR stages for attribute shown below (0-100 scale):
{stat_block}
Each stage must contain:
  • `threshold` integer (e.g. 25 / 50 / 75 / 90)
  • `cue` snake_case ≤2 words
  • `description` – max 16 words describing what the *player* might notice as the mask slips.
Return JSON array.
""",
        )
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except Exception as e:
        logger.error("mask-slippage GPT fallback: %s", e)
    return [
        {"threshold": 25, "cue": "tell_tale_smirk", "description": "Her smile lingers a notch too predatory."},
        {"threshold": 50, "cue": "commanding_tone", "description": "Orders slip into casual conversation."},
        {"threshold": 75, "cue": "predatory_gaze", "description": "Eyes sharpen, weighing the player's reactions."},
        {"threshold": 90, "cue": "open_assertion", "description": "Stops pretending; expects immediate obedience."},
    ]
# ───────────────────────────────────────────────────────────────────────────
# 2. Relationship stage ladder with regression notes
# ───────────────────────────────────────────────────────────────────────────

class RelStage(TypedDict):
    level: int
    name: str
    advance: str      # how the relationship *progresses* to next stage
    regress: str      # how it might *slip back*

@async_result_cache(maxsize=64)
async def get_relationship_stages(scenario: str, environment_desc: str = "") -> List[RelStage]:
    """Return 5 nuanced stages that support forward *and* backward movement."""
    try:
        env_block = qblock(environment_desc or "Default city")
        scenario_block = qblock(scenario)
        data = await _gpt_json(
            "Design bidirectional relationship ladders.",
            f"""
Setting summary:
{env_block}

Relationship scenario:
{scenario_block}

Return FIVE stages using levels 10/30/50/70/90.  Each stage must include:
  • `name` ≤3 words
  • `advance` ≤20 words (what pushes it *forward*)
  • `regress` ≤20 words (what drags it *backward*)
Return JSON array.
""",
        )
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except Exception as e:
        logger.error("relationship ladder GPT fallback: %s", e)
    return [
        {"level": 10, "name": "Polite Fog", "advance": "Shared secret surfaces", "regress": "One ghosts the other"},
        {"level": 30, "name": "Edges Show", "advance": "Inside joke succeeds", "regress": "Awkward silence grows"},
        {"level": 50, "name": "Terse Truce", "advance": "Joint victory celebrated", "regress": "Old wound reopened"},
        {"level": 70, "name": "Heat Flash", "advance": "Honest confession given", "regress": "Outside pressure mounts"},
        {"level": 90, "name": "Truth Bared", "advance": "Vow exchanged", "regress": "Betrayal revealed"},
    ]

# ───────────────────────────────────────────────────────────────────────────
# 3. Relationship‑specific memory snippet
# ───────────────────────────────────────────────────────────────────────────

async def generate_relationship_memory(npc_name: str, target_name: str, relationship: str, location: str, environment_desc: str = "") -> str | None:
    """Return a vivid first‑person memory (3‑5 sentences) for this interaction."""
    try:
        env_block = qblock(environment_desc or "Generic")
        details_block = qblock(f"NPC: {npc_name}\nTarget: {target_name}\nLocation: {location}\nRelationship: {relationship}")
        data = await _gpt_json(
            "Craft immersive NPC memory.",
            f"""
World:
{env_block}

Details:
{details_block}

Write ONE 3‑5 sentence first‑person memory of this interaction.
Include at least one sensory detail. Return {{"memory": "..."}}.
""",
        )
        if isinstance(data, dict) and "memory" in data:
            return str(data["memory"])
        if isinstance(data, str):
            return data
    except Exception as e:
        logger.warning("relationship memory fallback: %s", e)
    return None

# ───────────────────────────────────────────────────────────────────────────
# 4. Reciprocal label helper
# ───────────────────────────────────────────────────────────────────────────

@async_result_cache(maxsize=256)
async def get_reciprocal_label(label: str, archetype_summary: str = "") -> str:
    fixed = {"mother": "child", "sister": "sibling", "aunt": "niece/nephew"}
    if label.lower() in fixed:
        return fixed[label.lower()]
    try:
        label_block = qblock(label)
        archetype_block = qblock(archetype_summary)
        data = await _gpt_json(
            "Reciprocal term mapper.", 
            f"""
Label to find reciprocal for:
{label_block}

Other person's archetype:
{archetype_block}

Return reciprocal term (one‑two words).
"""
        )
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            return next(iter(data.values()))
    except Exception:
        pass
    return label

# ───────────────────────────────────────────────────────────────────────────
# 5. Calendar helpers (months + NEW day names)
# ───────────────────────────────────────────────────────────────────────────

@async_result_cache(maxsize=32)
async def get_calendar_months(environment_desc: str = "") -> List[str]:
    try:
        env_block = qblock(environment_desc)
        data = await _gpt_json(
            "Invent 12 month names.", 
            f"Setting:\n{env_block}\n\nReturn JSON list of 12 distinct month names."
        )
        if isinstance(data, list) and len(data) >= 12:
            return data[:12]
    except Exception as e:
        logger.error("month names GPT fallback: %s", e)
    return [
        "Frostmoon", "Windsong", "Bloomrise", "Dawnsveil", "Emberlight", "Goldencrest",
        "Shadowleaf", "Harvesttide", "Stormcall", "Nightwhisper", "Snowbound", "Yearsend",
    ]

@async_result_cache(maxsize=32)
async def get_calendar_day_names(environment_desc: str = "") -> List[str]:
    """Return 7 unique day names tied to the setting."""
    try:
        env_block = qblock(environment_desc)
        data = await _gpt_json(
            "Invent 7 day names.",
            f"Setting:\n{env_block}\n\nReturn JSON list of 7 distinct day names (short, ≤2 words).",
        )
        if isinstance(data, list) and len(data) >= 7:
            return data[:7]
    except Exception as e:
        logger.error("day names GPT fallback: %s", e)
    return ["Moonday", "Twosday", "Woten", "Thorsday", "Freyday", "Starday", "Sundim"]

# ───────────────────────────────────────────────────────────────────────────
# 6‑10. Beliefs, semantic topics, flashback triggers, trauma keywords, alt names
# ───────────────────────────────────────────────────────────────────────────

@async_result_cache(maxsize=128)
async def generate_core_beliefs(archetype_summary: str, personality_traits_str: str, environment_desc: str = "", *, n: int = 5) -> List[str]:
    """
    Generate core beliefs. Note: personality_traits_str should be a comma-separated string, not a list.
    """
    try:
        env_block = qblock(environment_desc)
        archetype_block = qblock(archetype_summary)
        traits_block = qblock(personality_traits_str)
        data = await _gpt_json(
            "NPC belief generator.",
            f"""World:
{env_block}

Archetype:
{archetype_block}

Traits:
{traits_block}

Generate {n} core first-person beliefs (≤14 words each).
Return as JSON array of belief strings.""",
        )
        if isinstance(data, list):
            return [str(b) for b in data][:n]
    except Exception as e:
        logger.error("belief GPT fallback: %s", e)
    return ["I must stay in control", "Others reveal themselves if I listen"]

@async_result_cache(maxsize=128)
async def get_semantic_seed_topics(archetype_summary: str, environment_desc: str = "") -> List[str]:
    try:
        env_block = qblock(environment_desc)
        archetype_block = qblock(archetype_summary)
        data = await _gpt_json(
            "Seed topics extractor.",
            f"""Setting:
{env_block}

Archetype summary:
{archetype_block}

List 4 noun phrases central to this character's knowledge graph.""",
        )
        if isinstance(data, list):
            return [str(x) for x in data][:5]
    except Exception:
        pass
    return ["Self", "Power", "Family", "Secrets"]

@async_result_cache(maxsize=64)
async def get_trauma_keywords(environment_desc: str = "") -> List[str]:
    try:
        env_block = qblock(environment_desc)
        data = await _gpt_json(
            "Trauma keyword list.", 
            f"Give 10 one‑word trauma flags for:\n{env_block}"
        )
        if isinstance(data, list):
            return data[:10]
    except Exception:
        pass
    return ["hurt", "pain", "suffer", "trauma", "abuse", "betray", "torture", "loss", "fear", "violate"]

@async_result_cache(maxsize=128)
async def get_alternative_names(gender: str = "female", environment_desc: str = "", *, n: int = 12) -> List[str]:
    try:
        env_block = qblock(environment_desc)
        data = await _gpt_json(
            "Name generator.", 
            f"Need {n} unique {gender} first names.\nSetting:\n{env_block}\n\nReturn JSON list."
        )
        if isinstance(data, list):
            return [str(x) for x in data][:n]
    except Exception:
        pass
    return [
        "Aurora", "Celeste", "Luna", "Nova", "Ivy", "Evelyn", "Isolde", "Marina", "Sable", "Lyra", "Nyx", "Mira",
    ]

# ───────────────────────────────────────────────────────────────────────────
# Convenience sync runner (unchanged)
# ───────────────────────────────────────────────────────────────────────────

def sync(awaitable):  # pragma: no cover
    return asyncio.get_event_loop().run_until_complete(awaitable)
