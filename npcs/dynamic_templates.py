# npcs/dynamic_templates.py
"""
Dynamic, environment‑aware template helpers for NPC generation.

This file centralises **all** the little hard‑coded tables and fallback text that used to live in
*npcs/new_npc_creation.py*.
Each helper below makes a **single GPT call** (cached) so data is:
  • **tailored to the current campaign setting** (environment description, culture, genre,…)
  • lightweight – only queried the first time it’s needed thanks to `functools.lru_cache`
  • resilient – each has a static fallback in case GPT is unavailable.

────────────────────────────────────────────────────────────────────────────
What changed in this revision
────────────────────────────────────────────────────────────────────────────
• **Mask‑slippage triggers** now return *criteria* (no memory text). Each row gives a cue the **player
  might notice** once the relevant stat crosses its threshold.
• **Relationship ladders** include *progression* **and** *regression* blurbs, acknowledging that bonds
  can strengthen or weaken over time.
• Added `get_calendar_day_names` for custom day names per‑setting.
• Helpers for beliefs, semantic topics, flashback triggers unchanged but kept for context.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
from typing import Any, Dict, List, TypedDict

try:
    from logic.chatgpt_integration import get_openai_client  # project‑level wrapper
except ImportError:  # unit‑tests / offline
    get_openai_client = None  # type: ignore

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# GPT wrapper
# ───────────────────────────────────────────────────────────────────────────

async def _gpt_json(system: str, user: str, *, model: str = "gpt-4o-mini") -> Any:
    """Call OpenAI forcing JSON output, three retries, else raise."""
    if get_openai_client is None:
        raise RuntimeError("OpenAI client unavailable – falling back")

    client = get_openai_client()
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.7,
                max_tokens=640,
                response_format={"type": "json_object"},
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # pragma: no cover
            logger.warning("GPT call failed (%s/3): %s", attempt + 1, e)
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError("GPT failed after 3 attempts")

_hash = lambda txt: hashlib.sha1(txt.encode()).hexdigest()[:12]  # campaign isolator

# ───────────────────────────────────────────────────────────────────────────
# 1. Mask‑slippage criteria (viewer‑aware)
# ───────────────────────────────────────────────────────────────────────────

class SlippageRow(TypedDict):
    threshold: int        # stat value at which cue becomes active
    cue: str              # snake_case identifier
    description: str      # what the player can notice

@functools.lru_cache(maxsize=64)
async def get_mask_slippage_triggers(stat: str, environment_desc: str = "") -> List[SlippageRow]:
    """Return 4 progressive rows describing how the NPC's façade begins cracking *to the player*."""
    try:
        data = await _gpt_json(
            "You output structured design data for mask‑slippage.",
            f"""
Setting:
{environment_desc or 'Generic tavern‑and‑swords fantasy'}

Create FOUR stages for attribute **{stat}** (0‑100 scale).  Each stage must contain:
  • `threshold` integer (e.g. 25 / 50 / 75 / 90)
  • `cue` snake_case ≤2 words
  • `description` – max 16 words describing what the *player* might notice as the mask slips.
Return JSON array.
""",
        )
        if isinstance(data, list):
            return data  # type: ignore[return-value]
    except Exception as e:
        logger.error("mask‑slippage GPT fallback: %s", e)
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

@functools.lru_cache(maxsize=64)
async def get_relationship_stages(scenario: str, environment_desc: str = "") -> List[RelStage]:
    """Return 5 nuanced stages that support forward *and* backward movement."""
    try:
        data = await _gpt_json(
            "Design bidirectional relationship ladders.",
            f"""
Setting summary:
{environment_desc or 'Default city'}

Relationship scenario: "{scenario}"
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
# 3. Relationship‑specific memory snippet (unchanged)
# ───────────────────────────────────────────────────────────────────────────

async def generate_relationship_memory(npc_name: str, target_name: str, relationship: str, location: str, environment_desc: str = "") -> str | None:
    """Return a vivid first‑person memory (3‑5 sentences) for this interaction."""
    try:
        data = await _gpt_json(
            "Craft immersive NPC memory.",
            f"""
World:
{environment_desc or 'Generic'}
Write ONE 3‑5 sentence first‑person memory of {npc_name} with {target_name} at {location}.
Relationship: {relationship}. Include at least one sensory detail. Return {{"memory": "..."}}.
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
# 4. Reciprocal label helper (unchanged)
# ───────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
async def get_reciprocal_label(label: str, archetype_summary: str = "") -> str:
    fixed = {"mother": "child", "sister": "sibling", "aunt": "niece/nephew"}
    if label.lower() in fixed:
        return fixed[label.lower()]
    try:
        data = await _gpt_json("Reciprocal term mapper.", f"Give reciprocal of '{label}' given other is {archetype_summary}. One‑two words.")
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

@functools.lru_cache(maxsize=32)
async def get_calendar_months(environment_desc: str = "") -> List[str]:
    try:
        data = await _gpt_json("Invent 12 month names.", f"Setting: {environment_desc}\nReturn JSON list of 12 distinct month names.")
        if isinstance(data, list) and len(data) >= 12:
            return data[:12]
    except Exception as e:
        logger.error("month names GPT fallback: %s", e)
    return [
        "Frostmoon", "Windsong", "Bloomrise", "Dawnsveil", "Emberlight", "Goldencrest",
        "Shadowleaf", "Harvesttide", "Stormcall", "Nightwhisper", "Snowbound", "Yearsend",
    ]

@functools.lru_cache(maxsize=32)
async def get_calendar_day_names(environment_desc: str = "") -> List[str]:
    """Return 7 unique day names tied to the setting."""
    try:
        data = await _gpt_json(
            "Invent 7 day names.",
            f"Setting: {environment_desc}\nReturn JSON list of 7 distinct day names (short, ≤2 words).",
        )
        if isinstance(data, list) and len(data) >= 7:
            return data[:7]
    except Exception as e:
        logger.error("day names GPT fallback: %s", e)
    return ["Moonday", "Twosday", "Woten", "Thorsday", "Freyday", "Starday", "Sundim"]

# ───────────────────────────────────────────────────────────────────────────
# 6‑10. Beliefs, semantic topics, flashback triggers, trauma keywords, alt names
#       (copied unchanged from previous revision for continuity)
# ───────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=128)
async def generate_core_beliefs(archetype_summary: str, personality_traits: List[str], environment_desc: str = "", *, n: int = 5) -> List[str]:
    try:
        data = await _gpt_json(
            "NPC belief generator.",
            f"World: {environment_desc}\nArchetype: {archetype_summary}\nTraits: {', '.join(personality_traits)}\nReturn {n} core first‑person beliefs (≤14 words).",
        )
        if isinstance(data, list):
            return [str(b) for b in data][:n]
    except Exception as e:
        logger.error("belief GPT fallback: %s", e)
    return ["I must stay in control", "Others reveal themselves if I listen"]

@functools.lru_cache(maxsize=128)
async def get_semantic_seed_topics(archetype_summary: str, environment_desc: str = "") -> List[str]:
    try:
        data = await _gpt_json(
            "Seed topics extractor.",
            f"Setting: {environment_desc}\nArchetype summary: {archetype_summary}\nList 4 noun phrases central to this character's knowledge graph.",
        )
        if isinstance(data, list):
            return [str(x) for x in data][:5]
    except Exception:
        pass
    return ["Self", "Power", "Family", "Secrets"]

@functools.lru_cache(maxsize=64)
async def get_trauma_keywords(environment_desc: str = "") -> List[str]:
    try:
        data = await _gpt_json("Trauma keyword list.", f"Give 10 one‑word trauma flags for: {environment_desc}")
        if isinstance(data, list):
            return data[:10]
    except Exception:
        pass
    return ["hurt", "pain", "suffer", "trauma", "abuse", "betray", "torture", "loss", "fear", "violate"]

@functools.lru_cache(maxsize=128)
async def get_alternative_names(gender: str = "female", environment_desc: str = "", *, n: int = 12) -> List[str]:
    try:
        data = await _gpt_json("Name generator.", f"Need {n} unique {gender} first names for: {environment_desc}")
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
