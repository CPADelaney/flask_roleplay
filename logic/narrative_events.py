# logic/narrative_events.py
"""
Narrative event system for NPC-specific progression.
This module contains functions for generating narrative events based on 
the state of multiple NPC relationships at different progression stages.
"""

import logging
import random
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

import asyncpg
from db.connection import get_db_connection_context
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    NPC_NARRATIVE_STAGES
)

logger = logging.getLogger(__name__)


async def get_relationship_overview(user_id: int, conversation_id: int) -> Dict[str, Any]:
    """
    Get an overview of all NPC relationship stages and progression.
    Useful for understanding the player's overall situation.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get all NPC progressions
            progressions = await conn.fetch("""
                SELECT 
                    np.npc_id,
                    ns.npc_name,
                    np.narrative_stage,
                    np.corruption,
                    np.dependency,
                    np.realization_level,
                    np.stage_entered_at,
                    sl.link_level,
                    sl.link_type
                FROM NPCNarrativeProgression np
                JOIN NPCStats ns ON np.npc_id = ns.npc_id
                LEFT JOIN SocialLinks sl ON (
                    sl.entity1_type = 'npc' AND sl.entity1_id = np.npc_id 
                    AND sl.entity2_type = 'player' AND sl.user_id = np.user_id 
                    AND sl.conversation_id = np.conversation_id
                )
                WHERE np.user_id = $1 AND np.conversation_id = $2
                ORDER BY np.corruption + np.dependency DESC
            """, user_id, conversation_id)
            
            # Categorize by stage
            by_stage = {}
            for stage in NPC_NARRATIVE_STAGES:
                by_stage[stage.name] = []
            
            # Overall statistics
            total_corruption = 0
            total_dependency = 0
            total_realization = 0
            
            relationships = []
            for prog in progressions:
                npc_data = {
                    'npc_id': prog['npc_id'],
                    'npc_name': prog['npc_name'],
                    'stage': prog['narrative_stage'],
                    'corruption': prog['corruption'],
                    'dependency': prog['dependency'],
                    'realization': prog['realization_level'],
                    'link_level': prog['link_level'] or 50,
                    'link_type': prog['link_type'] or 'neutral',
                    'days_in_stage': (datetime.now() - prog['stage_entered_at']).days if prog['stage_entered_at'] else 0
                }
                
                relationships.append(npc_data)
                by_stage[prog['narrative_stage']].append(npc_data)
                
                # Weight by relationship strength
                weight = (prog['link_level'] or 50) / 100.0
                total_corruption += prog['corruption'] * weight
                total_dependency += prog['dependency'] * weight
                total_realization += prog['realization_level'] * weight
            
            # Calculate averages
            num_relationships = len(relationships)
            if num_relationships > 0:
                avg_corruption = total_corruption / num_relationships
                avg_dependency = total_dependency / num_relationships
                avg_realization = total_realization / num_relationships
            else:
                avg_corruption = avg_dependency = avg_realization = 0
            
            return {
                'total_relationships': num_relationships,
                'by_stage': by_stage,
                'relationships': relationships,
                'aggregate_stats': {
                    'average_corruption': avg_corruption,
                    'average_dependency': avg_dependency,
                    'average_realization': avg_realization
                },
                'most_advanced_npcs': relationships[:3],  # Top 3
                'stage_distribution': {
                    stage: len(npcs) for stage, npcs in by_stage.items()
                }
            }
            
    except Exception as e:
        logger.error(f"Error getting relationship overview: {e}")
        return {
            'error': str(e),
            'total_relationships': 0,
            'by_stage': {},
            'relationships': []
        }

# Add this function to logic/narrative_events.py

async def generate_inner_monologue(
    user_id: int, 
    conversation_id: int,
    topic: Optional[str] = None
) -> str:
    """
    Generate an inner monologue for the player character.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID  
        topic: Optional topic to focus the monologue on
        
    Returns:
        Generated inner monologue text
    """
    from logic.chatgpt_integration import generate_text_completion
    
    try:
        # Get current player state for context
        async with get_db_connection_context() as conn:
            # Get player stats
            stats = await conn.fetchrow("""
                SELECT corruption, dependency, confidence, willpower, obedience
                FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                LIMIT 1
            """, user_id, conversation_id)
            
            # Get recent NPCs
            recent_npcs = await conn.fetch("""
                SELECT DISTINCT ns.npc_name, np.narrative_stage
                FROM NPCNarrativeProgression np
                JOIN NPCStats ns ON np.npc_id = ns.npc_id
                WHERE np.user_id = $1 AND np.conversation_id = $2
                ORDER BY np.stage_updated_at DESC
                LIMIT 3
            """, user_id, conversation_id)
        
        # Build context
        context_parts = []
        if stats:
            if stats['corruption'] > 50:
                context_parts.append("feeling increasingly corrupted")
            if stats['dependency'] > 50:
                context_parts.append("growing dependent on others")
            if stats['willpower'] < 30:
                context_parts.append("struggling with weakened willpower")
                
        npcs_context = ""
        if recent_npcs:
            npc_names = [npc['npc_name'] for npc in recent_npcs]
            npcs_context = f"Recent interactions with: {', '.join(npc_names)}"
        
        prompt = f"""Generate a brief inner monologue for Chase, a college student in a femdom game.

Context: {', '.join(context_parts) if context_parts else 'Early in the experience'}
{npcs_context}
Topic: {topic or 'reflecting on recent events'}

The monologue should:
- Be 1-2 sentences of internal thought
- Show genuine psychological state
- Feel like authentic self-reflection
- Avoid being overly dramatic

Example: "Why do I keep thinking about what she said? It shouldn't matter this much to me."

Generate the inner monologue:"""

        monologue = await generate_text_completion(
            system_prompt="You are generating authentic inner thoughts for a character gradually experiencing psychological changes.",
            user_prompt=prompt,
            task_type="reflection"
        )
        
        return monologue.strip() if monologue else "What's happening to me?"
        
    except Exception as e:
        logger.error(f"Error generating inner monologue: {e}")
        # Return a fallback monologue
        if topic and "dependency" in topic.lower():
            return "When did I start needing their approval so much?"
        elif topic and "control" in topic.lower():
            return "I'm losing control of my own life, bit by bit."
        else:
            return "Something feels different, but I can't quite place what."


async def check_for_personal_revelations(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a personal revelation.
    Considers revelations across all NPC relationships.
    """
    try:
        async with get_db_connection_context() as conn:
            # Check recent revelations
            recent_count = await conn.fetchval("""
                SELECT COUNT(*) FROM PlayerJournal
                WHERE user_id = $1 AND conversation_id = $2 
                AND entry_type = 'personal_revelation'
                AND timestamp > NOW() - INTERVAL '5 days'
            """, user_id, conversation_id) or 0
            
            if recent_count > 2:
                return None
            
            # Get all NPC relationships with their stages
            npcs = await conn.fetch("""
                SELECT np.npc_id, np.narrative_stage, np.corruption, np.dependency, 
                       np.realization_level, ns.npc_name, ns.dominance
                FROM NPCNarrativeProgression np
                JOIN NPCStats ns ON np.npc_id = ns.npc_id
                WHERE np.user_id = $1 AND np.conversation_id = $2
                AND np.narrative_stage != 'Innocent Beginning'
                ORDER BY np.corruption + np.dependency DESC
                LIMIT 3
            """, user_id, conversation_id)
            
            if not npcs:
                return None
            
            # Choose the most influential NPC for the revelation
            primary_npc = npcs[0]
            npc_id = primary_npc['npc_id']
            npc_name = primary_npc['npc_name']
            stage = primary_npc['narrative_stage']
            
            # Get recent stat changes across all relationships
            stat_changes = await conn.fetch("""
                SELECT 
                    SUM(CASE WHEN corruption > LAG(corruption) OVER (PARTITION BY npc_id ORDER BY stage_updated_at) 
                        THEN corruption - LAG(corruption) OVER (PARTITION BY npc_id ORDER BY stage_updated_at) 
                        ELSE 0 END) as corruption_increase,
                    SUM(CASE WHEN dependency > LAG(dependency) OVER (PARTITION BY npc_id ORDER BY stage_updated_at) 
                        THEN dependency - LAG(dependency) OVER (PARTITION BY npc_id ORDER BY stage_updated_at) 
                        ELSE 0 END) as dependency_increase
                FROM NPCNarrativeProgression
                WHERE user_id = $1 AND conversation_id = $2
                AND stage_updated_at > NOW() - INTERVAL '7 days'
            """, user_id, conversation_id)
            
            # Determine revelation type based on aggregate changes
            revelation_type = "dependency"  # default
            if stat_changes and stat_changes[0]:
                if stat_changes[0]['corruption_increase'] > stat_changes[0]['dependency_increase']:
                    revelation_type = "corruption"
            
            # Generate revelation considering multiple NPCs if applicable
            other_npcs = [npc['npc_name'] for npc in npcs[1:]]
            
            templates = {
                "dependency": [
                    f"I've been checking my phone constantly to see if {npc_name} has messaged me. When did I start needing her approval so much?",
                    f"I realized today that I haven't made a significant decision without consulting {npc_name} in weeks. Is that normal?",
                ] + ([f"Between {npc_name} and {', '.join(other_npcs)}, I barely have a moment to myself anymore. When did I become so dependent on them?"] if other_npcs else []),
                
                "corruption": [
                    f"I found myself enjoying the feeling of following {npc_name}'s instructions perfectly. The pride I felt at her approval was... intense.",
                    f"Last year, I would have been offended if someone treated me the way {npc_name} did today. Now I'm grateful for her attention.",
                ] + ([f"Each of them - {npc_name}, {', '.join(other_npcs)} - has changed me in their own way. I'm not who I was."] if other_npcs else []),
                
                "realization": [
                    f"I see it clearly now - {npc_name} has been guiding me, shaping me, this whole time.",
                    f"The pattern is obvious once you see it. {npc_name}'s control, how I've changed... it's all connected.",
                ] + ([f"They're all doing it - {npc_name}, {', '.join(other_npcs)}. Working together or separately, I can't tell, but the result is the same."] if other_npcs else [])
            }
            
            revelation_templates = templates.get(revelation_type, templates["dependency"])
            inner_monologue = random.choice(revelation_templates)
            
            # Create journal entry
            journal_id = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, revelation_types, timestamp)
                VALUES ($1, $2, 'personal_revelation', $3, $4, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, inner_monologue, revelation_type
            )
            
            return {
                "type": "personal_revelation",
                "npc_id": npc_id,
                "npc_name": npc_name,
                "other_npcs": other_npcs,
                "name": f"{revelation_type.capitalize()} Revelation",
                "inner_monologue": inner_monologue,
                "journal_id": journal_id
            }
            
    except Exception as e:
        logger.error(f"Error checking for personal revelations: {e}")
        return None


async def check_for_narrative_moments(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a narrative moment.
    Considers the mix of different NPC relationship stages.
    """
    try:
        async with get_db_connection_context() as conn:
            # Check recent moments
            recent_count = await conn.fetchval("""
                SELECT COUNT(*) FROM PlayerJournal
                WHERE user_id = $1 AND conversation_id = $2 
                AND entry_type = 'narrative_moment'
                AND timestamp > NOW() - INTERVAL '7 days'
            """, user_id, conversation_id) or 0
            
            if recent_count > 2:
                return None
            
            # Get NPCs at different stages
            npcs_by_stage = await conn.fetch("""
                SELECT np.narrative_stage, 
                       array_agg(ns.npc_name) as npc_names,
                       array_agg(np.npc_id) as npc_ids
                FROM NPCNarrativeProgression np
                JOIN NPCStats ns ON np.npc_id = ns.npc_id
                WHERE np.user_id = $1 AND np.conversation_id = $2
                GROUP BY np.narrative_stage
            """, user_id, conversation_id)
            
            if not npcs_by_stage:
                return None
            
            # Create a stage distribution
            stage_distribution = {}
            for row in npcs_by_stage:
                stage_distribution[row['narrative_stage']] = {
                    'names': row['npc_names'],
                    'ids': row['npc_ids']
                }
            
            # Generate moments based on stage diversity
            if len(stage_distribution) > 1:
                # Mixed stages create interesting dynamics
                advanced_npcs = []
                early_npcs = []
                
                for stage_name in ["Full Revelation", "Veil Thinning", "Creeping Realization"]:
                    if stage_name in stage_distribution:
                        advanced_npcs.extend(stage_distribution[stage_name]['names'])
                
                for stage_name in ["Innocent Beginning", "First Doubts"]:
                    if stage_name in stage_distribution:
                        early_npcs.extend(stage_distribution[stage_name]['names'])
                
                if advanced_npcs and early_npcs:
                    # Create a moment highlighting the contrast
                    advanced_name = random.choice(advanced_npcs)
                    early_name = random.choice(early_npcs)
                    
                    scene_text = f"You notice {advanced_name} and {early_name} exchanging glances. {advanced_name}'s knowing smile contrasts sharply with {early_name}'s seemingly innocent demeanor. Are they working together, or is {early_name} truly unaware of what's happening?"
                    moment_name = "Contrasting Masks"
                    player_realization = "Not everyone is at the same stage of revealing their true nature."
                else:
                    # Default to a general moment
                    return await _generate_general_narrative_moment(conn, user_id, conversation_id)
            else:
                # All NPCs at similar stage
                return await _generate_general_narrative_moment(conn, user_id, conversation_id)
            
            # Create journal entry
            journal_id = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, narrative_moment, timestamp)
                VALUES ($1, $2, 'narrative_moment', $3, $4, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, scene_text, moment_name
            )
            
            return {
                "type": "narrative_moment",
                "name": moment_name,
                "scene_text": scene_text,
                "player_realization": player_realization,
                "stage_distribution": stage_distribution,
                "journal_id": journal_id
            }
            
    except Exception as e:
        logger.error(f"Error checking for narrative moments: {e}")
        return None


async def _generate_general_narrative_moment(conn, user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """Generate a general narrative moment not tied to specific stage contrasts."""
    # Get a few NPCs for the moment
    npcs = await conn.fetch("""
        SELECT ns.npc_id, ns.npc_name, np.narrative_stage
        FROM NPCStats ns
        JOIN NPCNarrativeProgression np ON ns.npc_id = np.npc_id
        WHERE ns.user_id = $1 AND ns.conversation_id = $2
        AND np.narrative_stage != 'Innocent Beginning'
        ORDER BY RANDOM()
        LIMIT 2
    """, user_id, conversation_id)
    
    if not npcs:
        return None
    
    npc1 = npcs[0]
    templates = [
        {
            "name": "Orchestrated Encounter",
            "scene_text": f"The 'chance' meeting with {npc1['npc_name']} feels anything but random. The timing, the location, even her words seem carefully planned.",
            "player_realization": "Nothing happens by accident anymore."
        },
        {
            "name": "Subtle Coordination",
            "scene_text": f"You catch {npc1['npc_name']} glancing at her phone and smiling. Moments later, your own phone buzzes with a message that seems connected to your earlier conversation.",
            "player_realization": "They're coordinating, even when they seem to be acting independently."
        }
    ]
    
    if len(npcs) > 1:
        npc2 = npcs[1]
        templates.append({
            "name": "Shared Knowledge",
            "scene_text": f"{npc2['npc_name']} mentions something you only told {npc1['npc_name']}. The look they share when you react tells you everything.",
            "player_realization": "They share information about me. I have no secrets from any of them."
        })
    
    chosen = random.choice(templates)
    
    journal_id = await conn.fetchval(
        """
        INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, narrative_moment, timestamp)
        VALUES ($1, $2, 'narrative_moment', $3, $4, CURRENT_TIMESTAMP)
        RETURNING id
        """,
        user_id, conversation_id, chosen["scene_text"], chosen["name"]
    )
    
    return {
        "type": "narrative_moment",
        "name": chosen["name"],
        "scene_text": chosen["scene_text"],
        "player_realization": chosen["player_realization"],
        "journal_id": journal_id
    }


async def add_dream_sequence(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
    """
    Generate and add a dream sequence to the player's journal.
    Dreams reflect the mix of relationships at different stages.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get NPCs at various stages
            npcs = await conn.fetch("""
                SELECT ns.npc_id, ns.npc_name, np.narrative_stage,
                       np.corruption, np.dependency
                FROM NPCStats ns
                JOIN NPCNarrativeProgression np ON ns.npc_id = np.npc_id
                WHERE ns.user_id = $1 AND ns.conversation_id = $2
                ORDER BY np.corruption + np.dependency DESC
                LIMIT 3
            """, user_id, conversation_id)
            
            if not npcs:
                return None
            
            # Categorize NPCs by stage
            advanced_npcs = [npc for npc in npcs if npc['narrative_stage'] in ['Veil Thinning', 'Full Revelation']]
            developing_npcs = [npc for npc in npcs if npc['narrative_stage'] in ['First Doubts', 'Creeping Realization']]
            innocent_npcs = [npc for npc in npcs if npc['narrative_stage'] == 'Innocent Beginning']
            
            # Generate dream based on stage mix
            if advanced_npcs and innocent_npcs:
                # Contrast dream
                adv = advanced_npcs[0]
                inn = innocent_npcs[0]
                dream_text = f"In your dream, {adv['npc_name']} stands behind {inn['npc_name']}, hands on her shoulders. {inn['npc_name']} smiles at you warmly, but {adv['npc_name']}'s hands seem to be guiding her every movement like a puppeteer. You try to warn {inn['npc_name']}, but no sound comes out. {adv['npc_name']} just smiles and whispers, 'She'll understand soon enough, just like you did.'"
            elif advanced_npcs:
                # Full control dream
                names = [npc['npc_name'] for npc in advanced_npcs]
                dream_text = f"You're in a room with no doors. {' and '.join(names)} {'are' if len(names) > 1 else 'is'} there, watching you search for an exit that doesn't exist. 'Why are you still looking?' {names[0]} asks. 'You know there's nowhere else you'd rather be.' The worst part is, she's right."
            else:
                # Early stage dream - subtle unease
                npc = npcs[0]
                dream_text = f"You dream of following {npc['npc_name']} through an endless hallway. Each door you pass clicks locked behind you. You want to ask where you're going, but somehow you already know - wherever she leads."
            
            # Create journal entry
            journal_id = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES ($1, $2, 'dream_sequence', $3, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, dream_text
            )
            
            return {
                "type": "dream_sequence",
                "text": dream_text,
                "primary_npcs": [npc['npc_name'] for npc in npcs],
                "journal_id": journal_id
            }
            
    except Exception as e:
        logger.error(f"Error creating dream sequence: {e}")
        return None


async def add_moment_of_clarity(user_id: int, conversation_id: int, realization_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Add a moment of clarity to the player's journal.
    Considers the player's overall situation across all relationships.
    """
    try:
        # Get aggregate information about all relationships
        async with get_db_connection_context() as conn:
            # Count NPCs at each stage
            stage_counts = await conn.fetch("""
                SELECT narrative_stage, COUNT(*) as count
                FROM NPCNarrativeProgression
                WHERE user_id = $1 AND conversation_id = $2
                GROUP BY narrative_stage
            """, user_id, conversation_id)
            
            stage_dict = {row['narrative_stage']: row['count'] for row in stage_counts}
            total_npcs = sum(stage_dict.values())
            
            if not realization_text:
                # Generate realization based on overall situation
                if stage_dict.get('Full Revelation', 0) > 0:
                    realization_text = "Some of them don't even pretend anymore. The others... are they genuinely different, or just better at hiding it?"
                elif stage_dict.get('Veil Thinning', 0) >= total_npcs / 2:
                    realization_text = "I'm surrounded by people who want to control me. How did I let it get this far?"
                elif stage_dict.get('Creeping Realization', 0) > 0:
                    realization_text = "The patterns are becoming clear. It's not just one of them - they all have their ways of influencing me."
                else:
                    realization_text = "Something feels off about my relationships. I can't quite put my finger on it, but the dynamic isn't what I thought."
            
            # Create journal entry
            journal_id = await conn.fetchval(
                """
                INSERT INTO PlayerJournal (user_id, conversation_id, entry_type, entry_text, timestamp)
                VALUES ($1, $2, 'moment_of_clarity', $3, CURRENT_TIMESTAMP)
                RETURNING id
                """,
                user_id, conversation_id, realization_text
            )
            
            return {
                "type": "moment_of_clarity",
                "text": realization_text,
                "stage_distribution": stage_dict,
                "journal_id": journal_id
            }
            
    except Exception as e:
        logger.error(f"Error creating moment of clarity: {e}")
        return None


async def initialize_player_stats(user_id: int, conversation_id: int):
    """
    Initialize player stats if they don't exist.
    This remains global as it's about the player, not NPC relationships.
    """
    try:
        async with get_db_connection_context() as conn:
            # Check if stats exist
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM PlayerStats WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase')",
                user_id, conversation_id
            )
            
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO PlayerStats 
                    (user_id, conversation_id, player_name, corruption, dependency, confidence, 
                     willpower, obedience, lust, timestamp)
                    VALUES ($1, $2, 'Chase', 0, 0, 50, 50, 0, 0, CURRENT_TIMESTAMP)
                    """,
                    user_id, conversation_id
                )
                logger.info(f"Initialized default PlayerStats for user {user_id}, convo {conversation_id}")
                
    except Exception as e:
        logger.error(f"Error initializing player stats: {e}")


async def analyze_narrative_tone(narrative_text: str) -> Dict[str, Any]:
    """
    Analyze the tone of a narrative text.
    This remains useful for analyzing any narrative content.
    """
    from logic.chatgpt_integration import get_chatgpt_response
    import json
    
    prompt = f"""
    Analyze the tone and thematic elements of this narrative text from a femdom roleplaying game:

    {narrative_text}

    Consider which NPC relationships this might relate to and what stages they might be in.
    
    Return your analysis in JSON format with these fields:
    - dominant_tone: The primary tone of the text
    - power_dynamics: Analysis of power relationships
    - implied_stages: Dict of potential NPC stages based on the text
    - manipulation_techniques: Any manipulation methods identified
    - intensity_level: Rating from 1-5 of how intense the power dynamic is
    """
    
    try:
        response = await get_chatgpt_response(None, "narrative_analysis", prompt)
        
        if response and "function_args" in response:
            return response["function_args"]
        else:
            # Extract JSON from text response
            response_text = response.get("response", "{}")
            import re
            
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Fallback
        return {
            "dominant_tone": "unknown",
            "power_dynamics": "unclear",
            "implied_stages": {},
            "manipulation_techniques": [],
            "intensity_level": 1
        }
    except Exception as e:
        logger.error(f"Error analyzing narrative tone: {e}", exc_info=True)
        return {
            "dominant_tone": "error",
            "power_dynamics": "error",
            "implied_stages": {},
            "manipulation_techniques": [],
            "intensity_level": 0,
            "error": str(e)
        }
