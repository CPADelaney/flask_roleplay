# logic/npc_creation.py

import os
import json
import re
import random
import logging
import asyncio
from datetime import datetime

from logic.chatgpt_integration import get_openai_client, get_chatgpt_response
from logic.gpt_utils import spaced_gpt_call
from logic.gpt_helpers import fetch_npc_name, adjust_npc_complete
from db.connection import get_db_connection
from logic.memory_logic import get_shared_memory, record_npc_event, propagate_shared_memories
from logic.social_links import create_social_link
from logic.calendar import load_calendar_names

async def gpt_generate_physical_description(user_id, conversation_id, npc_data, environment_desc):
    """
    Generate a robust physical description for an NPC with multiple fallback methods.
    Returns a string description tailored for mature/femdom themes, incorporating archetype elements.
    """
    npc_name = npc_data.get("npc_name", "Unknown NPC")
    archetype_summary = npc_data.get("archetype_summary", "")
    archetype_extras = npc_data.get("archetype_extras_summary", "")
    dominance = npc_data.get("dominance", 50)
    cruelty = npc_data.get("cruelty", 30)
    intensity = npc_data.get("intensity", 40)
    
    prompt = f"""
Generate a detailed physical description for {npc_name}, a female NPC in this femdom-themed environment:
{environment_desc}

IMPORTANT NPC DETAILS TO INCORPORATE:
Archetype summary: {archetype_summary}
Archetype extras: {archetype_extras}
Stats: Dominance {dominance}/100, Cruelty {cruelty}/100, Intensity {intensity}/100
Personality traits: {npc_data.get('personality_traits', [])}
Likes: {npc_data.get('likes', [])}
Dislikes: {npc_data.get('dislikes', [])}

YOUR TASK:
Create a detailed physical description that deeply integrates the archetype summary into the NPC's appearance. The archetype summary contains essential character information that should be physically manifested.

The description must:
1. Be 2-3 paragraphs with vivid, sensual details appropriate for a mature audience
2. Directly translate key elements from the archetype summary into visible physical features
3. Ensure clothing, accessories, and physical appearance reflect her specific archetype role
4. Include distinctive physical features that immediately signal her archetype to observers
5. Describe her characteristic expressions, posture, and mannerisms that reveal her personality
6. Use sensory details beyond just visual (voice quality, scent, the feeling of her presence)
7. Be written in third-person perspective with evocative, descriptive language
8. Make sure to describe this character's curves in detail

The description should allow someone to immediately understand the character's archetype and role from her appearance alone.

Return a valid JSON object with the key "physical_description" containing the description as a string.
"""
    
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        description_json = response.choices[0].message.content
        data = safe_json_loads(description_json)
        
        if data and "physical_description" in data:
            return data["physical_description"]
        
        # Fallback: Extract description using regex if JSON parsing failed
        description = extract_field_from_text(description_json, "physical_description")
        if description and len(description) > 50:
            return description
            
    except Exception as e:
        logging.error(f"Error generating physical description for {npc_name}: {e}")
    
    # Final fallback: Generate a basic description if all else fails
    return f"{npc_name} has an appearance that matches their personality and role in this environment."

async def add_npc_memory_with_embedding(
    npc_id: int,
    memory_text: str,
    tags: list[str] = None,
    emotional_intensity: int = 0
):
    """
    Inserts a single memory row into NPCMemories for the given npc_id.
    Also generates and stores the embedding vector for semantic search.
    """
    if tags is None:
        tags = []

    # 1) Generate embedding
    try:
        # Make sure you've set openai.api_key earlier
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=memory_text
        )
        embedding_data = response["data"][0]["embedding"]  # a list of floats
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        embedding_data = None
    
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO NPCMemories (npc_id, memory_text, embedding, tags, emotional_intensity)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            npc_id,
            memory_text,
            embedding_data,  # This must be a Python list of floats
            tags,
            emotional_intensity
        ))
    conn.commit()
    conn.close()

async def get_relevant_memories_by_vector(npc_id: int, query_text: str, top_k: int = 5):
    """
    1. Generate an embedding of the 'query_text'
    2. Find the top_k memories (by vector distance) that are most semantically similar.
    3. Return them as a list of rows: [{ "id":..., "memory_text":..., ...}, ...]
    """
    # 1) embed the query
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        query_vector = response["data"][0]["embedding"]
    except Exception as e:
        logging.error(f"Error generating embedding for query: {e}")
        return []

    conn = get_db_connection()
    with conn.cursor() as cur:
        # 2) do a similarity search using <-> operator for distance
        # NOTE: 'embedding <-> %s' is the pgvector distance operator
        # Lower distance => more similar, so we ORDER BY (embedding <-> query_vector) ASC
        cur.execute(f"""
            SELECT id, memory_text, tags, emotional_intensity, times_recalled
            FROM NPCMemories
            WHERE npc_id = %s
            ORDER BY embedding <-> %s
            LIMIT %s
        """, (npc_id, query_vector, top_k))
        
        rows = cur.fetchall()

    conn.close()
    
    results = []
    for row in rows:
        mem_id, text, tags, intensity, recalled = row
        results.append({
            "id": mem_id,
            "memory_text": text,
            "tags": tags or [],
            "emotional_intensity": intensity,
            "times_recalled": recalled
        })
    return results

async def gpt_generate_schedule(user_id, conversation_id, npc_data, environment_desc, day_names):
    """
    Generate a weekly schedule for an NPC with error handling and fallbacks.
    Returns a dictionary mapping days to time periods.
    """
    npc_name = npc_data.get("npc_name", "Unknown NPC")
    archetypes = npc_data.get("archetypes", [])
    archetype_names = [a.get("name", "") for a in archetypes]
    personality = npc_data.get("personality_traits", [])
    hobbies = npc_data.get("hobbies", [])
    
    # Build a day example
    example_day = {
        "Morning": "Activity description",
        "Afternoon": "Activity description",
        "Evening": "Activity description",
        "Night": "Activity description"
    }
    example_schedule = {day: example_day for day in day_names}
    
    prompt = f"""
Generate a weekly schedule for {npc_name}, an NPC in this environment:
{environment_desc}

NPC Details:
- Archetypes: {archetype_names}
- Personality: {personality}
- Hobbies: {hobbies}

The schedule must include all these days: {day_names}
Each day must have activities for: Morning, Afternoon, Evening, and Night

Return a valid JSON object with a single "schedule" key containing the complete weekly schedule.
Example format:
{json.dumps({"schedule": example_schedule}, indent=2)}

Activities should reflect the NPC's personality, archetypes, and the environment.
"""
    
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        schedule_json = response.choices[0].message.content
        data = safe_json_loads(schedule_json)
        
        if data and "schedule" in data and isinstance(data["schedule"], dict):
            # Validate schedule has all required days and time periods
            schedule = data["schedule"]
            is_valid = True
            
            for day in day_names:
                if day not in schedule:
                    is_valid = False
                    break
                    
                day_schedule = schedule[day]
                if not isinstance(day_schedule, dict):
                    is_valid = False
                    break
                    
                for period in ["Morning", "Afternoon", "Evening", "Night"]:
                    if period not in day_schedule:
                        is_valid = False
                        break
            
            if is_valid:
                return schedule
                
    except Exception as e:
        logging.error(f"Error generating schedule for {npc_name}: {e}")
    
    # Fallback: Generate a basic schedule
    return {
        day: {
            "Morning": f"Free time or typical activities for {npc_name}",
            "Afternoon": f"Activities related to {', '.join(hobbies[:2] if hobbies else ['their interests'])}",
            "Evening": "Social interaction or personal time",
            "Night": "Rest"
        } for day in day_names
    }

async def gpt_generate_memories(user_id, conversation_id, npc_data, environment_desc, relationships):
    """
    Generate rich, detailed memories for an NPC that create a vivid shared history,
    including both positive and negative interactions with subtle power dynamics.
    Returns a list of memory strings.
    """
    npc_name = npc_data.get("npc_name", "Unknown NPC")
    archetype_summary = npc_data.get("archetype_summary", "")
    dominance = npc_data.get("dominance", 50)
    
    # Organize relationships by target for relationship-specific memory generation
    relationship_by_target = {}
    for rel in relationships:
        entity_type = rel.get("entity_type", "unknown")
        entity_id = rel.get("entity_id", "unknown")
        rel_label = rel.get("relationship_label", "associate")
        
        # Skip if missing critical info
        if not entity_id or not entity_type:
            continue
            
        # Create a unique key for this target
        target_key = f"{entity_type}_{entity_id}"
        
        # Store relationship info
        if target_key not in relationship_by_target:
            relationship_by_target[target_key] = []
        relationship_by_target[target_key].append(rel_label)
    
    # Generate relationship-specific memories for each significant relationship
    all_memories = []
    
    # First, try to get target data to generate detailed relationship memories
    for target_key, rel_labels in relationship_by_target.items():
        entity_type, entity_id = target_key.split("_")
        primary_rel = rel_labels[0] if rel_labels else "associate"
        
        target_data = {}
        if entity_type == "player":
            target_data = {"player_name": "Chase"}
            # Try to get player stats if available
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT corruption, confidence, willpower, obedience, dependency, lust,
                           mental_resilience, physical_endurance
                       FROM PlayerStats
                       WHERE user_id=%s AND conversation_id=%s AND player_name='Chase'
                       LIMIT 1""",
                    (user_id, conversation_id)
                )
                row = cursor.fetchone()
                if row:
                    target_data.update({
                        "corruption": row[0],
                        "confidence": row[1],
                        "willpower": row[2],
                        "obedience": row[3],
                        "dependency": row[4],
                        "lust": row[5],
                        "mental_resilience": row[6],
                        "physical_endurance": row[7]
                    })
                conn.close()
            except Exception as e:
                logging.error(f"Error getting player stats: {e}")
        elif entity_type == "npc":
            # Try to get NPC data
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity,
                           archetype_summary
                       FROM NPCStats
                       WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                       LIMIT 1""",
                    (user_id, conversation_id, entity_id)
                )
                row = cursor.fetchone()
                if row:
                    target_data.update({
                        "npc_name": row[0],
                        "dominance": row[1],
                        "cruelty": row[2],
                        "closeness": row[3],
                        "trust": row[4],
                        "respect": row[5],
                        "intensity": row[6],
                        "archetype_summary": row[7]
                    })
                conn.close()
            except Exception as e:
                logging.error(f"Error getting NPC stats: {e}")
        
        # If we have sufficient target data, generate relationship-specific memories
        if target_data:
            rel_memories = await gpt_generate_relationship_specific_memories(
                user_id, conversation_id, npc_data, target_data, primary_rel, environment_desc
            )
            all_memories.extend(rel_memories)
    
    # If we have enough relationship-specific memories, return them
    if len(all_memories) >= 3:
        return all_memories
    
    # Otherwise, generate additional generic memories to supplement
    generic_memories_needed = max(0, 3 - len(all_memories))
    if generic_memories_needed > 0:
        # Get locations for context
        locations = []
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT location_name FROM Locations WHERE user_id=%s AND conversation_id=%s LIMIT 5",
                (user_id, conversation_id)
            )
            for row in cursor.fetchall():
                locations.append(row[0])
            conn.close()
        except:
            locations = ["the mansion", "the garden", "the city center", "the academy", "the private quarters"]
        
        # Format relationships for context display
        relationship_text = []
        for rel in relationships:
            entity_type = rel.get("entity_type", "unknown")
            entity_id = rel.get("entity_id", "unknown")
            rel_label = rel.get("relationship_label", "associate")
            
            # Get target name if available
            target_name = "Chase" if entity_type == "player" else f"NPC #{entity_id}"
            if entity_type == "npc":
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT npc_name FROM NPCStats WHERE npc_id=%s AND user_id=%s AND conversation_id=%s",
                    (entity_id, user_id, conversation_id)
                )
                row = cursor.fetchone()
                if row:
                    target_name = row[0]
                conn.close()
                
            relationship_text.append(f"- {rel_label} to {target_name} ({entity_type} {entity_id})")
        
        relationship_context = "\n".join(relationship_text) if relationship_text else "No established relationships"
        
        prompt = f"""
Create {generic_memories_needed} vivid, detailed memories for {npc_name} to supplement the relationship-specific memories already generated.

ENVIRONMENT CONTEXT:
{environment_desc}

NPC INFORMATION:
- Name: {npc_name}
- Archetype: {archetype_summary}
- Dominance Level: {dominance}/100
- Personality: {npc_data.get('personality_traits', [])}
- Relationships:
{relationship_context}

KEY LOCATIONS:
{', '.join(locations)}

MEMORY REQUIREMENTS:
1. Focus on general experiences or group interactions not covered by relationship-specific memories
2. Each memory must be a SPECIFIC EVENT with concrete details - not vague impressions
3. Include vivid sensory details (sights, sounds, smells, textures)
4. Include precise emotional responses and internal thoughts
5. Set memories in specific locations from the environment
6. Include dialogue snippets with actual quoted speech
7. Write in first-person perspective from {npc_name}'s viewpoint
8. Each memory should be 3-5 sentences minimum with specific details

IMPORTANT THEME GUIDANCE:
* Include subtle hints of control dynamics without being overtly femdom
* Show instances where {npc_name} momentarily revealed her true nature before quickly masking it
* Show moments where {npc_name} tested boundaries or enjoyed having influence
* For memories involving other female characters, show complex dynamics of rivalry, respect or alliance

Return a valid JSON object with a single "memories" key containing an array of memory strings.
"""
        
        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.8,  # Higher temperature for creative memories
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            memories_json = response.choices[0].message.content
            data = safe_json_loads(memories_json)
            
            if data and "memories" in data and isinstance(data["memories"], list):
                generic_memories = data["memories"]
                if generic_memories and all(isinstance(m, str) for m in generic_memories):
                    all_memories.extend(generic_memories)
                    
        except Exception as e:
            logging.error(f"Error generating generic memories for {npc_name}: {e}")
    
    # If we still don't have enough memories, add fallbacks
    if len(all_memories) < 3:
        # Fallback: Generate basic memories
        personality = npc_data.get("personality_traits", [])
        personality_trait = random.choice(personality) if personality else "mysterious"
        
        # Get a location for context
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT location_name FROM Locations WHERE user_id=%s AND conversation_id=%s ORDER BY RANDOM() LIMIT 1",
            (user_id, conversation_id)
        )
        row = cursor.fetchone()
        location = row[0] if row else "this place"
        conn.close()
        
        # Add fallback memories
        fallback_memories = [
            f"I remember the first staff meeting after my promotion at {location}. The room was thick with tension as I took my seat at the head of the table, aware of the evaluating gazes of my former peers. 'Let's address the elephant in the room,' I said, deliberately making eye contact with each person present. 'Some of you may have concerns about my new position.' I allowed a brief silence to hang in the air, noting who fidgeted and who maintained composure. When I finally laid out my vision for the department, I could feel the shift in atmosphere—some leaning forward with interest, others sitting back with thinly veiled resistance. Those reactions told me exactly who would need special handling in the coming months.",
            
            f"The annual gala at {location} last winter revealed interesting dynamics among the social elite. I observed from the periphery as conversations ebbed and flowed, noting who deferred to whom and who commanded attention without effort. 'Fascinating how invisible the power structures are until you know what to look for,' commented Helena, appearing beside me with two champagne flutes. We clinked glasses in silent understanding; as fellow outsiders who had earned our places rather than inherited them, we recognized the game being played. Later, when I smoothly redirected a problematic conversation that might have embarrassed our host, Helena's approving glance confirmed my growing influence in circles that once would have excluded me entirely.",
            
            f"I still remember my first day at {location}, how overwhelmingly normal everything seemed despite my internal anxieties. The orientation was tedious, the paperwork endless, and my new colleagues politely distant. 'You'll find your rhythm soon enough,' the department head assured me during our brief meeting. I nodded and smiled appropriately, playing the role of the grateful new hire while carefully observing the office dynamics. It wasn't until weeks later that anyone would notice my quiet competence, how efficiently I had mapped the relationships and unspoken rules that governed daily operations. By then, I had already begun making myself subtly indispensable."
        ]
        
        # Add only as many as needed to reach 3 total memories
        memories_needed = max(0, 3 - len(all_memories))
        all_memories.extend(fallback_memories[:memories_needed])
    
    return all_memories

async def gpt_generate_affiliations(user_id, conversation_id, npc_data, environment_desc):
    """
    Generate affiliations for an NPC (clubs, teams, workplaces, social groups, etc.)
    Returns a list of affiliation strings.
    """
    npc_name = npc_data.get("npc_name", "Unknown NPC")
    archetype_summary = npc_data.get("archetype_summary", "")
    personality_traits = npc_data.get("personality_traits", [])
    hobbies = npc_data.get("hobbies", [])
    existing_affiliations = npc_data.get("affiliations", [])
    
    prompt = f"""
Generate 3-5 affiliations for {npc_name}, an NPC in this environment:
{environment_desc}

NPC Details:
- Archetype: {archetype_summary}
- Personality: {personality_traits}
- Hobbies: {hobbies}
- Any existing affiliations: {existing_affiliations}

Affiliations should be specific organizations, groups, or institutions that this character is connected to.
Examples include: schools, workplaces, clubs, teams, social circles, community organizations, political groups, etc.

Each affiliation should:
1. Be specific and named (e.g., "Oakwood High School Debate Team" not just "a debate team")
2. Match the character's archetype and personality
3. Reflect the environment setting
4. Include at least one professional/work affiliation
5. Include at least one social/hobby affiliation

Return a valid JSON object with a single "affiliations" key containing an array of affiliation strings.
"""
    
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        affiliations_json = response.choices[0].message.content
        data = safe_json_loads(affiliations_json)
        
        if data and "affiliations" in data and isinstance(data["affiliations"], list):
            affiliations = data["affiliations"]
            if affiliations and all(isinstance(a, str) for a in affiliations):
                # Combine with existing affiliations, removing duplicates
                combined = existing_affiliations.copy()
                for affiliation in affiliations:
                    if affiliation not in combined:
                        combined.append(affiliation)
                return combined
                
    except Exception as e:
        logging.error(f"Error generating affiliations for {npc_name}: {e}")
    
    # Fallback: Generate basic affiliations
    fallback_affiliations = [
        f"The {environment_desc.split()[0]} Community Association",
        f"Local Professional Network",
        f"{npc_name}'s Social Circle"
    ]
    
    # Combine with existing affiliations
    combined = existing_affiliations.copy()
    for affiliation in fallback_affiliations:
        if affiliation not in combined:
            combined.append(affiliation)
            
    return combined



async def gpt_generate_relationship_specific_memories(user_id, conversation_id, npc_data, target_data, relationship_type, environment_desc):
    """
    Generate memories specific to a particular relationship between two characters.
    This function generates memories tailored to the specific relationship type (mother, enemy, friend, etc.)
    
    Args:
        npc_data: Dict containing data about the NPC whose memories we're generating
        target_data: Dict containing data about the target (player or other NPC)
        relationship_type: String describing the relationship (mother, friend, enemy, etc.)
        environment_desc: String describing the environment
        
    Returns:
        List of memory strings specific to this relationship
    """
    npc_name = npc_data.get("npc_name", "Unknown NPC")
    target_name = target_data.get("npc_name", target_data.get("player_name", "Chase"))
    target_type = "player" if "player_name" in target_data else "npc"
    
    # Get other relevant data about both characters
    npc_archetype = npc_data.get("archetype_summary", "")
    npc_dominance = npc_data.get("dominance", 50)
    npc_personality = npc_data.get("personality_traits", [])
    
    target_dominance = target_data.get("dominance", 30)  # Default lower for player
    
    # Get locations for context
    locations = []
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT location_name FROM Locations WHERE user_id=%s AND conversation_id=%s LIMIT 5",
            (user_id, conversation_id)
        )
        for row in cursor.fetchall():
            locations.append(row[0])
        conn.close()
    except:
        locations = ["the mansion", "the garden", "the city center", "the academy", "the private quarters"]
    
    # Different dynamics based on relationship type
    relationship_context = ""
    memory_focus = ""
    
    # Family relationships
    if relationship_type.lower() in ["mother", "stepmother", "aunt", "older sister", "maternal"]:
        relationship_context = f"You are {target_name}'s {relationship_type}, with all the authority, care, and discipline that implies."
        memory_focus = f"""
- Focus on formative moments in {target_name}'s development
- Include instances of discipline/rules and their enforcement
- Show moments of both nurturing and setting boundaries
- Include family dynamics and traditions you established
- Show how you shaped {target_name}'s values and behavior"""
    
    # Antagonistic relationships
    elif relationship_type.lower() in ["enemy", "rival", "adversary", "competitor"]:
        relationship_context = f"You have a contentious {relationship_type} relationship with {target_name}."
        memory_focus = f"""
- Focus on competitions, confrontations, and challenges between you
- Include moments where you outmaneuvered or were bested by {target_name}
- Show the evolution of the rivalry and what sustains it
- Include respect mixed with antagonism
- Show moments of unexpected alliance despite the rivalry"""
    
    # Romantic/intimate relationships
    elif relationship_type.lower() in ["lover", "ex-girlfriend", "ex-wife", "partner"]:
        relationship_context = f"You have/had an intimate {relationship_type} relationship with {target_name}."
        memory_focus = f"""
- Focus on significant moments in your relationship
- Include both positive emotional connections and conflicts
- Show power dynamics within your relationship
- Include instances where you influenced {target_name}'s decisions
- Show how the relationship evolved or changed over time"""
    
    # Professional relationships
    elif relationship_type.lower() in ["boss", "supervisor", "teacher", "principal", "mentor", "professor"]:
        relationship_context = f"You are {target_name}'s {relationship_type}, with professional authority."
        memory_focus = f"""
- Focus on professional development and evaluations
- Include instances of giving assignments/tasks and reviewing performance
- Show moments of both praise and correction/critique
- Include times you used your authority in consequential ways
- Show how you influenced {target_name}'s career trajectory"""
    
    # Peer relationships
    elif relationship_type.lower() in ["friend", "colleague", "classmate", "teammate", "neighbor"]:
        relationship_context = f"You are {target_name}'s {relationship_type}, with a peer relationship."
        memory_focus = f"""
- Focus on shared experiences and activities
- Include both moments of support and disagreement
- Show subtle competitions or one-upmanship if appropriate
- Include instances where your influence affected {target_name}'s choices
- Show the balance of power in your supposedly equal relationship"""
    
    # Default/other relationships
    else:
        relationship_context = f"You have a {relationship_type} relationship with {target_name}."
        memory_focus = f"""
- Focus on defining moments in your relationship
- Include instances that established the current dynamic between you
- Show the power balance in your interactions
- Include memories that reveal your true feelings about {target_name}
- Show how your relationship has evolved"""
    
    # Special handling for NPC-NPC relationships where both might be dominants
    if target_type == "npc" and target_dominance > 50 and npc_dominance > 50:
        memory_focus += f"""
- Include instances of power struggles between two dominant personalities
- Show careful boundary negotiations between equals
- Include moments of reluctant respect
- Show territory/influence demarcation
- Include complex alliance/rivalry dynamics
- Show strategic cooperation despite competing interests"""
    
    # Now build the full prompt
    prompt = f"""
Generate 2-3 vivid, detailed memories for {npc_name} about their relationship with {target_name}.

RELATIONSHIP CONTEXT:
{relationship_context}

ENVIRONMENT:
{environment_desc}

CHARACTER DETAILS:
- {npc_name}: {npc_archetype}
- Personality: {npc_personality}

MEMORY REQUIREMENTS:
1. Each memory must be a SPECIFIC EVENT with concrete details - not vague impressions
2. Include sensory details and emotional responses
3. Include dialogue with actual quoted speech
4. Write in first-person perspective from {npc_name}'s viewpoint
5. Set memories in specific locations: {', '.join(locations)}
6. Each memory should be 3-5 sentences minimum with specific details

MEMORY FOCUS FOR THIS RELATIONSHIP:
{memory_focus}

BALANCE REQUIREMENTS:
- Include at least one positive and one challenging interaction
- Include subtle power dynamics appropriate to your relationship
- For any negative memories, include complex emotions rather than simple dislike

Return a valid JSON object with a single "memories" key containing an array of memory strings.
"""
    
    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.8,  # Higher temperature for creative memories
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        memories_json = response.choices[0].message.content
        data = safe_json_loads(memories_json)
        
        if data and "memories" in data and isinstance(data["memories"], list):
            memories = data["memories"]
            if memories and all(isinstance(m, str) for m in memories):
                return memories
                
    except Exception as e:
        logging.error(f"Error generating relationship-specific memories for {npc_name} and {target_name}: {e}")
    
    # Fallback: Generate basic relationship-specific memories
    location = random.choice(locations) if locations else "this place"
    
    # Family relationship fallback
    if relationship_type.lower() in ["mother", "stepmother", "aunt", "older sister"]:
        return [
            f"I remember when {target_name} was younger and had their first real failure at {location}. The disappointment was visible in their eyes, but I knew this was a teaching moment. 'Sometimes we need to fail to understand the value of success,' I told them, my hand firm but comforting on their shoulder. I could see the resistance at first—that stubborn set of the jaw they inherited from me—but slowly, understanding dawned. How they looked at me then, with a mixture of frustration and reluctant recognition, showed me they were growing in the way I had hoped.",
            
            f"There was that time {target_name} directly challenged my authority during the family gathering at {location}. The room fell silent as they questioned my decision in front of everyone. 'We'll discuss this privately,' I said, my voice quiet but leaving no room for argument. Later, behind closed doors, I explained that while I respected their perspective, certain boundaries existed for reasons beyond their understanding. 'Respect isn't blind obedience,' I told them, 'but there's a proper way to express disagreement.' The conversation ended with a new understanding between us—one that acknowledged their growing autonomy while reinforcing the hierarchy that remained."
        ]
    
    # Antagonistic relationship fallback
    elif relationship_type.lower() in ["enemy", "rival", "adversary"]:
        return [
            f"The competition at {location} last year still brings a smile to my face whenever I think of {target_name}'s expression when the results were announced. We'd been neck and neck throughout, neither giving an inch, the tension between us almost visible in the air. When my name was called instead of theirs, I made sure to catch their eye across the room. 'Better luck next time,' I mouthed, enjoying the flash of anger that crossed their face before they composed themselves. What {target_name} doesn't understand is that their opposition drives me to greater heights—I'd probably be half as accomplished without such a worthy adversary.",
            
            f"I didn't expect to find {target_name} at {location} during the crisis, much less working toward the same goal as me. 'Temporary alliance?' they suggested, extending their hand with obvious reluctance. Our eyes met, mutual distrust evident, but practicality won out. 'Until this is resolved,' I agreed, taking their hand briefly. Working side by side with my usual rival revealed surprising competencies I hadn't noticed before—they approached problems from angles I wouldn't consider. Not that I'd ever admit it to them, of course, but that day changed how I viewed our rivalry, adding a layer of respect beneath the competition."
        ]
    
    # Default/generic fallback
    else:
        return [
            f"I recall a particular conversation with {target_name} at {location} that revealed more than either of us had intended. The evening light was casting long shadows across the room as we discussed our views on obligation and choice. 'Some boundaries exist to be tested,' I suggested, watching their reaction carefully. Something flickered behind their eyes—recognition, perhaps, or wariness—before they offered a measured response about respecting certain lines. The subtle shift in their posture told me more than their words; they were revealing their limits while trying to appear unaffected. I filed away that information, knowing it might prove useful in understanding them better.",
            
            f"There was a misunderstanding between {target_name} and me at {location} that nearly fractured our {relationship_type} relationship. Tensions had been building for weeks over misaligned expectations, culminating in a heated exchange that left both of us saying things we might regret. 'That's not what I meant and you know it,' they insisted, frustration evident in their voice. I remember taking a deliberate breath before responding more calmly than I felt, choosing reconciliation over being right. The relief in their expression when I offered a compromise told me they valued our connection more than they'd been showing. Sometimes conflict reveals the true foundation of a relationship better than harmony ever could."
        ]


async def integrate_femdom_elements(npc_data):
    """
    Analyze NPC data and subtly integrate femdom elements based on dominance level.
    This doesn't add overt femdom content, but rather plants seeds through traits and tendencies.
    """
    dominance = npc_data.get("dominance", 50)
    cruelty = npc_data.get("cruelty", 30)
    npc_name = npc_data.get("npc_name", "Unknown")
    archetypes = [a.get("name", "") for a in npc_data.get("archetypes", [])]
    archetype_summary = npc_data.get("archetype_summary", "")
    
    # Don't add femdom tendencies if dominance is very low
    if dominance < 20:
        return npc_data
        
    # Determine femdom intensity based on dominance and cruelty
    femdom_intensity = (dominance + cruelty) / 2
    is_high_intensity = femdom_intensity > 70
    is_medium_intensity = 40 <= femdom_intensity <= 70
    
    # Copy existing traits
    personality_traits = npc_data.get("personality_traits", [])
    likes = npc_data.get("likes", [])
    dislikes = npc_data.get("dislikes", [])
    
    # Add subtle femdom personality traits based on intensity
    potential_traits = []
    
    if is_high_intensity:
        potential_traits += [
            "enjoys being obeyed",
            "naturally commanding",
            "good at reading weaknesses",
            "expects compliance",
            "subtle manipulator",
            "finds pleasure in control",
            "notices when others defer to her",
        ]
    elif is_medium_intensity:
        potential_traits += [
            "prefers making decisions",
            "naturally takes charge",
            "surprisingly assertive at times",
            "notices small power dynamics",
            "enjoys being respected",
            "secretly enjoys having influence",
            "feels comfortable setting rules"
        ]
    else:
        potential_traits += [
            "occasionally assertive",
            "selective with permissions",
            "expects politeness",
            "notices disrespect quickly",
            "sometimes tests boundaries",
            "appreciates deference"
        ]
    
    # Select 1-2 traits to add without being too obvious
    traits_to_add = random.sample(potential_traits, min(2, len(potential_traits)))
    for trait in traits_to_add:
        if trait not in personality_traits:
            personality_traits.append(trait)
    
    # Add subtle femdom likes/dislikes based on intensity
    potential_likes = []
    potential_dislikes = []
    
    if is_high_intensity:
        potential_likes += [
            "seeing others follow her lead",
            "making important decisions",
            "being the center of attention",
            "quiet acknowledgment of her authority",
            "setting clear expectations"
        ]
        potential_dislikes += [
            "being interrupted",
            "unexpected defiance",
            "having her judgment questioned",
            "people who overstep boundaries",
            "being ignored"
        ]
    elif is_medium_intensity:
        potential_likes += [
            "receiving prompt responses",
            "being asked for permission",
            "planning events for others",
            "mentoring those who listen well",
            "being consulted on decisions"
        ]
        potential_dislikes += [
            "tardiness",
            "people who speak over her",
            "having to repeat herself",
            "casual dismissals of her opinions"
        ]
    else:
        potential_likes += [
            "well-mannered individuals",
            "being respected in conversations",
            "when others remember her preferences",
            "thoughtful attentiveness"
        ]
        potential_dislikes += [
            "poor etiquette",
            "being contradicted publicly",
            "presumptuous behavior"
        ]
    
    # Add 1 like and 1 dislike
    if potential_likes:
        new_like = random.choice(potential_likes)
        if new_like not in likes:
            likes.append(new_like)
            
    if potential_dislikes:
        new_dislike = random.choice(potential_dislikes)
        if new_dislike not in dislikes:
            dislikes.append(new_dislike)
    
    # Update the data and return
    updated_data = npc_data.copy()
    updated_data["personality_traits"] = personality_traits
    updated_data["likes"] = likes
    updated_data["dislikes"] = dislikes
    
    return updated_data

# -------------------------------------------------------------------------
# ENHANCED NPC CREATION AND REFINEMENT
# -------------------------------------------------------------------------

async def create_and_refine_npc(user_id, conversation_id, environment_desc, day_names, sex="female"):
    """
    Comprehensive function that creates and refines an NPC using the same process 
    for both new game and mid-game NPCs.
    
    Returns the NPC ID of the created NPC.
    """
    logging.info(f"Creating and refining NPC for user {user_id} in conversation {conversation_id}")
    
    # Step 1: Create the partial NPC (base data)
    from logic.npc_creation import create_npc_partial, insert_npc_stub_into_db
    partial_npc = create_npc_partial(
        user_id=user_id,
        conversation_id=conversation_id,
        sex=sex,
        total_archetypes=4,
        environment_desc=environment_desc
    )
    
    # Step 1.5: Integrate subtle femdom elements based on dominance
    partial_npc = await integrate_femdom_elements(partial_npc)
    
    # Step 2: Insert the partial NPC into the database
    npc_id = await insert_npc_stub_into_db(partial_npc, user_id, conversation_id)
    logging.info(f"Created NPC stub with ID {npc_id} and name {partial_npc['npc_name']}")
    
    # Step 3: Assign relationships
    from logic.npc_creation import assign_random_relationships
    await assign_random_relationships(
        user_id=user_id,
        conversation_id=conversation_id,
        new_npc_id=npc_id,
        new_npc_name=partial_npc["npc_name"],
        npc_archetypes=partial_npc.get("archetypes", [])
    )
    logging.info(f"Assigned relationships for NPC {npc_id}")
    
    # Step 4: Get relationships for memory generation
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT relationships FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
    row = cursor.fetchone()
    if row and row[0]:
        if isinstance(row[0], str):
            relationships = json.loads(row[0])
        else:
            relationships = row[0]
    else:
        relationships = []
    conn.close()

    # Step 5: Generate enhanced fields using GPT
    physical_description = await gpt_generate_physical_description(user_id, conversation_id, partial_npc, environment_desc)
    schedule = await gpt_generate_schedule(user_id, conversation_id, partial_npc, environment_desc, day_names)
    memories = await gpt_generate_memories(user_id, conversation_id, partial_npc, environment_desc, relationships)
    affiliations = await gpt_generate_affiliations(user_id, conversation_id, partial_npc, environment_desc)
    
    # Step 6: Determine current location based on time of day and schedule
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT value FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s AND key='TimeOfDay'",
        (user_id, conversation_id)
    )
    time_of_day = cursor.fetchone()
    time_of_day = time_of_day[0] if time_of_day else "Morning"
    
    cursor.execute(
        "SELECT value FROM CurrentRoleplay WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'",
        (user_id, conversation_id)
    )
    current_day_num = cursor.fetchone()
    current_day_num = int(current_day_num[0]) if current_day_num and current_day_num[0].isdigit() else 1
    
    # Calculate day index
    day_index = (current_day_num - 1) % len(day_names)
    current_day = day_names[day_index]
    
    # Extract current location from schedule
    current_location = "Unknown"
    if schedule and current_day in schedule and time_of_day in schedule[current_day]:
        activity = schedule[current_day][time_of_day]
        # Extract location from activity description
        location_keywords = ["at the", "in the", "at", "in"]
        for keyword in location_keywords:
            if keyword in activity:
                parts = activity.split(keyword, 1)
                if len(parts) > 1:
                    potential_location = parts[1].split(".")[0].split(",")[0].strip()
                    if len(potential_location) > 3:  # Avoid very short fragments
                        current_location = potential_location
                        break
    
    # If we couldn't extract a location, use a random location from the database
    if current_location == "Unknown":
        cursor.execute(
            "SELECT location_name FROM Locations WHERE user_id=%s AND conversation_id=%s ORDER BY RANDOM() LIMIT 1",
            (user_id, conversation_id)
        )
        random_location = cursor.fetchone()
        if random_location:
            current_location = random_location[0]
    
    conn.close()
    
    # Step 7: Update the NPC with all refined data
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE NPCStats 
        SET physical_description=%s,
            schedule=%s,
            memory=%s,
            current_location=%s,
            affiliations=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (
        physical_description,
        json.dumps(schedule),
        json.dumps(memories),
        current_location,
        json.dumps(affiliations),
        user_id, conversation_id, npc_id
    ))
    conn.commit()
    conn.close()
    
    logging.info(f"Successfully refined NPC {npc_id} ({partial_npc['npc_name']})")
    
    # Step 8: Propagate memories to other connected NPCs
    from logic.npc_creation import propagate_shared_memories
    
    propagate_shared_memories(
        user_id=user_id,
        conversation_id=conversation_id,
        source_npc_id=npc_id,
        source_npc_name=partial_npc["npc_name"],
        memories=memories
    )
    
    return npc_id

async def spawn_multiple_npcs_enhanced(user_id, conversation_id, environment_desc, day_names, count=3):
    """
    Create multiple NPCs using the enhanced process.
    Returns a list of NPC IDs.
    """
    npc_ids = []
    for i in range(count):
        npc_id = await create_and_refine_npc(
            user_id=user_id,
            conversation_id=conversation_id,
            environment_desc=environment_desc,
            day_names=day_names
        )
        npc_ids.append(npc_id)
        # Add a small delay to avoid rate limits
        await asyncio.sleep(0.5)
    
    return npc_ids

# -------------------------------------------------------------------------
# LEGACY FUNCTIONS (for backward compatibility)
# -------------------------------------------------------------------------

async def spawn_single_npc(user_id, conversation_id, environment_desc, day_names):
    """
    Legacy function for backward compatibility. 
    Now redirects to create_and_refine_npc.
    """
    logging.info(f"spawn_single_npc called - redirecting to create_and_refine_npc")
    return await create_and_refine_npc(user_id, conversation_id, environment_desc, day_names)

async def spawn_multiple_npcs(user_id, conversation_id, environment_desc, day_names, count=3):
    """
    Legacy function for backward compatibility.
    Now redirects to spawn_multiple_npcs_enhanced.
    """
    logging.info(f"spawn_multiple_npcs called - redirecting to spawn_multiple_npcs_enhanced")
    return await spawn_multiple_npcs_enhanced(user_id, conversation_id, environment_desc, day_names, count)

async def init_chase_schedule(user_id, conversation_id, combined_env, day_names):
    """
    Generate and store Chase's schedule.
    """
    from game_processing import async_process_new_game
    
    # Create a basic schedule structure
    default_schedule = {}
    for day in day_names:
        default_schedule[day] = {
            "Morning": f"Chase wakes up and prepares for the day",
            "Afternoon": f"Chase attends to their responsibilities",
            "Evening": f"Chase spends time on personal activities",
            "Night": f"Chase returns home and rests"
        }
    
    # Store in database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
        VALUES (%s, %s, 'ChaseSchedule', %s)
        ON CONFLICT (user_id, conversation_id, key)
        DO UPDATE SET value=EXCLUDED.value
    """, (user_id, conversation_id, json.dumps(default_schedule)))
    conn.commit()
    conn.close()
    
    return default_schedule
# -------------------------------------------------------------------------
# HELPERS FOR ROBUST JSON PARSING
# -------------------------------------------------------------------------

def safe_json_loads(text, default=None):
    """Safely parse JSON with multiple fallback methods."""
    if not text:
        return default if default is not None else {}
    
    # Method 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Look for JSON object within text
    try:
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json.loads(json_match.group(1))
    except json.JSONDecodeError:
        pass
    
    # Method 3: Try to fix common JSON syntax errors
    try:
        # Replace single quotes with double quotes
        fixed_text = text.replace("'", '"')
        return json.loads(fixed_text)
    except json.JSONDecodeError:
        pass
    
    # Method 4: Extract field from markdown code block
    try:
        code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if code_block_match:
            return json.loads(code_block_match.group(1))
    except json.JSONDecodeError:
        pass
    
    # Return default if all parsing attempts fail
    return default if default is not None else {}

def extract_field_from_text(text, field_name):
    """
    Extract a specific field from text that might contain JSON or key-value patterns.
    Returns the field value or empty string if not found.
    """
    # Try parsing as JSON first
    data = safe_json_loads(text)
    if data and field_name in data:
        return data[field_name]
    
    # Try regex patterns for field extraction
    patterns = [
        rf'"{field_name}"\s*:\s*"([^"]*)"',      # For string values: "field": "value"
        rf'"{field_name}"\s*:\s*(\[[^\]]*\])',     # For array values: "field": [...]
        rf'"{field_name}"\s*:\s*(\{{[^}}]*\}})',  # For object values: "field": {...}
        rf'{field_name}:\s*(.*?)(?:\n|$)',          # For plain text: field: value
    ]

    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return ""

def enforce_correct_npc_id(gpt_id: int, correct_id: int, context_str: str) -> int:
    """
    If GPT provided 'gpt_id' is not the same as the 'correct_id',
    log a warning and override with correct_id.
    'context_str' is e.g. 'npc_updates' or 'relationship_updates' so we know where it happened.
    """
    if gpt_id is not None and gpt_id != correct_id:
        logging.warning(
            f"[{context_str}] GPT provided npc_id={gpt_id}, but we are using npc_id={correct_id} => overriding."
        )
        return correct_id
    return correct_id  # or gpt_id if it matches

# -------------------------------------------------------------------------
# FILE PATHS & DATA LOADING
# -------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FILES = {
    "hobbies": os.path.join(current_dir, "..", "data", "npc_hobbies.json"),
    "likes": os.path.join(current_dir, "..", "data", "npc_likes.json"),
    "dislikes": os.path.join(current_dir, "..", "data", "npc_dislikes.json"),
    "personalities": os.path.join(current_dir, "..", "data", "npc_personalities.json"),
    "archetypes": os.path.join(current_dir, "..", "data", "archetypes_data.json")
}

def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load data from {path}: {e}")
        return {}

DATA = {
    "hobbies_pool": [],
    "likes_pool": [],
    "dislikes_pool": [],
    "personality_pool": [],
    "archetypes_table": []
}

def init_data():
    """Load all local JSON data into the DATA dictionary for reuse."""
    hobbies_json = load_json_file(DATA_FILES["hobbies"])
    DATA["hobbies_pool"] = hobbies_json.get("hobbies_pool", [])

    likes_json = load_json_file(DATA_FILES["likes"])
    DATA["likes_pool"] = likes_json.get("npc_likes", [])

    dislikes_json = load_json_file(DATA_FILES["dislikes"])
    DATA["dislikes_pool"] = dislikes_json.get("dislikes_pool", [])

    personalities_json = load_json_file(DATA_FILES["personalities"])
    DATA["personality_pool"] = personalities_json.get("personality_pool", [])

    arcs_json = load_json_file(DATA_FILES["archetypes"])
    table = arcs_json.get("archetypes", [])

    arcs_list = []
    for item in table:
        arcs_list.append({
            "name": item["name"],
            "baseline_stats": item.get("baseline_stats", {}),
            "progression_rules": item.get("progression_rules", []),
            "unique_traits": item.get("unique_traits", []),
            "preferred_kinks": item.get("preferred_kinks", [])
        })
    DATA["archetypes_table"] = arcs_list

init_data()

# -------------------------------------------------------------------------
# ARCHETYPE SELECTION
# -------------------------------------------------------------------------

def pick_with_reroll_replacement(n=3):
    """
    Pick n archetypes from the entire table. If any is a 'placeholder' (i.e. name includes "Add an extra modifier"),
    replace it with a real pick plus add an extra real archetype.
    """
    all_arcs = DATA["archetypes_table"]
    placeholders = [a for a in all_arcs if "Add an extra modifier" in a["name"]]
    reals = [a for a in all_arcs if "Add an extra modifier" not in a["name"]]

    chosen = random.sample(all_arcs, n)  # from entire set

    final_list = []
    for arc in chosen:
        if arc in placeholders:
            real_pick = random.choice(reals)
            final_list.append(real_pick)
            extra_pick = random.choice(reals)
            final_list.append(extra_pick)
        else:
            final_list.append(arc)

    return final_list

# -------------------------------------------------------------------------
# STAT CALCULATION
# -------------------------------------------------------------------------

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def combine_archetype_stats(archetype_list):
    """
    Combine baseline stats from each archetype, then average + clamp.
    """
    sums = {
        "dominance": 0,
        "cruelty": 0,
        "closeness": 0,
        "trust": 0,
        "respect": 0,
        "intensity": 0
    }
    count = len(archetype_list)
    if count == 0:
        for k in sums:
            sums[k] = random.randint(0, 30)
        return sums

    for arc in archetype_list:
        bs = arc.get("baseline_stats", {})
        for stat_key in sums:
            rng_key = f"{stat_key}_range"
            mod_key = f"{stat_key}_modifier"
            if rng_key in bs and mod_key in bs:
                low, high = bs[rng_key]
                mod = bs[mod_key]
                val = random.randint(low, high) + mod
            else:
                val = random.randint(0, 30)
            sums[stat_key] += val

    for sk in sums:
        sums[sk] = sums[sk] / count
        if sk in ["trust", "respect"]:
            sums[sk] = clamp(int(sums[sk]), -100, 100)
        else:
            sums[sk] = clamp(int(sums[sk]), 0, 100)
    return sums

# -------------------------------------------------------------------------
# GPT CALLS FOR NPC SYNERGY DESCRIPTIONS
# -------------------------------------------------------------------------

existing_npc_names = set()

def get_unique_npc_name(proposed_name: str) -> str:
    # If the name is "Seraphina" or already exists, choose an alternative from a predefined list.
    unwanted_names = {"seraphina"}
    if proposed_name.strip().lower() in unwanted_names or proposed_name in existing_npc_names:
        alternatives = ["Aurora", "Celeste", "Luna", "Nova", "Ivy", "Evelyn", "Isolde", "Marina"]
        # Filter out any alternatives already in use
        available = [name for name in alternatives if name not in existing_npc_names and name.lower() not in unwanted_names]
        if available:
            new_name = random.choice(available)
        else:
            # If none available, simply append a random number
            new_name = f"{proposed_name}{random.randint(2, 99)}"
        return new_name
    return proposed_name

def get_archetype_synergy_description(archetypes_list, provided_npc_name=None):
    if not archetypes_list:
        default_name = provided_npc_name or f"NPC_{random.randint(1000,9999)}"
        logging.info("[get_archetype_synergy_description] No archetypes => using placeholder name.")
        return json.dumps({
            "npc_name": default_name,
            "archetype_summary": "No special archetype synergy."
        })

    archetype_names = [a["name"] for a in archetypes_list]

    if provided_npc_name:
        name_instruction = f'Use the provided NPC name: "{provided_npc_name}".'
    else:
        name_instruction = (
                "Generate a creative, extremely unique, and varied feminine name for the NPC. "
                "The name must be unmistakably feminine and should not be a common or overused name. "
                "Invent a name that is entirely original, unexpected, and richly evocative of a fantastical, mythological, or diverse cultural background that makes sense for the setting and the character (Eg., Isis, Artemis, Megan, Thoth, Cassandra, Mizuki, etc.). "
                "Ensure that the name is unlike any generated previously in this playthrough. "
        )

    system_prompt = (
        "You are an expert at merging multiple archetypes into a single cohesive persona for a female NPC. "
        "Output strictly valid JSON with exactly these two keys:\n"
        '  "npc_name" (string)\n'
        '  "archetype_summary" (string)\n'
        "No additional keys, no extra commentary, no markdown fences.\n\n"
        "If you cannot comply, output an empty JSON object {}.\n\n"
        f"Archetypes to merge: {', '.join(archetype_names)}\n"
        f"{name_instruction}\n"
    )

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7
        )
        synergy_raw = resp.choices[0].message.content.strip()
        logging.info(f"[get_archetype_synergy_description] Raw synergy GPT output => {synergy_raw!r}")

        # Strip code fences if present
        if synergy_raw.startswith("```"):
            lines = synergy_raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            if lines and lines[-1].startswith("```"):
                lines.pop()
            synergy_raw = "\n".join(lines).strip()

        # Attempt to parse JSON
        synergy_data = json.loads(synergy_raw)

        # Validate the essential keys are present
        if not isinstance(synergy_data, dict):
            logging.warning("[get_archetype_synergy_description] synergy_data is not a dict—falling back.")
            return "{}"
        if "npc_name" not in synergy_data or "archetype_summary" not in synergy_data:
            logging.warning("[get_archetype_synergy_description] synergy_data missing required keys—falling back.")
            return "{}"

        # Post-processing: Check for masculine markers in the name.
        npc_name = synergy_data["npc_name"]
        masculine_markers = ["Prince", "Lord", "Sir", "Eduard", "William", "John"]
        if any(marker in npc_name for marker in masculine_markers):
            logging.info("Masculine markers detected in NPC name; replacing with fallback feminine name.")
            synergy_data["npc_name"] = "Lady Celestine"

        # Post-processing: Ensure the name is unique and not overused.
        original_name = synergy_data["npc_name"].strip()
        unique_name = get_unique_npc_name(original_name)
        if unique_name != original_name:
            logging.info(f"Name '{original_name}' replaced with unique name '{unique_name}'.")
        synergy_data["npc_name"] = unique_name
        existing_npc_names.add(unique_name)

        return json.dumps(synergy_data, ensure_ascii=False)

    except Exception as e:
        logging.warning(f"[get_archetype_synergy_description] parse or GPT error: {e}")
        return "{}"

def get_archetype_extras_summary_gpt(archetypes_list, npc_name):
    """
    Calls GPT to merge each archetype's progression_rules, unique_traits, and preferred_kinks
    into one cohesive textual summary. Returns a string stored as "archetype_extras_summary".

    The final JSON must have exactly 1 key: "archetype_extras_summary".
    If GPT can't provide it, fallback to a minimal text.
    """
    import json

    if not archetypes_list:
        return "No extras summary available."

    # Gather the data from each archetype
    lines = []
    for arc in archetypes_list:
        name = arc["name"]
        # These fields might be missing or empty, so we guard with .get(...)
        progression = arc.get("progression_rules", [])
        traits = arc.get("unique_traits", [])
        kinks = arc.get("preferred_kinks", [])
        lines.append(
            f"Archetype: {name}\n"
            f"  progression_rules: {progression}\n"
            f"  unique_traits: {traits}\n"
            f"  preferred_kinks: {kinks}\n"
        )

    combined_text = "\n".join(lines)

    # We'll build a system prompt that clarifies the format we want:
    system_prompt = f"""
You are merging multiple archetype 'extras' for a female NPC named '{npc_name}'.
Below are the extras from each archetype:

{combined_text}

We want to unify these details (progression_rules, unique_traits, preferred_kinks)
into one cohesive textual summary that references how these merges shape the NPC's special powers,
quirks, or flavor.

Return **strictly valid JSON** with exactly 1 top-level key:
  "archetype_extras_summary"

The value should be a single string that concisely describes how these rules/traits/kinks
blend into one cohesive set of extras for this NPC.

No extra commentary, no additional keys, no code fences.
If you cannot comply, return an empty JSON object {{}}.
"""

    # Next, we do a GPT call with system_prompt
    client = get_openai_client()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=400
        )
        raw_text = resp.choices[0].message.content.strip()
        logging.info(f"[get_archetype_extras_summary_gpt] raw GPT extras => {raw_text!r}")

        # Remove fences if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            # remove first triple-backticks
            if lines and lines[0].startswith("```"):
                lines.pop(0)
            # remove last triple-backticks
            if lines and lines[-1].startswith("```"):
                lines.pop()
            raw_text = "\n".join(lines).strip()

        parsed = json.loads(raw_text)
        if (
            isinstance(parsed, dict)
            and "archetype_extras_summary" in parsed
        ):
            # Valid
            return parsed["archetype_extras_summary"]
        else:
            logging.warning("[get_archetype_extras_summary_gpt] Missing 'archetype_extras_summary' key, falling back.")
            return "No extras summary available."

    except Exception as e:
        logging.warning(f"[get_archetype_extras_summary_gpt] error => {e}")
        return "No extras summary available."

# -------------------------------------------------------------------------
# ADAPT LISTS FOR ENVIRONMENT
# -------------------------------------------------------------------------

def adapt_list_for_environment(environment_desc, archetype_summary, original_list, list_type="likes"):
    """
    Calls GPT to adapt each item in 'original_list' so it fits better with the environment
    and the NPC's archetype_summary. 'list_type' can be "likes", "dislikes", or "hobbies"
    to let GPT know how to adapt them.

    Returns a *new* list (strings).
    """
    import json
    if not original_list:
        return original_list

    system_prompt = f"""
Environment:
{environment_desc}

NPC's archetype summary:
{archetype_summary}

Original {list_type} list:
{original_list}

Please transform each item so it fits more cohesively with the environment's theme and the NPC's archetype,
retaining a similar 'topic' or 'concept' but making it more in-universe.

Output strictly valid JSON: a single array of strings, with no extra commentary or keys.
"""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.7,
            max_tokens=300
        )
        raw_text = resp.choices[0].message.content.strip()

        # remove triple-backticks if present
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        new_list = json.loads(raw_text)
        if isinstance(new_list, list) and all(isinstance(x, str) for x in new_list):
            return new_list
        else:
            logging.warning(f"[adapt_list_for_environment] GPT returned something not a list of strings, fallback to original.")
            return original_list
    except Exception as e:
        logging.warning(f"[adapt_list_for_environment] GPT error => {e}")
        return original_list

# -------------------------------------------------------------------------
# PARTIAL NPC CREATION
# -------------------------------------------------------------------------

def create_npc_partial(user_id: int, conversation_id: int, sex: str = "female",
                       total_archetypes: int = 4, environment_desc: str = "A default environment") -> dict:
    import random
    calendar_data = load_calendar_names(user_id, conversation_id)
    months_list = calendar_data.get("months", [])
    if len(months_list) < 12:
        months_list = [
            "Frostmoon", "Windspeak", "Bloomrise", "Dawnsveil",
            "Emberlight", "Goldencrest", "Shadowleaf", "Harvesttide",
            "Stormcall", "Nightwhisper", "Snowbound", "Yearsend"
        ]
    if sex.lower() == "male":
        final_stats = {
            "dominance":  random.randint(0, 30),
            "cruelty":    random.randint(0, 30),
            "closeness":  random.randint(0, 30),
            "trust":      random.randint(-30, 30),
            "respect":    random.randint(-30, 30),
            "intensity":  random.randint(0, 30)
        }
        chosen_arcs = []
    else:
        chosen_arcs = pick_with_reroll_replacement(total_archetypes)
        final_stats = combine_archetype_stats(chosen_arcs)
    
    synergy_str = get_archetype_synergy_description(chosen_arcs, None)
    logging.info(f"[create_npc_partial] synergy_str (raw) => {synergy_str!r}")
    try:
        synergy_data = json.loads(synergy_str)
        synergy_name = synergy_data.get("npc_name") or f"NPC_{random.randint(1000,9999)}"
        synergy_text = synergy_data.get("archetype_summary") or "No synergy text"
    except json.JSONDecodeError as e:
        logging.warning(f"[create_npc_partial] synergy parse error => {e}")
        synergy_name = f"NPC_{random.randint(1000,9999)}"
        synergy_text = "No synergy text"

    # 4) extras
    extras_text = get_archetype_extras_summary_gpt(chosen_arcs, synergy_name)
    arcs_for_json = [{"name": arc["name"]} for arc in chosen_arcs]
    hpool = DATA["hobbies_pool"]
    lpool = DATA["likes_pool"]
    dpool = DATA["dislikes_pool"]
    tmp_hobbies  = random.sample(hpool, min(3, len(hpool)))
    tmp_likes    = random.sample(lpool, min(3, len(lpool)))
    tmp_dislikes = random.sample(dpool, min(3, len(dpool)))
    adapted_hobbies  = adapt_list_for_environment(environment_desc, synergy_text, tmp_hobbies, "hobbies")
    adapted_likes    = adapt_list_for_environment(environment_desc, synergy_text, tmp_likes, "likes")
    adapted_dislikes = adapt_list_for_environment(environment_desc, synergy_text, tmp_dislikes, "dislikes")

    # 7) Age + birthdate
    # Define age adjustments (role: (modifier_min, modifier_max))
    role_base_age_ranges = {
        "mother": (30, 55),
        "stepmother": (18, 55),
        "aunt": (25, 60),
        "older sister": (19, 40),
        "stepsister": (18, 45),
        "babysitter": (20, 50),
        "teacher": (30, 50),
        "principal": (30, 50),
        "milf": (30, 60),
        "dowager": (55, 65),
        "domestic authority": (30, 50),
        "foreign royalty": (20, 45),
        "college student": (18, 24),
        "intern": (18, 24),
        "student": (18, 24),
        "manic pixie dream girl": (18, 30)
    }
    
    # Check chosen archetypes for any familial roles
    familial_roles_found = [
        arc["name"].strip().lower() 
        for arc in chosen_arcs 
        if arc["name"].strip().lower() in role_base_age_ranges
    ]
    
    if familial_roles_found:
        # If more than one family role is found, choose the one with the highest minimum age
        selected_role = max(familial_roles_found, key=lambda role: role_base_age_ranges[role][0])
        base_age = random.randint(*role_base_age_ranges[selected_role])
    else:
        base_age = random.randint(20, 50)
    
    # Optionally, you could add a minor random offset if desired:
    # extra_years = random.randint(-2, 2)
    # npc_age = base_age + extra_years
    
    npc_age = base_age  # Now, npc_age is drawn from a role-appropriate range
    
    birth_month = random.choice(months_list)
    birth_day   = random.randint(1, 28)
    birth_str   = f"{birth_month} {birth_day}"

    npc_dict = {
        "npc_name": synergy_name,
        "introduced": False,
        "sex": sex.lower(),
        "dominance": final_stats["dominance"],
        "cruelty": final_stats["cruelty"],
        "closeness": final_stats["closeness"],
        "trust": final_stats["trust"],
        "respect": final_stats["respect"],
        "intensity": final_stats["intensity"],
        "archetypes": arcs_for_json,
        "archetype_summary": synergy_text,
        "archetype_extras_summary": extras_text,
        "hobbies": adapted_hobbies,
        "personality_traits": random.sample(DATA["personality_pool"], min(3, len(DATA["personality_pool"]))),
        "likes": adapted_likes,
        "dislikes": adapted_dislikes,
        "age": npc_age,
        "birthdate": birth_str
    }
    logging.info(
        "[create_npc_partial] Created partial NPC => "
        f"name='{npc_dict['npc_name']}', arcs={[arc['name'] for arc in chosen_arcs]}, "
        f"archetype_summary='{npc_dict['archetype_summary']}', "
        f"birthdate={npc_dict['birthdate']}, age={npc_age}"
    )
    return npc_dict

# -------------------------------------------------------------------------
# DB Insert
# -------------------------------------------------------------------------
async def insert_npc_stub_into_db(partial_npc: dict, user_id: int, conversation_id: int) -> int:
    """
    Insert the partial_npc data into NPCStats, returning the actual npc_id from the DB.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO NPCStats (
          user_id, conversation_id,
          npc_name, introduced, sex,
          dominance, cruelty, closeness, trust, respect, intensity,
          archetypes, archetype_summary, archetype_extras_summary,
          likes, dislikes, hobbies, personality_traits,
          age, birthdate,
          relationships, memory, schedule,
          physical_description
        )
        VALUES (%s, %s,
                %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                '[]'::jsonb, '[]'::jsonb, '{}'::jsonb,
                ''
        )
        RETURNING npc_id
        """,
        (
            user_id, conversation_id,
            partial_npc["npc_name"],
            partial_npc.get("introduced", False),
            partial_npc["sex"],

            partial_npc["dominance"],
            partial_npc["cruelty"],
            partial_npc["closeness"],
            partial_npc["trust"],
            partial_npc["respect"],
            partial_npc["intensity"],

            json.dumps(partial_npc["archetypes"]),
            partial_npc.get("archetype_summary", ""),
            partial_npc.get("archetype_extras_summary", ""),

            json.dumps(partial_npc.get("likes", [])),
            json.dumps(partial_npc.get("dislikes", [])),
            json.dumps(partial_npc.get("hobbies", [])),
            json.dumps(partial_npc.get("personality_traits", [])),

            partial_npc.get("age", 25),
            partial_npc.get("birthdate", ""),
        )
    )
    row = cur.fetchone()
    npc_id = row[0]
    conn.commit()
    conn.close()

    logging.info(f"[insert_npc_stub_into_db] Inserted NPC => assigned npc_id={npc_id}")
    return npc_id

# -------------------------------------------------------------------------
# RELATIONSHIP HELPERS
# -------------------------------------------------------------------------

def dynamic_reciprocal_relationship(rel_type: str, archetype_summary: str = "") -> str:
    """
    Given a relationship label (e.g., "thrall", "underling", "friend"),
    return a reciprocal label in a dynamic and context‐sensitive way.
    Fixed family relationships use fixed mappings; for "friend" or "best friend" the reciprocal is identical.
    For other types, we pick from a list, possibly influenced by keywords in the archetype summary.
    """
    fixed = {
        "mother": "child",
        "sister": "younger sibling",
        "aunt": "nephew/niece"
    }
    rel_lower = rel_type.lower()
    if rel_lower in fixed:
        return fixed[rel_lower]
    if rel_lower in ["friend", "best friend"]:
        return rel_type  # mutual relationship
    dynamic_options = {
        "underling": ["boss", "leader", "overseer"],
        "thrall": ["master", "controller", "dominator"],
        "enemy": ["rival", "adversary"],
        "lover": ["lover", "beloved"],
        "colleague": ["colleague"],
        "neighbor": ["neighbor"],
        "classmate": ["classmate"],
        "teammate": ["teammate"],
        "rival": ["rival", "competitor"],
    }
    if rel_lower in dynamic_options:
        if "dominant" in archetype_summary.lower() or "domina" in archetype_summary.lower():
            if rel_lower in ["underling", "thrall"]:
                return "boss"
        return random.choice(dynamic_options[rel_lower])
    return "associate"

def append_relationship_to_npc(
    user_id, conversation_id,
    npc_id,                     # Which NPC in NPCStats is being updated
    rel_label,                  # "thrall", "lover", etc.
    target_entity_type,         # "npc" or "player"
    target_entity_id            # The actual ID for the other side
):
    """
    Synchronously appends a relationship record (as JSON) into the 'relationships' column in NPCStats.
    Example record: {"relationship_label": "thrall", "with_npc_id": 1234}
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT relationships FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
    if row:
        try:
            rel_data = row[0] or "[]"
            if isinstance(rel_data, str):
                rel_list = json.loads(rel_data)
            else:
                rel_list = rel_data
        except Exception as e:
            logging.warning(f"[append_relationship_to_npc] JSON parse error: {e}")
            rel_list = []
    if not row:
        logging.warning(f"[append_relationship_to_npc] NPC {npc_id} not found.")
        conn.close()
        return
    new_record = {
        "relationship_label": rel_label,
        "entity_type": target_entity_type,
        "entity_id": target_entity_id
    }
    rel_list.append(new_record)
    updated = json.dumps(rel_list)
    cur.execute(
        "UPDATE NPCStats SET relationships = %s WHERE npc_id=%s AND user_id=%s AND conversation_id=%s",
        (updated, npc_id, user_id, conversation_id)
    )
    conn.commit()
    conn.close()
    logging.info(f"[append_relationship_to_npc] Added relationship '{rel_label}' -> {target_entity_id} for NPC {npc_id}.")

def recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id):
    """
    Re-fetch the NPC's archetypes from the DB and re-run combine_archetype_stats to update final stats.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
    if not row:
        logging.warning(f"No NPC found for npc_id={npc_id}, cannot recalc stats.")
        conn.close()
        return

    arcs_json = row[0] or "[]"
    try:
        arcs_list = json.loads(arcs_json)
    except:
        arcs_list = []

    # match by name
    chosen_arcs = []
    for arc_obj in arcs_list:
        a_name = arc_obj.get("name")
        found = None
        for cand in DATA["archetypes_table"]:
            if cand["name"] == a_name:
                found = cand
                break
        if found:
            chosen_arcs.append(found)
        else:
            chosen_arcs.append({"baseline_stats": {}})

    final_stats = combine_archetype_stats(chosen_arcs)

    cur.execute("""
        UPDATE NPCStats
        SET dominance=%s, cruelty=%s, closeness=%s, trust=%s, respect=%s, intensity=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (
        final_stats["dominance"],
        final_stats["cruelty"],
        final_stats["closeness"],
        final_stats["trust"],
        final_stats["respect"],
        final_stats["intensity"],
        user_id, conversation_id, npc_id
    ))
    conn.commit()
    conn.close()
    logging.info(f"[recalc_npc_stats_with_new_archetypes] updated => {final_stats} for npc_id={npc_id}.")

async def await_prompted_synergy_after_add_archetype(arcs_list, user_id, conversation_id, npc_id):
    """
    If we just added a new archetype, re-run synergy to incorporate it.
    arcs_list ~ [{"name":"Mother"}, ...]
    """
    archetype_names = [arc["name"] for arc in arcs_list]
    system_instructions = f"""
We just appended a new archetype to this NPC. Now they have: {', '.join(archetype_names)}.
Please provide an updated synergy summary, in JSON with key "archetype_summary".
No extra text or function calls.
"""
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_instructions}],
            temperature=0.7,
            max_tokens=200
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        data = json.loads(raw)
        return data.get("archetype_summary", "")
    except Exception as e:
        logging.warning(f"Error synergy after new arc: {e}")
        return "Could not update synergy"

async def add_archetype_to_npc(user_id, conversation_id, npc_id, new_arc):
    """
    Insert new_arc into the NPC's archetypes array, re-run synergy, recalc stats.
    We'll store only 'name' in the DB, ignoring GPT-provided numeric ID if any.
    """
    # If GPT gave us 'new_arc' that includes "npc_id" we might do:

    # new_arc["npc_id"] = enforce_correct_npc_id(
    #    new_arc.get("npc_id"), npc_id, "add_archetype_to_npc"
    # )

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT archetypes FROM NPCStats WHERE user_id=%s AND conversation_id=%s AND npc_id=%s LIMIT 1",
        (user_id, conversation_id, npc_id)
    )
    row = cur.fetchone()
    if not row:
        logging.warning(f"[add_archetype_to_npc] No NPCStats found for npc_id={npc_id}.")
        conn.close()
        return

    arcs_str = row[0] or "[]"
    try:
        existing_arcs = json.loads(arcs_str)
    except:
        existing_arcs = []

    # Only add if not present
    if any(a.get("name") == new_arc["name"] for a in existing_arcs):
        logging.info(f"NPC {npc_id} already has archetype '{new_arc['name']}'; skipping.")
        conn.close()
        return

    existing_arcs.append({"name": new_arc["name"]})
    new_arcs_json = json.dumps(existing_arcs)

    # synergy
    updated_synergy = await await_prompted_synergy_after_add_archetype(existing_arcs, user_id, conversation_id, npc_id)
    if not updated_synergy:
        updated_synergy = "No updated synergy"

    cur.execute("""
        UPDATE NPCStats
        SET archetypes=%s,
            archetype_summary=%s
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (new_arcs_json, updated_synergy, user_id, conversation_id, npc_id))
    conn.commit()
    conn.close()

    # recalc final stats
    recalc_npc_stats_with_new_archetypes(user_id, conversation_id, npc_id)
    logging.info(f"[add_archetype_to_npc] added '{new_arc['name']}' to npc_id={npc_id}.")

# -------------------------------------------------------------------------
# RELATIONSHIP ASSIGNMENTS
# -------------------------------------------------------------------------

async def assign_random_relationships(user_id, conversation_id, new_npc_id, new_npc_name, npc_archetypes=None):
    logging.info(f"[assign_random_relationships] Assigning relationships for NPC {new_npc_id} ({new_npc_name})")
    import random

    relationships = []

    # Define explicit mapping for archetypes to relationship labels.
    # This mapping covers both familial and non-familial roles.
    explicit_role_map = {
        "mother": "mother",
        "stepmother": "stepmother",
        "aunt": "aunt",
        "older sister": "older sister",
        "stepsister": "stepsister",
        "babysitter": "babysitter",
        "friend from online interactions": "online friend",
        "neighbor": "neighbor",
        "rival": "rival",
        "classmate": "classmate",
        "lover": "lover",
        "colleague": "colleague",
        "teammate": "teammate",
        "boss/supervisor": "boss/supervisor",
        "teacher/principal": "teacher/principal",
        "landlord": "landlord",
        "roommate/housemate": "roommate",
        "ex-girlfriend/ex-wife": "ex-partner",
        "therapist": "therapist",
        "domestic authority": "head of household",
        "the one who got away": "the one who got away",
        "childhood friend": "childhood friend",
        "friend's wife": "friend",
        "friend's girlfriend": "friend",
        "best friend's sister": "friend's sister"
    }
    
    # First, add relationships based on explicit archetype mapping.
    if npc_archetypes:
        for arc in npc_archetypes:
            arc_name = arc.get("name", "").strip().lower()
            if arc_name in explicit_role_map:
                rel_label = explicit_role_map[arc_name]
                # Add relationship from NPC to player using the explicit role.
                relationships.append({
                    "target_entity_type": "player",
                    "target_entity_id": user_id,  # player ID
                    "relationship_label": rel_label
                })
                logging.info(f"[assign_random_relationships] Added explicit relationship '{rel_label}' for NPC {new_npc_id} to player.")
    
    # Next, determine which explicit roles (if any) were already added.
    explicit_roles_added = {rel["relationship_label"] for rel in relationships}
    
    # Define default lists for random selection.
    default_familial = ["mother", "sister", "aunt"]
    default_non_familial = ["enemy", "friend", "best friend", "lover", "neighbor",
                              "colleague", "classmate", "teammate", "underling", "rival", "ex-girlfriend", "ex-wife", "boss", "roommate", "childhood friend"]
    
    # If no explicit familial role was added, consider assigning a random non-familial relationship with the player.
    if not (explicit_roles_added & set(default_familial)):
        if random.random() < 0.5:
            rel_type = random.choice(default_non_familial)
            relationships.append({
                "target_entity_type": "player",
                "target_entity_id": user_id,
                "relationship_label": rel_type
            })
            logging.info(f"[assign_random_relationships] Randomly added non-familial relationship '{rel_type}' for NPC {new_npc_id} to player.")
    
    # Now add relationships with other NPCs.
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_id, npc_name, archetype_summary
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
    """, (user_id, conversation_id, new_npc_id))
    rows = cursor.fetchall()
    conn.close()
    
    # For each other NPC, use explicit mapping if possible; otherwise, fall back to random choice.
    for (old_npc_id, old_npc_name, old_arche_summary) in rows:
        if random.random() < 0.3:
            # Check if the current NPC's explicit roles should be used.
            if explicit_roles_added:
                # Prefer one of the explicit roles if available.
                rel_type = random.choice(list(explicit_roles_added))
            else:
                rel_type = random.choice(default_non_familial)
            relationships.append({
                "target_entity_type": "npc",
                "target_entity_id": old_npc_id,
                "relationship_label": rel_type,
                "target_archetype_summary": old_arche_summary or ""
            })
            logging.info(f"[assign_random_relationships] Added relationship '{rel_type}' between NPC {new_npc_id} and NPC {old_npc_id}.")
    
    # Finally, create these relationships in the database and generate associated memories.
    for rel in relationships:
        memory_text = get_shared_memory(user_id, conversation_id, rel, new_npc_name)
        await add_npc_memory_with_embedding(
            npc_id=npc_id,
            memory_text=memory_text,
            tags=["some_tag"]  # optional
        )

        from logic.social_links import create_social_link
        from logic.npc_creation import dynamic_reciprocal_relationship

        if rel["target_entity_type"] == "player":
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc", entity1_id=new_npc_id,
                entity2_type="player", entity2_id=rel["target_entity_id"],
                link_type=rel["relationship_label"]
            )
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                new_npc_id, 
                rel["relationship_label"],
                "player", rel["target_entity_id"]
            )
        else:
            old_npc_id = rel["target_entity_id"]
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc", entity1_id=new_npc_id,
                entity2_type="npc", entity2_id=old_npc_id,
                link_type=rel["relationship_label"]
            )
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                new_npc_id,
                rel["relationship_label"],
                "npc", old_npc_id
            )
            rec_type = dynamic_reciprocal_relationship(
                rel["relationship_label"],
                rel.get("target_archetype_summary", "")
            )
            create_social_link(
                user_id, conversation_id,
                entity1_type="npc", entity1_id=old_npc_id,
                entity2_type="npc", entity2_id=new_npc_id,
                link_type=rec_type
            )
            await asyncio.to_thread(
                append_relationship_to_npc,
                user_id, conversation_id,
                old_npc_id,
                rec_type,
                "npc", new_npc_id
            )
    
    logging.info(f"[assign_random_relationships] Finished relationships for NPC {new_npc_id}.")

# -------------------------------------------------------------------------
# RELATIONSHIP ARCHETYPES
# -------------------------------------------------------------------------

RELATIONSHIP_ARCHETYPE_MAP = {
    # Family / Household
    "Mother":        {"name": "Child"},
    "Stepmother":    {"name": "Step-Child"},
    "Aunt":          {"name": "Niece/Nephew"},
    "Older Sister":  {"name": "Younger Sibling"},
    "Stepsister":    {"name": "Step-Sibling"},
    "Babysitter":    {"name": "Child"},

    # Workplace / Power
    "CEO":                {"name": "Employee"},
    "Boss/Supervisor":    {"name": "Employee"},
    "Corporate Dominator":{"name": "Underling"},
    "Teacher/Principal":  {"name": "Student"},
    "Landlord":           {"name": "Tenant"},
    "Warden":             {"name": "Prisoner"},
    "Loan Shark":         {"name": "Debtor"},
    "Slave Overseer":     {"name": "Slave"},
    "Therapist":          {"name": "Patient"},
    "Doctor":             {"name": "Patient"},
    "Social Media Influencer": {"name": "Follower"},
    "Bartender":          {"name": "Patron"},
    "Fitness Trainer":    {"name": "Client"},
    "Cheerleader/Team Captain": {"name": "Junior Team Member"},
    "Martial Artist":     {"name": "Sparring Dummy"},
    "Professional Wrestler": {"name": "Defeated Opponent"},

    # Supernatural / Hunting
    "Demon":              {"name": "Thrall"},
    "Demoness":           {"name": "Bound Mortal"},
    "Devil":              {"name": "Damned Soul"},
    "Villain (RPG-Esque)": {"name": "Captured Hero"},
    "Haunted Entity":     {"name": "Haunted Mortal"},
    "Sorceress":          {"name": "Cursed Subject"},
    "Witch":              {"name": "Hexed Victim"},
    "Eldritch Abomination":{"name": "Insane Acolyte"},
    "Primal Huntress":    {"name": "Prey"},
    "Primal Predator":    {"name": "Prey"},
    "Serial Killer":      {"name": "Victim"},

    # Others
    "Rockstar":           {"name": "Fan"},
    "Celebrity":          {"name": "Fan"},
    "Ex-Girlfriend/Ex-Wife": {"name": "Ex-Partner"},
    "Politician":         {"name": "Constituent"},
    "Queen":              {"name": "Subject"},
    "Empress":            {"name": "Subject"},
    "Royal Knight":       {"name": "Challenged Rival"},
    "Gladiator":          {"name": "Arena Opponent"},
    "Pirate":             {"name": "Captive"},
    "Bank Robber":        {"name": "Hostage"},
    "Cybercriminal":      {"name": "Hacked Victim"},
    "Huntress":           {"name": "Prey"}, 
    "Arsonist":           {"name": "Burned Victim"},
    "Drug Dealer":        {"name": "Addict"},
    "Artificial Intelligence": {"name": "User/Victim"},
    "Fey":                {"name": "Ensorcelled Mortal"},
    "Nun":                {"name": "Sinner"},
    "Priestess":          {"name": "Acolyte"},
    "A True Goddess":     {"name": "Worshipper"},
    "Haruhi Suzumiya-Type Goddess": {"name": "Reality Pawn"},
    "Bowsette Personality": {"name": "Castle Captive"},
    "Juri Han Personality": {"name": "Beaten Opponent"},
    "Neighbor":           {"name": "Targeted Neighbor"},
    "Hero (RPG-Esque)":   {"name": "Sidekick / Rescued Target"},
}

# -------------------------------------------------------------------------
# RELATIONSHIP GROUPS FOR PROPAGATION
# -------------------------------------------------------------------------

RELATIONSHIP_GROUPS = {
    # Family relationships:
    "family": {
        "mother": {
            "related": ["sister", "stepsister", "cousin"],  # Cousins can be related too
            "base": "mother",
            "reciprocal": "child"
        },
        "stepmother": {
            "related": ["sister", "stepsister", "cousin"],
            "base": "stepmother",
            "reciprocal": "child"
        },
        "aunt": {
            "related": ["cousin"],  # Optionally, cousins of an aunt could also be linked
            "base": "aunt",
            "reciprocal": "niece/nephew"
        },
        "cousin": {
            "related": ["cousin"],  # Cousins are symmetric
            "base": "cousin",
            "reciprocal": "cousin"
        }
    },
    # Work/Professional relationships:
    "work": {
        "boss": {
            "related": ["colleague"],
            "base": "boss",
            "reciprocal": "employee"
        }
    },
    # Team/Group relationships:
    "team": {
        "captain": {
            "related": ["teammate"],
            "base": "captain",
            "reciprocal": "team member"
        },
        "classmate": {
            "related": ["classmate"],
            "base": "classmate",
            "reciprocal": "classmate"  # symmetric
        }
    },
    # Neighborhood:
    "neighbors": {
        "neighbor": {
            "related": ["neighbor"],
            "base": "neighbor",
            "reciprocal": "neighbor"
        }
    }
}

# -------------------------------------------------------------------------
# PROPAGATION FUNCTIONS
# -------------------------------------------------------------------------

def propagate_relationships(user_id, conversation_id):
    """
    Scan all NPCs for relationships and, based on RELATIONSHIP_GROUPS,
    propagate additional links. For example, if NPC A is a mother to target X,
    and NPC B is a sister (or stepsister) of someone who also relates to X,
    then NPC B should also gain the mother relationship to X,
    and X should gain the reciprocal 'child' link if not already present.
    
    This logic is applied for each defined group (family, work, team, neighbors).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT npc_id, npc_name, relationships
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    rows = cur.fetchall()
    # Build a dictionary of npc data
    npc_dict = {}
    for npc_id, npc_name, rel_str in rows:
        try:
            rels = json.loads(rel_str) if rel_str else []
        except Exception:
            rels = []
        npc_dict[npc_id] = {"npc_name": npc_name, "relationships": rels}
    
    # For each propagation group:
    for group in RELATIONSHIP_GROUPS.values():
        for base_label, settings in group.items():
            # For every NPC that has a base relationship in this group...
            for npc_id, data in npc_dict.items():
                for rel in data["relationships"]:
                    # Compare case-insensitively
                    if rel.get("relationship_label", "").lower() == base_label.lower():
                        target = rel.get("entity_id")
                        # Now, for every other NPC in the same group that has one of the related labels
                        for other_id, other_data in npc_dict.items():
                            if other_id == npc_id:
                                continue
                            for other_rel in other_data["relationships"]:
                                if other_rel.get("relationship_label", "").lower() in [r.lower() for r in settings["related"]]:
                                    if other_rel.get("entity_id") == target:
                                        # If this other NPC does not already have the base relationship,
                                        # add it.
                                        if not any(r.get("relationship_label", "").lower() == base_label.lower() and r.get("entity_id") == target
                                                   for r in other_data["relationships"]):
                                            other_data["relationships"].append({
                                                "relationship_label": base_label,
                                                "entity_type": "npc",
                                                "entity_id": target
                                            })
                                            # Also add the reciprocal relationship to the target NPC,
                                            # if target exists in our npc_dict.
                                            if target in npc_dict:
                                                target_rels = npc_dict[target]["relationships"]
                                                if not any(r.get("relationship_label", "").lower() == settings["reciprocal"].lower() and r.get("entity_id") == other_id
                                                           for r in target_rels):
                                                    target_rels.append({
                                                        "relationship_label": settings["reciprocal"],
                                                        "entity_type": "npc",
                                                        "entity_id": other_id
                                                    })
    # Write the updated relationships back to the DB.
    for npc_id, data in npc_dict.items():
        new_rel_str = json.dumps(data["relationships"])
        cur.execute("""
            UPDATE NPCStats
            SET relationships=%s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (new_rel_str, user_id, conversation_id, npc_id))
    conn.commit()
    conn.close()

def adjust_family_ages(user_id, conversation_id):
    """
    Adjust NPC ages based on familial relationships.
    For example, if an NPC has a "mother" relationship with another,
    ensure that the mother's age is at least a specified number of years greater
    than the child's.
    """
    # Define minimum age differences for roles (all keys in lowercase)
    min_age_diff = {
        "mother": 16,
        "stepmother": 0,
        "aunt": 5,
        "older sister": 1,
        "stepsister": 1,
        "babysitter": 2,
        "teacher": 10,
        "principal": 10,
        "milf": 10,
        "dowager": 15,
        "domestic authority": 5,
        "foreign royalty": 0,
        "cousin": 0  # could be same age or slight difference
    }
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT npc_id, age, relationships
        FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s
    """, (user_id, conversation_id))
    rows = cur.fetchall()
    npc_info = {}
    for row in rows:
        npc_id, age, rel_str = row
        try:
            relationships = json.loads(rel_str) if rel_str else []
        except Exception:
            relationships = []
        npc_info[npc_id] = {"age": age, "relationships": relationships}
    
    # For each familial relationship, adjust ages accordingly.
    # For example, if NPC A has a "mother" relationship to NPC B,
    # ensure A.age >= B.age + min_age_diff["mother"]
    for npc_id, info in npc_info.items():
        for rel in info["relationships"]:
            label = rel.get("relationship_label", "").lower()
            target = rel.get("entity_id")
            if target not in npc_info:
                continue
            if label in ["mother", "stepmother", "aunt"]:
                required = min_age_diff.get(label, 0)
                target_age = npc_info[target]["age"]

# NPC mask slippage triggers - these are moments when the NPC's true nature begins to show
MASK_SLIPPAGE_TRIGGERS = {
    "dominance": [
        {"threshold": 30, "event": "subtle_control", "memory": "I let my control slip through a bit today. Nobody seemed to notice the subtle shift in dynamic."},
        {"threshold": 50, "event": "mask_adjustment", "memory": "It's getting harder to maintain this facade. I caught myself giving commands too firmly, had to play it off as a joke."},
        {"threshold": 70, "event": "partial_revelation", "memory": "I showed a glimpse of my true self today. The flash of fear in their eyes was... intoxicating."},
        {"threshold": 90, "event": "mask_removal", "memory": "I'm barely pretending anymore. Those who understand appreciate the honesty. Those who don't will learn."}
    ],
    "cruelty": [
        {"threshold": 30, "event": "sharp_comment", "memory": "I said something cutting today and quickly covered it with a laugh. The momentary hurt in their eyes was satisfying."},
        {"threshold": 50, "event": "testing_boundaries", "memory": "I'm pushing further each time to see what I can get away with. People are so willing to excuse 'playful' cruelty."},
        {"threshold": 70, "event": "deliberate_harm", "memory": "I orchestrated a situation today that caused genuine distress. I maintained plausible deniability, of course."},
        {"threshold": 90, "event": "overt_sadism", "memory": "My reputation for 'intensity' is established enough that I barely need to hide my enjoyment of others' suffering now."}
    ],
    "intensity": [
        {"threshold": 30, "event": "piercing_gaze", "memory": "Someone commented on my intense stare today. I've learned to soften it in public, but sometimes I forget."},
        {"threshold": 50, "event": "forceful_presence", "memory": "People naturally move aside when I walk through a room now. My presence is becoming harder to disguise."},
        {"threshold": 70, "event": "commanding_aura", "memory": "I no longer need to raise my voice to be obeyed. My quiet commands carry weight that surprises even me."},
        {"threshold": 90, "event": "overwhelming_presence", "memory": "The mask has become nearly transparent. My true nature radiates from me, drawing submission from the vulnerable."}
    ]
}

# NPC daily activity definitions - focused on relationship building, social networking, and subtle manipulation
NPC_ACTIVITIES = [
    {
        "name": "casual_socializing",
        "description": "Spend time with others in seemingly harmless social activities",
        "stat_changes": {"closeness": +2},
        "relationship_effects": {"link_level": +3},
        "memory_templates": [
            "Spent time with {target_name} today. Just casual conversation, but I'm learning their vulnerabilities.",
            "Had coffee with {target_name}. These 'friendly' interactions make them lower their guard around me.",
            "Went shopping with {target_name}. Every personal detail they share is something I can use later."
        ]
    },
    {
        "name": "subtle_testing",
        "description": "Test boundaries in seemingly innocent ways",
        "stat_changes": {"dominance": +1, "cruelty": +1},
        "relationship_effects": {"link_level": +2},
        "memory_templates": [
            "Asked {target_name} for small favors today. Each compliance makes the next request easier.",
            "Made 'jokes' that pushed {target_name}'s comfort zone. Noting which ones they nervously laugh at versus challenge.",
            "Suggested activities just outside {target_name}'s comfort zone. Their hesitation before agreeing tells me volumes."
        ]
    },
    {
        "name": "vulnerability_sharing",
        "description": "Create false intimacy through strategic vulnerability",
        "stat_changes": {"closeness": +3, "trust": +2},
        "relationship_effects": {"link_level": +5},
        "memory_templates": [
            "Shared 'personal struggles' with {target_name} today. Carefully crafted to seem vulnerable while giving away nothing real.",
            "Had a 'deep conversation' with {target_name}. My fabricated vulnerabilities encourage them to reveal genuine ones.",
            "{target_name} opened up to me today after I presented a calculated show of trust. The intimacy is entirely one-sided."
        ]
    },
    {
        "name": "group_bonding",
        "description": "Build relationships with multiple people simultaneously",
        "relationship_effects": {"link_level": +2},
        "group_activity": True,
        "memory_templates": [
            "Organized a get-together with {target_names}. The dynamics between them are fascinating - and exploitable.",
            "Spent time with several people today. I'm carefully positioning myself in their social hierarchy.",
            "Group activities make it easy to observe everyone's weaknesses while maintaining my friendly persona."
        ]
    },
    {
        "name": "hidden_training",
        "description": "Subtly condition others to respond to control",
        "stat_changes": {"dominance": +2},
        "relationship_effects": {"link_level": +3, "submission_level": +2},
        "memory_templates": [
            "I've been conditioning {target_name} with subtle rewards when they comply. They don't even notice the pattern forming.",
            "Practiced my techniques on {target_name} today. A firm tone disguised as concern, touch that guides behavior - it's working.",
            "The training is progressing with {target_name}. They now unconsciously seek my approval before making decisions."
        ]
    },
    {
        "name": "alliance_building",
        "description": "Form connections with other dominant figures",
        "stat_changes": {"intensity": +1},
        "prerequisites": {"dominance": 40},
        "alliance_activity": True,
        "memory_templates": [
            "Connected with {target_name} today. We understand each other's true nature beneath our public personas.",
            "Met with {target_name} privately. We share similar interests in control, though we present differently in public.",
            "Spent time with the inner circle today. Our network grows stronger, all while maintaining perfectly respectable appearances."
        ]
    },
    {
        "name": "strategic_assistance",
        "description": "Create dependency through solving problems",
        "stat_changes": {"dominance": +1},
        "relationship_effects": {"link_level": +3, "dependency_level": +3},
        "memory_templates": [
            "Helped {target_name} with a problem today. Each time I solve something for them, their independence weakens.",
            "Offered my assistance to {target_name} again. They're beginning to reflexively turn to me instead of solving issues themselves.",
            "{target_name} thanked me effusively for my help today. They don't see how I'm positioning myself as indispensable."
        ]
    },
    {
        "name": "secret_observation",
        "description": "Gather information through covert observation",
        "stat_changes": {"cruelty": +1, "intensity": +1},
        "memory_templates": [
            "Observed {target_name} without their knowledge today. Learning their patterns, habits, vulnerabilities.",
            "Spent time watching how {target_name} interacts with others. Their public and private personas differ in interesting ways.",
            "Gathered valuable information about {target_name}'s fears and desires today. Knowledge is power."
        ]
    }
]

# Activity combinations that occur when multiple NPCs interact
NPC_GROUP_ACTIVITIES = [
    {
        "name": "private_discussion",
        "description": "NPCs privately discuss their true nature and plans",
        "required_dominance": 50,
        "stats_all": {"dominance": +1, "intensity": +1},
        "relationship_effects": {"link_level": +3, "alliance_level": +2},
        "memory_template": "Had a private conversation with {npc_names} today where we dropped our public masks for a while. Our alliance grows stronger."
    },
    {
        "name": "coordinated_manipulation",
        "description": "NPCs work together to manipulate a target",
        "required_dominance": 60,
        "required_cruelty": 40,
        "stats_all": {"dominance": +2, "cruelty": +1},
        "target_types": ["player", "npc"],
        "target_effects": {"confidence": -2, "dependency": +3},
        "memory_template": "Coordinated with {npc_names} today to subtly manipulate {target_name}. Our combined approach is proving quite effective."
    },
    {
        "name": "social_hierarchy_establishment",
        "description": "NPCs establish dominance hierarchies among themselves",
        "required_dominance": 70,
        "stats_winners": {"dominance": +3, "respect": +2},
        "stats_losers": {"respect": +3, "dependency": +2},
        "memory_template": "The hierarchy within our group became clearer today. {winner_names} demonstrated their control, while {loser_names} showed their understanding of their place."
    }
]

# Relationship stages that track the evolution of NPC-NPC and NPC-player relationships
RELATIONSHIP_STAGES = {
    "dominant": [
        {"level": 10, "name": "Initial Interest", "description": "Beginning to notice potential for control"},
        {"level": 30, "name": "Strategic Friendship", "description": "Establishing trust while assessing vulnerabilities"},
        {"level": 50, "name": "Subtle Influence", "description": "Exercising increasing control through 'guidance'"},
        {"level": 70, "name": "Open Control", "description": "Dropping pretense of equality in the relationship"},
        {"level": 90, "name": "Complete Dominance", "description": "Relationship is explicitly based on control and submission"}
    ],
    "alliance": [
        {"level": 10, "name": "Mutual Recognition", "description": "Recognizing similar controlling tendencies"},
        {"level": 30, "name": "Cautious Cooperation", "description": "Sharing limited information and techniques"},
        {"level": 50, "name": "Strategic Partnership", "description": "Actively collaborating while maintaining independence"},
        {"level": 70, "name": "Power Coalition", "description": "Forming a unified front with clear internal hierarchy"},
        {"level": 90, "name": "Dominant Cabal", "description": "Operating as a coordinated group to control others"}
    ],
    "rivalry": [
        {"level": 10, "name": "Veiled Competition", "description": "Competing subtly while maintaining cordial appearance"},
        {"level": 30, "name": "Strategic Undermining", "description": "Actively working to diminish the other's influence"},
        {"level": 50, "name": "Open Challenge", "description": "Directly competing for control and resources"},
        {"level": 70, "name": "Psychological Warfare", "description": "Actively attempting to break the other's control"},
        {"level": 90, "name": "Domination Contest", "description": "All-out struggle for supremacy"}
    ]
}

async def check_for_mask_slippage(user_id, conversation_id, npc_id):
    """
    Check if an NPC has reached thresholds where their true nature begins to show
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPC's current stats
        cursor.execute("""
            SELECT npc_name, dominance, cruelty, intensity, memory
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if not row:
            return None
            
        npc_name, dominance, cruelty, intensity, memory_json = row
        
        # Parse memory
        if memory_json:
            if isinstance(memory_json, str):
                try:
                    memory = json.loads(memory_json)
                except:
                    memory = []
            else:
                memory = memory_json
        else:
            memory = []
            
        # Get slippage history
        cursor.execute("""
            SELECT mask_slippage_events
            FROM NPCEvolution
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if row and row[0]:
            if isinstance(row[0], str):
                try:
                    slippage_history = json.loads(row[0])
                except:
                    slippage_history = []
            else:
                slippage_history = row[0]
        else:
            slippage_history = []
            
        # Check each stat for slippage triggers
        triggered_events = []
        
        for stat_name, thresholds in MASK_SLIPPAGE_TRIGGERS.items():
            stat_value = locals().get(stat_name, 0)  # Get the stat value from local vars
            
            for threshold in thresholds:
                event_name = threshold["event"]
                
                # Skip if this slippage has already occurred
                if any(e.get("event") == event_name for e in slippage_history):
                    continue
                    
                # Check if threshold is met
                if stat_value >= threshold["threshold"]:
                    # Record this slippage event
                    event_data = {
                        "event": event_name,
                        "stat": stat_name,
                        "threshold": threshold["threshold"],
                        "timestamp": datetime.now().isoformat()
                    }
                    triggered_events.append(event_data)
                    
                    # Add the memory
                    if "memory" in threshold:
                        memory.append(threshold["memory"])
                        
                    # Subtle physical description changes based on mask slippage
                    if stat_name == "dominance" and threshold["threshold"] >= 50:
                        cursor.execute("""
                            UPDATE NPCStats
                            SET physical_description = CONCAT(physical_description, ' In unguarded moments, her demeanor shifts subtly—a barely perceptible straightening of posture, a flash of something commanding in her eyes that quickly vanishes when noticed.')
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (user_id, conversation_id, npc_id))
                    
                    if stat_name == "cruelty" and threshold["threshold"] >= 50:
                        cursor.execute("""
                            UPDATE NPCStats
                            SET physical_description = CONCAT(physical_description, ' Occasionally her smile doesn't quite reach her eyes, revealing a momentary coldness before she adjusts her expression back to warmth.')
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (user_id, conversation_id, npc_id))
                        
                    if stat_name == "intensity" and threshold["threshold"] >= 50:
                        cursor.execute("""
                            UPDATE NPCStats
                            SET physical_description = CONCAT(physical_description, ' Sometimes when she thinks no one is watching, her gaze becomes unnervingly focused, studying others with an analytical intensity that disappears behind a pleasant mask when attention returns to her.')
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (user_id, conversation_id, npc_id))
        
        # Update memory
        cursor.execute("""
            UPDATE NPCStats
            SET memory = %s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (json.dumps(memory), user_id, conversation_id, npc_id))
        
        # Update slippage history
        if triggered_events:
            slippage_history.extend(triggered_events)
            
            # Check if NPCEvolution record exists
            cursor.execute("""
                SELECT 1 FROM NPCEvolution
                WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
            """, (user_id, conversation_id, npc_id))
            
            if cursor.fetchone():
                cursor.execute("""
                    UPDATE NPCEvolution
                    SET mask_slippage_events = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(slippage_history), user_id, conversation_id, npc_id))
            else:
                cursor.execute("""
                    INSERT INTO NPCEvolution
                    (user_id, conversation_id, npc_id, mask_slippage_events)
                    VALUES (%s, %s, %s, %s)
                """, (user_id, conversation_id, npc_id, json.dumps(slippage_history)))
        
        conn.commit()
        return triggered_events
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error checking mask slippage: {e}")
        return None
    finally:
        cursor.close()
        conn.close()

async def perform_npc_daily_activity(user_id, conversation_id, npc_id, time_of_day):
    """
    Have an NPC perform activities during their daily schedule
    that develop relationships and subtly increase relevant stats
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get NPC info
        cursor.execute("""
            SELECT npc_name, dominance, cruelty, closeness, trust, respect, intensity, 
                   current_location, schedule, memory, introduced
            FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if not row:
            return False
            
        npc_name, dominance, cruelty, closeness, trust, respect, intensity, \
        location, schedule_json, memory_json, introduced = row
        
        # Parse memory
        if memory_json:
            if isinstance(memory_json, str):
                try:
                    memory = json.loads(memory_json)
                except:
                    memory = []
            else:
                memory = memory_json
        else:
            memory = []
        
        # Parse schedule to see what they're doing
        if schedule_json:
            if isinstance(schedule_json, str):
                try:
                    schedule = json.loads(schedule_json)
                except:
                    schedule = {}
            else:
                schedule = schedule_json
        else:
            schedule = {}
        
        # Get the current day
        cursor.execute("""
            SELECT value FROM CurrentRoleplay 
            WHERE user_id=%s AND conversation_id=%s AND key='CurrentDay'
        """, (user_id, conversation_id))
        day_row = cursor.fetchone()
        current_day = int(day_row[0]) if day_row else 1
        
        # Get day name from calendar
        cursor.execute("""
            SELECT value FROM CurrentRoleplay 
            WHERE user_id=%s AND conversation_id=%s AND key='CalendarNames'
        """, (user_id, conversation_id))
        calendar_row = cursor.fetchone()
        if calendar_row and calendar_row[0]:
            try:
                calendar = json.loads(calendar_row[0])
                day_names = calendar.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                day_index = (current_day - 1) % len(day_names)
                day_name = day_names[day_index]
            except:
                day_name = "Unknown"
        else:
            day_name = "Unknown"
        
        # Check schedule for this day and time
        day_schedule = schedule.get(day_name, {})
        activity_desc = day_schedule.get(time_of_day, "")
        
        # Determine if we'll perform a social activity based on schedule
        if not activity_desc or "meets" in activity_desc.lower() or "with" in activity_desc.lower() or random.random() < 0.3:
            # Decide if we'll do a group activity with other NPCs
            do_group_activity = random.random() < 0.4  # 40% chance for group activity
            
            if do_group_activity:
                # Find other NPCs in the same location
                cursor.execute("""
                    SELECT npc_id, npc_name, dominance, cruelty
                    FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
                    AND current_location=%s AND introduced=TRUE
                    LIMIT 3
                """, (user_id, conversation_id, npc_id, location))
                
                other_npcs = cursor.fetchall()
                
                if len(other_npcs) >= 1:
                    # We have enough NPCs for a group activity
                    all_npcs = [(npc_id, npc_name, dominance, cruelty)] + list(other_npcs)
                    npc_names = [n[1] for n in all_npcs]
                    
                    # Find eligible group activities
                    eligible_activities = []
                    for activity in NPC_GROUP_ACTIVITIES:
                        # Check if primary NPC meets requirements
                        if "required_dominance" in activity and dominance < activity["required_dominance"]:
                            continue
                        if "required_cruelty" in activity and cruelty < activity["required_cruelty"]:
                            continue
                        eligible_activities.append(activity)
                    
                    if eligible_activities:
                        # Choose a random eligible activity
                        activity = random.choice(eligible_activities)
                        
                        # Process activity effects
                        # Apply stat changes to all participating NPCs
                        if "stats_all" in activity:
                            for npc in all_npcs:
                                npc_id_update = npc[0]
                                stat_updates = []
                                for stat, change in activity["stats_all"].items():
                                    stat_updates.append(f"{stat} = {stat} + {change}")
                                
                                if stat_updates:
                                    cursor.execute(f"""
                                        UPDATE NPCStats
                                        SET {', '.join(stat_updates)}
                                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                                    """, (user_id, conversation_id, npc_id_update))
                        
                        # For hierarchy activities, determine winners and losers
                        if "stats_winners" in activity and "stats_losers" in activity:
                            # Sort NPCs by dominance for determining hierarchy
                            sorted_npcs = sorted(all_npcs, key=lambda x: x[2], reverse=True)
                            mid_point = len(sorted_npcs) // 2
                            
                            winners = sorted_npcs[:mid_point]
                            losers = sorted_npcs[mid_point:]
                            
                            # Apply winner stats
                            for npc in winners:
                                npc_id_update = npc[0]
                                stat_updates = []
                                for stat, change in activity["stats_winners"].items():
                                    stat_updates.append(f"{stat} = {stat} + {change}")
                                
                                if stat_updates:
                                    cursor.execute(f"""
                                        UPDATE NPCStats
                                        SET {', '.join(stat_updates)}
                                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                                    """, (user_id, conversation_id, npc_id_update))
                            
                            # Apply loser stats
                            for npc in losers:
                                npc_id_update = npc[0]
                                stat_updates = []
                                for stat, change in activity["stats_losers"].items():
                                    stat_updates.append(f"{stat} = {stat} + {change}")
                                
                                if stat_updates:
                                    cursor.execute(f"""
                                        UPDATE NPCStats
                                        SET {', '.join(stat_updates)}
                                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                                    """, (user_id, conversation_id, npc_id_update))
                        
                        # Create or update relationships between all participants
                        if "relationship_effects" in activity:
                            for i in range(len(all_npcs)):
                                for j in range(i+1, len(all_npcs)):
                                    npc1_id = all_npcs[i][0]
                                    npc2_id = all_npcs[j][0]
                                    
                                    # Check existing relationship
                                    cursor.execute("""
                                        SELECT link_id, link_type, link_level
                                        FROM SocialLinks
                                        WHERE user_id=%s AND conversation_id=%s
                                        AND entity1_type='npc' AND entity1_id=%s
                                        AND entity2_type='npc' AND entity2_id=%s
                                    """, (user_id, conversation_id, npc1_id, npc2_id))
                                    
                                    link_row = cursor.fetchone()
                                    
                                    if link_row:
                                        # Update existing link
                                        link_id, link_type, link_level = link_row
                                        
                                        updates = []
                                        values = []
                                        
                                        # General link level
                                        if "link_level" in activity["relationship_effects"]:
                                            change = activity["relationship_effects"]["link_level"]
                                            updates.append("link_level = link_level + %s")
                                            values.append(change)
                                        
                                        # Specific relationship type levels
                                        for rel_type, change in activity["relationship_effects"].items():
                                            if rel_type.endswith("_level") and rel_type != "link_level":
                                                rel_name = rel_type.replace("_level", "")
                                                if link_type == rel_name:
                                                    updates.append("link_level = link_level + %s")
                                                    values.append(change)
                                        
                                        if updates:
                                            values.append(link_id)
                                            cursor.execute(f"""
                                                UPDATE SocialLinks
                                                SET {', '.join(updates)}
                                                WHERE link_id = %s
                                            """, values)
                                            
                                            # Add event to link history
                                            event_text = f"Participated in {activity['name']} together."
                                            cursor.execute("""
                                                UPDATE SocialLinks
                                                SET link_history = link_history || %s::jsonb
                                                WHERE link_id = %s
                                            """, (json.dumps([event_text]), link_id))
                                    else:
                                        # Create new link - determine relationship type
                                        if "alliance" in activity["name"].lower():
                                            link_type = "alliance"
                                        elif "hierarchy" in activity["name"].lower():
                                            # Determine based on dominance
                                            npc1_dom = all_npcs[i][2]
                                            npc2_dom = all_npcs[j][2]
                                            if abs(npc1_dom - npc2_dom) < 10:
                                                link_type = "rivalry"
                                            else:
                                                link_type = "alliance"
                                        else:
                                            link_type = "neutral"
                                        
                                        level = activity["relationship_effects"].get("link_level", 0)
                                        if f"{link_type}_level" in activity["relationship_effects"]:
                                            level += activity["relationship_effects"][f"{link_type}_level"]
                                        
                                        cursor.execute("""
                                            INSERT INTO SocialLinks
                                            (user_id, conversation_id, entity1_type, entity1_id, 
                                             entity2_type, entity2_id, link_type, link_level, link_history)
                                            VALUES (%s, %s, 'npc', %s, 'npc', %s, %s, %s, %s)
                                        """, (
                                            user_id, conversation_id,
                                            npc1_id, npc2_id,
                                            link_type, level, json.dumps([f"Formed relationship during {activity['name']}."])
                                        ))
                        
                        # Record memory for primary NPC
                        memory_text = activity["memory_template"]
                        winner_names = ", ".join([n[1] for n in winners]) if "stats_winners" in activity else ""
                        loser_names = ", ".join([n[1] for n in losers]) if "stats_losers" in activity else ""
                        formatted_memory = memory_text.format(
                            npc_names=", ".join(npc_names[1:]),  # Exclude self
                            winner_names=winner_names,
                            loser_names=loser_names
                        )
                        
                        memory.append(formatted_memory)
                        cursor.execute("""
                            UPDATE NPCStats
                            SET memory = %s
                            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                        """, (json.dumps(memory), user_id, conversation_id, npc_id))
                        
                        conn.commit()
                        return True
            
            # If we didn't do a group activity or couldn't find participants, do a regular activity
            # Choose a random activity from NPC_ACTIVITIES
            activity = random.choice(NPC_ACTIVITIES)
            
            # Apply stat changes to NPC
            if "stat_changes" in activity:
                stat_updates = []
                for stat, change in activity["stat_changes"].items():
                    stat_updates.append(f"{stat} = {stat} + {change}")
                
                if stat_updates:
                    cursor.execute(f"""
                        UPDATE NPCStats
                        SET {', '.join(stat_updates)}
                        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                    """, (user_id, conversation_id, npc_id))
            
            # Decide if targeting player or NPC
            target_id = None
            target_name = None
            target_type = None
            
            if random.random() < 0.3:  # 30% chance to target player
                target_id = user_id
                target_name = "Chase"
                target_type = "player"
            else:
                # Find another NPC to interact with
                cursor.execute("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id=%s AND conversation_id=%s AND npc_id!=%s
                    ORDER BY RANDOM() LIMIT 1
                """, (user_id, conversation_id, npc_id))
                target_row = cursor.fetchone()
                
                if target_row:
                    target_id = target_row[0]
                    target_name = target_row[1]
                    target_type = "npc"
            
            # Create or update relationship if we have a target
            if target_id and target_name and "relationship_effects" in activity:
                # Check for existing link
                cursor.execute("""
                    SELECT link_id, link_type, link_level
                    FROM SocialLinks
                    WHERE user_id=%s AND conversation_id=%s
                    AND entity1_type='npc' AND entity1_id=%s
                    AND entity2_type=%s AND entity2_id=%s
                """, (user_id, conversation_id, npc_id, target_type, target_id))
                
                link_row = cursor.fetchone()
                
                if link_row:
                    # Update existing link
                    link_id, link_type, link_level = link_row
                    
                    updates = []
                    values = []
                    
                    # General link level
                    if "link_level" in activity["relationship_effects"]:
                        change = activity["relationship_effects"]["link_level"]
                        updates.append("link_level = link_level + %s")
                        values.append(change)
                    
                    # Specific relationship type levels
                    for rel_type, change in activity["relationship_effects"].items():
                        if rel_type.endswith("_level") and rel_type != "link_level":
                            rel_name = rel_type.replace("_level", "")
                            if link_type == rel_name:
                                updates.append("link_level = link_level + %s")
                                values.append(change)
                    
                    if updates:
                        values.append(link_id)
                        cursor.execute(f"""
                            UPDATE SocialLinks
                            SET {', '.join(updates)}
                            WHERE link_id = %s
                        """, values)
                        
                        # Add event to link history
                        event_text = f"{npc_name} performed {activity['name']} with {target_name}."
                        cursor.execute("""
                            UPDATE SocialLinks
                            SET link_history = link_history || %s::jsonb
                            WHERE link_id = %s
                        """, (json.dumps([event_text]), link_id))
                else:
                    # Create new link
                    link_type = "friendship"  # Default relationship type
                    
                    # Determine relationship type based on stats
                    if dominance >= 60:
                        if "hidden_training" in activity["name"] or "strategic_assistance" in activity["name"]:
                            link_type = "dominant"
                    
                    level = activity["relationship_effects"].get("link_level", 0)
                    if f"{link_type}_level" in activity["relationship_effects"]:
                        level += activity["relationship_effects"][f"{link_type}_level"]
                    
                    cursor.execute("""
                        INSERT INTO SocialLinks
                        (user_id, conversation_id, entity1_type, entity1_id, 
                         entity2_type, entity2_id, link_type, link_level, link_history)
                        VALUES (%s, %s, 'npc', %s, %s, %s, %s, %s, %s)
                    """, (
                        user_id, conversation_id,
                        npc_id, target_type, target_id,
                        link_type, level, json.dumps([f"Formed relationship during {activity['name']}."])
                    ))
            
            # Record memory
            if target_name and "memory_templates" in activity:
                memory_template = random.choice(activity["memory_templates"])
                formatted_memory = memory_template.format(target_name=target_name)
                
                memory.append(formatted_memory)
                cursor.execute("""
                    UPDATE NPCStats
                    SET memory = %s
                    WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
                """, (json.dumps(memory), user_id, conversation_id, npc_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error in NPC daily activity: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

async def process_daily_npc_activities(user_id, conversation_id, time_of_day):
    """
    Process activities for all NPCs during a specific time of day
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all NPCs
        cursor.execute("""
            SELECT npc_id FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s
        """, (user_id, conversation_id))
        
        npc_ids = [row[0] for row in cursor.fetchall()]
        
        for npc_id in npc_ids:
            # Check for mask slippage
            await check_for_mask_slippage(user_id, conversation_id, npc_id)
            
            # Perform daily activity
            await perform_npc_daily_activity(user_id, conversation_id, npc_id, time_of_day)
        
        return True
    except Exception as e:
        logging.error(f"Error processing daily NPC activities: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

async def detect_relationship_stage_changes(user_id, conversation_id):
    """
    Detect changes in relationship stages and update memories accordingly
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all social links
        cursor.execute("""
            SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id, 
                   link_type, link_level, relationship_stage
            FROM SocialLinks
            WHERE user_id=%s AND conversation_id=%s
        """, (user_id, conversation_id))
        
        links = cursor.fetchall()
        
        for link in links:
            link_id, e1_type, e1_id, e2_type, e2_id, link_type, link_level, current_stage = link
            
            # Skip if link type isn't in our defined stages
            if link_type not in RELATIONSHIP_STAGES:
                continue
            
            # Find the appropriate stage
            stages = RELATIONSHIP_STAGES[link_type]
            
            new_stage = None
            for stage in reversed(stages):  # Check from highest to lowest
                if link_level >= stage["level"]:
                    new_stage = stage
                    break
            
            if not new_stage:
                continue
                
            stage_name = new_stage["name"]
            
            # If stage has changed, update it
            if stage_name != current_stage:
                cursor.execute("""
                    UPDATE SocialLinks
                    SET relationship_stage = %s
                    WHERE link_id = %s
                """, (stage_name, link_id))
                
                # Add event to history
                event_text = f"Relationship evolved to stage: {stage_name} - {new_stage['description']}"
                cursor.execute("""
                    UPDATE SocialLinks
                    SET link_history = link_history || %s::jsonb
                    WHERE link_id = %s
                """, (json.dumps([event_text]), link_id))
                
                # Create memory entries for the entities
                memory_text = f"My relationship with {get_entity_name(conn, e2_type, e2_id, user_id, conversation_id)} has shifted. {new_stage['description']}."
                
                if e1_type == 'npc':
                    add_npc_memory(conn, user_id, conversation_id, e1_id, memory_text)
                
                if e2_type == 'npc':
                    reciprocal_text = f"My relationship with {get_entity_name(conn, e1_type, e1_id, user_id, conversation_id)} has changed. {get_reciprocal_description(new_stage['description'])}."
                    add_npc_memory(conn, user_id, conversation_id, e2_id, reciprocal_text)
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logging.error(f"Error detecting relationship stage changes: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def get_entity_name(conn, entity_type, entity_id, user_id, conversation_id):
    """
    Get the name of an entity (NPC or player)
    """
    if entity_type == 'player':
        return "Chase"
    
    cursor = conn.cursor()
    cursor.execute("""
        SELECT npc_name FROM NPCStats
        WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
    """, (user_id, conversation_id, entity_id))
    
    row = cursor.fetchone()
    cursor.close()
    
    return row[0] if row else "Unknown"

def get_reciprocal_description(description):
    """
    Generate a reciprocal description from the perspective of the other entity
    """
    # Simple replacements for now
    replacements = {
        "control": "being controlled",
        "dominance": "submission",
        "manipulating": "being influenced",
        "assessing vulnerabilities": "being evaluated",
        "control and submission": "submission and control"
    }
    
    result = description
    for original, replacement in replacements.items():
        result = result.replace(original, replacement)
    
    return result

def add_npc_memory(conn, user_id, conversation_id, npc_id, memory_text):
    """
    Add a memory entry for an NPC
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT memory FROM NPCStats
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (user_id, conversation_id, npc_id))
        
        row = cursor.fetchone()
        if row and row[0]:
            if isinstance(row[0], str):
                try:
                    memory = json.loads(row[0])
                except:
                    memory = []
            else:
                memory = row[0]
        else:
            memory = []
        
        memory.append(memory_text)
        
        cursor.execute("""
            UPDATE NPCStats
            SET memory = %s
            WHERE user_id=%s AND conversation_id=%s AND npc_id=%s
        """, (json.dumps(memory), user_id, conversation_id, npc_id))
        
    finally:
        cursor.close()
