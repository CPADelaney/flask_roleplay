# logic/activities_logic.py

import random
from db.connection import get_db_connection

def get_all_activities():
    """Fetches all Activities from DB. Returns a list of dicts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name, purpose, stat_integration, intensity_tiers, setting_variants
        FROM Activities
    """)
    rows = cursor.fetchall()
    conn.close()

    # Convert rows to list of dicts
    activities = []
    for r in rows:
        name, purpose, stat_i, tiers, variants = r
        activities.append({
            "name": name,
            "purpose": purpose,  # JSONB => a Python list
            "stat_integration": stat_i,
            "intensity_tiers": tiers,
            "setting_variants": variants
        })
    return activities

def filter_activities_for_npc(npc_archetypes=[], meltdown_level=0, user_stats=None, setting=""):
    """
    Grabs all Activities, then filters or weighs them by:
      - NPC archetypes (e.g. "Giantess", "Mommy Domme", etc.)
      - meltdown level
      - user stats or setting
    Returns a short random selection of e.g. 3–5 activities.
    """
    all_acts = get_all_activities()

    # Example approach:
    # 1) Create a "score" for each Activity based on how many relevant keywords match the NPC’s archetypes or setting
    # 2) Possibly add a meltdown factor if meltdown_level > 3 => prefer extreme acts
    # 3) We'll keep it naive for demonstration: just partial matching.

    def calc_score(act):
        # Start with 0
        score = 0

        # If meltdown is high, we might prefer acts with "public humiliation" or "intensity"
        # You could parse the text or apply your own logic

        # Check if any of npc_archetypes appear in purpose or setting_variants
        combined_text = " ".join(act["purpose"]) + " ".join(act["setting_variants"]) + act["name"].lower()

        for arc in npc_archetypes:
            if arc.lower() in combined_text.lower():
                score += 3  # or some weighting

        # If meltdown > 3 => prefer activities that mention "public", "extreme", or "maximum"
        if meltdown_level > 3:
            # If they have a "maximum" tier or mention "extreme humiliation" in purpose, bonus
            if "maximum" in " ".join(act["intensity_tiers"]).lower() or "extreme" in " ".join(act["purpose"]).lower():
                score += 2

        # If setting is included in any setting_variants
        if setting and any(setting.lower() in sv.lower() for sv in act["setting_variants"]):
            score += 2

        # If user_stats has e.g. shame > 70 => we might prefer "public" or "forced" tasks
        if user_stats and user_stats.get("shame", 0) > 70:
            # naive check if 'public' is found in intensity_tiers or purpose
            if "public" in combined_text.lower():
                score += 1

        return score

    scored_acts = [(calc_score(a), a) for a in all_acts]
    # Sort descending by score
    scored_acts.sort(key=lambda x: x[0], reverse=True)

    # Filter out only those with score > 0, or keep them all if you want random variety
    filtered = [a for (score, a) in scored_acts if score > 0]
    if not filtered:
        # fallback to random sample from the entire set, so we always have something
        filtered = all_acts

    # Now pick e.g. 3–5 at random from the filtered
    num_to_pick = min(5, len(filtered))
    chosen = random.sample(filtered, num_to_pick)

    # Return just the Activity dict
    return [c for c in chosen]


def build_short_summary(activity):
    """
    Returns a concise one-liner for GPT referencing:
     - name
     - a snippet from 'purpose' or 'intensity_tiers'
    """
    name = activity["name"]
    # e.g. first sentence of purpose
    short_purpose = activity["purpose"][0] if activity["purpose"] else "No purpose text."
    # maybe a snippet from intensity_tiers
    example_tier = ""
    if activity["intensity_tiers"]:
        example_tier = activity["intensity_tiers"][0]  # just the first line
    # Combine
    return f"{name} -> {short_purpose} (e.g. {example_tier})"

