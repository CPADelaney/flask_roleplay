# logic/npc_agents/

import random
import logging
from datetime import datetime, timedelta
from memory.wrapper import MemorySystem
from db.connection import get_db_connection

logger = logging.getLogger(__name__)

class BehaviorEvolution:
    """
    Evolves NPC behavior over time, modifying their tactics based on past events.
    NPCs will develop hidden agendas, adjust their manipulation strategies, 
    and attempt to control the world around them.
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None

    async def get_memory_system(self) -> MemorySystem:
        """Lazy-load the memory system."""
        if self.memory_system is None:
            self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        return self.memory_system

    # In BehaviorEvolution
    async def evaluate_npc_scheming_with_user_model(self, npc_id: int) -> dict:
        """Evolve NPC behavior considering user preferences"""
        # Get base evolution
        base_adjustments = await self.evaluate_npc_scheming(npc_id)
        
        # Get user model from Nyx
        user_model = UserModelManager(self.user_id, self.conversation_id)
        user_guidance = await user_model.get_response_guidance()
        
        # Adjust scheming based on user preferences
        if user_guidance.get("suggested_dominance", 0.5) > 0.7:
            # User prefers dominant characters, enhance scheming
            base_adjustments["scheme_level"] += 2
        elif user_guidance.get("suggested_dominance", 0.5) < 0.3:
            # User prefers less dominant characters, reduce scheming
            base_adjustments["scheme_level"] = max(0, base_adjustments["scheme_level"] - 1)
        
        return base_adjustments

    async def evaluate_npc_scheming(self, npc_id: int) -> dict:
        """
        Periodically evaluate if an NPC should adjust their behavior, escalate plans, or set new secret goals.
        This is called every X hours/days.

        Returns:
            Dict containing the NPC's updated behavior.
        """

        try:
            memory_system = await self.get_memory_system()
            
            # Retrieve NPC history
            npc_data = await self._get_npc_data(npc_id)
            if not npc_data:
                return {"error": "NPC data not found"}

            name = npc_data["npc_name"]
            dominance = npc_data["dominance"]
            cruelty = npc_data["cruelty"]
            paranoia = "paranoid" in npc_data.get("personality_traits", [])
            deceptive = "manipulative" in npc_data.get("personality_traits", [])

            # Retrieve past manipulations & betrayals
            betrayals = await memory_system.recall(npc_id, "betrayal", limit=5)
            successful_lies = await memory_system.recall(npc_id, "deception success", limit=5)
            failed_lies = await memory_system.recall(npc_id, "deception failure", limit=3)
            loyalty_tests = await memory_system.recall(npc_id, "tested loyalty", limit=3)

            # Define NPC evolution behavior
            adjustments = {
                "scheme_level": 0,  # Determines how aggressive their plotting is
                "trust_modifiers": {},
                "loyalty_tests": 0,
                "betrayal_planning": False,
                "targeting_player": False,
                "npc_recruits": []
            }

            # Adjust scheming level based on success rate
            if successful_lies:
                adjustments["scheme_level"] += len(successful_lies)
            if failed_lies:
                adjustments["scheme_level"] -= len(failed_lies)  # Punishes failures
            if betrayals:
                adjustments["scheme_level"] += len(betrayals) * 2  # Increases scheming if they’ve been betrayed

            # If their deception is failing often, they become either cautious or reckless
            if failed_lies and paranoia:
                adjustments["scheme_level"] += 3  # Paranoia increases scheming

            # If an NPC has tested loyalty and found **weak** targets, they begin **manipulating more.**
            if loyalty_tests:
                adjustments["loyalty_tests"] += len(loyalty_tests)
                weak_targets = [test["npc_id"] for test in loyalty_tests if test.get("failed_loyalty_check")]
                if weak_targets:
                    adjustments["npc_recruits"].extend(weak_targets)

            # Dominant NPCs escalate manipulation if they see success
            if dominance > 70 and successful_lies:
                adjustments["scheme_level"] += 2

            # Cruel NPCs escalate based on betrayals
            if cruelty > 70 and betrayals:
                adjustments["betrayal_planning"] = True

            # Paranoid NPCs will **target** anyone they suspect of deception
            if paranoia and failed_lies:
                adjustments["targeting_player"] = True

            # **Final checks: If the NPC is in full scheming mode, they begin long-term plans**
            if adjustments["scheme_level"] >= 5:
                logger.info(f"{name} is entering full scheming mode.")

                # Set a **secret goal**
                secret_goal = f"{name} is planning to manipulate the world around them."
                await memory_system.create_memory(npc_id, secret_goal, importance="high", emotional=True)

                # If **deceptive**, they will now **actively deceive the player**
                if deceptive:
                    adjustments["targeting_player"] = True

                # NPC starts actively **recruiting allies** if they aren’t already doing so
                if not adjustments["npc_recruits"]:
                    all_npcs = await self._get_all_npcs()
                    potential_recruits = [n["npc_id"] for n in all_npcs if n["dominance"] < 50]
                    adjustments["npc_recruits"].extend(potential_recruits[:2])

            return adjustments

        except Exception as e:
            logger.error(f"Error evaluating NPC scheming: {e}")
            return {"error": str(e)}

    async def _get_npc_data(self, npc_id: int):
        """Retrieve NPC data from database."""
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT npc_id, npc_name, dominance, cruelty, personality_traits FROM NPCStats WHERE npc_id = %s", (npc_id,))
            row = cursor.fetchone()
            if row:
                return {"npc_id": row[0], "npc_name": row[1], "dominance": row[2], "cruelty": row[3], "personality_traits": row[4]}
        return None
