# logic/resource_management.py

import logging
import json
from db.connection import get_db_connection
from datetime import datetime
from logic.currency_generator import CurrencyGenerator

class ResourceManager:
    """
    Manages player resources (money, supplies, influence) and hunger.
    Provides methods for adding, removing, and checking resources.
    """
    
    def __init__(self, user_id, conversation_id, player_name="Chase"):
        """Initialize the resource manager."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.player_name = player_name


    async def get_resources(self):
        """Get current resources for the player."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT money, supplies, influence, updated_at
                FROM PlayerResources
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Initialize with defaults if not found
                self.create_default_resources()
                return {
                    "money": 100,
                    "supplies": 20,
                    "influence": 10,
                    "updated_at": datetime.now()
                }
            
            return {
                "money": row[0],
                "supplies": row[1],
                "influence": row[2],
                "updated_at": row[3]
            }
        finally:
            cursor.close()
            conn.close()
    
    async def get_vitals(self):
        """Get current vitals for the player."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT energy, hunger, last_update
                FROM PlayerVitals
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Initialize with defaults if not found
                self.create_default_vitals()
                return {
                    "energy": 100,
                    "hunger": 100,
                    "last_update": datetime.now()
                }
            
            return {
                "energy": row[0],
                "hunger": row[1],
                "last_update": row[2]
            }
        finally:
            cursor.close()
            conn.close()
    
    async def get_formatted_money(self, amount=None):
        """
        Get the player's money formatted according to the current currency system.
        
        Args:
            amount: Optional specific amount to format. If None, uses current balance.
        
        Returns:
            Formatted currency string
        """
        if amount is None:
            # Get current balance
            resources = await self.get_resources()
            amount = resources.get("money", 0)
        
        # Use CurrencyGenerator to format it
        currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
        return await currency_generator.format_currency(amount)
    
    # Update the modify_money method to return formatted currency
    async def modify_money(self, amount, source, description=None):
        """
        Modify player's money.
        
        Args:
            amount: Amount to add (positive) or remove (negative)
            source: Source of the change (e.g., "quest", "conflict", "work")
            description: Optional description of the transaction
                
        Returns:
            Dict with success status, new balance, and formatted amounts
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current value
            cursor.execute("""
                SELECT money FROM PlayerResources
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                FOR UPDATE
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Create default resources if not found
                self.create_default_resources()
                old_value = 100
            else:
                old_value = row[0]
            
            new_value = max(0, old_value + amount)
            
            # Update resources
            cursor.execute("""
                UPDATE PlayerResources
                SET money=%s, updated_at=CURRENT_TIMESTAMP
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (new_value, self.user_id, self.conversation_id, self.player_name))
            
            # Log the change
            cursor.execute("""
                INSERT INTO ResourceHistoryLog 
                (user_id, conversation_id, player_name, resource_type, 
                 old_value, new_value, amount_changed, source, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (self.user_id, self.conversation_id, self.player_name, 
                  "money", old_value, new_value, amount, source, description))
            
            conn.commit()
            
            return {
                "success": True,
                "old_value": old_value,
                "new_value": new_value,
                "change": amount
            }
            
        if "success" in result and result["success"]:
            # Initialize currency generator
            currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
            
            # Add formatted values
            result["formatted_old_value"] = await currency_generator.format_currency(result["old_value"])
            result["formatted_new_value"] = await currency_generator.format_currency(result["new_value"])
            result["formatted_change"] = await currency_generator.format_currency(result["change"])
        
        return result
    
    async def modify_supplies(self, amount, source, description=None):
        """
        Modify player's supplies.
        
        Args:
            amount: Amount to add (positive) or remove (negative)
            source: Source of the change
            description: Optional description of the transaction
            
        Returns:
            Dict with success status and new balance
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current value
            cursor.execute("""
                SELECT supplies FROM PlayerResources
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                FOR UPDATE
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Create default resources if not found
                self.create_default_resources()
                old_value = 20
            else:
                old_value = row[0]
            
            new_value = max(0, old_value + amount)
            
            # Update resources
            cursor.execute("""
                UPDATE PlayerResources
                SET supplies=%s, updated_at=CURRENT_TIMESTAMP
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (new_value, self.user_id, self.conversation_id, self.player_name))
            
            # Log the change
            cursor.execute("""
                INSERT INTO ResourceHistoryLog 
                (user_id, conversation_id, player_name, resource_type, 
                 old_value, new_value, amount_changed, source, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (self.user_id, self.conversation_id, self.player_name, 
                  "supplies", old_value, new_value, amount, source, description))
            
            conn.commit()
            
            return {
                "success": True,
                "old_value": old_value,
                "new_value": new_value,
                "change": amount
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error modifying supplies: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            cursor.close()
            conn.close()
    
    async def modify_influence(self, amount, source, description=None):
        """
        Modify player's influence.
        
        Args:
            amount: Amount to add (positive) or remove (negative)
            source: Source of the change
            description: Optional description of the transaction
            
        Returns:
            Dict with success status and new balance
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current value
            cursor.execute("""
                SELECT influence FROM PlayerResources
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                FOR UPDATE
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Create default resources if not found
                self.create_default_resources()
                old_value = 10
            else:
                old_value = row[0]
            
            new_value = max(0, old_value + amount)
            
            # Update resources
            cursor.execute("""
                UPDATE PlayerResources
                SET influence=%s, updated_at=CURRENT_TIMESTAMP
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (new_value, self.user_id, self.conversation_id, self.player_name))
            
            # Log the change
            cursor.execute("""
                INSERT INTO ResourceHistoryLog 
                (user_id, conversation_id, player_name, resource_type, 
                 old_value, new_value, amount_changed, source, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (self.user_id, self.conversation_id, self.player_name, 
                  "influence", old_value, new_value, amount, source, description))
            
            conn.commit()
            
            return {
                "success": True,
                "old_value": old_value,
                "new_value": new_value,
                "change": amount
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error modifying influence: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            cursor.close()
            conn.close()
    
    async def modify_hunger(self, amount, source=None, description=None):
        """
        Modify player's hunger level.
        
        Args:
            amount: Amount to add (positive) or remove (negative)
            source: Source of the change (e.g., "eating", "activity")
            description: Optional description
            
        Returns:
            Dict with success status and new hunger level
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current value
            cursor.execute("""
                SELECT hunger FROM PlayerVitals
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                FOR UPDATE
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Create default vitals if not found
                self.create_default_vitals()
                old_value = 100
            else:
                old_value = row[0]
            
            # Hunger is capped between 0 and 100
            new_value = max(0, min(100, old_value + amount))
            
            # Update vitals
            cursor.execute("""
                UPDATE PlayerVitals
                SET hunger=%s, last_update=CURRENT_TIMESTAMP
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (new_value, self.user_id, self.conversation_id, self.player_name))
            
            # Log the change
            cursor.execute("""
                INSERT INTO ResourceHistoryLog 
                (user_id, conversation_id, player_name, resource_type, 
                 old_value, new_value, amount_changed, source, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (self.user_id, self.conversation_id, self.player_name, 
                  "hunger", old_value, new_value, amount, source or "activity", description))
            
            conn.commit()
            
            return {
                "success": True,
                "old_value": old_value,
                "new_value": new_value,
                "change": amount
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error modifying hunger: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            cursor.close()
            conn.close()
    
    async def modify_energy(self, amount, source=None, description=None):
        """
        Modify player's energy level.
        
        Args:
            amount: Amount to add (positive) or remove (negative)
            source: Source of the change (e.g., "rest", "activity")
            description: Optional description
            
        Returns:
            Dict with success status and new energy level
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current value
            cursor.execute("""
                SELECT energy FROM PlayerVitals
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
                FOR UPDATE
            """, (self.user_id, self.conversation_id, self.player_name))
            
            row = cursor.fetchone()
            
            if not row:
                # Create default vitals if not found
                self.create_default_vitals()
                old_value = 100
            else:
                old_value = row[0]
            
            # Energy is capped between 0 and 100
            new_value = max(0, min(100, old_value + amount))
            
            # Update vitals
            cursor.execute("""
                UPDATE PlayerVitals
                SET energy=%s, last_update=CURRENT_TIMESTAMP
                WHERE user_id=%s AND conversation_id=%s AND player_name=%s
            """, (new_value, self.user_id, self.conversation_id, self.player_name))
            
            # Log the change
            cursor.execute("""
                INSERT INTO ResourceHistoryLog 
                (user_id, conversation_id, player_name, resource_type, 
                 old_value, new_value, amount_changed, source, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (self.user_id, self.conversation_id, self.player_name, 
                  "energy", old_value, new_value, amount, source or "activity", description))
            
            conn.commit()
            
            return {
                "success": True,
                "old_value": old_value,
                "new_value": new_value,
                "change": amount
            }
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error modifying energy: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            cursor.close()
            conn.close()
    
    async def check_resources(self, money=0, supplies=0, influence=0):
        """
        Check if player has enough resources.
        
        Args:
            money: Required money amount
            supplies: Required supplies amount
            influence: Required influence amount
            
        Returns:
            Dict with success status and missing resources if any
        """
        resources = await self.get_resources()
        
        missing = {}
        if resources["money"] < money:
            missing["money"] = money - resources["money"]
        if resources["supplies"] < supplies:
            missing["supplies"] = supplies - resources["supplies"]
        if resources["influence"] < influence:
            missing["influence"] = influence - resources["influence"]
        
        if missing:
            return {
                "has_resources": False,
                "missing": missing,
                "current": resources
            }
        else:
            return {
                "has_resources": True,
                "current": resources
            }
    
    async def commit_resources_to_conflict(self, conflict_id, money=0, supplies=0, influence=0):
        """
        Commit resources to a conflict.
        
        Args:
            conflict_id: The conflict ID
            money: Amount of money to commit
            supplies: Amount of supplies to commit
            influence: Amount of influence to commit
            
        Returns:
            Dict with success status and results
        """
        # First check if we have enough resources
        check_result = await self.check_resources(money, supplies, influence)
        if not check_result["has_resources"]:
            return {
                "success": False,
                "error": "Insufficient resources",
                "missing": check_result["missing"]
            }
        
        # Then deduct resources
        results = {
            "success": True,
            "money_result": None,
            "supplies_result": None,
            "influence_result": None
        }
        
        if money > 0:
            results["money_result"] = await self.modify_money(
                -money, 
                "conflict", 
                f"Committed to conflict #{conflict_id}"
            )
        
        if supplies > 0:
            results["supplies_result"] = await self.modify_supplies(
                -supplies, 
                "conflict", 
                f"Committed to conflict #{conflict_id}"
            )
        
        if influence > 0:
            results["influence_result"] = await self.modify_influence(
                -influence, 
                "conflict", 
                f"Committed to conflict #{conflict_id}"
            )
        
        return results
    
    def create_default_resources(self):
        """Create default resources for a player if not exists."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO PlayerResources (user_id, conversation_id, player_name, money, supplies, influence)
                VALUES (%s, %s, %s, 100, 20, 10)
                ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
            """, (self.user_id, self.conversation_id, self.player_name))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error creating default resources: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def create_default_vitals(self):
        """Create default vitals for a player if not exists."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO PlayerVitals (user_id, conversation_id, player_name, energy, hunger)
                VALUES (%s, %s, %s, 100, 100)
                ON CONFLICT (user_id, conversation_id, player_name) DO NOTHING
            """, (self.user_id, self.conversation_id, self.player_name))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error creating default vitals: {e}")
        finally:
            cursor.close()
            conn.close()
