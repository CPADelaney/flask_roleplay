# logic/currency_generator.py

import json
import re
import logging
from db.connection import get_db_connection
from logic.chatgpt_integration import get_chatgpt_response

class CurrencyGenerator:
    """
    Generates and manages setting-specific currency systems.
    """
    
    def __init__(self, user_id, conversation_id):
        """Initialize the currency generator."""
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    async def get_currency_system(self):
        """
        Get the current currency system for this game world.
        If none exists, generate one based on the setting.
        """
        # Check if we already have a currency system
        existing_system = await self._get_existing_currency()
        if existing_system:
            return existing_system
        
        # If not, generate a new one
        setting_context = await self._get_setting_context()
        currency_system = await self._generate_currency_system(setting_context)
        
        # Store it for future use
        await self._store_currency_system(currency_system)
        
        return currency_system
    
    async def format_currency(self, amount):
        """
        Format a currency amount according to the current system.
        
        Args:
            amount: The amount to format (integer)
            
        Returns:
            Formatted currency string (e.g., "50 credits" or "£25")
        """
        currency_system = await self.get_currency_system()
        
        # Handle negative amounts
        is_negative = amount < 0
        abs_amount = abs(amount)
        
        # Format according to the template
        template = currency_system.get("format_template", "{{amount}} {{currency}}")
        currency_name = currency_system.get("currency_name", "money")
        currency_plural = currency_system.get("currency_plural", currency_name + "s")
        symbol = currency_system.get("currency_symbol", "")
        
        # Choose singular or plural form
        currency = currency_name if abs_amount == 1 else currency_plural
        
        # Apply the template
        formatted = template.replace("{{amount}}", str(abs_amount))
        formatted = formatted.replace("{{currency}}", currency)
        formatted = formatted.replace("{{symbol}}", symbol if symbol else "")
        
        # Add negative sign if needed
        if is_negative:
            formatted = "-" + formatted
        
        return formatted
    
    async def _get_existing_currency(self):
        """Check if a currency system already exists for this game."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT currency_name, currency_plural, minor_currency_name, minor_currency_plural,
                       exchange_rate, currency_symbol, format_template, description
                FROM CurrencySystem
                WHERE user_id=%s AND conversation_id=%s
                LIMIT 1
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    "currency_name": row[0],
                    "currency_plural": row[1],
                    "minor_currency_name": row[2],
                    "minor_currency_plural": row[3],
                    "exchange_rate": row[4],
                    "currency_symbol": row[5],
                    "format_template": row[6],
                    "description": row[7]
                }
            
            return None
            
        finally:
            cursor.close()
            conn.close()
    
    async def _get_setting_context(self):
        """Get the current setting context for currency generation."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT value 
                FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='EnvironmentDesc'
                LIMIT 1
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if row:
                return row[0]
            
            # If no environment description, try getting current setting name
            cursor.execute("""
                SELECT value 
                FROM CurrentRoleplay
                WHERE user_id=%s AND conversation_id=%s AND key='CurrentSetting'
                LIMIT 1
            """, (self.user_id, self.conversation_id))
            
            row = cursor.fetchone()
            
            if row:
                return row[0]
            
            # Fallback
            return "A modern setting with a standard economy"
            
        finally:
            cursor.close()
            conn.close()
    
    async def _generate_currency_system(self, setting_context):
        """
        Use GPT to generate an appropriate currency system for the setting.
        """
        prompt = f"""
        Create a unique, immersive currency system for this game world:
        
        Setting: {setting_context}
        
        Design a currency system that feels authentic and fits naturally with this setting.
        Consider the technology level, social structure, and cultural elements.
        
        Examples:
        - For cyberpunk: "credits" with symbol "₢", perhaps with minor units like "bits"
        - For fantasy medieval: "gold sovereigns" and "silver pieces" with exchange rate 20:1
        - For post-apocalyptic: "scrip" or "ration tickets" with no symbol
        
        Create a realistic, setting-appropriate currency that would be used for daily transactions.
        
        Provide your response in this exact JSON format:
        {{
          "currency_name": "singular name of primary currency unit",
          "currency_plural": "plural form of primary currency",
          "minor_currency_name": "singular name of secondary currency unit (optional)",
          "minor_currency_plural": "plural form of secondary currency (optional)",
          "exchange_rate": integer (how many minor units = 1 major unit),
          "currency_symbol": "symbol if appropriate (optional)",
          "format_template": "how to format (e.g., '{{symbol}}{{amount}}' or '{{amount}} {{currency}}')",
          "description": "brief explanation of the currency system"
        }}
        
        Focus on creating something memorable that enhances immersion.
        """
        
        try:
            gpt_response = await get_chatgpt_response(
                self.conversation_id,
                setting_context,
                prompt
            )
            
            # Extract JSON data
            if gpt_response.get("type") == "function_call":
                currency_system = gpt_response.get("function_args", {})
            else:
                response_text = gpt_response.get("response", "")
                # Try to extract JSON from the response
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
                if json_match:
                    try:
                        currency_system = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        currency_system = {}
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'{[\s\S]*}', response_text)
                    if json_match:
                        try:
                            currency_system = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            currency_system = {}
                    else:
                        currency_system = {}
            
            # Ensure we have the required fields
            if not currency_system.get("currency_name"):
                # Extract currency name using regex as fallback
                currency_name_match = re.search(r'"currency_name":\s*"([^"]+)"', response_text)
                if currency_name_match:
                    currency_system["currency_name"] = currency_name_match.group(1)
                else:
                    # Last resort fallback
                    if "cyberpunk" in setting_context.lower():
                        currency_system["currency_name"] = "credit"
                    elif "fantasy" in setting_context.lower() or "medieval" in setting_context.lower():
                        currency_system["currency_name"] = "gold piece"
                    elif "apocalyptic" in setting_context.lower():
                        currency_system["currency_name"] = "scrip"
                    else:
                        currency_system["currency_name"] = "coin"
            
            # Set plural form if missing
            if not currency_system.get("currency_plural"):
                name = currency_system.get("currency_name", "coin")
                # Apply simple pluralization rules
                if name.endswith("y"):
                    currency_system["currency_plural"] = name[:-1] + "ies"
                elif name.endswith("s") or name.endswith("x") or name.endswith("z") or name.endswith("ch") or name.endswith("sh"):
                    currency_system["currency_plural"] = name + "es"
                else:
                    currency_system["currency_plural"] = name + "s"
            
            # Set format template if missing
            if not currency_system.get("format_template"):
                if currency_system.get("currency_symbol"):
                    currency_system["format_template"] = "{{symbol}}{{amount}}"
                else:
                    currency_system["format_template"] = "{{amount}} {{currency}}"
            
            # Set exchange rate if missing
            if not currency_system.get("exchange_rate"):
                currency_system["exchange_rate"] = 100
            
            # Set description if missing
            if not currency_system.get("description"):
                currency_system["description"] = f"Standard currency for this setting: {currency_system.get('currency_name', 'coin')}"
            
            return currency_system
                
        except Exception as e:
            logging.error(f"Error generating currency system: {e}")
            # Default fallback currency system
            return {
                "currency_name": "coin",
                "currency_plural": "coins",
                "minor_currency_name": None,
                "minor_currency_plural": None,
                "exchange_rate": 100,
                "currency_symbol": None,
                "format_template": "{{amount}} {{currency}}",
                "description": "Standard currency used in this world"
            }
    
    async def _store_currency_system(self, currency_system):
        """Store the currency system in the database."""
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            setting_context = await self._get_setting_context()
            
            cursor.execute("""
                INSERT INTO CurrencySystem 
                (user_id, conversation_id, currency_name, currency_plural, 
                 minor_currency_name, minor_currency_plural, exchange_rate, 
                 currency_symbol, format_template, description, setting_context)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET 
                    currency_name = EXCLUDED.currency_name,
                    currency_plural = EXCLUDED.currency_plural,
                    minor_currency_name = EXCLUDED.minor_currency_name,
                    minor_currency_plural = EXCLUDED.minor_currency_plural,
                    exchange_rate = EXCLUDED.exchange_rate,
                    currency_symbol = EXCLUDED.currency_symbol,
                    format_template = EXCLUDED.format_template,
                    description = EXCLUDED.description,
                    setting_context = EXCLUDED.setting_context
            """, (
                self.user_id, self.conversation_id, 
                currency_system.get("currency_name"), currency_system.get("currency_plural"),
                currency_system.get("minor_currency_name"), currency_system.get("minor_currency_plural"),
                currency_system.get("exchange_rate"), currency_system.get("currency_symbol"),
                currency_system.get("format_template"), currency_system.get("description"),
                setting_context
            ))
            
            conn.commit()
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error storing currency system: {e}")
        finally:
            cursor.close()
            conn.close()
