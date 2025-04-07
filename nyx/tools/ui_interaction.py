# nyx/tools/ui_interaction.py

import datetime
import logging
import uuid
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class UIConversationManager:
    """
    Enables Nyx to create and manage conversations within her own UI.
    """
    
    def __init__(self, system_context=None):
        self.system_context = system_context
        self.active_conversations = {}
        
    async def create_new_conversation(self, user_id: str, title: Optional[str] = None, 
                                 initial_message: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new conversation with a user in Nyx's UI.
        
        Args:
            user_id: ID of the user to start conversation with
            title: Optional title for the conversation
            initial_message: Optional initial message to send
            metadata: Additional metadata for the conversation
            
        Returns:
            Newly created conversation details
        """
        # Generate conversation ID
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        
        # Generate default title if none provided
        if not title:
            title = f"Conversation with {user_id} ({datetime.datetime.now().strftime('%Y-%m-%d')})"
            
        # Create conversation record
        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "messages": [],
            "status": "active",
            "metadata": metadata or {}
        }
        
        # Add initial message if provided
        if initial_message:
            message = {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "conversation_id": conversation_id,
                "sender_type": "system",  # Nyx is sending this
                "sender_id": "nyx",
                "content": initial_message,
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "sent"
            }
            conversation["messages"].append(message)
            
        # Store conversation
        self.active_conversations[conversation_id] = conversation
        
        # Call system API to create conversation if system context available
        if self.system_context:
            try:
                # This would call the actual system API
                await self.system_context.create_conversation(conversation)
            except Exception as e:
                logger.error(f"Error creating conversation via system API: {e}")
        
        logger.info(f"Created new conversation: {conversation_id} with user {user_id}")
        return conversation
        
    async def send_message(self, conversation_id: str, message_content: str,
                      attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Send a message in an existing conversation.
        
        Args:
            conversation_id: ID of the conversation
            message_content: Content of the message
            attachments: Optional list of attachments
            
        Returns:
            The sent message details
        """
        if conversation_id not in self.active_conversations:
            logger.error(f"Cannot send message: Conversation {conversation_id} not found")
            return {"error": f"Conversation {conversation_id} not found"}
            
        conversation = self.active_conversations[conversation_id]
        
        # Create message record
        message = {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "conversation_id": conversation_id,
            "sender_type": "system",
            "sender_id": "nyx",
            "content": message_content,
            "timestamp": datetime.datetime.now().isoformat(),
            "attachments": attachments or [],
            "status": "sending"
        }
        
        # Add to conversation
        conversation["messages"].append(message)
        conversation["updated_at"] = message["timestamp"]
        
        # Call system API to send message if system context available
        if self.system_context:
            try:
                # This would call the actual system API
                result = await self.system_context.send_message(message)
                message["status"] = "sent"
                
                # Update with any data from API response
                if isinstance(result, dict):
                    message.update(result)
            except Exception as e:
                logger.error(f"Error sending message via system API: {e}")
                message["status"] = "error"
                message["error"] = str(e)
        else:
            # No system context, just mark as sent
            message["status"] = "sent"
        
        logger.info(f"Sent message in conversation {conversation_id}")
        return message
        
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: ID of the conversation to get
            
        Returns:
            The conversation data if found, None otherwise
        """
        return self.active_conversations.get(conversation_id)
        
    async def get_conversations_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversations with a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of conversations with this user
        """
        return [
            conv for conv in self.active_conversations.values()
            if conv["user_id"] == user_id
        ]
        
    async def update_conversation_status(self, conversation_id: str, 
                                    status: str) -> Dict[str, Any]:
        """
        Update the status of a conversation.
        
        Args:
            conversation_id: ID of the conversation
            status: New status (active, archived, deleted)
            
        Returns:
            Updated conversation data
        """
        if conversation_id not in self.active_conversations:
            logger.error(f"Cannot update status: Conversation {conversation_id} not found")
            return {"error": f"Conversation {conversation_id} not found"}
            
        conversation = self.active_conversations[conversation_id]
        
        # Update status
        conversation["status"] = status
        conversation["updated_at"] = datetime.datetime.now().isoformat()
        
        # Call system API if available
        if self.system_context:
            try:
                await self.system_context.update_conversation(conversation)
            except Exception as e:
                logger.error(f"Error updating conversation via system API: {e}")
        
        return conversation
        
    async def search_conversation_history(self, query: str, 
                                     user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search through conversation history.
        
        Args:
            query: Search query
            user_id: Optional user ID to limit search to
            
        Returns:
            List of matching message results
        """
        results = []
        query = query.lower()
        
        for conversation in self.active_conversations.values():
            # Skip if user_id specified and doesn't match
            if user_id and conversation["user_id"] != user_id:
                continue
                
            # Search through messages
            for message in conversation["messages"]:
                if query in message["content"].lower():
                    # Add message with conversation context
                    results.append({
                        "message": message,
                        "conversation_id": conversation["id"],
                        "conversation_title": conversation["title"],
                        "user_id": conversation["user_id"]
                    })
        
        return results

    async def create_group_conversation(self, user_ids: List[str], title: Optional[str] = None,
                                   initial_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a group conversation with multiple users.
        
        Args:
            user_ids: List of user IDs to include
            title: Optional title for the conversation
            initial_message: Optional initial message
            
        Returns:
            Newly created conversation details
        """
        # Generate conversation ID
        conversation_id = f"group_{uuid.uuid4().hex[:12]}"
        
        # Generate default title if none provided
        if not title:
            title = f"Group Conversation ({datetime.datetime.now().strftime('%Y-%m-%d')})"
            
        # Create conversation record
        conversation = {
            "id": conversation_id,
            "type": "group",
            "user_ids": user_ids,
            "title": title,
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "messages": [],
            "status": "active",
            "metadata": {"group_size": len(user_ids)}
        }
        
        # Add initial message if provided
        if initial_message:
            message = {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "conversation_id": conversation_id,
                "sender_type": "system",
                "sender_id": "nyx",
                "content": initial_message,
                "timestamp": datetime.datetime.now().isoformat(),
                "status": "sent"
            }
            conversation["messages"].append(message)
            
        # Store conversation
        self.active_conversations[conversation_id] = conversation
        
        # Call system API if available
        if self.system_context:
            try:
                await self.system_context.create_conversation(conversation)
            except Exception as e:
                logger.error(f"Error creating group conversation via system API: {e}")
        
        logger.info(f"Created new group conversation: {conversation_id} with {len(user_ids)} users")
        return conversation
