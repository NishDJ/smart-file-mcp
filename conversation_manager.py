"""
Conversation manager module for the MCP server.
Manages assistant conversations for each file.

This module is responsible for managing conversations with AI assistants
for each monitored file. It creates and maintains separate conversations
for each file, stores the conversation history, and handles interactions
with different AI providers (Anthropic Claude and OpenAI).

Key components:
- Message: Represents a message in a conversation
- Conversation: Manages a conversation for a specific file
- AIProvider: Base class for AI providers
- AnthropicProvider: Provider for Anthropic Claude
- OpenAIProvider: Provider for OpenAI
- ConversationManager: Main class for managing all conversations

Architecture:
The ConversationManager maintains a collection of Conversation objects,
each tied to a specific file. It delegates AI interactions to the appropriate
provider based on configuration. Conversations and messages are persisted to disk
and can be loaded on startup.

Usage:
    conversation_manager = ConversationManager()
    
    # Add a file change to a conversation
    conversation_manager.add_file_change(file_path, change_data)
    
    # Query about a file
    response = conversation_manager.query_file(file_path, query)
    
    # Get conversation history
    history = conversation_manager.get_conversation_history(file_path)
    
    # Clean up inactive conversations
    conversation_manager.cleanup_inactive_conversations(days=7)

Provider Architecture:
The module uses a provider pattern to abstract away the differences between
AI providers. Each provider implements a common interface (AIProvider) and
handles the specifics of interacting with its respective AI service.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Import AI providers conditionally
try:
    # Always import both for type checking, but use conditionally
    import anthropic
    import openai
    
    PROVIDERS_AVAILABLE = {
        "anthropic": bool(config.ANTHROPIC_API_KEY),
        "openai": bool(config.OPENAI_API_KEY)
    }
except ImportError as e:
    logger.warning(f"Could not import AI provider packages: {e}")
    # Create empty dicts so we can still check keys
    PROVIDERS_AVAILABLE = {"anthropic": False, "openai": False}

class Message:
    """
    Class to represent a message in a conversation.
    
    This class represents a single message in a conversation, including
    its role (system, user, or assistant), content, and timestamp.
    
    Attributes:
        role (str): Role of the message sender (system, user, or assistant)
        content (str): Content of the message
        timestamp (float): Unix timestamp of when the message was created
    """
    def __init__(self, role: str, content: str, timestamp: float = None):
        """
        Initialize a message.
        
        Args:
            role: Role of the message sender (system, user, or assistant)
            content: Content of the message
            timestamp: Unix timestamp of when the message was created (default: current time)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create a message from dictionary.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            A new Message instance
        """
        timestamp = time.mktime(datetime.fromisoformat(data['timestamp']).timetuple()) if 'timestamp' in data else None
        return cls(data['role'], data['content'], timestamp)

class Conversation:
    """
    Class to represent an assistant conversation for a file.
    
    This class manages the conversation history for a specific file,
    including system, user, and assistant messages. It provides methods
    to add messages, retrieve message history, and manage conversation
    cleanup.
    
    Attributes:
        file_path (str): Path to the file this conversation is about
        messages (List[Message]): List of messages in the conversation
        last_activity (float): Unix timestamp of the last activity
        created_at (float): Unix timestamp of when the conversation was created
        metadata (Dict[str, Any]): Additional metadata about the conversation
    """
    def __init__(self, file_path: str):
        """
        Initialize a conversation for a file.
        
        Args:
            file_path: Path to the file
        """
        self.file_path = file_path
        self.messages: List[Message] = []
        self.last_activity: float = time.time()
        self.created_at: float = time.time()
        self.metadata: Dict[str, Any] = {
            "provider": config.AI_PROVIDER,
            "model": config.MODEL_CONFIGS[config.AI_PROVIDER]["model"]
        }
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.
        
        Args:
            role: Role of the message sender (system, user, or assistant)
            content: Content of the message
        """
        message = Message(role, content)
        self.messages.append(message)
        self.last_activity = message.timestamp
    
    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.
        
        Args:
            content: Content of the system message
        """
        self.add_message("system", content)
    
    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            content: Content of the user message
        """
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: Content of the assistant message
        """
        self.add_message("assistant", content)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the conversation.
        
        Returns:
            List of dictionaries representing all messages
        """
        return [message.to_dict() for message in self.messages]
    
    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent messages in the conversation.
        
        Args:
            count: Maximum number of recent messages to return
            
        Returns:
            List of dictionaries representing the most recent messages
        """
        return [message.to_dict() for message in self.messages[-count:]]
    
    def clear_messages(self) -> None:
        """
        Clear all messages in the conversation.
        
        This method removes all non-system messages, keeping only
        the system messages (initialization and context).
        """
        # Keep system messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        self.messages = system_messages
    
    def prune_old_messages(self, max_messages: int = 50) -> None:
        """
        Prune old messages to keep conversation size manageable.
        
        This method reduces the size of the conversation by removing old
        non-system messages while keeping all system messages.
        
        Args:
            max_messages: Maximum number of messages to keep
        """
        if len(self.messages) <= max_messages:
            return
        
        # Always keep system messages
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        other_messages = [msg for msg in self.messages if msg.role != "system"]
        
        # Keep the most recent messages
        if len(other_messages) > max_messages - len(system_messages):
            other_messages = other_messages[-(max_messages - len(system_messages)):]
        
        # Recombine messages
        self.messages = system_messages + other_messages
    
    def is_inactive(self, max_inactive_time: float = 3600 * 24 * 7) -> bool:
        """
        Check if the conversation is inactive for a long time.
        
        Args:
            max_inactive_time: Maximum inactive time in seconds
            
        Returns:
            True if the conversation is inactive, False otherwise
        """
        return time.time() - self.last_activity > max_inactive_time
    
    def get_age(self) -> float:
        """
        Get the age of the conversation in seconds.
        
        Returns:
            Age of the conversation in seconds
        """
        return time.time() - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            "file_path": self.file_path,
            "messages": [message.to_dict() for message in self.messages],
            "last_activity": datetime.fromtimestamp(self.last_activity).isoformat(),
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Create a conversation from dictionary.
        
        Args:
            data: Dictionary containing conversation data
            
        Returns:
            A new Conversation instance
        """
        conversation = cls(data['file_path'])
        conversation.messages = [Message.from_dict(msg) for msg in data['messages']]
        conversation.last_activity = time.mktime(datetime.fromisoformat(data['last_activity']).timetuple())
        
        # Handle optional fields
        if 'created_at' in data:
            conversation.created_at = time.mktime(datetime.fromisoformat(data['created_at']).timetuple())
        
        if 'metadata' in data:
            conversation.metadata = data['metadata']
        
        return conversation

class AIProvider:
    """
    Base class for AI providers.
    
    This abstract class defines the interface that all AI providers
    must implement. It provides methods for initialization and querying
    the AI model.
    
    Attributes:
        name (str): Name of the provider
        is_available (bool): Whether the provider is available
        client: Client for interacting with the AI service
    """
    def __init__(self):
        """Initialize the provider"""
        self.name = "base"
        self.is_available = False
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize the provider.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        return False
    
    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Query the AI provider.
        
        Args:
            messages: List of messages to send to the AI
            **kwargs: Additional arguments for the provider
            
        Returns:
            Response from the AI provider
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("This method must be implemented by subclasses")

class AnthropicProvider(AIProvider):
    """
    Anthropic Claude provider.
    
    This class implements the AIProvider interface for Anthropic Claude.
    It handles initialization of the Anthropic client and querying the
    Claude model.
    
    Attributes:
        name (str): Name of the provider ("anthropic")
        is_available (bool): Whether Anthropic is available
        client: Anthropic client instance
    """
    def __init__(self):
        """Initialize the Anthropic provider"""
        super().__init__()
        self.name = "anthropic"
        self.is_available = PROVIDERS_AVAILABLE.get("anthropic", False)
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize the Anthropic provider.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.is_available:
            logger.warning("Anthropic API key not available")
            return False
        
        try:
            self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            return True
        except Exception as e:
            logger.error(f"Error initializing Anthropic client: {e}")
            return False
    
    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Query Claude.
        
        Args:
            messages: List of messages to send to Claude
            **kwargs: Additional arguments for Claude
            
        Returns:
            Response from Claude containing the generated content
        """
        if not self.client:
            if not self.initialize():
                return {"error": "Anthropic client not available"}
        
        try:
            model_config = config.MODEL_CONFIGS["anthropic"]
            response = self.client.messages.create(
                model=model_config["model"],
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Extract the assistant's response
            return {"content": response.content[0].text}
        except Exception as e:
            logger.error(f"Error querying Anthropic: {e}")
            return {"error": f"Error querying Anthropic: {str(e)}"}

class OpenAIProvider(AIProvider):
    """
    OpenAI provider.
    
    This class implements the AIProvider interface for OpenAI.
    It handles initialization of the OpenAI client and querying the
    OpenAI models.
    
    Attributes:
        name (str): Name of the provider ("openai")
        is_available (bool): Whether OpenAI is available
        client: OpenAI client instance
    """
    def __init__(self):
        """Initialize the OpenAI provider"""
        super().__init__()
        self.name = "openai"
        self.is_available = PROVIDERS_AVAILABLE.get("openai", False)
        self.client = None
    
    def initialize(self) -> bool:
        """
        Initialize the OpenAI provider.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.is_available:
            logger.warning("OpenAI API key not available")
            return False
        
        try:
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            return False
    
    def query(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Query OpenAI.
        
        Args:
            messages: List of messages to send to OpenAI
            **kwargs: Additional arguments for OpenAI
            
        Returns:
            Response from OpenAI containing the generated content
        """
        if not self.client:
            if not self.initialize():
                return {"error": "OpenAI client not available"}
        
        try:
            model_config = config.MODEL_CONFIGS["openai"]
            response = self.client.chat.completions.create(
                model=model_config["model"],
                max_tokens=model_config["max_tokens"],
                temperature=model_config["temperature"],
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            # Extract the assistant's response
            return {"content": response.choices[0].message.content}
        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return {"error": f"Error querying OpenAI: {str(e)}"}

class ConversationManager:
    """
    Manager for file-specific assistant conversations.
    
    This class manages conversations for files, handling creation,
    persistence, querying, and cleanup of conversations. It delegates
    AI interactions to the appropriate provider based on configuration.
    
    Attributes:
        conversations_file (str): Path to the file where conversations are saved
        conversations (Dict[str, Conversation]): Dictionary mapping file paths to conversations
        max_conversations (int): Maximum number of conversations to keep
        cleanup_interval (int): How often to clean up conversations, in seconds
        last_cleanup (float): When conversations were last cleaned up
        providers (Dict[str, AIProvider]): Dictionary of available AI providers
        active_provider_name (str): Name of the active AI provider
        active_provider (AIProvider): Active AI provider instance
    """
    def __init__(self, 
                 conversations_file: str = "conversations.json",
                 max_conversations: int = 100,
                 cleanup_interval: int = 3600):  # 1 hour
        """
        Initialize the conversation manager.
        
        Args:
            conversations_file: Path to the file where conversations are saved
            max_conversations: Maximum number of conversations to keep
            cleanup_interval: How often to clean up conversations, in seconds
        """
        self.conversations_file = conversations_file
        self.conversations: Dict[str, Conversation] = {}
        self.max_conversations = max_conversations
        self.cleanup_interval = cleanup_interval
        self.last_cleanup: float = time.time()
        
        # Initialize AI providers
        self.providers = {
            "anthropic": AnthropicProvider(),
            "openai": OpenAIProvider()
        }
        
        # Set active provider
        self.active_provider_name = config.AI_PROVIDER
        self.active_provider = self.providers.get(self.active_provider_name)
        if not self.active_provider:
            logger.error(f"Provider {self.active_provider_name} not available")
            # Try to fall back to an available provider
            for name, provider in self.providers.items():
                if provider.is_available:
                    logger.info(f"Falling back to provider: {name}")
                    self.active_provider_name = name
                    self.active_provider = provider
                    break
        
        # Initialize the client
        if self.active_provider:
            self.active_provider.initialize()
            logger.info(f"Using AI provider: {self.active_provider_name}")
        else:
            logger.warning("No AI provider available")
        
        self.load_conversations()
    
    def load_conversations(self) -> None:
        """
        Load conversations from file if it exists.
        
        This method attempts to load previously saved conversations from
        a JSON file. If the file doesn't exist or there's an error loading
        it, the conversations will start from scratch.
        """
        if os.path.exists(self.conversations_file):
            try:
                with open(self.conversations_file, 'r') as f:
                    data = json.load(f)
                
                for conv_data in data:
                    conversation = Conversation.from_dict(conv_data)
                    self.conversations[conversation.file_path] = conversation
                
                logger.info(f"Loaded {len(self.conversations)} conversations from {self.conversations_file}")
            except Exception as e:
                logger.error(f"Error loading conversations: {e}")
                # Start with empty conversations on error
                self.conversations = {}
    
    def save_conversations(self) -> None:
        """
        Save conversations to file.
        
        This method saves the current conversations to a JSON file for
        persistence. It converts each conversation to a dictionary
        representation and writes them to the file.
        """
        try:
            conversations_list = [conv.to_dict() for conv in self.conversations.values()]
            
            with open(self.conversations_file, 'w') as f:
                json.dump(conversations_list, f, indent=2)
            
            logger.debug(f"Saved {len(self.conversations)} conversations to {self.conversations_file}")
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def get_conversation(self, file_path: str) -> Conversation:
        """
        Get or create a conversation for a file.
        
        This method returns an existing conversation for a file, or creates
        a new one if it doesn't exist. It also checks if cleanup is needed
        and removes old conversations if there are too many.
        
        Args:
            file_path: Path to the file
            
        Returns:
            The conversation for the file
        """
        # Check if we need to do cleanup
        self._check_cleanup()
        
        if file_path not in self.conversations:
            # Create a new conversation
            conversation = Conversation(file_path)
            # Add initial system message
            conversation.add_system_message(
                f"You are an assistant monitoring changes to the file '{file_path}'. "
                f"Your task is to analyze changes, provide insights, and answer questions about this file."
            )
            self.conversations[file_path] = conversation
            
            # If we've exceeded max conversations, remove the oldest inactive one
            if len(self.conversations) > self.max_conversations:
                self._remove_oldest_inactive_conversation()
        
        return self.conversations[file_path]
    
    def _check_cleanup(self) -> None:
        """
        Check if we need to do cleanup of old conversations.
        
        This method checks if it's time to clean up old conversations
        based on the cleanup interval, and runs the cleanup if needed.
        """
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_conversations()
            self.last_cleanup = time.time()
    
    def _cleanup_conversations(self) -> None:
        """
        Cleanup old and inactive conversations.
        
        This method prunes old messages in all conversations and removes
        inactive conversations if there are too many.
        """
        # First, prune old messages in all conversations
        for conversation in self.conversations.values():
            conversation.prune_old_messages()
        
        # Check if we need to remove inactive conversations
        if len(self.conversations) <= self.max_conversations:
            return
        
        # Get a list of inactive conversations
        inactive_conversations = [
            file_path for file_path, conv in self.conversations.items()
            if conv.is_inactive()
        ]
        
        # Sort by last activity time (oldest first)
        inactive_conversations.sort(
            key=lambda fp: self.conversations[fp].last_activity
        )
        
        # Remove oldest inactive conversations until we're under the limit
        conversations_to_remove = len(self.conversations) - self.max_conversations
        if conversations_to_remove > 0:
            for file_path in inactive_conversations[:conversations_to_remove]:
                logger.info(f"Removing inactive conversation for file: {file_path}")
                del self.conversations[file_path]
        
        # Save after cleanup
        self.save_conversations()
    
    def _remove_oldest_inactive_conversation(self) -> None:
        """
        Remove the oldest inactive conversation.
        
        This method finds and removes the oldest conversation based on
        last activity time to make room for new conversations.
        """
        if not self.conversations:
            return
        
        # Find oldest conversation
        oldest_file_path = None
        oldest_activity = float('inf')
        
        for file_path, conversation in self.conversations.items():
            if conversation.last_activity < oldest_activity:
                oldest_activity = conversation.last_activity
                oldest_file_path = file_path
        
        if oldest_file_path:
            logger.info(f"Removing oldest conversation for file: {oldest_file_path}")
            del self.conversations[oldest_file_path]
    
    def add_file_change(self, file_path: str, change_data: Dict[str, Any]) -> None:
        """
        Add a file change to the relevant conversation.
        
        This method creates a message about a file change and adds it to
        the conversation for that file.
        
        Args:
            file_path: Path to the file
            change_data: Data about the change
        """
        conversation = self.get_conversation(file_path)
        
        # Create a user message about the change
        event_type = change_data['event_type']
        timestamp = change_data['timestamp']
        
        message = f"File {event_type} at {timestamp}."
        
        # Add diff information if available
        if change_data.get('diff'):
            if isinstance(change_data['diff'], str):
                message += f"\nDiff: {change_data['diff']}"
            else:
                message += "\nChanges:"
                for change in change_data['diff']:
                    if 'type' in change and 'line' in change and 'content' in change:
                        # New diff format
                        change_type = change.get('type', 'unknown')
                        line_num = change.get('line', 'unknown')
                        content = change.get('content', '')
                        message += f"\n- {change_type.capitalize()} at line {line_num}: '{content}'"
                    else:
                        # Old diff format
                        line_num = change.get('line', 'unknown')
                        before = change.get('before', 'none')
                        after = change.get('after', 'none')
                        message += f"\n- Line {line_num}: '{before}' -> '{after}'"
        
        # Add file hash information
        if change_data.get('file_hash_before'):
            message += f"\nFile hash before: {change_data['file_hash_before']}"
        if change_data.get('file_hash_after'):
            message += f"\nFile hash after: {change_data['file_hash_after']}"
        
        # Add the message
        conversation.add_user_message(message)
        
        # Save conversations
        self.save_conversations()
    
    def query_file(self, file_path: str, query: str) -> Dict[str, Any]:
        """
        Query about a file and get a response.
        
        This method sends a query to the AI provider about a file and
        returns the response. The query is added to the conversation
        for that file.
        
        Args:
            file_path: Path to the file
            query: Query about the file
            
        Returns:
            Response from the AI provider
        """
        conversation = self.get_conversation(file_path)
        
        # Add the query as a user message
        conversation.add_user_message(query)
        
        # Make sure we have an active provider
        if not self.active_provider:
            logger.error("No AI provider available")
            error_message = "No AI provider available"
            
            # Add an error message to the conversation
            conversation.add_assistant_message(json.dumps({
                "error": error_message,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Save conversations
            self.save_conversations()
            
            return {
                "error": error_message,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }
        
        # Prepare messages for the API
        formatted_messages = []
        for message in conversation.messages:
            formatted_messages.append({
                "role": message.role,
                "content": message.content
            })
        
        # Call AI provider
        response = self.active_provider.query(formatted_messages)
        
        # Check if we got an error
        if "error" in response:
            logger.error(f"Error querying AI provider: {response['error']}")
            
            # Add an error message to the conversation
            error_message = f"Error querying AI provider: {response['error']}"
            conversation.add_assistant_message(json.dumps({
                "error": error_message,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Save conversations
            self.save_conversations()
            
            return {
                "error": error_message,
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            }
        
        # Extract the assistant's response
        assistant_response = response["content"]
        
        # Add the response to the conversation
        conversation.add_assistant_message(assistant_response)
        
        # Parse JSON response
        try:
            parsed_response = json.loads(assistant_response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse assistant response as JSON: {assistant_response}")
            parsed_response = {
                "error": "Failed to parse response as JSON",
                "raw_response": assistant_response
            }
        
        # Save conversations
        self.save_conversations()
        
        return parsed_response
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        Get all conversations.
        
        Returns:
            List of dictionaries representing all conversations
        """
        return [conv.to_dict() for conv in self.conversations.values()]
    
    def get_conversation_history(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get conversation history for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of dictionaries representing the conversation history
        """
        if file_path not in self.conversations:
            return []
        
        return self.conversations[file_path].get_messages()
    
    def clear_conversation(self, file_path: str) -> bool:
        """
        Clear a conversation.
        
        This method clears all non-system messages from a conversation,
        keeping only the system messages.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the conversation was cleared, False if it wasn't found
        """
        if file_path in self.conversations:
            conversation = self.conversations[file_path]
            conversation.clear_messages()
            self.save_conversations()
            return True
        
        return False
    
    def delete_conversation(self, file_path: str) -> bool:
        """
        Delete a conversation.
        
        This method completely removes a conversation.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the conversation was deleted, False if it wasn't found
        """
        if file_path in self.conversations:
            del self.conversations[file_path]
            self.save_conversations()
            return True
        
        return False
    
    def get_inactive_conversations(self, days: int = 7) -> List[str]:
        """
        Get a list of inactive conversations.
        
        Args:
            days: Number of days of inactivity
            
        Returns:
            List of file paths for inactive conversations
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        
        inactive = []
        for file_path, conversation in self.conversations.items():
            if conversation.last_activity < cutoff_time:
                inactive.append(file_path)
        
        return inactive
    
    def cleanup_inactive_conversations(self, days: int = 7) -> int:
        """
        Clean up inactive conversations.
        
        Args:
            days: Number of days of inactivity
            
        Returns:
            Number of conversations that were cleaned up
        """
        inactive = self.get_inactive_conversations(days)
        
        for file_path in inactive:
            self.delete_conversation(file_path)
        
        return len(inactive) 