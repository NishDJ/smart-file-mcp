#!/usr/bin/env python3
"""
Unit tests for the MCP server components.
Tests the core functionality of each component.
"""

import os
import sys
import json
import time
import unittest
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path to make imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import components to test
import config
from file_monitor import FileMonitor, FileChange, FileContentCache
from conversation_manager import ConversationManager, Conversation, Message
from response_handler import ResponseHandler

class TestFileContentCache(unittest.TestCase):
    """Test the FileContentCache class"""
    
    def setUp(self):
        """Set up test environment"""
        self.cache = FileContentCache(max_size=5)
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        # Test set and get
        self.cache.set("file1.py", "content1")
        self.assertEqual(self.cache.get("file1.py"), "content1")
        
        # Test empty get
        self.assertEqual(self.cache.get("nonexistent.py"), "")
        
        # Test has_file
        self.assertTrue(self.cache.has_file("file1.py"))
        self.assertFalse(self.cache.has_file("nonexistent.py"))
        
        # Test remove
        self.cache.remove("file1.py")
        self.assertFalse(self.cache.has_file("file1.py"))
    
    def test_cache_pruning(self):
        """Test cache pruning functionality"""
        # Add items to cache
        for i in range(10):
            self.cache.set(f"file{i}.py", f"content{i}")
        
        # Check that some items were pruned (max size is 5)
        self.assertLessEqual(len(self.cache.cache), 5)
        
        # Check that most recently accessed items are still in cache
        for i in range(5, 10):
            self.assertTrue(self.cache.has_file(f"file{i}.py"))

class TestFileChange(unittest.TestCase):
    """Test the FileChange class"""
    
    def setUp(self):
        """Set up test environment"""
        self.change = FileChange("modified", "test.py")
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        change_dict = self.change.to_dict()
        self.assertEqual(change_dict["event_type"], "modified")
        self.assertEqual(change_dict["file_path"], "test.py")
        self.assertIsNotNone(change_dict["timestamp"])
    
    def test_calculate_hash(self):
        """Test file hash calculation"""
        content = "test content"
        hash_value = self.change.calculate_hash(content)
        # MD5 hash of "test content"
        expected_hash = "05a671c66aefea124cc08b76ea6d30bb"
        self.assertEqual(hash_value, expected_hash)
    
    def test_calculate_diff(self):
        """Test diff calculation"""
        self.change.content_before = "line 1\nline 2\nline 3\n"
        self.change.content_after = "line 1\nmodified line\nline 3\n"
        self.change.calculate_diff()
        
        # Check that diff was calculated
        self.assertIsNotNone(self.change.diff)
        self.assertIsInstance(self.change.diff, list)
        
        # Check that the right line was detected as changed
        found_change = False
        for diff_item in self.change.diff:
            if diff_item["type"] == "removed" and "line 2" in diff_item["content"]:
                found_change = True
        self.assertTrue(found_change)

class TestMessage(unittest.TestCase):
    """Test the Message class"""
    
    def test_message_creation(self):
        """Test creating a message"""
        message = Message("user", "test message")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "test message")
        self.assertIsNotNone(message.timestamp)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        message = Message("user", "test message")
        message_dict = message.to_dict()
        self.assertEqual(message_dict["role"], "user")
        self.assertEqual(message_dict["content"], "test message")
        self.assertIsNotNone(message_dict["timestamp"])
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        timestamp = datetime.now().isoformat()
        message_dict = {
            "role": "assistant",
            "content": "test response",
            "timestamp": timestamp
        }
        message = Message.from_dict(message_dict)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "test response")
        self.assertIsNotNone(message.timestamp)

class TestConversation(unittest.TestCase):
    """Test the Conversation class"""
    
    def setUp(self):
        """Set up test environment"""
        self.conversation = Conversation("test.py")
    
    def test_add_messages(self):
        """Test adding messages to the conversation"""
        # Add system message
        self.conversation.add_system_message("test system message")
        self.assertEqual(len(self.conversation.messages), 1)
        self.assertEqual(self.conversation.messages[0].role, "system")
        
        # Add user message
        self.conversation.add_user_message("test user message")
        self.assertEqual(len(self.conversation.messages), 2)
        self.assertEqual(self.conversation.messages[1].role, "user")
        
        # Add assistant message
        self.conversation.add_assistant_message("test assistant message")
        self.assertEqual(len(self.conversation.messages), 3)
        self.assertEqual(self.conversation.messages[2].role, "assistant")
    
    def test_get_messages(self):
        """Test getting messages from the conversation"""
        self.conversation.add_user_message("message 1")
        self.conversation.add_assistant_message("message 2")
        self.conversation.add_user_message("message 3")
        
        messages = self.conversation.get_messages()
        self.assertEqual(len(messages), 3)
        
        recent_messages = self.conversation.get_recent_messages(2)
        self.assertEqual(len(recent_messages), 2)
        self.assertEqual(recent_messages[1]["content"], "message 3")
    
    def test_clear_messages(self):
        """Test clearing messages in the conversation"""
        # Add messages
        self.conversation.add_system_message("system message")
        self.conversation.add_user_message("user message")
        self.conversation.add_assistant_message("assistant message")
        
        # Clear messages
        self.conversation.clear_messages()
        
        # Check that only system messages remain
        self.assertEqual(len(self.conversation.messages), 1)
        self.assertEqual(self.conversation.messages[0].role, "system")
    
    def test_prune_old_messages(self):
        """Test pruning old messages"""
        # Add system message
        self.conversation.add_system_message("system message")
        
        # Add 10 messages
        for i in range(10):
            self.conversation.add_user_message(f"message {i}")
        
        # Prune to 5 messages
        self.conversation.prune_old_messages(5)
        
        # Check that we have 5 messages total (4 user + 1 system)
        self.assertEqual(len(self.conversation.messages), 5)
        
        # Check that system message is still there
        self.assertEqual(self.conversation.messages[0].role, "system")
        
        # Check that most recent user messages are there
        self.assertEqual(self.conversation.messages[-1].content, "message 9")
    
    def test_is_inactive(self):
        """Test checking if conversation is inactive"""
        # Set last activity to 8 days ago
        eight_days_ago = time.time() - (8 * 24 * 3600)
        self.conversation.last_activity = eight_days_ago
        
        # Check with 7 days threshold
        self.assertTrue(self.conversation.is_inactive(7 * 24 * 3600))
        
        # Check with 10 days threshold
        self.assertFalse(self.conversation.is_inactive(10 * 24 * 3600))

class TestResponseHandler(unittest.TestCase):
    """Test the ResponseHandler class"""
    
    def test_format_response(self):
        """Test formatting a response"""
        data = {
            "analysis": "Test analysis",
            "confidence": 0.8
        }
        
        formatted = ResponseHandler.format_response(data)
        
        # Check that required fields were added
        self.assertIn("file_path", formatted)
        self.assertIn("timestamp", formatted)
    
    def test_validate_response(self):
        """Test validating a response"""
        # Valid response
        valid_response = {
            "file_path": "test.py",
            "changes": [],
            "analysis": "Test analysis",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        self.assertTrue(ResponseHandler.validate_response(valid_response))
        
        # Invalid response (missing required field)
        invalid_response = {
            "file_path": "test.py",
            "analysis": "Test analysis",
            "timestamp": datetime.now().isoformat()
        }
        self.assertFalse(ResponseHandler.validate_response(invalid_response))
        
        # Invalid response (wrong type)
        invalid_type_response = {
            "file_path": "test.py",
            "changes": "not a list",
            "analysis": "Test analysis",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
        self.assertFalse(ResponseHandler.validate_response(invalid_type_response))
    
    def test_standardize_response(self):
        """Test standardizing a response"""
        raw_response = {
            "analysis": "Test analysis",
            "extra_field": "extra value"
        }
        
        standardized = ResponseHandler.standardize_response(raw_response)
        
        # Check that required fields were added with default values
        self.assertEqual(standardized["file_path"], "unknown")
        self.assertEqual(standardized["changes"], [])
        self.assertEqual(standardized["confidence"], 0.0)
        
        # Check that existing fields were preserved
        self.assertEqual(standardized["analysis"], "Test analysis")
        
        # Check that extra fields were preserved
        self.assertEqual(standardized["extra_field"], "extra value")
    
    def test_error_response(self):
        """Test creating an error response"""
        error_message = "Test error"
        file_path = "test.py"
        
        error_response = ResponseHandler.error_response(error_message, file_path)
        
        # Check error response format
        self.assertEqual(error_response["error"], error_message)
        self.assertEqual(error_response["file_path"], file_path)
        self.assertIn("timestamp", error_response)

if __name__ == "__main__":
    unittest.main() 