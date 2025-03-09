#!/usr/bin/env python3
"""
Integration test script for the MCP server.
This script tests the entire flow of the MCP server.
"""

import os
import sys
import json
import time
import argparse
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default server URL
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"

class MCPTester:
    """Test the MCP server"""
    
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        self.server_url = server_url
        self.test_file_path = "test_files/test_sample.py"
        self.test_content_1 = """
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
"""
        self.test_content_2 = """
def hello():
    print("Hello, World!")

def add(a, b):
    return a + b

if __name__ == "__main__":
    hello()
    print(f"2 + 2 = {add(2, 2)}")
"""
    
    def check_server_health(self) -> bool:
        """Check if the server is running and healthy"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200 and response.json().get("status") == "ok":
                logger.info("Server is running and healthy")
                return True
            else:
                logger.error(f"Server is not healthy: {response.status_code} - {response.text}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to server")
            return False
    
    def create_test_file(self, content: str) -> bool:
        """Create a test file"""
        try:
            # Ensure test directory exists
            os.makedirs(os.path.dirname(self.test_file_path), exist_ok=True)
            
            # Write the file
            with open(self.test_file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Created test file: {self.test_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating test file: {e}")
            return False
    
    def modify_test_file(self, content: str) -> bool:
        """Modify the test file"""
        try:
            with open(self.test_file_path, 'w') as f:
                f.write(content)
            
            logger.info(f"Modified test file: {self.test_file_path}")
            return True
        except Exception as e:
            logger.error(f"Error modifying test file: {e}")
            return False
    
    def delete_test_file(self) -> bool:
        """Delete the test file"""
        try:
            if os.path.exists(self.test_file_path):
                os.remove(self.test_file_path)
                logger.info(f"Deleted test file: {self.test_file_path}")
                return True
            else:
                logger.warning(f"Test file doesn't exist: {self.test_file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting test file: {e}")
            return False
    
    def wait_for_file_detection(self, seconds: int = 5) -> None:
        """Wait for the file monitor to detect changes"""
        logger.info(f"Waiting {seconds} seconds for file detection...")
        time.sleep(seconds)
    
    def notify_change(self, event_type: str, content_before: Optional[str] = None, content_after: Optional[str] = None) -> bool:
        """Manually notify about a file change"""
        try:
            payload = {
                "file_path": self.test_file_path,
                "event_type": event_type,
                "content_before": content_before,
                "content_after": content_after,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(f"{self.server_url}/notify", json=payload)
            
            if response.status_code == 200:
                logger.info(f"Change notification sent: {response.json().get('message')}")
                return True
            else:
                logger.error(f"Error notifying change: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error notifying change: {e}")
            return False
    
    def query_file(self, query: str) -> Dict[str, Any]:
        """Query about the test file"""
        try:
            payload = {
                "file_path": self.test_file_path,
                "query": query
            }
            
            response = requests.post(f"{self.server_url}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Query response received: {json.dumps(data)[:100]}...")
                return data
            else:
                logger.error(f"Error querying file: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error querying file: {e}")
            return {"error": str(e)}
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the test file"""
        try:
            encoded_path = requests.utils.quote(self.test_file_path)
            response = requests.get(f"{self.server_url}/files/{encoded_path}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Got file info for: {data['file_path']}")
                return data
            else:
                logger.error(f"Error getting file info: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"error": str(e)}
    
    def get_conversation(self) -> Dict[str, Any]:
        """Get conversation for the test file"""
        try:
            encoded_path = requests.utils.quote(self.test_file_path)
            response = requests.get(f"{self.server_url}/conversations/{encoded_path}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Got conversation for: {data['file_path']}")
                return data
            else:
                logger.error(f"Error getting conversation: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return {"error": str(e)}
    
    def clear_conversation(self) -> bool:
        """Clear conversation for the test file"""
        try:
            encoded_path = requests.utils.quote(self.test_file_path)
            response = requests.delete(f"{self.server_url}/conversations/{encoded_path}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(data.get("message", "Conversation cleared"))
                return True
            else:
                logger.error(f"Error clearing conversation: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False
    
    def run_full_test(self) -> bool:
        """Run a full test of the MCP server flow"""
        logger.info("Starting full MCP server test...")
        
        # Check server health
        if not self.check_server_health():
            logger.error("Server health check failed. Aborting test.")
            return False
        
        # Step 1: Create a new file
        logger.info("Step 1: Creating test file...")
        if not self.create_test_file(self.test_content_1):
            return False
        
        # Wait for detection
        self.wait_for_file_detection()
        
        # Step 2: Manually notify about file creation
        logger.info("Step 2: Notifying about file creation...")
        if not self.notify_change("created", None, self.test_content_1):
            return False
        
        # Step 3: Query about the file
        logger.info("Step 3: Querying about the new file...")
        query_response = self.query_file("What does this file do?")
        if "error" in query_response:
            logger.error(f"Error in query response: {query_response.get('error')}")
            return False
        
        # Step 4: Get file info
        logger.info("Step 4: Getting file info...")
        file_info = self.get_file_info()
        if "error" in file_info:
            logger.error(f"Error getting file info: {file_info.get('error')}")
            return False
        
        # Step 5: Modify the file
        logger.info("Step 5: Modifying the file...")
        if not self.modify_test_file(self.test_content_2):
            return False
        
        # Wait for detection
        self.wait_for_file_detection()
        
        # Step 6: Manually notify about file modification
        logger.info("Step 6: Notifying about file modification...")
        if not self.notify_change("modified", self.test_content_1, self.test_content_2):
            return False
        
        # Step 7: Query about the changes
        logger.info("Step 7: Querying about the changes...")
        query_response = self.query_file("What changed in this file?")
        if "error" in query_response:
            logger.error(f"Error in query response: {query_response.get('error')}")
            return False
        
        # Step 8: Get conversation
        logger.info("Step 8: Getting conversation...")
        conversation = self.get_conversation()
        if "error" in conversation:
            logger.error(f"Error getting conversation: {conversation.get('error')}")
            return False
        
        # Step 9: Clear conversation
        logger.info("Step 9: Clearing conversation...")
        if not self.clear_conversation():
            return False
        
        # Step 10: Delete the file
        logger.info("Step 10: Deleting the file...")
        if not self.delete_test_file():
            return False
        
        # Wait for detection
        self.wait_for_file_detection()
        
        # Step 11: Manually notify about file deletion
        logger.info("Step 11: Notifying about file deletion...")
        if not self.notify_change("deleted", self.test_content_2, None):
            return False
        
        logger.info("Test completed successfully!")
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MCP Server Tester")
    parser.add_argument("--url", default=DEFAULT_SERVER_URL, help="MCP server URL")
    args = parser.parse_args()
    
    tester = MCPTester(args.url)
    success = tester.run_full_test()
    
    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 