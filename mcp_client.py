"""
Client interface for the MCP server.
Provides a command-line interface for interacting with the server.
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
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server URL
SERVER_URL = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"

class MCPClient:
    """Client interface for the MCP server"""
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        self.server_url = server_url
        self.api_key = api_key
        self.headers = {}
        
        # Add API key to headers if provided
        if self.api_key:
            self.headers["X-API-Key"] = self.api_key
    
    def format_json(self, data: Dict[str, Any]) -> str:
        """Format JSON data for display"""
        return json.dumps(data, indent=2)

    def check_server_status(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                logger.info("Server is running")
                data = response.json()
                # Check if API key is required but not provided
                if data.get("components", {}).get("auth_required", False) and not self.api_key:
                    logger.warning("Server requires API key, but none provided")
                return True
            else:
                logger.error(f"Server is not healthy: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to server")
            return False

    def list_files(self) -> Dict[str, Any]:
        """List all monitored files"""
        try:
            response = requests.get(f"{self.server_url}/files", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Retrieved {len(data.get('files', []))} files")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            else:
                logger.error(f"Error listing files: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return {"error": str(e)}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a specific file"""
        try:
            encoded_path = requests.utils.quote(file_path)
            response = requests.get(f"{self.server_url}/files/{encoded_path}", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Got file info for: {data['file_path']}")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            else:
                logger.error(f"Error getting file info: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {"error": str(e)}

    def list_conversations(self) -> Dict[str, Any]:
        """List all conversations"""
        try:
            response = requests.get(f"{self.server_url}/conversations", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Retrieved {len(data.get('conversations', []))} conversations")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            else:
                logger.error(f"Error listing conversations: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
            return {"error": str(e)}

    def get_conversation(self, file_path: str) -> Dict[str, Any]:
        """Get conversation for a specific file"""
        try:
            encoded_path = requests.utils.quote(file_path)
            response = requests.get(f"{self.server_url}/conversations/{encoded_path}", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Got conversation for: {data['file_path']}")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            elif response.status_code == 404:
                logger.error(f"Conversation not found for file: {file_path}")
                return {"error": "Conversation not found", "file_path": file_path}
            else:
                logger.error(f"Error getting conversation: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error getting conversation: {e}")
            return {"error": str(e)}

    def query_file(self, file_path: str, query: str) -> Dict[str, Any]:
        """Query about a specific file"""
        try:
            payload = {
                "file_path": file_path,
                "query": query
            }
            
            response = requests.post(f"{self.server_url}/query", json=payload, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Query response received for: {file_path}")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            elif response.status_code == 404:
                logger.error(f"File not found: {file_path}")
                return {"error": "File not found", "file_path": file_path}
            else:
                logger.error(f"Error querying file: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error querying file: {e}")
            return {"error": str(e)}

    def notify_change(self, file_path: str, event_type: str, content_before: Optional[str] = None, content_after: Optional[str] = None) -> Dict[str, Any]:
        """Notify about a file change"""
        try:
            payload = {
                "file_path": file_path,
                "event_type": event_type,
                "content_before": content_before,
                "content_after": content_after,
                "timestamp": datetime.now().isoformat()
            }
            
            response = requests.post(f"{self.server_url}/notify", json=payload, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Change notification sent: {data.get('message')}")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            else:
                logger.error(f"Error notifying change: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error notifying change: {e}")
            return {"error": str(e)}

    def clear_conversation(self, file_path: str) -> Dict[str, Any]:
        """Clear conversation for a specific file"""
        try:
            encoded_path = requests.utils.quote(file_path)
            response = requests.delete(f"{self.server_url}/conversations/{encoded_path}", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(data.get("message", "Conversation cleared"))
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            elif response.status_code == 404:
                logger.error(f"Conversation not found for file: {file_path}")
                return {"error": "Conversation not found", "file_path": file_path}
            else:
                logger.error(f"Error clearing conversation: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return {"error": str(e)}

    def cleanup_conversations(self, days: int = 7) -> Dict[str, Any]:
        """Clean up inactive conversations"""
        try:
            payload = {
                "days": days
            }
            
            response = requests.post(f"{self.server_url}/conversations/cleanup", json=payload, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Cleaned up {data.get('message')}")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            else:
                logger.error(f"Error cleaning up conversations: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")
            return {"error": str(e)}

    def get_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        try:
            response = requests.get(f"{self.server_url}/config", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Retrieved server configuration")
                return data
            elif response.status_code == 401:
                logger.error("Authentication failed. API key required.")
                return {"error": "Authentication failed. API key required."}
            else:
                logger.error(f"Error getting config: {response.status_code} - {response.text}")
                return {"error": f"Status {response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            return {"error": str(e)}

def manual_file_change(client: MCPClient, file_path: str, event_type: str) -> bool:
    """Manually create a file change notification"""
    content_before = None
    content_after = None
    
    # For modifications, read the file content
    if event_type == "modified" and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                content_after = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
    
    # For creations, read the file content
    elif event_type == "created" and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                content_after = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
    
    # Send notification
    response = client.notify_change(file_path, event_type, content_before, content_after)
    return "error" not in response

def display_results(data: Dict[str, Any]) -> None:
    """Display results in a formatted manner"""
    if "error" in data:
        print(f"Error: {data['error']}")
        if "details" in data:
            print(f"Details: {data['details']}")
        return
    
    print(json.dumps(data, indent=2))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="MCP Client")
    parser.add_argument("--url", default=SERVER_URL, help="MCP server URL")
    parser.add_argument("--api-key", help="API key for authentication")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Status command
    subparsers.add_parser("status", help="Check server status")
    
    # List files command
    subparsers.add_parser("files", help="List all monitored files")
    
    # Get file info command
    file_info_parser = subparsers.add_parser("file", help="Get information about a specific file")
    file_info_parser.add_argument("file_path", help="Path to the file")
    
    # List conversations command
    subparsers.add_parser("conversations", help="List all conversations")
    
    # Get conversation command
    conversation_parser = subparsers.add_parser("conversation", help="Get conversation for a specific file")
    conversation_parser.add_argument("file_path", help="Path to the file")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query about a specific file")
    query_parser.add_argument("file_path", help="Path to the file")
    query_parser.add_argument("query", help="Query about the file")
    
    # Notify command
    notify_parser = subparsers.add_parser("notify", help="Notify about a file change")
    notify_parser.add_argument("file_path", help="Path to the file")
    notify_parser.add_argument("event_type", choices=["created", "modified", "deleted", "moved"], help="Type of change event")
    
    # Clear conversation command
    clear_parser = subparsers.add_parser("clear", help="Clear conversation for a specific file")
    clear_parser.add_argument("file_path", help="Path to the file")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up inactive conversations")
    cleanup_parser.add_argument("--days", type=int, default=7, help="Number of days of inactivity")
    
    # Config command
    subparsers.add_parser("config", help="Get server configuration")
    
    args = parser.parse_args()
    
    # Initialize client
    client = MCPClient(args.url, args.api_key)
    
    # Check server status for all commands
    if not client.check_server_status():
        print("Server is not running. Please start the server first.")
        sys.exit(1)
    
    # Execute command
    if args.command == "status":
        print("Server is running")
    elif args.command == "files":
        results = client.list_files()
        display_results(results)
    elif args.command == "file":
        results = client.get_file_info(args.file_path)
        display_results(results)
    elif args.command == "conversations":
        results = client.list_conversations()
        display_results(results)
    elif args.command == "conversation":
        results = client.get_conversation(args.file_path)
        display_results(results)
    elif args.command == "query":
        results = client.query_file(args.file_path, args.query)
        display_results(results)
    elif args.command == "notify":
        success = manual_file_change(client, args.file_path, args.event_type)
        if not success:
            print("Failed to notify about file change.")
            sys.exit(1)
    elif args.command == "clear":
        results = client.clear_conversation(args.file_path)
        display_results(results)
    elif args.command == "cleanup":
        results = client.cleanup_conversations(args.days)
        display_results(results)
    elif args.command == "config":
        results = client.get_config()
        display_results(results)
    else:
        parser.print_help()

if __name__ == "__main__":
    sys.exit(main()) 