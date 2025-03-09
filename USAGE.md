# Smart File MCP Server - Usage Guide

This guide demonstrates how to use the Model Context Protocol (MCP) server to monitor and query file changes.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file to add your Anthropic API key and configure the directories to watch.

## Starting the Server

Start the MCP server:

```bash
python mcp_server.py
```

The server will start monitoring the specified directories for file changes.

## Using the Client

The `mcp_client.py` script provides a command-line interface for interacting with the server.

### Check Server Status

```bash
python mcp_client.py status
```

### List Monitored Files

```bash
python mcp_client.py files
```

### Get File Information

```bash
python mcp_client.py file test_files/sample.py
```

### List Conversations

```bash
python mcp_client.py conversations
```

### Get Conversation for a File

```bash
python mcp_client.py conversation test_files/sample.py
```

### Query About a File

```bash
python mcp_client.py query test_files/sample.py "What does this file do?"
```

### Manually Notify About a File Change

```bash
python mcp_client.py notify test_files/sample.py modified
```

### Clear a Conversation

```bash
python mcp_client.py clear test_files/sample.py
```

## API Endpoints

The MCP server provides the following API endpoints:

- `GET /health` - Check server health
- `GET /files` - List all monitored files
- `GET /files/{file_path}` - Get information about a specific file
- `GET /conversations` - List all conversations
- `GET /conversations/{file_path}` - Get conversation for a specific file
- `POST /query` - Query about a specific file
- `POST /notify` - Notify about a file change
- `DELETE /conversations/{file_path}` - Clear conversation for a specific file

## Example Workflow

1. Start the server
2. Make changes to a file in the monitored directory
3. Query the server about the file changes
4. Get intelligent insights about the changes

## Programmatic Usage

You can also use the MCP server programmatically:

```python
import requests

# Server URL
SERVER_URL = "http://127.0.0.1:8000"

# Query about a file
def query_file(file_path, query):
    payload = {
        "file_path": file_path,
        "query": query
    }
    
    response = requests.post(f"{SERVER_URL}/query", json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")

# Example query
response = query_file("test_files/sample.py", "What does this file do?")
print(response)
``` 