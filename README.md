# Smart File MCP (Model Context Protocol) Server

A server designed to monitor and manage file changes within a specified zone of the local file system. For each file, the MCP server initiates a new assistant conversation, facilitating intelligent responses to queries about file modifications.

## Features

- File change monitoring in specified directories
- File content caching with intelligent diff calculation
- Assistant conversation management for each file
- Structured responses to client queries
- Context-aware information about file changes
- Robust error handling and resilience
- Automatic cleanup of inactive conversations
- Comprehensive logging
- Multiple AI provider support (Anthropic Claude, OpenAI GPT)
- API key authentication
- Background tasks for cleanup
- Extensive unit and integration tests
- Metrics collection and reporting
- Docker support for easy deployment

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure your environment variables (copy `.env.example` to `.env` and fill in your API keys)

## Usage

### Running Locally

1. Start the server:
   ```
   python mcp_server.py
   ```
2. Use the client interface to query information about file changes:
   ```
   python mcp_client.py query test_files/sample.py "What changes were made to this file recently?"
   ```

### Running with Docker

1. Configure your environment variables in a `.env` file
2. Use Docker Compose to start the server:
   ```
   docker-compose up -d
   ```

## Client Usage Examples

### With API Key Authentication
When API key authentication is enabled, use the `--api-key` parameter:
```
python mcp_client.py --api-key YOUR_API_KEY query test_files/sample.py "Analyze changes"
```

### Basic Commands
```
# Check server status
python mcp_client.py status

# List all monitored files
python mcp_client.py files

# Get information about a specific file
python mcp_client.py file test_files/sample.py

# Query about file changes
python mcp_client.py query test_files/sample.py "Summarize the recent changes"

# Manually notify about a change
python mcp_client.py notify test_files/sample.py modified

# Get server metrics
python mcp_client.py config
```

## Testing

### Unit Tests
Run unit tests for components:
```
python test_components.py
```

### Integration Tests
Run integration tests for the full system:
```
python test_mcp.py
```

These tests verify various aspects of the system, from individual component functionality to the entire flow of file creation, modification, querying, and deletion.

## Metrics and Monitoring

The server collects metrics about:
- API requests and response times
- File change events
- Query performance
- System health

Access metrics via the `/metrics` endpoint (requires API key if enabled).

## Configuration

Configure the monitored directories and other settings in your `.env` file and `config.py`.

### Key Configuration Options

- `WATCH_DIRECTORIES`: Directories to monitor for file changes
- `FILE_PATTERNS`: File patterns to monitor (e.g., *.py, *.js, *.txt)
- `AI_PROVIDER`: AI provider to use (anthropic or openai)
- `API_KEY_ENABLED`: Whether to enable API key authentication
- `MAX_HISTORY_ENTRIES`: Maximum number of file change entries to keep
- `MAX_CACHE_SIZE`: Maximum number of files to keep in the content cache
- `ANTHROPIC_API_KEY`: API key for Claude assistant
- `OPENAI_API_KEY`: API key for OpenAI (optional)

## Components

- `mcp_server.py`: Main server module with API endpoints
- `file_monitor.py`: Tracks file changes with content caching
- `conversation_manager.py`: Manages assistant conversations with context
- `response_handler.py`: Handles structured responses
- `config.py`: Configuration settings
- `metrics.py`: Collects and reports metrics
- `mcp_client.py`: Client interface for testing
- `test_mcp.py`: Integration testing script
- `test_components.py`: Unit tests for components

## API Endpoints

The server provides the following endpoints:

- `GET /health`: Health check endpoint
- `GET /metrics`: Get server metrics
- `GET /files`: List all monitored files
- `GET /files/{file_path}`: Get information about a specific file
- `GET /conversations`: List all conversations
- `GET /conversations/{file_path}`: Get conversation for a specific file
- `POST /query`: Query about a specific file
- `POST /notify`: Notify about a file change
- `DELETE /conversations/{file_path}`: Clear conversation for a specific file
- `POST /conversations/cleanup`: Clean up inactive conversations
- `GET /config`: Get server configuration

All endpoints (except for root and health) require API key authentication when `API_KEY_ENABLED=true`.

## Security

- API key authentication for endpoints (configurable)
- Input validation on all endpoints
- Error handling and logging of security issues
- Configurable access controls

## License

MIT 