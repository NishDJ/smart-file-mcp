"""
Configuration module for the MCP server.
Loads configuration from environment variables and provides defaults.

This module is responsible for loading and validating configuration settings
for the MCP server from environment variables. It provides sensible defaults
where appropriate and validates critical settings.

Key components:
- Server configuration (host, port)
- API security settings
- AI provider configuration
- File monitoring settings
- Logging configuration
- Model configurations for different AI providers
- Response format specification
- History and cache settings

Architecture:
Configuration settings are loaded from environment variables using the python-dotenv
package. Critical settings are validated, and paths are expanded where needed.
The module provides centralized access to configuration values for all other components.

Usage:
Import the module to access configuration values:
    import config
    
    # Access configuration values
    host = config.SERVER_HOST
    port = config.SERVER_PORT
    
    # Access nested configurations
    model_name = config.MODEL_CONFIGS[config.AI_PROVIDER]["model"]

Environment Variables:
- SERVER_HOST: Host to bind the server to (default: 127.0.0.1)
- SERVER_PORT: Port to bind the server to (default: 8000)
- API_KEY_ENABLED: Whether API key authentication is enabled (default: false)
- API_KEY: API key for authentication (required if API_KEY_ENABLED is true)
- AI_PROVIDER: AI provider to use (default: anthropic, options: anthropic, openai)
- ANTHROPIC_API_KEY: API key for Anthropic Claude (required if AI_PROVIDER is anthropic)
- OPENAI_API_KEY: API key for OpenAI (required if AI_PROVIDER is openai)
- WATCH_DIRECTORIES: Comma-separated list of directories to watch (default: current directory)
- FILE_PATTERNS: Comma-separated list of file patterns to watch (default: *.py,*.js,*.txt)
- LOG_LEVEL: Logging level (default: INFO)
- LOG_FILE: Log file path (default: mcp_server.log)
- ASSISTANT_MODEL: Anthropic model to use (default: claude-3-opus-20240229)
- ASSISTANT_TEMPERATURE: Temperature for AI generation (default: 0.7)
- ASSISTANT_MAX_TOKENS: Maximum tokens for AI generation (default: 1000)
- OPENAI_MODEL: OpenAI model to use (default: gpt-4o)
- MAX_HISTORY_ENTRIES: Maximum number of history entries per file (default: 100)
- HISTORY_FILE: File to store history in (default: file_history.json)
- CACHE_EXPIRY: Cache expiry time in seconds (default: 3600)
- MAX_CACHE_SIZE: Maximum number of files to cache (default: 1000)
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# API security settings
API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
if API_KEY_ENABLED and not API_KEY:
    raise ValueError("API_KEY environment variable is required when API_KEY_ENABLED is true")

# AI Provider configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "anthropic").lower()
VALID_PROVIDERS = ["anthropic", "openai"]
if AI_PROVIDER not in VALID_PROVIDERS:
    raise ValueError(f"AI_PROVIDER must be one of: {', '.join(VALID_PROVIDERS)}")

# API keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required API keys based on provider
if AI_PROVIDER == "anthropic" and not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required when AI_PROVIDER is anthropic")
if AI_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required when AI_PROVIDER is openai")

# File monitoring configuration
WATCH_DIRECTORIES = os.getenv("WATCH_DIRECTORIES", "").split(",")
if not WATCH_DIRECTORIES or WATCH_DIRECTORIES == [""]:
    WATCH_DIRECTORIES = [os.getcwd()]  # Default to current directory
else:
    # Expand paths (e.g., handle ~)
    WATCH_DIRECTORIES = [os.path.expanduser(d) for d in WATCH_DIRECTORIES]

# File patterns to monitor
FILE_PATTERNS = os.getenv("FILE_PATTERNS", "*.py,*.js,*.txt").split(",")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "mcp_server.log")

# Assistant configuration
ASSISTANT_MODEL = os.getenv("ASSISTANT_MODEL", "claude-3-opus-20240229")
ASSISTANT_TEMPERATURE = float(os.getenv("ASSISTANT_TEMPERATURE", "0.7"))
ASSISTANT_MAX_TOKENS = int(os.getenv("ASSISTANT_MAX_TOKENS", "1000"))

# Provider-specific model mappings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Model configurations for different providers
MODEL_CONFIGS = {
    "anthropic": {
        "model": ASSISTANT_MODEL,
        "temperature": ASSISTANT_TEMPERATURE,
        "max_tokens": ASSISTANT_MAX_TOKENS
    },
    "openai": {
        "model": OPENAI_MODEL,
        "temperature": ASSISTANT_TEMPERATURE,
        "max_tokens": ASSISTANT_MAX_TOKENS
    }
}

# Response configuration
DEFAULT_RESPONSE_FORMAT = {
    "type": "json",
    "structure": {
        "file_path": "str",
        "changes": "list",
        "analysis": "str",
        "confidence": "float",
        "timestamp": "str"
    }
}

# History configuration
MAX_HISTORY_ENTRIES = int(os.getenv("MAX_HISTORY_ENTRIES", "100"))
HISTORY_FILE = os.getenv("HISTORY_FILE", "file_history.json")

# Cache configuration
CACHE_EXPIRY = int(os.getenv("CACHE_EXPIRY", "3600"))  # seconds
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))  # entries 