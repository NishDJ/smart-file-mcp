"""
Main server module for the MCP server.
Provides API endpoints for interacting with the file monitoring system.

This module serves as the central component of the MCP (Model Context Protocol) server,
implementing the RESTful API interface using FastAPI. It coordinates between the file monitoring
system, conversation management, and response handling components.

Key components:
- FastAPI application with endpoint definitions
- API key authentication via middleware
- Background tasks for automated cleanup and metrics collection
- Metrics middleware for request tracking
- Error handling and validation

Architecture:
The server acts as the orchestrator, receiving client requests, validating them,
routing them to the appropriate components, and returning structured responses.
It maintains no state of its own but delegates to specialized components.

Usage:
To start the server, simply run:
    python mcp_server.py

The server will start on the host and port specified in the configuration,
monitor the specified directories for file changes, and provide API endpoints
for querying information about those changes.

API Endpoints:
- GET /: Basic server information
- GET /health: Health check endpoint
- GET /metrics: Get server metrics (requires authentication if enabled)
- GET /files: List all monitored files (requires authentication if enabled)
- GET /files/{file_path}: Get information about a specific file (requires authentication if enabled)
- GET /conversations: List all conversations (requires authentication if enabled)
- GET /conversations/{file_path}: Get conversation for a specific file (requires authentication if enabled)
- POST /query: Query about a specific file (requires authentication if enabled)
- POST /notify: Notify about a file change (requires authentication if enabled)
- DELETE /conversations/{file_path}: Clear conversation for a specific file (requires authentication if enabled)
- POST /conversations/cleanup: Clean up inactive conversations (requires authentication if enabled)
- GET /config: Get server configuration (requires authentication if enabled)
"""

import os
import sys
import json
import logging
import traceback
import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Body, Request, Depends, status, Header, Security, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

import config
from file_monitor import FileMonitor
from conversation_manager import ConversationManager
from response_handler import ResponseHandler
from metrics import metrics

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol server for monitoring and managing file changes",
    version="1.0.0"
)

# Middleware for tracking request metrics
class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track metrics for each request.
    Records request counts, response times, and error rates.
    """
    async def dispatch(self, request: Request, call_next):
        # Record start time
        start_time = time.time()
        
        # Record request
        endpoint = request.url.path
        metrics.record_api_request(endpoint)
        
        # Process request
        try:
            response = await call_next(request)
            
            # Record response time
            duration = time.time() - start_time
            metrics.record_response_time(endpoint, duration)
            
            # Record error if applicable
            if response.status_code >= 400:
                metrics.record_api_error(endpoint)
            
            return response
        except Exception as e:
            # Record error
            metrics.record_api_error(endpoint)
            raise e

# Add middlewares
app.add_middleware(MetricsMiddleware)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Pydantic models for request validation
class FileQuery(BaseModel):
    """Model for file query requests"""
    file_path: str = Field(..., description="Path to the file to query")
    query: str = Field(..., description="Query about the file")

class ChangeNotification(BaseModel):
    """Model for file change notifications"""
    file_path: str = Field(..., description="Path to the file that changed")
    event_type: str = Field(..., description="Type of change event (created, modified, deleted, moved)")
    content_before: Optional[str] = Field(None, description="File content before the change")
    content_after: Optional[str] = Field(None, description="File content after the change")
    timestamp: Optional[str] = Field(None, description="Timestamp of the change event")

class CleanupRequest(BaseModel):
    """Model for conversation cleanup requests"""
    days: int = Field(7, description="Number of days of inactivity before cleaning up conversations")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Timestamp of the error")

# Initialize components
try:
    file_monitor = FileMonitor()
    conversation_manager = ConversationManager()
    response_handler = ResponseHandler()
    logger.info("Components initialized successfully")
except Exception as e:
    logger.critical(f"Error initializing components: {e}")
    traceback.print_exc()
    sys.exit(1)

# Background task to periodically clean up inactive conversations
async def cleanup_task():
    """
    Background task to periodically clean up inactive conversations.
    Runs on a schedule and removes conversations that have been inactive
    for a specified period of time.
    """
    cleanup_interval = 24 * 3600  # 24 hours
    inactive_days = 7  # 7 days of inactivity
    
    while True:
        try:
            logger.info(f"Running scheduled cleanup of inactive conversations (inactive for {inactive_days} days)")
            count = conversation_manager.cleanup_inactive_conversations(inactive_days)
            logger.info(f"Cleaned up {count} inactive conversations")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
        
        # Sleep for the specified interval
        await asyncio.sleep(cleanup_interval)

# Background task to update metrics
async def metrics_update_task():
    """
    Background task to update metrics periodically.
    Updates metrics about file monitoring and conversation management.
    """
    update_interval = 60  # Update metrics every minute
    
    while True:
        try:
            # Update file metrics
            metrics.update_monitored_files(len(file_monitor.active_files))
            metrics.update_active_conversations(len(conversation_manager.conversations))
            
            logger.debug("Updated metrics")
        except Exception as e:
            logger.error(f"Error in metrics update task: {e}")
        
        # Sleep for the specified interval
        await asyncio.sleep(update_interval)

# Background task for a specific action
def background_save_conversations():
    """
    Background task to save conversations.
    Called after API operations that modify conversations to ensure
    changes are persisted asynchronously.
    """
    try:
        conversation_manager.save_conversations()
        logger.debug("Conversations saved in background")
    except Exception as e:
        logger.error(f"Error saving conversations in background: {e}")

# Custom exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle all uncaught exceptions.
    Provides a consistent error response format for any unhandled exceptions.
    """
    logger.error(f"Uncaught exception: {exc}")
    logger.error(traceback.format_exc())
    
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_handler.error_response(f"Internal server error: {str(exc)}")
    )

# API key validation
async def get_api_key(api_key: str = Security(api_key_header)):
    """
    Validate API key if enabled.
    Used as a dependency for protected endpoints.
    """
    if not config.API_KEY_ENABLED:
        return None
    
    if api_key == config.API_KEY:
        return api_key
    
    logger.warning(f"Invalid API key attempt: {api_key}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key"
    )

# Event handlers
@app.on_event("startup")
async def startup_event():
    """
    Start the file monitor and background tasks on server startup.
    Initializes all necessary components and background tasks.
    """
    logger.info("Starting MCP server")
    try:
        # Start file monitor
        file_monitor.start()
        logger.info("File monitor started successfully")
        
        # Start background cleanup task
        asyncio.create_task(cleanup_task())
        logger.info("Background cleanup task started")
        
        # Start metrics update task
        asyncio.create_task(metrics_update_task())
        logger.info("Metrics update task started")
    except Exception as e:
        logger.critical(f"Error during startup: {e}")
        logger.critical(traceback.format_exc())
        # Don't exit here as we want the API to be available even if startup tasks fail
        # We'll try to recover on the next request

@app.on_event("shutdown")
async def shutdown_event():
    """
    Stop the file monitor on server shutdown.
    Ensures all components are properly shut down and data is saved.
    """
    logger.info("Stopping MCP server")
    try:
        # Stop file monitor
        file_monitor.stop()
        logger.info("File monitor stopped successfully")
        
        # Save conversations one last time
        conversation_manager.save_conversations()
        logger.info("Conversations saved during shutdown")
        
        # Save metrics one last time
        metrics.stop()
        logger.info("Metrics saved during shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        logger.error(traceback.format_exc())

# API endpoints
@app.get("/")
async def root():
    """
    Root endpoint.
    Provides basic information about the server status.
    """
    return {
        "message": "MCP Server is running",
        "version": "1.0.0",
        "documentation": "/docs",
        "monitoring_status": "active" if file_monitor.observers else "inactive",
        "watched_directories": file_monitor.watch_dirs,
        "file_patterns": file_monitor.patterns,
        "active_files_count": len(file_monitor.active_files),
        "conversations_count": len(conversation_manager.conversations),
        "auth_required": config.API_KEY_ENABLED
    }

@app.get("/health")
async def health():
    """
    Health check endpoint.
    Used by monitoring systems to check if the server is operational.
    Returns 200 OK if healthy, 503 Service Unavailable if not.
    """
    is_healthy = len(file_monitor.observers) > 0  # Check if file monitor is running
    
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    response = {
        "status": "ok" if is_healthy else "error",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "file_monitor": "active" if file_monitor.observers else "inactive",
            "conversation_manager": "active" if conversation_manager.active_provider is not None else "inactive",
            "active_files": len(file_monitor.active_files),
            "conversations": len(conversation_manager.conversations),
            "auth_required": config.API_KEY_ENABLED
        }
    }
    
    return JSONResponse(status_code=status_code, content=response)

@app.get("/metrics", dependencies=[Depends(get_api_key)])
async def get_metrics():
    """
    Get server metrics.
    Returns metrics about server performance, file changes, and queries.
    Protected by API key authentication if enabled.
    """
    try:
        return metrics.get_metrics_report()
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error getting metrics: {str(e)}")
        )

@app.get("/files", dependencies=[Depends(get_api_key)])
async def get_files():
    """
    Get list of monitored files.
    Returns a list of all files being monitored by the system.
    Protected by API key authentication if enabled.
    """
    return {
        "files": file_monitor.get_active_files(),
        "count": len(file_monitor.active_files),
        "patterns": file_monitor.patterns,
        "watched_directories": file_monitor.watch_dirs
    }

@app.get("/files/{file_path:path}", dependencies=[Depends(get_api_key)])
async def get_file_info(file_path: str):
    """
    Get information about a specific file.
    Returns history and details about a specific monitored file.
    Protected by API key authentication if enabled.
    """
    # Check if file exists
    if file_path not in file_monitor.active_files:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=response_handler.error_response("File not found", file_path)
        )
    
    try:
        # Get file history
        history = file_monitor.get_file_history(file_path)
        
        # Get current file content
        content = file_monitor.get_file_content(file_path)
        
        # Get conversation if it exists
        conversation_exists = file_path in conversation_manager.conversations
        
        return {
            "file_path": file_path,
            "history": history,
            "content_preview": content[:1000] if content else None,  # First 1000 chars
            "content_length": len(content) if content else 0,
            "has_conversation": conversation_exists,
            "history_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error getting file info: {str(e)}", file_path)
        )

@app.get("/conversations", dependencies=[Depends(get_api_key)])
async def get_conversations():
    """
    Get all conversations.
    Returns a list of all active conversations in the system.
    Protected by API key authentication if enabled.
    """
    try:
        conversations = conversation_manager.get_all_conversations()
        
        # Add message counts
        for conv in conversations:
            conv["message_count"] = len(conv["messages"])
            # Remove the actual messages to reduce response size
            conv["messages"] = []
        
        return {
            "conversations": conversations,
            "count": len(conversations),
            "inactive_count": len(conversation_manager.get_inactive_conversations())
        }
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error getting conversations: {str(e)}")
        )

@app.get("/conversations/{file_path:path}", dependencies=[Depends(get_api_key)])
async def get_conversation(file_path: str):
    """
    Get conversation for a specific file.
    Returns the conversation history for a specific file.
    Protected by API key authentication if enabled.
    """
    try:
        # Get conversation history
        history = conversation_manager.get_conversation_history(file_path)
        
        if not history:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=response_handler.error_response("Conversation not found", file_path)
            )
        
        return {
            "file_path": file_path,
            "conversation_history": history,
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting conversation: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error getting conversation: {str(e)}", file_path)
        )

@app.post("/query", dependencies=[Depends(get_api_key)])
async def query_file(query: FileQuery, background_tasks: BackgroundTasks):
    """
    Query about a specific file.
    This is the main endpoint for getting AI-powered insights about file changes.
    It sends the query to the appropriate AI provider via the conversation manager.
    Protected by API key authentication if enabled.
    """
    logger.info(f"Received query for file {query.file_path}: {query.query}")
    
    try:
        # Record query start time
        start_time = time.time()
        
        # Check if file exists
        if query.file_path not in file_monitor.active_files:
            metrics.record_query(time.time() - start_time, error=True)
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=response_handler.error_response("File not found", query.file_path)
            )
        
        # Process query
        response = conversation_manager.query_file(query.file_path, query.query)
        
        # Record query completion
        query_time = time.time() - start_time
        metrics.record_query(query_time, error="error" in response)
        
        # Schedule a background save
        background_tasks.add_task(background_save_conversations)
        
        # Check for error in response
        if "error" in response:
            logger.error(f"Error in assistant response: {response['error']}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=response_handler.error_response(response['error'], query.file_path)
            )
        
        # Process and standardize response
        processed_response = response_handler.standardize_response(response)
        
        # Enhance response with additional data
        enhanced_response = response_handler.enhance_response(
            processed_response,
            {
                "query": query.query,
                "file_path": query.file_path,
                "query_time_seconds": query_time
            }
        )
        
        return enhanced_response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        metrics.record_query(time.time() - start_time, error=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error processing query: {str(e)}", query.file_path)
        )

@app.post("/notify", dependencies=[Depends(get_api_key)])
async def notify_change(change: ChangeNotification, background_tasks: BackgroundTasks):
    """
    Notify about a file change.
    Allows external systems to notify the MCP server about file changes.
    This is useful when the file monitor cannot detect changes directly.
    Protected by API key authentication if enabled.
    """
    logger.info(f"Received change notification for file {change.file_path}: {change.event_type}")
    
    try:
        # Validate event type
        valid_event_types = ["created", "modified", "deleted", "moved"]
        if change.event_type not in valid_event_types:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=response_handler.error_response(
                    f"Invalid event type. Must be one of: {', '.join(valid_event_types)}",
                    change.file_path
                )
            )
        
        # Record file change in metrics
        metrics.record_file_change(change.event_type)
        
        # Convert the notification to a file change
        change_data = {
            "event_type": change.event_type,
            "file_path": change.file_path,
            "timestamp": change.timestamp or datetime.now().isoformat(),
            "content_before": change.content_before,
            "content_after": change.content_after
        }
        
        # Add to conversation
        conversation_manager.add_file_change(change.file_path, change_data)
        
        # Schedule a background save
        background_tasks.add_task(background_save_conversations)
        
        return {
            "status": "ok",
            "message": f"Change notification processed for file {change.file_path}",
            "file_path": change.file_path,
            "event_type": change.event_type
        }
    except Exception as e:
        logger.error(f"Error processing change notification: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error processing change notification: {str(e)}", change.file_path)
        )

@app.delete("/conversations/{file_path:path}", dependencies=[Depends(get_api_key)])
async def clear_conversation(file_path: str, background_tasks: BackgroundTasks):
    """
    Clear conversation for a specific file.
    Resets the conversation history for a file while keeping the system messages.
    Protected by API key authentication if enabled.
    """
    try:
        success = conversation_manager.clear_conversation(file_path)
        
        if not success:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=response_handler.error_response("Conversation not found", file_path)
            )
        
        # Schedule a background save
        background_tasks.add_task(background_save_conversations)
        
        return {
            "status": "ok",
            "message": f"Conversation cleared for file {file_path}"
        }
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error clearing conversation: {str(e)}", file_path)
        )

@app.post("/conversations/cleanup", dependencies=[Depends(get_api_key)])
async def cleanup_conversations(request: CleanupRequest, background_tasks: BackgroundTasks):
    """
    Clean up inactive conversations.
    Removes conversations that have been inactive for a specified number of days.
    Protected by API key authentication if enabled.
    """
    try:
        # Validate days parameter
        if request.days < 1:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content=response_handler.error_response("Days parameter must be at least 1")
            )
        
        count = conversation_manager.cleanup_inactive_conversations(request.days)
        
        # Schedule a background save
        background_tasks.add_task(background_save_conversations)
        
        return {
            "status": "ok",
            "message": f"Cleaned up {count} inactive conversations",
            "inactive_threshold_days": request.days,
            "remaining_conversations": len(conversation_manager.conversations)
        }
    except Exception as e:
        logger.error(f"Error cleaning up conversations: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error cleaning up conversations: {str(e)}")
        )

@app.get("/config", dependencies=[Depends(get_api_key)])
async def get_config():
    """
    Get server configuration.
    Returns the current server configuration settings.
    Protected by API key authentication if enabled.
    """
    try:
        # Return a subset of configuration options that are safe to expose
        return {
            "server": {
                "host": config.SERVER_HOST,
                "port": config.SERVER_PORT,
                "auth_required": config.API_KEY_ENABLED
            },
            "monitoring": {
                "watch_directories": config.WATCH_DIRECTORIES,
                "file_patterns": config.FILE_PATTERNS
            },
            "ai_provider": {
                "provider": config.AI_PROVIDER,
                "model": config.MODEL_CONFIGS[config.AI_PROVIDER]["model"]
            },
            "history": {
                "max_entries": config.MAX_HISTORY_ENTRIES
            },
            "cache": {
                "expiry": config.CACHE_EXPIRY,
                "max_size": config.MAX_CACHE_SIZE
            },
            "response_format": config.DEFAULT_RESPONSE_FORMAT
        }
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_handler.error_response(f"Error getting configuration: {str(e)}")
        )

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    try:
        uvicorn.run(
            "mcp_server:app",
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
            reload=True  # Enable auto-reload during development
        )
    except Exception as e:
        logger.critical(f"Error starting server: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1) 