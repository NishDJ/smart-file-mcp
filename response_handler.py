"""
Response handler module for the MCP server.
Handles formatting and structuring of responses.

This module is responsible for handling the formatting, validation, and
enhancement of responses from AI providers before they are returned to
the client. It ensures that responses have a consistent structure and
contain all required fields.

Key components:
- ResponseHandler: Class for formatting, validating, and enhancing responses

Architecture:
The ResponseHandler class provides static methods for processing responses.
It works with the response format defined in the configuration and ensures
that all responses conform to that format.

Usage:
    # Format a response
    formatted = ResponseHandler.format_response(data)
    
    # Validate a response
    is_valid = ResponseHandler.validate_response(response)
    
    # Standardize a response
    standardized = ResponseHandler.standardize_response(raw_response)
    
    # Enhance a response with additional data
    enhanced = ResponseHandler.enhance_response(response, additional_data)
    
    # Create an error response
    error = ResponseHandler.error_response("Error message", "file_path")
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

class ResponseHandler:
    """
    Handler for formatting and structuring responses.
    
    This class provides methods for formatting, validating, standardizing,
    and enhancing responses from AI providers. It ensures that responses
    have a consistent structure and contain all required fields.
    
    All methods are static and can be called without instantiating the class.
    """
    
    @staticmethod
    def format_response(data: Dict[str, Any], format_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format a response according to a format specification.
        
        This method ensures that the response contains all required fields
        according to the format specification. If no format specification is
        provided, the default from the configuration is used.
        
        Args:
            data: The response data to format
            format_spec: The format specification to use (default: from config)
            
        Returns:
            The formatted response
        """
        if format_spec is None:
            format_spec = config.DEFAULT_RESPONSE_FORMAT
        
        # Ensure required fields exist
        if "file_path" not in data and "file_path" in format_spec.get("structure", {}):
            data["file_path"] = "unknown"
        
        if "timestamp" not in data and "timestamp" in format_spec.get("structure", {}):
            data["timestamp"] = datetime.now().isoformat()
        
        # If format type is JSON, ensure it's properly formatted
        if format_spec.get("type") == "json":
            # This function already returns a dict, which will be JSON serialized,
            # so there's nothing special to do here
            return data
        
        # Handle other format types as needed
        return data
    
    @staticmethod
    def validate_response(response: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate a response against a schema.
        
        This method checks that the response contains all required fields
        and that the fields have the correct types according to the schema.
        If no schema is provided, the default from the configuration is used.
        
        Args:
            response: The response to validate
            schema: The schema to validate against (default: from config)
            
        Returns:
            True if the response is valid, False otherwise
        """
        if schema is None:
            schema = config.DEFAULT_RESPONSE_FORMAT.get("structure", {})
        
        # Simple validation - check that required fields exist and have the right type
        for field, expected_type in schema.items():
            if field not in response:
                logger.warning(f"Missing required field: {field}")
                return False
            
            # Check type
            if expected_type == "str":
                if not isinstance(response[field], str):
                    logger.warning(f"Field {field} should be a string, got {type(response[field])}")
                    return False
            elif expected_type == "int":
                if not isinstance(response[field], int):
                    logger.warning(f"Field {field} should be an integer, got {type(response[field])}")
                    return False
            elif expected_type == "float":
                if not isinstance(response[field], (int, float)):
                    logger.warning(f"Field {field} should be a number, got {type(response[field])}")
                    return False
            elif expected_type == "bool":
                if not isinstance(response[field], bool):
                    logger.warning(f"Field {field} should be a boolean, got {type(response[field])}")
                    return False
            elif expected_type == "list":
                if not isinstance(response[field], list):
                    logger.warning(f"Field {field} should be a list, got {type(response[field])}")
                    return False
            elif expected_type == "dict" or expected_type == "object":
                if not isinstance(response[field], dict):
                    logger.warning(f"Field {field} should be an object, got {type(response[field])}")
                    return False
        
        return True
    
    @staticmethod
    def standardize_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize a response to ensure it meets the expected format.
        
        This method adds default values for missing fields and ensures that
        the response has a consistent structure. It preserves all existing
        fields and adds any required fields that are missing.
        
        Args:
            raw_response: The raw response from the AI provider
            
        Returns:
            The standardized response
        """
        standardized = {}
        
        # Standard fields from config
        for field in config.DEFAULT_RESPONSE_FORMAT.get("structure", {}).keys():
            if field in raw_response:
                standardized[field] = raw_response[field]
            else:
                # Add default values for missing fields
                if field == "file_path":
                    standardized[field] = "unknown"
                elif field == "changes":
                    standardized[field] = []
                elif field == "analysis":
                    standardized[field] = "No analysis available"
                elif field == "confidence":
                    standardized[field] = 0.0
                elif field == "timestamp":
                    standardized[field] = datetime.now().isoformat()
                else:
                    standardized[field] = None
        
        # Include any additional fields from the raw response
        for key, value in raw_response.items():
            if key not in standardized:
                standardized[key] = value
        
        return standardized
    
    @staticmethod
    def enhance_response(response: Dict[str, Any], additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhance a response with additional data.
        
        This method adds additional data to the response, such as query
        information, file paths, and metadata.
        
        Args:
            response: The response to enhance
            additional_data: Additional data to add to the response
            
        Returns:
            The enhanced response
        """
        enhanced = response.copy()
        
        if additional_data:
            for key, value in additional_data.items():
                if key not in enhanced:
                    enhanced[key] = value
        
        # Add metadata
        if "metadata" not in enhanced:
            enhanced["metadata"] = {}
        
        enhanced["metadata"]["processed_at"] = datetime.now().isoformat()
        enhanced["metadata"]["response_format_version"] = "1.0"
        
        return enhanced
    
    @staticmethod
    def serialize_response(response: Dict[str, Any]) -> str:
        """
        Serialize a response to JSON string.
        
        Args:
            response: The response to serialize
            
        Returns:
            The serialized response as a JSON string
        """
        try:
            return json.dumps(response, indent=2)
        except Exception as e:
            logger.error(f"Error serializing response: {e}")
            return json.dumps({"error": "Error serializing response"})
    
    @staticmethod
    def error_response(message: str, file_path: str = "unknown") -> Dict[str, Any]:
        """
        Create an error response.
        
        This method creates a standardized error response that can be
        returned to the client.
        
        Args:
            message: The error message
            file_path: The file path related to the error
            
        Returns:
            The error response
        """
        response = {
            "error": message,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "success": False
        }
        
        return ResponseHandler.standardize_response(response) 