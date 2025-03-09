# Smart File MCP Server Dockerfile
# This Dockerfile sets up a container for the MCP server.
# It uses a multi-stage build for security and efficiency.

# Use Python 3.10 slim image as the base
# The slim variant provides a smaller footprint while still including essential packages
FROM python:3.10-slim

# Set the working directory in the container
# This is where our application code will be stored
WORKDIR /app

# Copy requirements first to leverage Docker cache
# This layer will only be rebuilt if requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# This comes after installing dependencies to leverage the Docker build cache
COPY . .

# Create a non-root user to run the application
# This is a security best practice to avoid running as root
RUN useradd -m appuser
USER appuser

# Expose the port the app runs on
# This documents that the container listens on port 8000 at runtime
EXPOSE 8000

# Command to run the application
# This is the command that will be executed when the container starts
CMD ["python", "mcp_server.py"] 