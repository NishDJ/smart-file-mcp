"""
File monitoring module for the MCP server.
Uses watchdog to track file changes in specified directories.

This module is responsible for monitoring file changes within specified directories.
It detects file creation, modification, deletion, and moving events, and maintains
a history of these changes. It also provides a content cache to enable accurate diff
calculation between file versions.

Key components:
- FileContentCache: Caches file contents to enable diff calculation
- FileChange: Represents a change to a file with diff calculation
- FileHistory: Manages the history of file changes
- FileChangeHandler: Watchdog event handler for file events
- FileMonitor: Main monitoring class that coordinates the components

Architecture:
The FileMonitor initializes watchdog observers for each directory, which in turn use
FileChangeHandler to process file events. When a change is detected, it's recorded
in FileHistory and processed through the content cache to calculate diffs.

Usage:
    monitor = FileMonitor()
    monitor.start()  # Start monitoring
    
    # Get active files
    active_files = monitor.get_active_files()
    
    # Get history for a file
    history = monitor.get_file_history("path/to/file.py")
    
    # Stop monitoring
    monitor.stop()

Dependencies:
- watchdog: For file system monitoring
- difflib: For diff calculation
- hashlib: For content hashing
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, List, Set, Callable, Any
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
import difflib
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

class FileContentCache:
    """
    Cache for file contents to enable diff calculation.
    
    This cache stores the content of files to enable diff calculation
    when files are modified. It implements a simple LRU (Least Recently Used)
    eviction policy to manage memory usage.
    
    Attributes:
        cache (Dict[str, str]): Dictionary mapping file paths to content
        max_size (int): Maximum number of entries in the cache
        last_accessed (Dict[str, float]): Dictionary tracking last access time
    """
    def __init__(self, max_size: int = config.MAX_CACHE_SIZE):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.last_accessed: Dict[str, float] = {}
    
    def get(self, file_path: str) -> str:
        """
        Get cached content for a file.
        Updates the last accessed time and returns the content.
        Returns empty string if not in cache.
        
        Args:
            file_path: Path to the file
            
        Returns:
            The file content or empty string if not cached
        """
        self.last_accessed[file_path] = time.time()
        return self.cache.get(file_path, "")
    
    def set(self, file_path: str, content: str) -> None:
        """
        Cache content for a file.
        Sets the content in the cache and prunes if necessary.
        
        Args:
            file_path: Path to the file
            content: Content to cache
        """
        self.cache[file_path] = content
        self.last_accessed[file_path] = time.time()
        
        # Prune cache if it exceeds maximum size
        if len(self.cache) > self.max_size:
            self._prune_cache()
    
    def _prune_cache(self) -> None:
        """
        Remove least recently accessed entries from cache.
        Removes the oldest 10% of entries when the cache is full.
        """
        if not self.last_accessed:
            return
        
        # Sort by last accessed time
        sorted_items = sorted(self.last_accessed.items(), key=lambda x: x[1])
        
        # Remove the oldest 10% of entries
        entries_to_remove = max(1, len(sorted_items) // 10)
        for file_path, _ in sorted_items[:entries_to_remove]:
            self.cache.pop(file_path, None)
            self.last_accessed.pop(file_path, None)
    
    def remove(self, file_path: str) -> None:
        """
        Remove a file from the cache.
        
        Args:
            file_path: Path to the file to remove
        """
        self.cache.pop(file_path, None)
        self.last_accessed.pop(file_path, None)
    
    def has_file(self, file_path: str) -> bool:
        """
        Check if a file is in the cache.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file is cached, False otherwise
        """
        return file_path in self.cache

class FileChange:
    """
    Class to represent a file change.
    
    Represents a change to a file, including the type of change
    (created, modified, deleted, moved), the content before and after,
    and a calculated diff.
    
    Attributes:
        event_type (str): Type of change event
        file_path (str): Path to the file
        timestamp (float): Unix timestamp of the change
        content_before (str): Content before the change
        content_after (str): Content after the change
        diff (List[Dict] or str): Calculated diff between before and after
        file_hash_before (str): MD5 hash of content before
        file_hash_after (str): MD5 hash of content after
    """
    def __init__(self, event_type: str, file_path: str, timestamp: float = None):
        self.event_type = event_type
        self.file_path = file_path
        self.timestamp = timestamp or time.time()
        self.content_before = None
        self.content_after = None
        self.diff = None
        self.file_hash_before = None
        self.file_hash_after = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert file change to dictionary.
        
        Returns:
            Dictionary representation of the file change
        """
        return {
            "event_type": self.event_type,
            "file_path": self.file_path,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "content_before": None,  # Don't store full content in history to save space
            "content_after": None,   # Don't store full content in history to save space
            "diff": self.diff,
            "file_hash_before": self.file_hash_before,
            "file_hash_after": self.file_hash_after
        }
    
    def calculate_hash(self, content: str) -> str:
        """
        Calculate hash of file content.
        Uses MD5 for simplicity and speed (not for security).
        
        Args:
            content: Content to hash
            
        Returns:
            MD5 hex digest or None if content is None
        """
        if content is None:
            return None
        return hashlib.md5(content.encode()).hexdigest()
    
    def calculate_diff(self) -> None:
        """
        Calculate the diff between before and after content.
        Uses Python's difflib to compute a unified diff and then
        converts it to a structured format.
        """
        if self.content_before is None or self.content_after is None:
            return
        
        # Calculate file hashes
        self.file_hash_before = self.calculate_hash(self.content_before)
        self.file_hash_after = self.calculate_hash(self.content_after)
        
        # If hashes are identical, no changes
        if self.file_hash_before == self.file_hash_after:
            self.diff = "No changes"
            return
        
        # Use difflib for better diff calculation
        before_lines = self.content_before.splitlines()
        after_lines = self.content_after.splitlines()
        
        # Generate unified diff
        diff_generator = difflib.unified_diff(
            before_lines, 
            after_lines,
            fromfile='before',
            tofile='after',
            lineterm=''
        )
        
        # Parse the diff to a more structured format
        changes = []
        current_line = None
        
        for line in diff_generator:
            if line.startswith('---') or line.startswith('+++'):
                continue
            
            if line.startswith('@@'):
                # Extract line numbers from hunk header
                try:
                    line_info = line.split('@@')[1].strip()
                    before_info, after_info = line_info.split(' ')
                    before_start = int(before_info.split(',')[0][1:])
                    after_start = int(after_info.split(',')[0][1:])
                    current_line = after_start
                except Exception as e:
                    logger.error(f"Error parsing diff hunk header: {e}")
                    current_line = 0
                continue
            
            if current_line is None:
                current_line = 0
                
            if line.startswith('-'):
                changes.append({
                    "type": "removed",
                    "line": current_line,
                    "content": line[1:]
                })
            elif line.startswith('+'):
                changes.append({
                    "type": "added",
                    "line": current_line,
                    "content": line[1:]
                })
                current_line += 1
            else:
                current_line += 1
        
        self.diff = changes

class FileHistory:
    """
    Class to manage file change history.
    
    Maintains a history of changes for each file and provides methods
    to save and load this history from disk.
    
    Attributes:
        history_file (str): Path to the file where history is saved
        file_changes (Dict[str, List[FileChange]]): Dictionary mapping file paths to lists of changes
    """
    def __init__(self, history_file: str = config.HISTORY_FILE):
        self.history_file = history_file
        self.file_changes: Dict[str, List[FileChange]] = {}
        self.load_history()
    
    def load_history(self) -> None:
        """
        Load history from file if it exists.
        Restores the change history from a JSON file.
        """
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                
                for file_path, changes in data.items():
                    self.file_changes[file_path] = []
                    for change_data in changes:
                        change = FileChange(
                            change_data['event_type'],
                            change_data['file_path'],
                            # Parse ISO format timestamp back to float
                            time.mktime(datetime.fromisoformat(change_data['timestamp']).timetuple())
                        )
                        change.file_hash_before = change_data.get('file_hash_before')
                        change.file_hash_after = change_data.get('file_hash_after')
                        change.diff = change_data.get('diff')
                        self.file_changes[file_path].append(change)
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                # Start with empty history on error
                self.file_changes = {}
    
    def save_history(self) -> None:
        """
        Save history to file.
        Persists the change history to a JSON file.
        """
        try:
            history_dict = {
                file_path: [change.to_dict() for change in changes]
                for file_path, changes in self.file_changes.items()
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(history_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def add_change(self, change: FileChange) -> None:
        """
        Add a change to the history.
        Adds a change to the history for a file and saves the history.
        
        Args:
            change: The FileChange object to add
        """
        if change.file_path not in self.file_changes:
            self.file_changes[change.file_path] = []
        
        # Add change
        self.file_changes[change.file_path].append(change)
        
        # Limit history size
        if len(self.file_changes[change.file_path]) > config.MAX_HISTORY_ENTRIES:
            self.file_changes[change.file_path] = self.file_changes[change.file_path][-config.MAX_HISTORY_ENTRIES:]
        
        # Save after each change
        self.save_history()
    
    def get_changes(self, file_path: str) -> List[FileChange]:
        """
        Get all changes for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of FileChange objects or empty list if none
        """
        return self.file_changes.get(file_path, [])
    
    def get_recent_changes(self, file_path: str, count: int = 10) -> List[FileChange]:
        """
        Get recent changes for a file.
        
        Args:
            file_path: Path to the file
            count: Maximum number of changes to return
            
        Returns:
            List of the most recent FileChange objects or empty list if none
        """
        changes = self.get_changes(file_path)
        return changes[-count:] if changes else []

class FileChangeHandler(FileSystemEventHandler):
    """
    Handler for file system events.
    
    Handles file system events from watchdog and creates FileChange objects.
    
    Attributes:
        callback (Callable): Function to call with each FileChange
        patterns (List[str]): List of file patterns to monitor
        content_cache (FileContentCache): Cache for file contents
        last_modified_times (Dict[str, float]): Dictionary tracking last modification times
    """
    def __init__(self, callback: Callable[[FileChange], None], patterns: List[str], content_cache: FileContentCache):
        self.callback = callback
        self.patterns = patterns
        self.content_cache = content_cache
        self.last_modified_times: Dict[str, float] = {}
    
    def should_process_file(self, file_path: str) -> bool:
        """
        Check if the file should be processed based on patterns.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be processed, False otherwise
        """
        if not self.patterns:
            return True
        
        # Handle glob patterns properly
        from fnmatch import fnmatch
        filename = os.path.basename(file_path)
        return any(fnmatch(filename, pattern) for pattern in self.patterns)
    
    def read_file_content(self, file_path: str) -> str:
        """
        Read content of a file safely.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content or empty string on error
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return ""
    
    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle file creation events.
        
        Args:
            event: The file system event
        """
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not self.should_process_file(file_path):
            return
        
        logger.info(f"File created: {file_path}")
        change = FileChange("created", file_path)
        
        # Read content of new file
        content = self.read_file_content(file_path)
        change.content_after = content
        
        # Calculate file hash
        change.file_hash_after = change.calculate_hash(content)
        
        # Cache the content
        self.content_cache.set(file_path, content)
        
        self.callback(change)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Handle file modification events.
        
        Args:
            event: The file system event
        """
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not self.should_process_file(file_path):
            return
        
        # Watchdog can fire multiple modify events rapidly
        # Use a timestamp check to avoid duplicates
        current_time = time.time()
        last_modified = self.last_modified_times.get(file_path, 0)
        if current_time - last_modified < 1:  # Ignore if < 1 second since last modify
            return
        
        self.last_modified_times[file_path] = current_time
        logger.info(f"File modified: {file_path}")
        
        change = FileChange("modified", file_path)
        
        # Get content before the change (if available in cache)
        change.content_before = self.content_cache.get(file_path)
        
        # Read current content
        current_content = self.read_file_content(file_path)
        change.content_after = current_content
        
        # Calculate diff if we have before content
        if change.content_before:
            change.calculate_diff()
        
        # Update cache with new content
        self.content_cache.set(file_path, current_content)
        
        self.callback(change)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """
        Handle file deletion events.
        
        Args:
            event: The file system event
        """
        if event.is_directory:
            return
        
        file_path = event.src_path
        if not self.should_process_file(file_path):
            return
        
        logger.info(f"File deleted: {file_path}")
        change = FileChange("deleted", file_path)
        
        # Get content before deletion
        change.content_before = self.content_cache.get(file_path)
        change.file_hash_before = change.calculate_hash(change.content_before)
        
        # Remove from cache
        self.content_cache.remove(file_path)
        
        self.callback(change)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """
        Handle file move events.
        
        Args:
            event: The file system event
        """
        if event.is_directory:
            return
        
        src_path = event.src_path
        dest_path = event.dest_path
        
        if not (self.should_process_file(src_path) or self.should_process_file(dest_path)):
            return
        
        logger.info(f"File moved from {src_path} to {dest_path}")
        
        # Handle as deletion of source and creation of destination
        src_change = FileChange("deleted", src_path)
        src_change.content_before = self.content_cache.get(src_path)
        src_change.file_hash_before = src_change.calculate_hash(src_change.content_before)
        
        # Remove source from cache
        self.content_cache.remove(src_path)
        
        # Create destination change
        dest_change = FileChange("created", dest_path)
        dest_content = self.read_file_content(dest_path)
        dest_change.content_after = dest_content
        dest_change.file_hash_after = dest_change.calculate_hash(dest_content)
        
        # Cache destination content
        self.content_cache.set(dest_path, dest_content)
        
        # Notify both changes
        self.callback(src_change)
        self.callback(dest_change)

class FileMonitor:
    """
    Monitor file changes in specified directories.
    
    This is the main class for file monitoring. It sets up observers for
    the specified directories, handles file change events, and maintains
    a history of changes.
    
    Attributes:
        watch_dirs (List[str]): Directories to watch
        patterns (List[str]): File patterns to monitor
        observers (List[Observer]): Watchdog observers
        history (FileHistory): History of file changes
        active_files (Set[str]): Set of currently monitored files
        content_cache (FileContentCache): Cache for file contents
    """
    def __init__(self, 
                 watch_dirs: List[str] = config.WATCH_DIRECTORIES,
                 patterns: List[str] = config.FILE_PATTERNS):
        self.watch_dirs = watch_dirs
        self.patterns = patterns
        self.observers: List[Observer] = []
        self.history = FileHistory()
        self.active_files: Set[str] = set()
        self.content_cache = FileContentCache()
        
        # Initialize file list and cache
        for watch_dir in self.watch_dirs:
            self._scan_directory(watch_dir)
    
    def _scan_directory(self, directory: str) -> None:
        """
        Scan directory to build initial file list and cache content.
        
        Args:
            directory: Directory to scan
        """
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Check if file matches patterns
                    from fnmatch import fnmatch
                    if any(fnmatch(file, pattern) for pattern in self.patterns):
                        self.active_files.add(file_path)
                        
                        # Read and cache file content
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                self.content_cache.set(file_path, f.read())
                        except Exception as e:
                            logger.error(f"Error reading file {file_path} during scan: {e}")
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
    
    def _handle_file_change(self, change: FileChange) -> None:
        """
        Handle a file change event.
        
        Args:
            change: The FileChange object
        """
        # Update active files set
        if change.event_type == "created":
            self.active_files.add(change.file_path)
        elif change.event_type == "deleted":
            self.active_files.discard(change.file_path)
        
        # Add to history
        self.history.add_change(change)
    
    def start(self) -> None:
        """
        Start monitoring file changes.
        Initializes and starts watchdog observers for each directory.
        """
        for watch_dir in self.watch_dirs:
            if not os.path.exists(watch_dir):
                logger.warning(f"Watch directory does not exist: {watch_dir}")
                continue
            
            observer = Observer()
            event_handler = FileChangeHandler(self._handle_file_change, self.patterns, self.content_cache)
            observer.schedule(event_handler, watch_dir, recursive=True)
            observer.start()
            self.observers.append(observer)
            logger.info(f"Started monitoring directory: {watch_dir}")
    
    def stop(self) -> None:
        """
        Stop monitoring file changes.
        Stops all watchdog observers.
        """
        for observer in self.observers:
            observer.stop()
        
        for observer in self.observers:
            observer.join()
        
        self.observers = []
        logger.info("Stopped file monitoring")
    
    def get_active_files(self) -> List[str]:
        """
        Get list of active files.
        
        Returns:
            List of paths to actively monitored files
        """
        return list(self.active_files)
    
    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Get history for a specific file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of change dictionaries
        """
        changes = self.history.get_changes(file_path)
        return [change.to_dict() for change in changes]
    
    def get_recent_changes(self, file_path: str, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent changes for a specific file.
        
        Args:
            file_path: Path to the file
            count: Maximum number of changes to return
            
        Returns:
            List of the most recent change dictionaries
        """
        changes = self.history.get_recent_changes(file_path, count)
        return [change.to_dict() for change in changes]
    
    def get_file_content(self, file_path: str) -> str:
        """
        Get current content of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Current file content or empty string on error
        """
        # Try to get from cache first
        content = self.content_cache.get(file_path)
        
        # If not in cache, read from file
        if not content and file_path in self.active_files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    # Update cache
                    self.content_cache.set(file_path, content)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
        
        return content or "" 