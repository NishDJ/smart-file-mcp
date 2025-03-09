"""
Metrics module for the MCP server.
Collects and reports metrics about the server operation.

This module is responsible for collecting, storing, and reporting metrics
about the MCP server's operation. It tracks API requests, file changes,
query performance, and general system health, providing insights into
server usage and performance.

Key components:
- API metrics collection (requests, errors, response times)
- File metrics collection (changes, monitored files)
- Query metrics collection (counts, performance)
- Metrics reporting and formatting
- Automatic periodic saving of metrics

Architecture:
The Metrics class is implemented as a singleton that all other components
can access. It uses thread-safe counters and collections to track metrics,
and provides methods for recording various events. Metrics are periodically
saved to disk and can be loaded on startup.

Usage:
    from metrics import metrics
    
    # Record API request
    metrics.record_api_request("/files")
    
    # Record file change
    metrics.record_file_change("modified")
    
    # Record query performance
    start_time = time.time()
    # ... query execution ...
    metrics.record_query(time.time() - start_time)
    
    # Get metrics report
    report = metrics.get_metrics_report()

Thread Safety:
This module uses thread locking to ensure that metrics can be safely
recorded from multiple threads concurrently. All public methods acquire
the metrics_lock before modifying shared data structures.
"""

import time
import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=config.LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

class Metrics:
    """
    Metrics collector for the MCP server.
    
    This class collects and reports metrics about the MCP server's operation,
    including API usage, file changes, and query performance. It uses thread-safe
    collections to ensure metrics can be recorded from multiple threads.
    
    Attributes:
        metrics_file (str): Path to the file where metrics are saved
        save_interval (int): How often to save metrics, in seconds
        metrics_lock (threading.RLock): Lock for thread-safe access to metrics
        api_requests (defaultdict): Counts of API requests by endpoint
        api_errors (defaultdict): Counts of API errors by endpoint
        response_times (defaultdict): Response times by endpoint
        file_changes (defaultdict): Counts of file changes by event type
        monitored_files (int): Number of files being monitored
        active_conversations (int): Number of active conversations
        query_count (int): Total number of queries executed
        query_errors (int): Total number of query errors
        query_times (deque): Response times for recent queries
        start_time (float): When the metrics collector was started
        last_save_time (float): When metrics were last saved
        save_thread (threading.Thread): Background thread for saving metrics
    """
    def __init__(self, metrics_file: str = "metrics.json", save_interval: int = 300):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_file: Path to the file where metrics are saved
            save_interval: How often to save metrics, in seconds (default: 5 minutes)
        """
        self.metrics_file = metrics_file
        self.save_interval = save_interval  # Save metrics every 5 minutes by default
        
        # Locks
        self.metrics_lock = threading.RLock()
        
        # API Metrics
        self.api_requests = defaultdict(int)  # Endpoint -> count
        self.api_errors = defaultdict(int)    # Endpoint -> count
        self.response_times = defaultdict(list)  # Endpoint -> list of response times
        self.max_response_times = 1000  # Max number of response times to keep per endpoint
        
        # File Metrics
        self.file_changes = defaultdict(int)  # Event type -> count
        self.monitored_files = 0
        self.active_conversations = 0
        
        # Query Metrics
        self.query_count = 0
        self.query_errors = 0
        self.query_times = deque(maxlen=1000)  # Last 1000 query times
        
        # System Metrics
        self.start_time = time.time()
        self.last_save_time = time.time()
        
        # Load existing metrics if available
        self.load_metrics()
        
        # Start metrics saving thread
        self.should_run = True
        self.save_thread = threading.Thread(target=self._save_metrics_loop, daemon=True)
        self.save_thread.start()
    
    def load_metrics(self) -> None:
        """
        Load metrics from file if it exists.
        
        This method attempts to load previously saved metrics from a JSON file.
        If the file doesn't exist or there's an error loading it, the metrics
        will start from scratch.
        """
        if not os.path.exists(self.metrics_file):
            return
        
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            with self.metrics_lock:
                # Load API metrics
                if "api_metrics" in data:
                    self.api_requests = defaultdict(int, data["api_metrics"].get("requests", {}))
                    self.api_errors = defaultdict(int, data["api_metrics"].get("errors", {}))
                
                # Load file metrics
                if "file_metrics" in data:
                    self.file_changes = defaultdict(int, data["file_metrics"].get("changes", {}))
                    self.monitored_files = data["file_metrics"].get("monitored_files", 0)
                    self.active_conversations = data["file_metrics"].get("active_conversations", 0)
                
                # Load query metrics
                if "query_metrics" in data:
                    self.query_count = data["query_metrics"].get("count", 0)
                    self.query_errors = data["query_metrics"].get("errors", 0)
                
                logger.info(f"Loaded metrics from {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def save_metrics(self) -> None:
        """
        Save metrics to file.
        
        This method saves the current metrics to a JSON file for persistence.
        It includes a timestamp and calculates some aggregate metrics like
        average response times.
        """
        try:
            with self.metrics_lock:
                data = {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": time.time() - self.start_time,
                    
                    "api_metrics": {
                        "requests": dict(self.api_requests),
                        "errors": dict(self.api_errors),
                        "avg_response_times": {
                            endpoint: sum(times) / len(times) if times else 0
                            for endpoint, times in self.response_times.items()
                        }
                    },
                    
                    "file_metrics": {
                        "changes": dict(self.file_changes),
                        "monitored_files": self.monitored_files,
                        "active_conversations": self.active_conversations
                    },
                    
                    "query_metrics": {
                        "count": self.query_count,
                        "errors": self.query_errors,
                        "avg_time": sum(self.query_times) / len(self.query_times) if self.query_times else 0
                    }
                }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.last_save_time = time.time()
            logger.debug(f"Saved metrics to {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _save_metrics_loop(self) -> None:
        """
        Background loop to save metrics periodically.
        
        This method runs in a background thread and saves metrics at regular
        intervals defined by self.save_interval.
        """
        while self.should_run:
            time.sleep(10)  # Check every 10 seconds
            
            if time.time() - self.last_save_time > self.save_interval:
                self.save_metrics()
    
    def stop(self) -> None:
        """
        Stop the metrics collector.
        
        This method stops the background saving thread and saves metrics
        one last time before shutdown.
        """
        self.should_run = False
        self.save_metrics()  # Save metrics one last time
        if self.save_thread.is_alive():
            self.save_thread.join(timeout=5)
    
    # API Metrics Methods
    def record_api_request(self, endpoint: str) -> None:
        """
        Record an API request.
        
        Args:
            endpoint: The API endpoint that was requested
        """
        with self.metrics_lock:
            self.api_requests[endpoint] += 1
    
    def record_api_error(self, endpoint: str) -> None:
        """
        Record an API error.
        
        Args:
            endpoint: The API endpoint where the error occurred
        """
        with self.metrics_lock:
            self.api_errors[endpoint] += 1
    
    def record_response_time(self, endpoint: str, duration: float) -> None:
        """
        Record response time for an endpoint.
        
        Args:
            endpoint: The API endpoint
            duration: The response time in seconds
        """
        with self.metrics_lock:
            self.response_times[endpoint].append(duration)
            # Prune if necessary
            if len(self.response_times[endpoint]) > self.max_response_times:
                self.response_times[endpoint] = self.response_times[endpoint][-self.max_response_times:]
    
    # File Metrics Methods
    def record_file_change(self, event_type: str) -> None:
        """
        Record a file change event.
        
        Args:
            event_type: The type of file change (created, modified, deleted, etc.)
        """
        with self.metrics_lock:
            self.file_changes[event_type] += 1
    
    def update_monitored_files(self, count: int) -> None:
        """
        Update the count of monitored files.
        
        Args:
            count: The current number of monitored files
        """
        with self.metrics_lock:
            self.monitored_files = count
    
    def update_active_conversations(self, count: int) -> None:
        """
        Update the count of active conversations.
        
        Args:
            count: The current number of active conversations
        """
        with self.metrics_lock:
            self.active_conversations = count
    
    # Query Metrics Methods
    def record_query(self, duration: float, error: bool = False) -> None:
        """
        Record a query execution.
        
        Args:
            duration: The query execution time in seconds
            error: Whether the query resulted in an error
        """
        with self.metrics_lock:
            self.query_count += 1
            self.query_times.append(duration)
            if error:
                self.query_errors += 1
    
    # Report Methods
    def get_metrics_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive metrics report.
        
        Returns:
            A dictionary containing all collected metrics in a structured format
        """
        with self.metrics_lock:
            uptime = time.time() - self.start_time
            uptime_str = str(timedelta(seconds=int(uptime)))
            
            total_requests = sum(self.api_requests.values())
            total_errors = sum(self.api_errors.values())
            error_rate = (total_errors / total_requests) * 100 if total_requests > 0 else 0
            
            avg_query_time = sum(self.query_times) / len(self.query_times) if self.query_times else 0
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "uptime": uptime_str,
                "uptime_seconds": uptime,
                
                "api_metrics": {
                    "total_requests": total_requests,
                    "requests_by_endpoint": dict(self.api_requests),
                    "error_rate": f"{error_rate:.2f}%",
                    "avg_response_times": {
                        endpoint: f"{sum(times) / len(times):.4f}s" if times else "0s"
                        for endpoint, times in self.response_times.items()
                    }
                },
                
                "file_metrics": {
                    "monitored_files": self.monitored_files,
                    "active_conversations": self.active_conversations,
                    "changes_by_type": dict(self.file_changes)
                },
                
                "query_metrics": {
                    "total_queries": self.query_count,
                    "error_rate": f"{(self.query_errors / self.query_count) * 100:.2f}%" if self.query_count > 0 else "0%",
                    "avg_query_time": f"{avg_query_time:.4f}s"
                }
            }
            
            return report
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of metrics.
        
        Returns:
            A string containing a formatted summary of the most important metrics
        """
        report = self.get_metrics_report()
        
        summary = [
            f"MCP Server Metrics Summary ({report['timestamp']})",
            f"Uptime: {report['uptime']}",
            f"",
            f"API Metrics:",
            f"  Total Requests: {report['api_metrics']['total_requests']}",
            f"  Error Rate: {report['api_metrics']['error_rate']}",
            f"",
            f"File Metrics:",
            f"  Monitored Files: {report['file_metrics']['monitored_files']}",
            f"  Active Conversations: {report['file_metrics']['active_conversations']}",
            f"  File Changes: {sum(report['file_metrics']['changes_by_type'].values())}",
            f"",
            f"Query Metrics:",
            f"  Total Queries: {report['query_metrics']['total_queries']}",
            f"  Error Rate: {report['query_metrics']['error_rate']}",
            f"  Avg Query Time: {report['query_metrics']['avg_query_time']}"
        ]
        
        return "\n".join(summary)

# Singleton instance
metrics = Metrics() 