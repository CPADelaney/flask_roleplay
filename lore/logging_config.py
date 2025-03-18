"""Logging configuration for the lore system."""

import logging
import logging.handlers
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path

class LogConfig:
    """Logging configuration manager."""
    
    def __init__(self, log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        self.log_dir = log_dir
        self.log_level = log_level
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._setup_log_dir()
    
    def _setup_log_dir(self):
        """Create log directory if it doesn't exist."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating log directory: {str(e)}")
            raise
    
    def configure_logging(self):
        """Configure logging for the system."""
        try:
            # Create formatters
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
            
            # Create handlers
            # File handler for all logs
            all_logs_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.log_dir, 'all.log'),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            all_logs_handler.setFormatter(file_formatter)
            all_logs_handler.setLevel(self.log_level)
            
            # File handler for errors only
            error_logs_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.log_dir, 'error.log'),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            error_logs_handler.setFormatter(file_formatter)
            error_logs_handler.setLevel(logging.ERROR)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(self.log_level)
            
            # Configure root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(self.log_level)
            root_logger.addHandler(all_logs_handler)
            root_logger.addHandler(error_logs_handler)
            root_logger.addHandler(console_handler)
            
            # Create JSON log handler for structured logging
            json_log_handler = logging.handlers.RotatingFileHandler(
                os.path.join(self.log_dir, 'structured.log'),
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            json_log_handler.setFormatter(JsonFormatter())
            json_log_handler.setLevel(self.log_level)
            root_logger.addHandler(json_log_handler)
            
            # Log configuration success
            logging.info("Logging system configured successfully")
            
        except Exception as e:
            print(f"Error configuring logging: {str(e)}")
            raise
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance with the specified name."""
        return logging.getLogger(name)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        try:
            # Create base log entry
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = {
                    'type': str(record.exc_info[0].__name__),
                    'message': str(record.exc_info[1])
                }
            
            # Add extra fields if present
            if hasattr(record, 'extra'):
                log_entry.update(record.extra)
            
            return json.dumps(log_entry)
        except Exception as e:
            # Fallback to basic formatting if JSON conversion fails
            return f"{record.levelname}: {record.getMessage()}"

class LogManager:
    """Log manager for handling log operations."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._setup_log_dir()
    
    def _setup_log_dir(self):
        """Create log directory if it doesn't exist."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating log directory: {str(e)}")
            raise
    
    def get_log_files(self) -> Dict[str, str]:
        """Get paths to all log files."""
        try:
            log_files = {}
            for file in os.listdir(self.log_dir):
                if file.endswith('.log'):
                    log_files[file] = os.path.join(self.log_dir, file)
            return log_files
        except Exception as e:
            logging.error(f"Error getting log files: {str(e)}")
            return {}
    
    def get_log_content(self, log_file: str, 
                       lines: Optional[int] = None,
                       search: Optional[str] = None) -> List[str]:
        """Get content from a log file with optional filtering."""
        try:
            file_path = os.path.join(self.log_dir, log_file)
            if not os.path.exists(file_path):
                return []
            
            with open(file_path, 'r') as f:
                if lines:
                    # Get last N lines
                    return list(f)[-lines:]
                elif search:
                    # Get lines containing search string
                    return [line for line in f if search in line]
                else:
                    # Get all lines
                    return list(f)
        except Exception as e:
            logging.error(f"Error reading log file {log_file}: {str(e)}")
            return []
    
    def clear_logs(self, log_file: Optional[str] = None):
        """Clear logs, optionally for a specific file."""
        try:
            if log_file:
                file_path = os.path.join(self.log_dir, log_file)
                if os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write('')
            else:
                for file in os.listdir(self.log_dir):
                    if file.endswith('.log'):
                        file_path = os.path.join(self.log_dir, file)
                        with open(file_path, 'w') as f:
                            f.write('')
        except Exception as e:
            logging.error(f"Error clearing logs: {str(e)}")
    
    def archive_logs(self, archive_dir: Optional[str] = None):
        """Archive current logs to a timestamped directory."""
        try:
            if archive_dir is None:
                archive_dir = os.path.join(self.log_dir, 'archives')
            os.makedirs(archive_dir, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            archive_path = os.path.join(archive_dir, f'logs_{timestamp}')
            os.makedirs(archive_path)
            
            for file in os.listdir(self.log_dir):
                if file.endswith('.log'):
                    src = os.path.join(self.log_dir, file)
                    dst = os.path.join(archive_path, file)
                    os.rename(src, dst)
            
            logging.info(f"Logs archived to {archive_path}")
        except Exception as e:
            logging.error(f"Error archiving logs: {str(e)}")
    
    def get_log_statistics(self, log_file: str) -> Dict[str, Any]:
        """Get statistics for a log file."""
        try:
            file_path = os.path.join(self.log_dir, log_file)
            if not os.path.exists(file_path):
                return {}
            
            stats = {
                'size': os.path.getsize(file_path),
                'created': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                'line_count': 0,
                'level_counts': defaultdict(int)
            }
            
            with open(file_path, 'r') as f:
                for line in f:
                    stats['line_count'] += 1
                    # Count log levels
                    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                        if f" - {level} - " in line:
                            stats['level_counts'][level] += 1
                            break
            
            return stats
        except Exception as e:
            logging.error(f"Error getting log statistics: {str(e)}")
            return {} 