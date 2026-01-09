"""
Enhanced Logging Utility for Test Automation Framework
Provides detailed console logs, progress tracking, and performance monitoring

Features:
- Colored console output with emoji indicators
- Progress tracking with time estimates
- Performance metrics and timing
- Structured logging with context
- Error tracking and diagnostics
- Session-based logging with unique IDs
"""

import logging
import sys
import time
import functools
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
import threading
import traceback
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration for consistent logging"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColorCodes:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class EmojiIndicators:
    """Emoji indicators for different log types"""
    START = "🚀"
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    DEBUG = "🔍"
    PROGRESS = "⏳"
    COMPLETE = "🎉"
    PROCESSING = "⚙️"
    NETWORK = "🌐"
    DATABASE = "🗄️"
    FILE = "📁"
    ROBOT = "🤖"
    AI = "🧠"
    TEST = "🧪"
    PERFORMANCE = "⚡"
    SEARCH = "🔎"
    SAVE = "💾"
    LOAD = "📥"
    EXPORT = "📤"
    STOP = "🛑"
    PAUSE = "⏸️"
    PLAY = "▶️"
    CLOCK = "⏰"
    CHECKMARK = "✓"
    CROSSMARK = "✗"
    ARROW_RIGHT = "→"
    ARROW_LEFT = "←"
    ARROW_UP = "↑"
    ARROW_DOWN = "↓"
    FOLDER = "📂"
    DOCUMENT = "📄"
    CHART = "📊"
    LOCK = "🔒"
    UNLOCK = "🔓"
    KEY = "🔑"
    LINK = "🔗"
    CHAIN = "⛓️"
    SPARKLES = "✨"
    FIRE = "🔥"
    LIGHT_BULB = "💡"
    ROCKET = "🚀"
    TARGET = "🎯"
    TROPHY = "🏆"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""

    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.BRIGHT_BLACK,
        logging.INFO: ColorCodes.BRIGHT_BLUE,
        logging.WARNING: ColorCodes.BRIGHT_YELLOW,
        logging.ERROR: ColorCodes.BRIGHT_RED,
        logging.CRITICAL: ColorCodes.BG_RED + ColorCodes.WHITE,
    }

    LEVEL_EMOJIS = {
        logging.DEBUG: EmojiIndicators.DEBUG,
        logging.INFO: EmojiIndicators.INFO,
        logging.WARNING: EmojiIndicators.WARNING,
        logging.ERROR: EmojiIndicators.ERROR,
        logging.CRITICAL: EmojiIndicators.STOP,
    }

    def format(self, record):
        # Add color based on log level
        color = self.LEVEL_COLORS.get(record.levelno, ColorCodes.RESET)
        emoji = self.LEVEL_EMOJIS.get(record.levelno, "")

        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        # Add emoji to message if not already present
        message = record.getMessage()
        if not any(emoji in message for emoji in EmojiIndicators.__dict__.values() if isinstance(emoji, str)):
            message = f"{emoji} {message}"

        # Build formatted message
        formatted = f"{color}{timestamp} | {record.levelname:8s} | {record.name:20s} | {message}{ColorCodes.RESET}"

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{ColorCodes.RED}{self.formatException(record.exc_info)}{ColorCodes.RESET}"

        return formatted


class ProgressTracker:
    """Track progress of operations with time estimates"""

    def __init__(self, total: int, description: str = "", logger: Optional[logging.Logger] = None):
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 1.0  # Update every second minimum

    def update(self, increment: int = 1, message: str = ""):
        """Update progress"""
        self.current += increment
        current_time = time.time()

        # Only update if interval has passed or completed
        if current_time - self.last_update_time >= self.update_interval or self.current >= self.total:
            elapsed = current_time - self.start_time
            percentage = (self.current / self.total * 100) if self.total > 0 else 0

            # Calculate ETA
            if self.current > 0 and self.current < self.total:
                avg_time_per_item = elapsed / self.current
                remaining_items = self.total - self.current
                eta_seconds = avg_time_per_item * remaining_items
                eta = str(timedelta(seconds=int(eta_seconds)))
            else:
                eta = "N/A"

            # Build progress message
            progress_msg = f"{EmojiIndicators.PROGRESS} {self.description} [{self.current}/{self.total}] {percentage:.1f}%"
            if eta != "N/A" and self.current < self.total:
                progress_msg += f" | ETA: {eta}"
            if message:
                progress_msg += f" | {message}"

            self.logger.info(progress_msg)
            self.last_update_time = current_time

    def complete(self, message: str = ""):
        """Mark progress as complete"""
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        complete_msg = f"{EmojiIndicators.COMPLETE} {self.description} completed in {elapsed_str}"
        if message:
            complete_msg += f" | {message}"

        self.logger.info(complete_msg)


class PerformanceTimer:
    """Track performance metrics for operations"""

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        self.metrics: Dict[str, Any] = {}

    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.logger.info(f"{EmojiIndicators.START} Starting: {self.operation_name}")
        return self

    def stop(self, log_result: bool = True):
        """Stop timing"""
        if self.start_time is None:
            self.logger.warning(f"{EmojiIndicators.WARNING} Timer was not started for: {self.operation_name}")
            return

        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.metrics['duration_seconds'] = duration

        if log_result:
            self._log_result(duration)

        return duration

    def _log_result(self, duration: float):
        """Log timing result"""
        if duration < 1:
            duration_str = f"{duration * 1000:.2f}ms"
            emoji = EmojiIndicators.PERFORMANCE
        elif duration < 60:
            duration_str = f"{duration:.2f}s"
            emoji = EmojiIndicators.SUCCESS
        else:
            duration_str = str(timedelta(seconds=int(duration)))
            emoji = EmojiIndicators.CLOCK

        self.logger.info(f"{emoji} {self.operation_name} completed in {duration_str}")

    def add_metric(self, key: str, value: Any):
        """Add a custom metric"""
        self.metrics[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return self.metrics


class EnhancedLogger:
    """Enhanced logger with additional features"""

    def __init__(self, name: str, level: int = logging.INFO, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)

        # File handler if specified (without colors)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Always log everything to file
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self._timers: Dict[str, PerformanceTimer] = {}

    # Standard logging methods
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **kwargs)

    # Enhanced logging methods
    def start_operation(self, operation: str, **context):
        """Log operation start"""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()]) if context else ""
        msg = f"{EmojiIndicators.START} Starting operation: {operation}"
        if context_str:
            msg += f" | {context_str}"
        self.logger.info(msg)

    def complete_operation(self, operation: str, **context):
        """Log operation completion"""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()]) if context else ""
        msg = f"{EmojiIndicators.COMPLETE} Completed operation: {operation}"
        if context_str:
            msg += f" | {context_str}"
        self.logger.info(msg)

    def log_step(self, step_num: int, total_steps: int, description: str):
        """Log a step in a process"""
        percentage = (step_num / total_steps * 100) if total_steps > 0 else 0
        self.logger.info(
            f"{EmojiIndicators.ARROW_RIGHT} Step {step_num}/{total_steps} ({percentage:.0f}%): {description}"
        )

    def log_success(self, message: str):
        """Log success message"""
        self.logger.info(f"{EmojiIndicators.SUCCESS} {message}")

    def log_failure(self, message: str):
        """Log failure message"""
        self.logger.error(f"{EmojiIndicators.ERROR} {message}")

    def log_progress(self, current: int, total: int, description: str = ""):
        """Log progress"""
        percentage = (current / total * 100) if total > 0 else 0
        msg = f"{EmojiIndicators.PROGRESS} Progress: [{current}/{total}] {percentage:.1f}%"
        if description:
            msg += f" | {description}"
        self.logger.info(msg)

    def log_network_request(self, method: str, url: str, status_code: Optional[int] = None):
        """Log network request"""
        msg = f"{EmojiIndicators.NETWORK} {method} {url}"
        if status_code:
            msg += f" | Status: {status_code}"
        self.logger.info(msg)

    def log_database_query(self, query: str, duration: Optional[float] = None):
        """Log database query"""
        msg = f"{EmojiIndicators.DATABASE} Query: {query[:100]}..."
        if duration:
            msg += f" | Duration: {duration:.3f}s"
        self.logger.info(msg)

    def log_file_operation(self, operation: str, file_path: str):
        """Log file operation"""
        self.logger.info(f"{EmojiIndicators.FILE} {operation}: {file_path}")

    def log_ai_operation(self, operation: str, model: Optional[str] = None, tokens: Optional[int] = None):
        """Log AI operation"""
        msg = f"{EmojiIndicators.AI} {operation}"
        if model:
            msg += f" | Model: {model}"
        if tokens:
            msg += f" | Tokens: {tokens:,}"
        self.logger.info(msg)

    def log_test_result(self, test_name: str, status: str, duration: Optional[float] = None):
        """Log test result"""
        emoji = EmojiIndicators.SUCCESS if status.lower() == "pass" else EmojiIndicators.ERROR
        msg = f"{emoji} Test: {test_name} | Status: {status}"
        if duration:
            msg += f" | Duration: {duration:.2f}s"
        self.logger.info(msg)

    def create_progress_tracker(self, total: int, description: str = "") -> ProgressTracker:
        """Create a progress tracker"""
        return ProgressTracker(total, description, self.logger)

    def start_timer(self, operation_name: str) -> PerformanceTimer:
        """Start a performance timer"""
        timer = PerformanceTimer(operation_name, self.logger)
        timer.start()
        self._timers[operation_name] = timer
        return timer

    def stop_timer(self, operation_name: str) -> Optional[float]:
        """Stop a performance timer"""
        if operation_name in self._timers:
            timer = self._timers[operation_name]
            duration = timer.stop()
            del self._timers[operation_name]
            return duration
        else:
            self.logger.warning(f"{EmojiIndicators.WARNING} No timer found for: {operation_name}")
            return None

    @contextmanager
    def operation_context(self, operation_name: str, **context):
        """Context manager for logging operations"""
        timer = self.start_timer(operation_name)
        self.start_operation(operation_name, **context)
        try:
            yield timer
            self.complete_operation(operation_name, **context)
        except Exception as e:
            self.log_failure(f"{operation_name}: {str(e)}")
            self.exception(f"Exception in {operation_name}")
            raise
        finally:
            timer.stop(log_result=False)  # Don't log twice


def timed_operation(logger: Optional[EnhancedLogger] = None):
    """Decorator to time function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = EnhancedLogger(func.__module__)

            operation_name = f"{func.__name__}"
            timer = logger.start_timer(operation_name)

            try:
                result = func(*args, **kwargs)
                timer.stop()
                return result
            except Exception as e:
                timer.stop(log_result=False)
                logger.log_failure(f"{operation_name}: {str(e)}")
                logger.exception(f"Exception in {operation_name}")
                raise

        return wrapper
    return decorator


def log_exceptions(logger: Optional[EnhancedLogger] = None):
    """Decorator to log exceptions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = EnhancedLogger(func.__module__)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.log_failure(f"Exception in {func.__name__}: {str(e)}")
                logger.exception(f"Full traceback for {func.__name__}")
                raise

        return wrapper
    return decorator


# Global logger registry
_logger_registry: Dict[str, EnhancedLogger] = {}
_registry_lock = threading.Lock()


def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> EnhancedLogger:
    """Get or create an enhanced logger"""
    with _registry_lock:
        if name not in _logger_registry:
            _logger_registry[name] = EnhancedLogger(name, level, log_file)
        return _logger_registry[name]


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Configure global logging settings"""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


# Example usage
if __name__ == "__main__":
    # Configure logging
    configure_logging(level=logging.INFO, log_file="test_enhanced_logging.log")

    # Get logger
    logger = get_logger("TestModule")

    # Test various log types
    logger.info("This is a standard info message")
    logger.start_operation("Test Operation", user="test_user", session_id="12345")

    # Test progress tracking
    progress = logger.create_progress_tracker(100, "Processing items")
    for i in range(100):
        time.sleep(0.01)
        progress.update(1, f"Item {i+1}")
    progress.complete("All items processed successfully")

    # Test performance timing
    with logger.operation_context("Complex Operation", operation_type="data_processing"):
        time.sleep(0.5)
        logger.log_ai_operation("Generate test cases", model="gpt-4", tokens=1500)
        logger.log_database_query("SELECT * FROM users WHERE id = ?", duration=0.023)
        logger.log_network_request("GET", "https://api.example.com/data", status_code=200)

    logger.log_success("All tests completed successfully!")

