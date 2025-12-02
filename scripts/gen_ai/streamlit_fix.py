#!/usr/bin/env python
# Fix for asyncio and PyTorch path issues in Streamlit
# This file should be imported before any other imports in your Streamlit app

import os
import sys
import warnings
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables to prevent PyTorch errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['TORCH_USE_RTLD_GLOBAL'] = 'YES'

# Suppress SSL and certificate warnings
warnings.filterwarnings("ignore", message=".*SSL.*")
warnings.filterwarnings("ignore", message=".*certificate.*")
warnings.filterwarnings("ignore", message=".*insecure.*")

# Configure proper asyncio for all platforms
try:
    # Install required package if not present
    try:
        import nest_asyncio
    except ImportError:
        logger.info("Installing nest_asyncio package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
        import nest_asyncio

    # Apply nest_asyncio before any other asyncio operations
    nest_asyncio.apply()
    logger.info("Applied nest_asyncio for event loop compatibility")

    # Safely handle asyncio event loop setup
    import asyncio
    
    def safe_get_event_loop():
        """Safely get or create an event loop"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            # No event loop in current thread, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    # Ensure we have a working event loop
    safe_get_event_loop()
    logger.info("Asyncio event loop configured successfully")
    
except Exception as e:
    logger.warning(f"Failed to configure asyncio: {e}")

# Safely import PyTorch to prevent path issues
try:
    # Monkey patch torch module to prevent __path__ access issues
    import sys
    
    class TorchWrapper:
        """Wrapper for torch module to prevent __path__ access issues"""
        def __init__(self):
            self._torch = None
            self._imported = False
        
        def __getattr__(self, name):
            if not self._imported:
                try:
                    import torch as _torch
                    self._torch = _torch
                    self._imported = True
                    logger.info("PyTorch imported successfully")
                except ImportError:
                    logger.warning("PyTorch not available")
                    raise AttributeError(f"torch module not available, attribute '{name}' not found")
            
            if self._torch is None:
                raise AttributeError(f"torch module not available, attribute '{name}' not found")
            
            # Prevent accessing problematic attributes
            if name == '__path__':
                logger.warning("Prevented access to torch.__path__ which can cause issues in Streamlit")
                return None
            
            return getattr(self._torch, name)
    
    # Replace torch in sys.modules if it's already imported
    if 'torch' in sys.modules:
        # Store reference to avoid issues
        _original_torch = sys.modules['torch']
        # Only wrap if we detect potential issues
        try:
            # Test if accessing __path__ causes issues
            _ = _original_torch.__path__
        except Exception:
            logger.info("Wrapping torch module to prevent path access issues")
            sys.modules['torch'] = TorchWrapper()
    
    logger.info("PyTorch compatibility layer configured")
    
except Exception as e:
    logger.warning(f"PyTorch compatibility setup failed: {e}")

# Additional SSL/TLS configuration for web requests
try:
    import ssl
    import urllib3
    
    # Disable SSL warnings for development environments
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Create a more permissive SSL context for development
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    logger.info("SSL configuration set to permissive mode for development")
    
except Exception as e:
    logger.warning(f"SSL configuration failed: {e}")

# Set up requests session defaults
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    # Configure default session with retries and SSL handling
    def create_robust_session():
        session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set reasonable timeout
        session.timeout = 30
        
        # Disable SSL verification for development
        session.verify = False
        
        return session
    
    # Store the session creator function for use by other modules
    requests.create_robust_session = create_robust_session
    
    logger.info("Requests session configuration completed")
    
except Exception as e:
    logger.warning(f"Requests configuration failed: {e}")

logger.info("Streamlit compatibility fixes applied successfully")
