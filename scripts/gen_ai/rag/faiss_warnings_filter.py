"""
Filter to suppress common Faiss warnings.

This module should be imported early in your application startup
to configure proper warning filters for Faiss GPU-related warnings.
"""

import warnings
import logging

# Configure specific logger for Faiss
faiss_logger = logging.getLogger("faiss")
faiss_logger.setLevel(logging.ERROR)  # Only show ERROR and above, not WARNING

# Add specific warning filter for Faiss GPU warning
warnings.filterwarnings("ignore", message=".*Failed to load GPU Faiss.*")
warnings.filterwarnings("ignore", message=".*GpuIndexIVFFlat.*")
warnings.filterwarnings("ignore", message=".*Will not load constructor refs for GPU indexes.*")

# Helper function to be called by applications
def suppress_faiss_gpu_warnings():
    """
    Call this function to ensure Faiss GPU-related warnings are suppressed.
    """
    # The filters are already applied when the module is imported,
    # but this function can be called explicitly if needed
    pass
