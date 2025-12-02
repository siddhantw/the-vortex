"""
Streamlit-compatible wrapper for RAG services that avoids PyTorch module loading issues.
"""

import importlib.util
import logging
import os
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGHandler")


# Add a fix for Streamlit's file watching of torch modules
# This prevents Streamlit from trying to access torch.__path__._path
# which causes the "Tried to instantiate class '__path__._path'" error
def _patch_torch_for_streamlit():
    """
    Apply patches to avoid Streamlit's file watcher problems with PyTorch.
    This function prevents Streamlit from trying to watch PyTorch modules.
    """
    try:
        import sys
        from types import ModuleType
        import warnings

        # Suppress any warnings from PyTorch
        warnings.filterwarnings("ignore", message=".*__path__._path.*")
        warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
        warnings.filterwarnings("ignore", message=".*but it does not exist.*")

        # Create a class that mimics the expected behavior for __path__
        class PathWrapper(list):
            def __init__(self):
                self.path = [os.path.join(os.path.dirname(__file__), 'torch_stub')]
                super().__init__(self.path)

            def __iter__(self):
                return iter(self.path)

            def __getitem__(self, idx):
                return self.path[idx]

            def __getattr__(self, name):
                # Return a dummy function for any other attribute access
                if name == '_path':
                    return self
                return lambda *args, **kwargs: None

        # Create a fake torch module if it doesn't exist
        if 'torch' not in sys.modules:
            fake_torch = ModuleType('torch')
            fake_torch.__path__ = PathWrapper()
            sys.modules['torch'] = fake_torch
        else:
            # Ensure the existing torch module has a proper __path__ attribute
            torch_module = sys.modules['torch']
            if not hasattr(torch_module, '__path__') or torch_module.__path__ == []:
                torch_module.__path__ = PathWrapper()

            # Fix the __class__ attribute to prevent instantiation issues
            if hasattr(torch_module.__path__, '__class__'):
                # Wrap the existing __path__ with our safer version
                try:
                    original_path = list(torch_module.__path__)
                    torch_module.__path__ = PathWrapper()
                    if original_path:
                        torch_module.__path__.path = original_path
                except Exception:
                    # If we can't convert the existing path, just use our wrapped version
                    torch_module.__path__ = PathWrapper()

        # Create the stub directory if it doesn't exist
        stub_path = os.path.join(os.path.dirname(__file__), 'torch_stub')
        os.makedirs(stub_path, exist_ok=True)

        # Create an empty __init__.py in the stub directory to make it a proper package
        init_path = os.path.join(stub_path, '__init__.py')
        if not os.path.exists(init_path):
            with open(init_path, 'w') as f:
                f.write("# PyTorch stub for Streamlit compatibility\n")
                f.write("class _path(list):\n")
                f.write("    def __init__(self, *args, **kwargs):\n")
                f.write("        super().__init__(*args, **kwargs)\n")
                f.write("    def __getattr__(self, name):\n")
                f.write("        return self\n")

        # Also explicitly create the torch.nn module and CrossEntropyLoss class
        # This fixes the "cannot import name 'CrossEntropyLoss' from 'torch.nn'" error
        if 'torch.nn' not in sys.modules:
            nn_module = ModuleType('torch.nn')
            nn_module.__path__ = PathWrapper()
            sys.modules['torch.nn'] = nn_module

        # Add the CrossEntropyLoss class to torch.nn module
        nn_module = sys.modules.get('torch.nn')
        if nn_module and not hasattr(nn_module, 'CrossEntropyLoss'):
            class CrossEntropyLoss:
                def __init__(self, *args, **kwargs):
                    pass

                def __call__(self, *args, **kwargs):
                    return None

            setattr(nn_module, 'CrossEntropyLoss', CrossEntropyLoss)

        # Create stub for nn module
        nn_path = os.path.join(stub_path, 'nn')
        os.makedirs(nn_path, exist_ok=True)
        nn_init_path = os.path.join(nn_path, '__init__.py')
        if not os.path.exists(nn_init_path):
            with open(nn_init_path, 'w') as f:
                f.write("# PyTorch nn stub for Streamlit compatibility\n")
                f.write("class CrossEntropyLoss:\n")
                f.write("    def __init__(self, *args, **kwargs):\n")
                f.write("        pass\n")
                f.write("    def __call__(self, *args, **kwargs):\n")
                f.write("        return None\n")

        # Create a special class_resolver that intercepts torch::class_ registrations
        if hasattr(sys.modules.get('torch', None), 'class_'):
            def class_resolver(*args, **kwargs):
                return None

            setattr(sys.modules['torch'], 'class_', class_resolver)

        logger.info("Applied enhanced patch for PyTorch compatibility with Streamlit file watcher")
    except Exception as e:
        logger.warning(f"Failed to patch torch for streamlit: {e}")


# Apply the patch when this module is imported
_patch_torch_for_streamlit()


class RAGHandler:
    """
    A wrapper class for RAG services that avoids direct imports that could trigger
    PyTorch module loading issues with Streamlit's file watcher.
    """

    def __init__(self):
        self.rag_service = None
        self.rag_config = None
        self.initialized = False

    def initialize_service(self, config_path: str) -> bool:
        """
        Initializes the RAG service on demand only when needed.

        Args:
            config_path: Path to the RAG configuration file

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Make sure PyTorch is patched before any imports
            _patch_torch_for_streamlit()

            # Dynamically import the RAG modules to avoid early loading
            if importlib.util.find_spec("rag.rag_config") is None:
                logger.error("Failed to find rag_config module")
                return False

            # Use a safer import approach
            def safe_import(module_name):
                """Import a module safely, handling exceptions"""
                try:
                    spec = importlib.util.find_spec(module_name)
                    if not spec:
                        logger.error(f"Could not find spec for {module_name}")
                        return None

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    return module
                except Exception as e:
                    logger.error(f"Error importing {module_name}: {e}")
                    return None

            config_module = safe_import("rag.rag_config")
            service_module = safe_import("rag.rag_service")

            if not config_module or not service_module:
                return False

            # Load configuration
            self.rag_config = config_module.load_config(config_path)

            # Initialize RAG service
            self.rag_service = service_module.RAGService(self.rag_config)

            self.initialized = True
            logger.info(f"RAG service initialized with config from {config_path}")
            return True

        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            return False

    def enhance_context(self, query: str, scope: str, test_type: str, components: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve and enhance context from various sources.

        Args:
            query: The query or requirement text
            scope: Test scope (Functional, Non-Functional, or Both)
            test_type: Test type (Component, Integration, Acceptance, or All)
            components: Optional list of specific components/modules

        Returns:
            Dictionary with enhanced context from all sources
        """
        if not self.initialized or not self.rag_service:
            logger.warning("RAG service not initialized")
            return {}

        try:
            return self.rag_service.enhance_context(query, scope, test_type, components)
        except Exception as e:
            logger.error(f"Error enhancing context: {e}")
            return {}

    def enhance_prompt(self, original_prompt: str, query: str, scope: str, test_type: str,
                       components: List[str] = None) -> str:
        """
        Enhance a prompt using the RAG service.

        Args:
            original_prompt: The original prompt template
            query: The query or requirement text
            scope: Test scope (Functional, Non-Functional, or Both)
            test_type: Test type (Component, Integration, Acceptance, or All)
            components: Optional list of components

        Returns:
            Enhanced prompt with context
        """
        if not self.initialized or not self.rag_service:
            logger.warning("RAG service not initialized")
            return original_prompt

        try:
            # Get enhanced context
            enhanced_context = self.rag_service.enhance_context(query, scope, test_type, components)

            # Create enhanced prompt
            return self.rag_service.create_enhanced_prompt(original_prompt, enhanced_context)
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return original_prompt

    def learn_from_urls(self, urls: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Learn from content at provided URLs.

        Args:
            urls: List of URLs to learn from
            labels: Optional labels to categorize the URL sources

        Returns:
            Dictionary with learning statistics and status
        """
        if not self.initialized or not self.rag_service:
            logger.warning("RAG service not initialized")
            return {"status": "error", "message": "RAG service not initialized"}

        try:
            return self.rag_service.learn_from_urls(urls, labels)
        except Exception as e:
            logger.error(f"Error learning from URLs: {e}")
            return {"status": "error", "message": str(e)}

    def extract_url_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single URL.

        Args:
            url: The URL to extract content from

        Returns:
            Dictionary with extracted content
        """
        if not self.initialized or not self.rag_service:
            logger.warning("RAG service not initialized")
            return {"status": "error", "message": "RAG service not initialized"}

        try:
            urls = [url]
            crawl_results = self.rag_service.crawl_and_index_urls(urls)
            return crawl_results
        except Exception as e:
            logger.error(f"Error extracting URL content: {e}")
            return {"status": "error", "message": str(e)}

    def get_rag_config(self) -> Dict[str, Any]:
        """
        Get the current RAG configuration.

        Returns:
            Dictionary with the RAG configuration
        """
        if not self.initialized or not self.rag_config:
            logger.warning("RAG service not initialized or config not loaded")
            return {}

        try:
            return self.rag_config.to_dict()
        except Exception as e:
            logger.error(f"Error getting RAG config: {e}")
            return {}

    def save_rag_config(self, config_path: str) -> bool:
        """
        Save the current RAG configuration to a file.

        Args:
            config_path: Path to save the RAG configuration

        Returns:
            True if saving was successful, False otherwise
        """
        if not self.initialized or not self.rag_config:
            logger.warning("RAG service not initialized or config not loaded")
            return False

        try:
            self.rag_config.save(config_path)
            logger.info(f"RAG config saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving RAG config: {e}")
            return False

    def reset_service(self):
        """
        Reset the RAG service and clear the current configuration.
        """
        self.rag_service = None
        self.rag_config = None
        self.initialized = False
        logger.info("RAG service has been reset")

    def is_initialized(self) -> bool:
        """
        Check if the RAG service is initialized.
        :return: True if the RAG service is initialized, False otherwise
        """
        return self.initialized and self.rag_service is not None

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the RAG service.
        :return:
            Dictionary with service status, including initialization state and config details
        """
        return {
            "initialized": self.initialized,
            "rag_config": self.rag_config.to_dict() if self.rag_config else None
        }

    def __del__(self):
        """
        Cleanup method to reset the service when the handler is deleted.
        """
        self.reset_service()
        logger.info("RAGHandler instance has been deleted and service reset")
