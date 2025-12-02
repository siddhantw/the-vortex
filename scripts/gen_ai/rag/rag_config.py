"""
RAG Service initializer and configuration utilities.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RAGConfig")

DEFAULT_CONFIG = {
    # Cache settings
    'enable_cache': True,
    'cache_dir': os.path.join(os.getcwd(), 'rag_cache'),
    'cache_ttl': 3600,  # 1 hour in seconds

    # Vector DB settings
    'vector_db_type': 'faiss',  # Options: faiss, chroma, pinecone, weaviate
    'vector_db_path': os.path.join(os.getcwd(), 'rag_cache', 'vector_db'),
    'faiss_index_path': os.path.join(os.getcwd(), 'rag_cache', 'faiss_index.bin'),
    'chroma_path': os.path.join(os.getcwd(), 'rag_cache', 'chroma_db'),
    'collection_name': 'test_knowledge',
    'similarity_threshold': 0.7,
    'max_results': 10,

    # Vector Database Advanced Settings
    'vector_db_config': {
        'faiss': {
            'index_type': 'IndexFlatL2',  # Options: IndexFlatL2, IndexIVFFlat, IndexHNSWFlat
            'dimension': 384,  # Embedding dimension
            'nlist': 100,  # Number of clusters for IVF index
            'nprobe': 10,  # Number of clusters to search
            'hnsw_m': 16,  # HNSW parameter
            'hnsw_ef_construction': 200,  # HNSW parameter
            'hnsw_ef_search': 50  # HNSW parameter
        },
        'chroma': {
            'host': 'localhost',
            'port': 8000,
            'ssl': False,
            'headers': {},
            'distance_function': 'cosine'  # Options: cosine, l2, ip
        },
        'pinecone': {
            'api_key': '',
            'environment': 'us-west1-gcp',
            'index_name': 'test-knowledge',
            'dimension': 384,
            'metric': 'cosine',
            'pod_type': 'p1.x1'
        },
        'weaviate': {
            'url': 'http://localhost:8080',
            'api_key': '',
            'class_name': 'TestKnowledge',
            'distance_metric': 'cosine'
        }
    },

    # Retriever settings
    'retriever_type': 'langchain',  # Options: langchain, llama_index, haystack
    'embedding_model': 'all-MiniLM-L6-v2',
    'chunk_size': 1000,
    'chunk_overlap': 200,
    'chunk_strategy': 'recursive',  # Options: recursive, semantic, fixed_size

    # Advanced Embedding Settings
    'embedding_config': {
        'model_options': [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'sentence-transformers/all-roberta-large-v1',
            'text-embedding-ada-002'  # OpenAI
        ],
        'batch_size': 32,
        'normalize_embeddings': True,
        'use_gpu': False,
        'openai_api_key': '',
        'azure_openai_config': {
            'api_key': '',
            'api_base': '',
            'api_version': '2023-05-15',
            'deployment_name': ''
        }
    },

    # Domain knowledge
    'knowledge_dir': os.path.join(os.getcwd(), 'rag_cache', 'knowledge'),

    # Brand Help Pages Configuration
    'brand_help_pages': {
        'enable': True,
        'brands': {
            'bluehost': {
                'base_url': 'https://www.bluehost.com/help',
                'sitemap_url': 'https://www.bluehost.com/sitemap.xml',
                'crawl_patterns': [
                    '/help/*',
                    '/tutorials/*',
                    '/support/*'
                ],
                'exclude_patterns': [
                    '/login*',
                    '/checkout*',
                    '/admin*'
                ],
                'max_pages': 500,
                'crawl_depth': 3,
                'update_frequency': 'weekly'
            },
            'hostgator': {
                'base_url': 'https://www.hostgator.com/help',
                'sitemap_url': 'https://www.hostgator.com/sitemap.xml',
                'crawl_patterns': [
                    '/help/*',
                    '/tutorials/*',
                    '/support/*'
                ],
                'exclude_patterns': [
                    '/login*',
                    '/checkout*',
                    '/admin*'
                ],
                'max_pages': 500,
                'crawl_depth': 3,
                'update_frequency': 'weekly'
            },
            'networksolutions.com': {
                'base_url': 'https://www.networksolutions.com/help',
                'sitemap_url': 'https://www.networksolutions.com/sitemap.xml',
                'crawl_patterns': [
                    '/help/*',
                    '/support/*',
                    '/kb/*'
                ],
                'exclude_patterns': [
                    '/login*',
                    '/checkout*',
                    '/admin*'
                ],
                'max_pages': 300,
                'crawl_depth': 2,
                'update_frequency': 'monthly'
            }
        },
        'default_brand': 'bluehost',  # Fallback brand if not specified
        'crawl_enabled': True,
        'crawl_config': {
            'delay_between_requests': 1.0,
            'user_agent': 'RAG-TestBot/1.0',
            'timeout': 30,
            'max_retries': 3,
            'respect_robots_txt': True,
            'follow_redirects': True
        }
    },

    # Custom URLs Configuration
    'custom_urls': {
        'enable': True,
        'url_groups': {
            'testing_resources': {
                'urls': [
                    'https://www.guru99.com/software-testing.html',
                    'https://www.softwaretestinghelp.com/',
                    'https://testautomationu.applitools.com/',
                    'https://www.ministryoftesting.com/',
                    'https://www.qatestingtools.com/',
                    'https://www.softwaretestingmaterial.com/',
                    'https://www.testingexcellence.com/',
                    'https://www.toolsqa.com/'
                ],
                'category': 'testing_best_practices',
                'priority': 'high',
                'update_frequency': 'monthly'
            },
            'selenium_docs': {
                'urls': [
                    'https://selenium-python.readthedocs.io/',
                    'https://www.selenium.dev/documentation/',
                    'https://robotframework.org/SeleniumLibrary/',
                    'https://www.selenium.dev/selenium/docs/api/py/',
                    'https://www.selenium.dev/selenium/docs/api/java/'
                ],
                'category': 'automation_frameworks',
                'priority': 'high',
                'update_frequency': 'weekly'
            },
            'api_testing': {
                'urls': [
                    'https://restfulapi.net/',
                    'https://www.postman.com/api-testing/',
                    'https://httpbin.org/',
                    'https://www.soapui.org/',
                    'https://www.loadrunner.com/api-testing/',
                    'https://www.jmeter.apache.org/usermanual/component_reference.html#HTTP_Request'
                ],
                'category': 'api_testing',
                'priority': 'medium',
                'update_frequency': 'monthly'
            }
        },
        'crawl_config': {
            'max_depth': 2,
            'max_pages_per_domain': 100,
            'extract_links': True,
            'follow_external_links': False
        }
    },

    # Intelligent Content Processing
    'content_processing': {
        'enable_semantic_chunking': True,
        'enable_metadata_extraction': True,
        'enable_content_classification': True,
        'content_types': {
            'documentation': {
                'chunk_size': 1500,
                'chunk_overlap': 300,
                'priority_boost': 1.2
            },
            'tutorials': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'priority_boost': 1.1
            },
            'faq': {
                'chunk_size': 500,
                'chunk_overlap': 100,
                'priority_boost': 1.3
            },
            'code_examples': {
                'chunk_size': 800,
                'chunk_overlap': 150,
                'priority_boost': 1.4
            }
        },
        'language_detection': True,
        'supported_languages': ['en', 'es', 'fr', 'de', 'it'],
        'quality_scoring': {
            'enable': True,
            'min_quality_score': 0.6,
            'factors': {
                'content_length': 0.2,
                'readability': 0.3,
                'information_density': 0.3,
                'freshness': 0.2
            }
        }
    },

    # Smart Query Enhancement
    'query_enhancement': {
        'enable_query_expansion': True,
        'enable_intent_detection': True,
        'enable_context_awareness': True,
        'query_rewriting': {
            'enable': True,
            'techniques': ['synonym_expansion', 'spelling_correction', 'abbreviation_expansion']
        },
        'intent_categories': [
            'test_case_generation',
            'bug_reproduction',
            'automation_help',
            'best_practices',
            'troubleshooting'
        ]
    },

    # Performance and Monitoring
    'performance': {
        'enable_caching': True,
        'cache_query_results': True,
        'cache_embeddings': True,
        'max_cache_size': '1GB',
        'enable_metrics': True,
        'log_queries': True,
        'track_usage_patterns': True
    },

    # API integrations
    'apis': [
        # Example API configuration - replace with your actual APIs
        # {
        #     'name': 'company_kb',
        #     'url': 'https://api.company.com/knowledge',
        #     'api_key': 'your-api-key-here',
        #     'headers': {'Content-Type': 'application/json'},
        #     'rate_limit': {'requests_per_minute': 60}
        # }
    ],

    # Advanced RAG Strategies
    'rag_strategies': {
        'default_strategy': 'hybrid',  # Options: simple, hybrid, adaptive, multi_modal
        'strategies': {
            'simple': {
                'description': 'Basic retrieval and generation',
                'retrieval_count': 5,
                'reranking': False
            },
            'hybrid': {
                'description': 'Combines dense and sparse retrieval',
                'retrieval_count': 10,
                'reranking': True,
                'dense_weight': 0.7,
                'sparse_weight': 0.3
            },
            'adaptive': {
                'description': 'Adapts strategy based on query type',
                'use_query_classification': True,
                'fallback_strategy': 'hybrid'
            },
            'multi_modal': {
                'description': 'Handles text, images, and code',
                'enable_image_processing': False,
                'enable_code_understanding': True
            }
        }
    },

    # Security and Privacy
    'security': {
        'enable_content_filtering': True,
        'pii_detection': True,
        'content_sanitization': True,
        'allowed_domains': [],  # Empty means all domains allowed
        'blocked_domains': [],
        'max_file_size': '10MB',
        'allowed_file_types': ['.txt', '.md', '.pdf', '.docx', '.html']
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load RAG configuration from a file or use defaults.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)

            # Update default config with user-provided values
            config.update(user_config)
            logger.info(f"Loaded RAG configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading RAG configuration: {e}")

    # Create necessary directories
    os.makedirs(config['cache_dir'], exist_ok=True)
    os.makedirs(config['vector_db_path'], exist_ok=True)
    os.makedirs(config['knowledge_dir'], exist_ok=True)

    return config

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save RAG configuration to a file.

    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved RAG configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving RAG configuration: {e}")
        return False


class RAGConfig:
    """
    RAG Configuration class to manage settings and initialization.
    """

    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        """
        Initialize RAG configuration.

        Args:
            config_data: Optional configuration dictionary
        """
        self.config = config_data or DEFAULT_CONFIG.copy()
        self._validate_config()
        self._setup_directories()

    def _validate_config(self):
        """Validate the configuration settings."""
        required_keys = ['cache_dir', 'vector_db_type', 'retriever_type']
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Missing required config key: {key}, using default")
                self.config[key] = DEFAULT_CONFIG.get(key)

    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.get('cache_dir'),
            self.config.get('vector_db_path'),
            self.config.get('knowledge_dir')
        ]

        for directory in directories:
            if directory:
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.debug(f"Created directory: {directory}")
                except Exception as e:
                    logger.error(f"Failed to create directory {directory}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        logger.debug(f"Set config {key} = {value}")

    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with dictionary.

        Args:
            config_dict: Dictionary of configuration updates
        """
        self.config.update(config_dict)
        self._validate_config()
        self._setup_directories()
        logger.info("Configuration updated")

    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def save(self, config_path: str) -> bool:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration

        Returns:
            True if successful, False otherwise
        """
        return save_config(self.config, config_path)

    @classmethod
    def from_file(cls, config_path: str) -> 'RAGConfig':
        """
        Create RAGConfig from file.

        Args:
            config_path: Path to configuration file

        Returns:
            RAGConfig instance
        """
        config_data = load_config(config_path)
        return cls(config_data)

    def is_valid(self) -> bool:
        """
        Check if configuration is valid.

        Returns:
            True if configuration is valid
        """
        try:
            self._validate_config()
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


def get_rag_service() -> Optional['RAGService']:
    """
    Get RAG service instance with default configuration.

    Returns:
        RAGService instance or None if initialization fails
    """
    try:
        # Import here to avoid circular imports
        from .rag_service import RAGService

        config = RAGConfig()
        service = RAGService(config)
        return service
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        return None


# Backward compatibility functions
def create_rag_config(config_path: Optional[str] = None) -> RAGConfig:
    """
    Create RAG configuration instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        RAGConfig instance
    """
    if config_path:
        return RAGConfig.from_file(config_path)
    else:
        return RAGConfig()


def get_default_config() -> Dict[str, Any]:
    """
    Get default RAG configuration.

    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()
