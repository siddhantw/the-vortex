"""
Retrieval-Augmented Generation (RAG) service for enhancing test case generation
with external knowledge sources.
"""

import os
import logging
import hashlib
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta

# Add requests with connection pooling for better performance
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("rag_service.log"), logging.StreamHandler()]
)
logger = logging.getLogger("RAGService")

# Create a session with connection pooling and retry strategy
def create_requests_session(retries=3, backoff_factor=0.3,
                          status_forcelist=(500, 502, 503, 504),
                          pool_connections=10, pool_maxsize=20):
    """Create a requests session with connection pooling and retry strategy"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
        max_retries=retry
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Global session for requests
http_session = create_requests_session()

class RAGService:
    """
    Service for retrieving relevant context from external knowledge sources
    and enhancing the prompt for test case generation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RAG service with configuration.

        Args:
            config: Configuration dictionary with settings for RAG services
        """
        self.config = config or {}
        self.cache_dir = self.config.get('cache_dir', os.path.join(os.getcwd(), 'rag_cache'))
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # Default: 1 hour in seconds
        self.enable_cache = self.config.get('enable_cache', True)
        self.vector_db = None
        self.document_store = None
        self.retriever = None
        self.encoder = None  # Initialize encoder attribute
        self.vector_db_type = self.config.get('vector_db_type', 'faiss')  # Store vector_db_type as class attribute

        # Create necessary directories upfront to avoid permission errors later
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Fallback document storage when no vector database is available
        self.fallback_storage_enabled = self.config.get('fallback_storage_enabled', True)
        self.documents = []
        self.documents_path = os.path.join(self.cache_dir, 'indexed_documents.json')

        # Try to load existing documents
        if self.fallback_storage_enabled and os.path.exists(self.documents_path):
            try:
                with open(self.documents_path, 'r') as f:
                    self.documents = json.load(f)
                logger.info(f"Loaded {len(self.documents)} documents from fallback storage")
            except Exception as e:
                logger.warning(f"Error loading documents from fallback storage: {e}")

        # URL crawler configuration
        self.url_cache_dir = os.path.join(self.cache_dir, 'url_cache')
        self.url_cache_ttl = self.config.get('url_cache_ttl', 86400)  # Default: 1 day in seconds

        # Critical crawler configuration - explicit log these values to help debug
        self.max_crawl_depth = self.config.get('max_crawl_depth', 2)
        self.max_pages_per_domain = self.config.get('max_pages_per_domain', 50)
        self.max_pages_per_brand = self.config.get('max_pages_per_brand', self.max_pages_per_domain)

        self.crawl_delay = self.config.get('crawl_delay', 1.0)  # Delay between requests in seconds

        logger.info(f"Crawler configured with: max_crawl_depth={self.max_crawl_depth}, " +
                   f"max_pages_per_domain={self.max_pages_per_domain}, " +
                   f"max_pages_per_brand={self.max_pages_per_brand}")

        self.irrelevant_patterns = self.config.get('irrelevant_patterns', [
            r'contact\s+us',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'cookie\s+policy',
            r'\b(?:phone|email|fax)\b\s*:\s*[\w\d.@+-]+',
            r'\baddress\b\s*:\s*[^.]+\.',
            r'\bCopyright\b.*\d{4}'
        ])

        # Create cache directories if they don't exist
        if self.enable_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.enable_cache and not os.path.exists(self.url_cache_dir):
            os.makedirs(self.url_cache_dir, exist_ok=True)

        # Check for PyTorch availability early to avoid issues later
        self._check_pytorch_availability()

        # Initialize external services based on config
        self._initialize_services()

        logger.info(f"RAG Service initialized with cache_dir: {self.cache_dir}, cache_ttl: {self.cache_ttl}s")

    def _check_pytorch_availability(self):
        """Check if PyTorch is properly installed and functioning"""
        try:
            import torch
            import torch.nn as nn

            # Verify torch.nn contains the expected classes
            has_cross_entropy = hasattr(nn, 'CrossEntropyLoss')
            if not has_cross_entropy:
                logger.warning("PyTorch installation seems incomplete. "
                              "torch.nn.CrossEntropyLoss not found. "
                              "Try reinstalling PyTorch: pip uninstall -y torch torchvision torchaudio && "
                              "pip install torch torchvision torchaudio")
                return False

            logger.info(f"PyTorch {torch.__version__} is properly installed")
            return True
        except ImportError as e:
            logger.warning(f"PyTorch import error: {e}. "
                          "Consider installing PyTorch: pip install torch")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error checking PyTorch: {e}")
            return False

    def _initialize_services(self):
        """Initialize external services based on configuration"""
        # Vector database for semantic search
        vector_db_type = self.config.get('vector_db_type', 'faiss')
        vector_db_initialized = False

        if vector_db_type == 'faiss':
            vector_db_initialized = self._initialize_faiss()
        elif vector_db_type == 'chroma':
            vector_db_initialized = self._initialize_chroma()
        elif vector_db_type == 'pinecone':
            vector_db_initialized = self._initialize_pinecone()

        # Document retriever - only initialize if we have a vector DB
        if vector_db_initialized:
            retriever_type = self.config.get('retriever_type', 'langchain')
            if retriever_type == 'langchain':
                self._initialize_langchain()
            elif retriever_type == 'llama_index':
                self._initialize_llama_index()
            elif retriever_type == 'haystack':
                self._initialize_haystack()
        else:
            logger.warning("Vector database initialization failed. Document retrieval will be limited.")

    def _initialize_faiss(self):
        """Initialize FAISS vector database"""
        try:
            import faiss

            # Check if FAISS is properly installed
            if not hasattr(faiss, 'IndexFlatL2'):
                logger.error("FAISS not properly installed. IndexFlatL2 class not found.")
                return False

            # First, ensure PyTorch is properly installed and SentenceTransformers can be imported
            try:
                # Import necessary dependencies early to catch issues
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                logger.error(f"SentenceTransformer import failed: {e}")
                logger.error("Install required packages with: pip install sentence-transformers")
                self.encoder = None
                self.vector_db = None
                return False
            except Exception as e:
                logger.error(f"Unexpected error importing SentenceTransformer: {e}")
                self.encoder = None
                self.vector_db = None
                return False

            # Gracefully handle GPU Faiss import errors
            self.use_gpu = False
            try:
                # Only try to import GPU version if available
                res = faiss.get_num_gpus()
                if res > 0:
                    try:
                        import faiss.gpu
                        self.use_gpu = True
                        logger.info(f"FAISS GPU support found with {res} GPUs available")
                    except (ImportError, AttributeError) as e:
                        logger.debug(f"FAISS GPU support not available: {e}. Using CPU version.")
            except Exception as e:
                logger.debug(f"Error checking FAISS GPU support: {e}. Using CPU version.")

            # Path to the FAISS index
            index_path = self.config.get('faiss_index_path')
            if not index_path:
                index_path = os.path.join(self.cache_dir, 'faiss_index.bin')
                logger.info(f"No FAISS index path provided, using default: {index_path}")

            # Initialize sentence transformer for encoding
            try:
                model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                self.encoder = SentenceTransformer(model_name)
                logger.info(f"Successfully initialized SentenceTransformer with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize SentenceTransformer encoder: {e}")
                self.encoder = None
                self.vector_db = None
                return False

            # Load or create FAISS index
            if index_path and os.path.exists(index_path):
                try:
                    self.vector_db = faiss.read_index(index_path)
                    logger.info(f"FAISS index loaded from {index_path}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index from {index_path}: {e}")
                    logger.info("Creating a new FAISS index instead")
                    # Create a new FAISS index since loading failed
                    embedding_dim = self.encoder.get_sentence_embedding_dimension()
                    self.vector_db = faiss.IndexFlatL2(embedding_dim)
                    logger.info(f"New FAISS index created with dimension {embedding_dim}")
            else:
                # Create a new FAISS index
                embedding_dim = self.encoder.get_sentence_embedding_dimension()
                self.vector_db = faiss.IndexFlatL2(embedding_dim)
                logger.info(f"New FAISS index created with dimension {embedding_dim}")

            # Move index to GPU if supported and enabled in config
            if self.use_gpu and self.config.get('use_gpu', False):
                try:
                    gpu_resource = faiss.StandardGpuResources()
                    self.vector_db = faiss.index_cpu_to_gpu(gpu_resource, 0, self.vector_db)
                    logger.info("FAISS index moved to GPU")
                except Exception as e:
                    logger.warning(f"Failed to move FAISS index to GPU: {e}. Using CPU version instead.")

            # Save the FAISS index path for future use
            self.config['faiss_index_path'] = index_path

            logger.info("FAISS vector database initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"FAISS not installed or import error: {e}. Run: pip install faiss-cpu sentence-transformers")
            self.encoder = None
            self.vector_db = None
            return False
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.encoder = None
            self.vector_db = None
            return False

    def _initialize_chroma(self):
        """Initialize Chroma vector database"""
        try:
            import chromadb

            # Initialize Chroma client
            chroma_path = self.config.get('chroma_path', os.path.join(self.cache_dir, 'chroma_db'))
            os.makedirs(chroma_path, exist_ok=True)

            self.vector_db = chromadb.PersistentClient(path=chroma_path)

            # Create or get collection
            collection_name = self.config.get('collection_name', 'test_knowledge')
            self.collection = self.vector_db.get_or_create_collection(name=collection_name)

            logger.info(f"Chroma vector database initialized with collection: {collection_name}")
            return True
        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.vector_db = None
            return False

    def _initialize_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            import pinecone
            from sentence_transformers import SentenceTransformer

            # Initialize Pinecone
            api_key = self.config.get('pinecone_api_key')
            environment = self.config.get('pinecone_environment')
            index_name = self.config.get('pinecone_index_name')

            if not all([api_key, environment, index_name]):
                raise ValueError("Missing Pinecone configuration: api_key, environment, and index_name required")

            pinecone.init(api_key=api_key, environment=environment)

            # Connect to index
            self.vector_db = pinecone.Index(index_name)

            # Initialize sentence transformer for encoding
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            self.encoder = SentenceTransformer(model_name)

            logger.info(f"Pinecone vector database initialized with index: {index_name}")
            return True
        except ImportError as e:
            logger.error(f"Pinecone or sentence_transformers not installed: {e}. Run: pip install pinecone-client sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.vector_db = None
            return False

    def _initialize_langchain(self):
        """Initialize LangChain for document retrieval"""
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            from langchain.vectorstores import FAISS as LangchainFAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader

            # Initialize embeddings model
            model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get('chunk_size', 1000),
                chunk_overlap=self.config.get('chunk_overlap', 200)
            )

            # Initialize document loaders (to be used as needed)
            self.loaders = {
                'txt': TextLoader,
                'pdf': PyPDFLoader,
                'md': UnstructuredMarkdownLoader
            }

            logger.info("LangChain document retrieval initialized")
            self.retriever = 'langchain'
            return True
        except ImportError as e:
            logger.error(f"LangChain not installed: {e}. Run: pip install langchain")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LangChain: {e}")
            self.retriever = None
            return False

    def _initialize_llama_index(self):
        """Initialize LlamaIndex for document retrieval"""
        try:
            from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
            from llama_index.node_parser import SimpleNodeParser
            import torch

            # Check for GPU availability
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

            # Initialize parser
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.config.get('chunk_size', 1024),
                chunk_overlap=self.config.get('chunk_overlap', 200)
            )

            # Initialize service context
            self.service_context = ServiceContext.from_defaults(
                llm=None,  # We're not using LLMs here, just retrieval
                node_parser=self.node_parser
            )

            # Initialize reader (to be used as needed)
            self.directory_reader = SimpleDirectoryReader

            logger.info("LlamaIndex document retrieval initialized")
            self.retriever = 'llama_index'
            return True
        except ImportError as e:
            logger.error(f"LlamaIndex not installed: {e}. Run: pip install llama-index")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex: {e}")
            self.retriever = None
            return False

    def _initialize_haystack(self):
        """Initialize Haystack for document retrieval"""
        try:
            from haystack.document_stores import InMemoryDocumentStore
            from haystack.nodes import BM25Retriever

            # Initialize document store
            self.document_store = InMemoryDocumentStore(
                use_bm25=True,
                similarity=self.config.get('similarity', 'cosine')
            )

            # Initialize retriever
            self.haystack_retriever = BM25Retriever(document_store=self.document_store)

            logger.info("Haystack document retrieval initialized")
            self.retriever = 'haystack'
            return True
        except ImportError as e:
            logger.error(f"Haystack not installed: {e}. Run: pip install farm-haystack")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Haystack: {e}")
            self.retriever = None
            return False

    def _get_cache_key(self, query: str, scope: str, test_type: str) -> str:
        """
        Generate a unique cache key for the given query parameters.

        Args:
            query: The query string
            scope: Test scope (Functional, Non-Functional, or Both)
            test_type: Test type (Component, Integration, Acceptance, or All)

        Returns:
            A unique hash string to use as cache key
        """
        combined = f"{query}|{scope}|{test_type}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if there's a valid cache entry for the given key.

        Args:
            cache_key: The cache key to look for

        Returns:
            Cached data if found and valid, None otherwise
        """
        if not self.enable_cache:
            return None

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)

                # Check if cache is still valid
                timestamp = cached_data.get('timestamp', 0)
                if time.time() - timestamp <= self.cache_ttl:
                    logger.info(f"Cache hit for key: {cache_key}")
                    return cached_data.get('data')
                else:
                    logger.info(f"Cache expired for key: {cache_key}")
            except Exception as e:
                logger.warning(f"Error reading cache: {e}")

        return None

    def _update_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """
        Update the cache with new data.

        Args:
            cache_key: The cache key
            data: The data to cache
        """
        if not self.enable_cache:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")

        try:
            cache_entry = {
                'timestamp': time.time(),
                'data': data
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f)

            logger.info(f"Updated cache for key: {cache_key}")
        except Exception as e:
            logger.warning(f"Error updating cache: {e}")

    def retrieve_from_api(self, query: str, scope: str, test_type: str) -> Dict[str, Any]:
        """
        Retrieve relevant information from external APIs.

        Args:
            query: The query or requirement text
            scope: Test scope (Functional, Non-Functional, or Both)
            test_type: Test type (Component, Integration, Acceptance, or All)

        Returns:
            Dictionary with retrieved information
        """
        api_results = {}

        # Get API configurations
        apis = self.config.get('apis', [])

        for api_config in apis:
            api_name = api_config.get('name', 'unknown_api')
            api_url = api_config.get('url')
            api_key = api_config.get('api_key')

            if not api_url:
                logger.warning(f"Missing URL for API: {api_name}")
                continue

            try:
                # Prepare request parameters
                headers = {'Content-Type': 'application/json'}
                if api_key:
                    headers['Authorization'] = f"Bearer {api_key}"

                payload = {
                    'query': query,
                    'scope': scope,
                    'test_type': test_type
                }

                # Make API request
                response = http_session.post(api_url, json=payload, headers=headers, timeout=10)

                if response.status_code == 200:
                    api_results[api_name] = response.json()
                    logger.info(f"Successfully retrieved data from API: {api_name}")
                else:
                    logger.warning(f"API request failed for {api_name}: {response.status_code}")

            except Exception as e:
                logger.error(f"Error retrieving from API {api_name}: {e}")

        return api_results

    def retrieve_from_vector_db(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant documents from vector database.

        Args:
            query: The query or requirement text
            top_k: Number of top results to retrieve

        Returns:
            Dictionary with retrieved documents
        """
        results = {}

        if not self.vector_db:
            logger.warning("Vector database not initialized, skipping retrieval")
            return results

        try:
            if self.retriever == 'langchain':
                from langchain.vectorstores import FAISS as LangchainFAISS

                # Vector DB path from config
                vector_db_path = self.config.get('vector_db_path')

                if os.path.exists(vector_db_path):
                    # Load existing vectorstore
                    vectorstore = LangchainFAISS.load_local(
                        vector_db_path,
                        self.embeddings
                    )

                    # Query for similar documents
                    docs = vectorstore.similarity_search(query, k=top_k)

                    # Extract content
                    results['documents'] = [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        }
                        for doc in docs
                    ]

                    logger.info(f"Retrieved {len(results['documents'])} documents from LangChain vector DB")

            elif self.retriever == 'llama_index':
                from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage

                # Vector DB path from config
                storage_dir = self.config.get('vector_db_path')

                if os.path.exists(storage_dir):
                    # Load existing index
                    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
                    index = load_index_from_storage(storage_context)

                    # Query the index
                    retriever = index.as_retriever(similarity_top_k=top_k)
                    nodes = retriever.retrieve(query)

                    # Extract content
                    results['documents'] = [
                        {
                            'content': node.text,
                            'metadata': node.metadata
                        }
                        for node in nodes
                    ]

                    logger.info(f"Retrieved {len(results['documents'])} documents from LlamaIndex vector DB")

            elif self.vector_db_type == 'faiss':
                # Check if encoder is available before attempting to encode
                if self.encoder is None:
                    logger.error("Encoder is not initialized. Make sure FAISS was properly initialized")
                    return results

                try:
                    # Encode query
                    query_embedding = self.encoder.encode([query])[0]

                    # Search FAISS index
                    distances, indices = self.vector_db.search(
                        query_embedding.reshape(1, -1).astype('float32'),
                        top_k
                    )

                    # Get document contents (requires a separate document store)
                    if hasattr(self, 'documents') and self.documents:
                        results['documents'] = [
                            self.documents[idx] for idx in indices[0] if idx < len(self.documents)
                        ]

                        logger.info(f"Retrieved {len(results['documents'])} documents from FAISS vector DB")
                    else:
                        logger.warning("FAISS index found but no document store available")
                except Exception as e:
                    logger.error(f"Error searching FAISS index: {e}")
                    return results

            elif self.vector_db_type == 'chroma':
                # Query Chroma collection
                results_chroma = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )

                # Extract documents
                if results_chroma.get('documents'):
                    results['documents'] = [
                        {
                            'content': doc,
                            'metadata': {'id': id}
                        }
                        for doc, id in zip(
                            results_chroma['documents'][0],
                            results_chroma['ids'][0]
                        )
                    ]

                    logger.info(f"Retrieved {len(results['documents'])} documents from Chroma DB")

            elif self.vector_db_type == 'pinecone':
                # Check if encoder is available
                if self.encoder is None:
                    logger.error("Encoder is not initialized for Pinecone queries")
                    return results

                try:
                    # Encode query
                    query_embedding = self.encoder.encode(query).tolist()

                    # Query Pinecone
                    pinecone_results = self.vector_db.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True
                    )

                    # Extract documents
                    if pinecone_results.get('matches'):
                        results['documents'] = [
                            {
                                'content': match.get('metadata', {}).get('text', ''),
                                'metadata': match.get('metadata', {})
                            }
                            for match in pinecone_results['matches']
                        ]

                        logger.info(f"Retrieved {len(results['documents'])} documents from Pinecone DB")
                except Exception as e:
                    logger.error(f"Error querying Pinecone: {e}")
                    return results

        except Exception as e:
            logger.error(f"Error retrieving from vector DB: {e}")

        return results

    def index_document(self, document: Dict[str, Any]) -> bool:
        """
        Index a document in the vector database.

        Args:
            document: Dictionary with document content and metadata

        Returns:
            True if indexing was successful, False otherwise
        """
        if not document or not document.get('content'):
            logger.warning("Empty document or no content provided, skipping indexing")
            return False

        if not self.vector_db and not self.fallback_storage_enabled:
            logger.warning("Vector database not initialized and fallback storage is disabled, skipping indexing")
            return False

        try:
            content = document.get('content', '')
            metadata = document.get('metadata', {})

            if not content:
                logger.warning("Empty document content, skipping indexing")
                return False

            if self.retriever == 'langchain':
                from langchain.schema import Document
                from langchain.vectorstores import FAISS as LangchainFAISS

                # Vector DB path from config
                vector_db_path = self.config.get('vector_db_path')

                # Create LangChain document
                langchain_doc = Document(page_content=content, metadata=metadata)

                # Split document into chunks
                docs = self.text_splitter.split_documents([langchain_doc])

                if os.path.exists(vector_db_path):
                    # Load existing vectorstore
                    vectorstore = LangchainFAISS.load_local(
                        vector_db_path,
                        self.embeddings
                    )

                    # Add documents to existing index
                    vectorstore.add_documents(docs)
                else:
                    # Create new vectorstore
                    vectorstore = LangchainFAISS.from_documents(
                        docs,
                        self.embeddings
                    )

                # Save vectorstore
                os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
                vectorstore.save_local(vector_db_path)

                logger.info(f"Document indexed in LangChain vector DB with {len(docs)} chunks")
                return True

            elif self.retriever == 'llama_index':
                from llama_index import Document as LlamaDocument
                from llama_index import VectorStoreIndex, StorageContext

                # Vector DB path from config
                storage_dir = self.config.get('vector_db_path')

                # Create LlamaIndex document
                llama_doc = LlamaDocument(text=content, metadata=metadata)

                # Parse nodes
                nodes = self.node_parser.get_nodes_from_documents([llama_doc])

                # Create or update index
                if os.path.exists(storage_dir):
                    # Load existing index
                    from llama_index import load_index_from_storage
                    storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
                    index = load_index_from_storage(storage_context)

                    # Insert nodes
                    for node in nodes:
                        index.insert(node)
                else:
                    # Create new index
                    os.makedirs(storage_dir, exist_ok=True)
                    index = VectorStoreIndex(
                        nodes,
                        service_context=self.service_context
                    )

                # Save index
                index.storage_context.persist(persist_dir=storage_dir)

                logger.info(f"Document indexed in LlamaIndex vector DB with {len(nodes)} nodes")
                return True

            elif self.vector_db_type == 'faiss':
                # Check if encoder is available before attempting to encode
                if self.encoder is None:
                    logger.error("Encoder is not initialized. Make sure FAISS was properly initialized")
                    # Try to initialize the encoder if possible
                    try:
                        from sentence_transformers import SentenceTransformer
                        model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                        logger.info(f"Attempting to initialize encoder with model: {model_name}")
                        self.encoder = SentenceTransformer(model_name)
                        logger.info(f"Successfully initialized encoder model: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize encoder: {e}")
                        # Fall back to local storage if enabled
                        if self.fallback_storage_enabled:
                            return self._store_document_in_fallback(document)
                        return False

                try:
                    # Encode document
                    embedding = self.encoder.encode([content])[0]

                    # Add to FAISS index
                    self.vector_db.add(embedding.reshape(1, -1).astype('float32'))

                    # Store document for retrieval (FAISS only stores vectors)
                    if not hasattr(self, 'documents'):
                        self.documents = []

                    self.documents.append({
                        'content': content,
                        'metadata': metadata
                    })

                    # Save FAISS index
                    index_path = self.config.get('faiss_index_path')
                    if index_path:
                        try:
                            import faiss
                            os.makedirs(os.path.dirname(index_path), exist_ok=True)
                            faiss.write_index(self.vector_db, index_path)
                        except Exception as e:
                            logger.warning(f"Failed to save FAISS index: {e}")

                    # Save documents
                    documents_path = os.path.join(self.cache_dir, 'faiss_documents.json')
                    try:
                        with open(documents_path, 'w') as f:
                            json.dump(self.documents, f)
                    except Exception as e:
                        logger.warning(f"Failed to save document store: {e}")

                    logger.info("Document indexed in FAISS vector DB")
                    return True
                except Exception as e:
                    logger.error(f"Error encoding document: {e}")
                    # Fall back to local storage if enabled
                    if self.fallback_storage_enabled:
                        return self._store_document_in_fallback(document)
                    return False

            elif self.vector_db_type == 'chroma':
                # Generate a unique ID
                doc_id = str(int(time.time() * 1000))

                # Add to Chroma collection
                self.collection.add(
                    documents=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )

                logger.info("Document indexed in Chroma vector DB")
                return True

            elif self.vector_db_type == 'pinecone':
                # Check if encoder is available before attempting to encode
                if self.encoder is None:
                    logger.error("Encoder is not initialized for Pinecone. Make sure it was properly initialized")
                    # Try to initialize the encoder
                    try:
                        from sentence_transformers import SentenceTransformer
                        model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
                        self.encoder = SentenceTransformer(model_name)
                        logger.info(f"Successfully initialized encoder model for Pinecone: {model_name}")
                    except Exception as e:
                        logger.error(f"Failed to initialize encoder for Pinecone: {e}")
                        # Fall back to local storage if enabled
                        if self.fallback_storage_enabled:
                            return self._store_document_in_fallback(document)
                        return False

                try:
                    # Encode document
                    embedding = self.encoder.encode(content).tolist()

                    # Generate a unique ID
                    doc_id = str(int(time.time() * 1000))

                    # Add to Pinecone
                    self.vector_db.upsert(
                        vectors=[(doc_id, embedding, {'text': content, **metadata})]
                    )

                    logger.info("Document indexed in Pinecone vector DB")
                    return True
                except Exception as e:
                    logger.error(f"Error encoding document for Pinecone: {e}")
                    # Fall back to local storage if enabled
                    if self.fallback_storage_enabled:
                        return self._store_document_in_fallback(document)
                    return False

            elif self.fallback_storage_enabled:
                return self._store_document_in_fallback(document)
            else:
                logger.warning(f"Unsupported vector DB type: {self.config.get('vector_db_type')}")
                return False

        except Exception as e:
            logger.error(f"Error indexing document: {e}")
            # Try fallback storage as a last resort
            if self.fallback_storage_enabled:
                return self._store_document_in_fallback(document)
            return False

    def _store_document_in_fallback(self, document: Dict[str, Any]) -> bool:
        """Helper method to store document in fallback storage"""
        try:
            # Fallback to local storage
            if not hasattr(self, 'documents'):
                self.documents = []

            content = document.get('content', '')
            metadata = document.get('metadata', {})

            self.documents.append({
                'content': content,
                'metadata': metadata
            })

            # Save documents to fallback storage
            try:
                os.makedirs(os.path.dirname(self.documents_path), exist_ok=True)
                with open(self.documents_path, 'w') as f:
                    json.dump(self.documents, f)
            except Exception as e:
                logger.warning(f"Failed to save fallback storage: {e}")

            logger.info("Document indexed in fallback storage")
            return True
        except Exception as e:
            logger.error(f"Error storing document in fallback storage: {e}")
            return False

    def retrieve_domain_knowledge(self, query: str, scope: str, test_type: str) -> Dict[str, Any]:
        """
        Retrieve domain-specific knowledge based on scope and test type.

        Args:
            query: The query or requirement text
            scope: Test scope (Functional, Non-Functional, or Both)
            test_type: Test type (Component, Integration, Acceptance, or All)

        Returns:
            Dictionary with retrieved domain knowledge
        """
        # Domain knowledge directories from config
        knowledge_dir = self.config.get('knowledge_dir')
        if not knowledge_dir:
            logger.warning("No knowledge directory configured, skipping domain knowledge retrieval")
            return {}
        # Check if knowledge directory exists
        import os
        if not knowledge_dir or not os.path.exists(knowledge_dir):
            logger.warning(f"Domain knowledge directory not found: {knowledge_dir}")
            return {}

        try:
            import os
            from glob import glob

            domain_knowledge = {}

            # Determine which subdirectories to search based on scope and test type
            scope_dir = os.path.join(knowledge_dir, scope.lower()) if scope != "Both" else knowledge_dir
            test_type_dir = os.path.join(scope_dir, test_type.lower()) if test_type != "All" else scope_dir

            # Get all markdown and text files in the directory
            knowledge_files = []
            for ext in ['*.md', '*.txt', '*.json']:
                if os.path.exists(test_type_dir):
                    knowledge_files.extend(glob(os.path.join(test_type_dir, ext)))
                if os.path.exists(scope_dir) and scope_dir != test_type_dir:
                    knowledge_files.extend(glob(os.path.join(scope_dir, ext)))
                if knowledge_dir != scope_dir and knowledge_dir != test_type_dir:
                    knowledge_files.extend(glob(os.path.join(knowledge_dir, ext)))

            # Extract and process content from files
            for file_path in knowledge_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()

                    filename = os.path.basename(file_path)
                    if content:
                        domain_knowledge[filename] = content
                except Exception as e:
                    logger.warning(f"Error reading domain knowledge file {file_path}: {e}")

            logger.info(f"Retrieved {len(domain_knowledge)} domain knowledge files")
            return {'domain_knowledge': domain_knowledge}

        except Exception as e:
            logger.error(f"Error retrieving domain knowledge: {e}")

        return {}

    def retrieve_test_patterns(self, test_type: str) -> Dict[str, Any]:
        """
        Retrieve common test patterns based on test type.

        Args:
            test_type: Test type (Component, Integration, Acceptance, or All)

        Returns:
            Dictionary with relevant test patterns
        """
        # Define common test patterns for different test types
        test_patterns = {
            'Component': [
                {
                    'name': 'Input Validation',
                    'description': 'Tests that verify input validation rules are correctly applied',
                    'template': 'Verify {component} handles {valid/invalid} {input_type} correctly'
                },
                {
                    'name': 'Output Verification',
                    'description': 'Tests that verify the output matches expected results',
                    'template': 'Verify {component} produces correct {output_type} when {condition}'
                },
                {
                    'name': 'Error Handling',
                    'description': 'Tests that verify proper error handling for invalid inputs or exceptional cases',
                    'template': 'Verify {component} handles {error_condition} gracefully'
                },
                {
                    'name': 'Boundary Tests',
                    'description': 'Tests that verify behavior at the boundaries of acceptable input ranges',
                    'template': 'Verify {component} handles {boundary_condition} correctly'
                },
                {
                    'name': 'State Management',
                    'description': 'Tests that verify the component maintains correct internal state',
                    'template': 'Verify {component} maintains correct state after {operation}'
                }
            ],
            'Integration': [
                {
                    'name': 'Interface Compatibility',
                    'description': 'Tests that verify interfaces between components are compatible',
                    'template': 'Verify {component1} correctly interfaces with {component2}'
                },
                {
                    'name': 'Data Flow',
                    'description': 'Tests that verify data flows correctly between components',
                    'template': 'Verify data flows correctly from {source} to {destination}'
                },
                {
                    'name': 'Transaction Management',
                    'description': 'Tests that verify transactions are managed correctly across components',
                    'template': 'Verify transaction {operation} is correctly managed across {components}'
                },
                {
                    'name': 'Error Propagation',
                    'description': 'Tests that verify errors are correctly propagated between components',
                    'template': 'Verify error in {component1} is correctly propagated to {component2}'
                },
                {
                    'name': 'Dependency Verification',
                    'description': 'Tests that verify all dependencies are correctly resolved',
                    'template': 'Verify {component} correctly resolves dependencies on {dependencies}'
                }
            ],
            'Acceptance': [
                {
                    'name': 'User Story Validation',
                    'description': 'Tests that verify the system satisfies user stories',
                    'template': 'Verify user can {action} to achieve {goal}'
                },
                {
                    'name': 'Business Rule Compliance',
                    'description': 'Tests that verify the system complies with business rules',
                    'template': 'Verify system enforces business rule: {rule}'
                },
                {
                    'name': 'End-to-End Workflow',
                    'description': 'Tests that verify complete end-to-end workflows',
                    'template': 'Verify end-to-end workflow: {workflow}'
                },
                {
                    'name': 'User Interface',
                    'description': 'Tests that verify the user interface meets requirements',
                    'template': 'Verify {UI_element} displays correctly and is functional'
                },
                {
                    'name': 'Performance Criteria',
                    'description': 'Tests that verify the system meets performance criteria',
                    'template': 'Verify {operation} completes within {time_limit}'
                }
            ]
        }

        # Return patterns based on test type
        if test_type != 'All':
            return {'test_patterns': test_patterns.get(test_type, [])}
        else:
            return {'test_patterns': [pattern for patterns in test_patterns.values() for pattern in patterns]}

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
        # Generate cache key
        cache_key = self._get_cache_key(query, scope, test_type)

        # Check cache first
        cached_context = self._check_cache(cache_key)
        if cached_context:
            return cached_context

        # Initialize enhanced context
        enhanced_context = {}

        # Add component information if provided
        if components:
            enhanced_context['components'] = components

        # Retrieve information from various sources
        try:
            # 1. Retrieve from APIs
            api_results = self.retrieve_from_api(query, scope, test_type)
            if api_results:
                enhanced_context['api_context'] = api_results

            # 2. Retrieve from vector database
            vector_results = self.retrieve_from_vector_db(query)
            if vector_results:
                enhanced_context['vector_context'] = vector_results

            # 3. Retrieve domain knowledge
            domain_knowledge = self.retrieve_domain_knowledge(query, scope, test_type)
            if domain_knowledge:
                enhanced_context['domain_context'] = domain_knowledge

            # 4. Retrieve test patterns
            test_patterns = self.retrieve_test_patterns(test_type if test_type != 'All' else 'Component')
            if test_patterns:
                enhanced_context['test_patterns'] = test_patterns

            # 5. Add timestamp
            enhanced_context['retrieved_at'] = datetime.now().isoformat()

            # Update cache
            self._update_cache(cache_key, enhanced_context)

            logger.info(f"Enhanced context created with {len(enhanced_context)} sources")

        except Exception as e:
            logger.error(f"Error enhancing context: {e}")

        return enhanced_context

    def format_context_for_prompt(self, enhanced_context: Dict[str, Any]) -> str:
        """
        Format the enhanced context into a string suitable for prompt injection.

        Args:
            enhanced_context: The enhanced context dictionary

        Returns:
            Formatted context string for prompt injection
        """
        if not enhanced_context:
            return ""

        context_parts = []

        # Format API context
        if 'api_context' in enhanced_context:
            api_context = enhanced_context['api_context']
            context_parts.append("EXTERNAL API KNOWLEDGE:")

            for api_name, api_data in api_context.items():
                context_parts.append(f"From {api_name}:")

                if isinstance(api_data, dict):
                    for key, value in api_data.items():
                        if isinstance(value, str):
                            context_parts.append(f"- {key}: {value}")
                        elif isinstance(value, (list, dict)):
                            context_parts.append(f"- {key}: {json.dumps(value, indent=2)}")
                elif isinstance(api_data, list):
                    for item in api_data:
                        context_parts.append(f"- {item}")
                else:
                    context_parts.append(f"- {api_data}")

                context_parts.append("")

        # Format vector DB context
        if 'vector_context' in enhanced_context:
            vector_context = enhanced_context['vector_context']

            if 'documents' in vector_context and vector_context['documents']:
                context_parts.append("RELEVANT KNOWLEDGE BASE DOCUMENTS:")

                for i, doc in enumerate(vector_context['documents']):
                    context_parts.append(f"Document {i+1}:")
                    context_parts.append(doc.get('content', 'No content'))

                    if 'metadata' in doc and doc['metadata']:
                        context_parts.append("Metadata:")
                        for key, value in doc['metadata'].items():
                            context_parts.append(f"- {key}: {value}")

                    context_parts.append("")

        # Format domain context
        if 'domain_context' in enhanced_context:
            domain_context = enhanced_context['domain_context']

            if 'domain_knowledge' in domain_context and domain_context['domain_knowledge']:
                context_parts.append("DOMAIN-SPECIFIC KNOWLEDGE:")

                for filename, content in domain_context['domain_knowledge'].items():
                    # Limit content length to avoid overloading the prompt
                    if len(content) > 1000:
                        content = content[:997] + "..."

                    context_parts.append(f"From {filename}:")
                    context_parts.append(content)
                    context_parts.append("")

        # Format test patterns
        if 'test_patterns' in enhanced_context:
            test_patterns = enhanced_context['test_patterns']
            context_parts.append("RECOMMENDED TEST PATTERNS:")

            for pattern in test_patterns.get('test_patterns', []):
                context_parts.append(f"Pattern: {pattern['name']}")
                context_parts.append(f"Description: {pattern['description']}")
                context_parts.append(f"Template: {pattern['template']}")
                context_parts.append("")

        # Join all parts with newlines
        formatted_context = "\n".join(context_parts)

        # Truncate if too long
        max_length = 8000  # Adjust based on token limits of your LLM
        if len(formatted_context) > max_length:
            logger.warning(f"Context too long ({len(formatted_context)} chars), truncating to {max_length}")
            formatted_context = formatted_context[:max_length - 100] + "\n...(truncated)..."

        return formatted_context

    def create_enhanced_prompt(self, original_prompt: str, enhanced_context: Dict[str, Any]) -> str:
        """
        Create an enhanced prompt by injecting the retrieved context.

        Args:
            original_prompt: The original prompt template
            enhanced_context: The enhanced context dictionary

        Returns:
            Enhanced prompt with injected context
        """
        formatted_context = self.format_context_for_prompt(enhanced_context)

        if not formatted_context:
            return original_prompt

        # Find a good injection point in the original prompt
        # Good points are usually after the instruction but before examples

        # Common markers that indicate good injection points
        markers = [
            "Here are some examples:",
            "For example:",
            "Generate the following:",
            "Based on the requirements:",
            "Given the following requirements:"
        ]

        injection_point = -1

        for marker in markers:
            pos = original_prompt.find(marker)
            if pos > 0:
                injection_point = pos
                break

        # If no marker found, append context to the end of the prompt
        if injection_point < 0:
            enhanced_prompt = f"{original_prompt}\n\nADDITIONAL CONTEXT:\n{formatted_context}"
        else:
            # Insert context at the injection point
            enhanced_prompt = (
                f"{original_prompt[:injection_point]}\n\n"
                f"ADDITIONAL CONTEXT:\n{formatted_context}\n\n"
                f"{original_prompt[injection_point:]}"
            )

        logger.info(f"Created enhanced prompt ({len(enhanced_prompt)} chars)")
        return enhanced_prompt

    def crawl_and_index_urls(self, urls: List[str], labels: Optional[List[str]] = None, force_refresh: bool = False, max_depth_override: Optional[int] = None, ignore_robots: bool = False) -> Dict[str, Any]:
        """
        Crawl and index content from a list of URLs.

        Args:
            urls: List of URLs to crawl and index
            labels: Optional labels to associate with each URL for better organization
            force_refresh: If True, ignores cache and forces fresh crawling
            max_depth_override: Optional override for maximum crawl depth
            ignore_robots: If True, ignores robots.txt restrictions

        Returns:
            Dictionary with crawling statistics and status
        """
        if not urls:
            logger.warning("No URLs provided for crawling")
            return {"status": "error", "message": "No URLs provided"}

        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urlparse, urljoin, urldefrag
            import time
            from collections import deque

            # Initialize results dictionary
            results = {
                "status": "success",
                "total_urls": len(urls),
                "successful_crawls": 0,
                "failed_crawls": 0,
                "total_pages_indexed": 0,
                "brand_pages_indexed": 0,
                "urls_status": {}
            }

            # Keep track of branded domains for brand-specific counters
            brand_domains = {}

            # Use the override depth if provided, otherwise use configured depth
            # Important: Convert to integer if provided as a string
            if max_depth_override is not None:
                try:
                    # Convert to int if it's a string
                    actual_max_depth = int(max_depth_override)
                    logger.info(f"Using overridden crawl depth: {actual_max_depth}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid max_depth_override value: {max_depth_override}, using default")
                    actual_max_depth = self.max_crawl_depth
            else:
                actual_max_depth = self.max_crawl_depth

            # Log the configuration values being used
            logger.info(f"Crawling with configuration: max_crawl_depth={actual_max_depth}, "
                        f"max_pages_per_domain={self.max_pages_per_domain}, "
                        f"max_pages_per_brand={self.max_pages_per_brand}, "
                        f"force_refresh={force_refresh}, ignore_robots={ignore_robots}")

            # Process each URL
            for i, url in enumerate(urls):
                url_label = labels[i] if labels and i < len(labels) else f"source_{i+1}"
                is_brand_source = "brand" in url_label.lower() or "brand" in url.lower()

                # Normalize URL to prevent duplicates (remove fragments)
                url = urldefrag(url)[0]

                # Get domain info
                parsed_url = urlparse(url)
                domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

                # Initialize brand-specific page counters if this is a branded source
                if is_brand_source and domain not in brand_domains:
                    brand_domains[domain] = {
                        "page_count": 0,
                        "max_pages": self.config.get('max_pages_per_brand', self.max_pages_per_domain)
                    }

                try:
                    logger.info(f"Crawling URL: {url} with label: {url_label}")

                    # Check URL cache first (unless force_refresh is True)
                    url_cache_key = hashlib.md5(url.encode()).hexdigest()
                    url_cache_file = os.path.join(self.url_cache_dir, f"{url_cache_key}.json")

                    # Check if we should use the cache
                    if not force_refresh and os.path.exists(url_cache_file) and self.enable_cache:
                        try:
                            with open(url_cache_file, 'r') as f:
                                cached_data = json.load(f)

                            # Check if cache is still valid
                            timestamp = cached_data.get('timestamp', 0)
                            if time.time() - timestamp <= self.url_cache_ttl:
                                # Check if the cached crawl used the same depth
                                cached_depth = cached_data.get('max_depth', self.max_crawl_depth)
                                if cached_depth == actual_max_depth:
                                    logger.info(f"Using cached data for URL: {url}")

                                    # Update statistics
                                    results["successful_crawls"] += 1
                                    pages_indexed = cached_data.get('pages_indexed', 0)
                                    brand_pages_indexed = cached_data.get('brand_pages_indexed', 0)

                                    results["total_pages_indexed"] += pages_indexed
                                    results["brand_pages_indexed"] += brand_pages_indexed
                                    results["urls_status"][url] = {
                                        "status": "cached",
                                        "pages_indexed": pages_indexed,
                                        "brand_pages_indexed": brand_pages_indexed,
                                        "visited_urls_count": len(cached_data.get('visited_urls', [])),
                                    }

                                    # Update brand domain counter if appropriate
                                    if is_brand_source and domain in brand_domains:
                                        brand_domains[domain]["page_count"] += pages_indexed

                                    continue
                                else:
                                    logger.info(f"Cached crawl used different depth ({cached_depth} vs {actual_max_depth}), will crawl fresh")
                            else:
                                logger.info(f"Cache expired, will crawl fresh: {url}")
                        except Exception as e:
                            logger.warning(f"Error reading cache, will crawl fresh: {e}")
                            # Continue with fresh crawl if cache read fails

                    # Initialize crawler state for this domain
                    visited_urls = set()
                    failed_urls = set()
                    retry_count = {}  # Track retry attempts for problematic URLs

                    # Store URLs with their depth level for better crawling [(url, depth_level, is_brand_page)]
                    # Use deque for faster operations
                    urls_to_visit = deque([(url, 0, is_brand_source)])

                    pages_indexed = 0
                    brand_pages_indexed = 0
                    domain_documents = []
                    pages_visited = 0
                    urls_by_depth = {0: 1}  # Track URLs at each depth level

                    # Determine max pages for this crawl based on domain type
                    max_pages_for_domain = brand_domains[domain]["max_pages"] if is_brand_source and domain in brand_domains else self.max_pages_per_domain

                    # Respect robots.txt (unless ignore_robots is True)
                    robot_parser = None
                    if not ignore_robots:
                        try:
                            import urllib.robotparser
                            robot_parser = urllib.robotparser.RobotFileParser()
                            robot_parser.set_url(urljoin(domain, "/robots.txt"))
                            robot_parser.read()
                            logger.info(f"Loaded robots.txt for {domain}")
                        except Exception as e:
                            robot_parser = None
                            logger.warning(f"Could not parse robots.txt for {domain}: {e}")
                    else:
                        logger.info(f"Ignoring robots.txt for {domain} as requested")

                    # Brand-related keywords to identify brand pages
                    brand_keywords = ['brand', 'product', 'catalog', 'collection', 'model', 'series',
                                      'specs', 'specification', 'feature']

                    logger.info(f"Starting crawl for {domain} with max_depth={actual_max_depth}, max_pages={max_pages_for_domain}")

                    # Continue crawling while there are URLs to visit and we haven't reached the limit
                    while urls_to_visit and pages_indexed < max_pages_for_domain:
                        # Get the next URL, its depth level, and brand status
                        page_url, depth_level, is_brand_page = urls_to_visit.popleft()

                        # Normalize URL to prevent duplicates (remove fragments)
                        page_url = urldefrag(page_url)[0]

                        # Skip if already visited or failed
                        if page_url in visited_urls or page_url in failed_urls:
                            continue

                        # Update depth tracking
                        if depth_level not in urls_by_depth:
                            urls_by_depth[depth_level] = 0
                        urls_by_depth[depth_level] += 1

                        # Don't go beyond max depth
                        # Use actual_max_depth instead of self.max_crawl_depth here
                        if depth_level > actual_max_depth:
                            continue

                        # For brand domains, check if we've hit the brand-specific page limit
                        if is_brand_source and domain in brand_domains and brand_domains[domain]["page_count"] >= brand_domains[domain]["max_pages"]:
                            logger.info(f"Reached maximum brand pages limit ({brand_domains[domain]['max_pages']}) for brand domain {domain}")
                            break

                        # Check if crawling is allowed by robots.txt
                        if robot_parser and not ignore_robots:
                            try:
                                if not robot_parser.can_fetch("*", page_url):
                                    logger.info(f"Skipping {page_url} - disallowed by robots.txt")
                                    continue
                            except Exception as e:
                                logger.warning(f"Error checking robots.txt for {page_url}: {e}")
                                # Continue anyway on robots.txt error

                        # Add to visited set and increment counter
                        visited_urls.add(page_url)
                        pages_visited += 1

                        # Log progress periodically
                        if pages_visited % 10 == 0:
                            logger.info(f"Progress update: Visited {pages_visited} pages, indexed {pages_indexed}, "
                                      f"queue size: {len(urls_to_visit)}, depth stats: {urls_by_depth}")

                        # Check if it's potentially a brand page based on URL
                        current_is_brand_page = is_brand_page or any(keyword in page_url.lower() for keyword in brand_keywords)

                        try:
                            # Respect crawl delay
                            time.sleep(self.crawl_delay)

                            # Fetch page content with proper headers and timeout
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (compatible; TestAutomationBot/1.0; +https://example.com/bot)',
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                                'Accept-Language': 'en-US,en;q=0.5',
                                'Connection': 'keep-alive',
                                'Upgrade-Insecure-Requests': '1',
                                'Cache-Control': 'max-age=0',
                                'TE': 'Trailers'
                            }

                            logger.info(f"Fetching {page_url} (depth: {depth_level}" +
                                      (", BRAND PAGE" if current_is_brand_page else "") + ")")

                            response = http_session.get(page_url, headers=headers, timeout=30,
                                                 allow_redirects=True, verify=False)

                            # Follow redirects manually if needed
                            redirect_count = 0
                            max_redirects = 5
                            while (300 <= response.status_code < 400 and
                                  'location' in response.headers and
                                  redirect_count < max_redirects):
                                redirect_url = urljoin(page_url, response.headers['location'])
                                logger.info(f"Following redirect from {page_url} to {redirect_url}")
                                response = http_session.get(redirect_url, headers=headers, timeout=30,
                                                     allow_redirects=False, verify=False)
                                redirect_count += 1
                                page_url = redirect_url  # Update the current URL

                            if response.status_code != 200:
                                logger.warning(f"Failed to fetch {page_url}: HTTP {response.status_code}")
                                failed_urls.add(page_url)
                                continue

                            # Check content type
                            content_type = response.headers.get('Content-Type', '')
                            if 'text/html' not in content_type.lower() and 'application/xhtml+xml' not in content_type.lower():
                                logger.info(f"Skipping non-HTML content at {page_url}: {content_type}")
                                continue

                            # Parse HTML
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Additional brand detection based on page content
                            page_text = soup.get_text().lower()
                            meta_tags = soup.find_all('meta')
                            meta_content = ' '.join([tag.get('content', '').lower() for tag in meta_tags if tag.get('content')])

                            # Check meta tags and page content for brand indicators
                            if not current_is_brand_page:
                                for keyword in brand_keywords:
                                    if keyword in meta_content or keyword in page_text[:1000]:
                                        current_is_brand_page = True
                                        logger.info(f"Detected brand page based on content: {page_url}")
                                        break

                            # Extract useful content and filter out irrelevant parts
                            content = self._extract_useful_content(soup)

                            if content and len(content) > 100:  # Only index content with meaningful length
                                # Create document for indexing
                                document = {
                                    'content': content,
                                    'metadata': {
                                        'url': page_url,
                                        'domain': domain,
                                        'label': url_label,
                                        'title': soup.title.string if soup.title else page_url,
                                        'crawled_at': datetime.now().isoformat(),
                                        'depth': depth_level,
                                        'is_brand_page': current_is_brand_page
                                    }
                                }

                                # Index the document
                                if self.index_document(document):
                                    pages_indexed += 1

                                    # Update brand-specific counters
                                    if current_is_brand_page:
                                        brand_pages_indexed += 1
                                        if is_brand_source and domain in brand_domains:
                                            brand_domains[domain]["page_count"] += 1

                                    domain_documents.append(document)
                                    logger.info(f"Indexed content from {page_url}" +
                                              (" (BRAND PAGE)" if current_is_brand_page else ""))
                                else:
                                    logger.warning(f"Failed to index content from {page_url}")

                            # Find links to other pages on the same domain if we haven't reached max depth
                            if depth_level < actual_max_depth:  # Use actual_max_depth here
                                # Improved link discovery strategy
                                discovered_links = self._discover_links(soup, page_url, domain, visited_urls, failed_urls)

                                # Prioritize brand pages by adding them first
                                priority_links = []
                                regular_links = []

                                for href, link_text in discovered_links:
                                    is_link_brand_page = current_is_brand_page or any(
                                        keyword in href.lower() or keyword in link_text.lower()
                                        for keyword in brand_keywords
                                    )

                                    if is_link_brand_page:
                                        priority_links.append((href, depth_level + 1, is_link_brand_page))
                                    else:
                                        regular_links.append((href, depth_level + 1, is_link_brand_page))

                                # Log the discoveries
                                logger.info(f"Discovered {len(discovered_links)} links from {page_url}: " +
                                          f"{len(priority_links)} priority links, {len(regular_links)} regular links")

                                # Add priority links first (brand pages)
                                for link in priority_links:
                                    urls_to_visit.append(link)

                                # Then add regular links
                                for link in regular_links:
                                    urls_to_visit.append(link)

                            # Check if we've reached maximum pages
                            if pages_indexed >= max_pages_for_domain:
                                logger.info(f"Reached maximum pages limit ({max_pages_for_domain}) for domain {domain}")
                                break

                        except requests.exceptions.Timeout:
                            logger.warning(f"Timeout while fetching {page_url} - skipping")
                            failed_urls.add(page_url)
                        except requests.exceptions.TooManyRedirects:
                            logger.warning(f"Too many redirects for {page_url} - skipping")
                            failed_urls.add(page_url)
                        except requests.exceptions.RequestException as e:
                            logger.warning(f"Request failed for {page_url}: {str(e)}")
                            failed_urls.add(page_url)
                        except Exception as e:
                            logger.warning(f"Error processing {page_url}: {str(e)}")
                            failed_urls.add(page_url)

                    # Cache the results - include the actual max depth that was used
                    cache_data = {
                        'timestamp': time.time(),
                        'domain': domain,
                        'pages_indexed': pages_indexed,
                        'brand_pages_indexed': brand_pages_indexed,
                        'visited_urls': list(visited_urls),
                        'failed_urls': list(failed_urls),
                        'is_brand_source': is_brand_source,
                        'depth_statistics': urls_by_depth,
                        'max_depth': actual_max_depth  # Store the actual max depth that was used
                    }

                    with open(url_cache_file, 'w') as f:
                        json.dump(cache_data, f)

                    # Update statistics
                    results["successful_crawls"] += 1
                    results["total_pages_indexed"] += pages_indexed
                    results["brand_pages_indexed"] += brand_pages_indexed
                    results["urls_status"][url] = {
                        "status": "success",
                        "pages_indexed": pages_indexed,
                        "brand_pages_indexed": brand_pages_indexed,
                        "pages_visited": len(visited_urls),
                        "failed_pages": len(failed_urls),
                        "depth_statistics": urls_by_depth,
                        "max_depth_used": actual_max_depth  # Also include in results
                    }

                    logger.info(f"Completed crawling {url}: visited {len(visited_urls)} pages, " +
                              f"indexed {pages_indexed} pages ({brand_pages_indexed} brand pages)")
                    logger.info(f"Depth statistics: {urls_by_depth}")

                except Exception as e:
                    logger.error(f"Failed to crawl {url}: {str(e)}")
                    results["failed_crawls"] += 1
                    results["urls_status"][url] = {"status": "error", "message": str(e)}

            # Add brand domain statistics to results
            results["brand_domains"] = {domain: stats["page_count"] for domain, stats in brand_domains.items()}

            # Include the actual max depth that was used in the overall results
            results["max_depth_used"] = actual_max_depth

            # Print final statistics
            logger.info(f"Crawling complete. Total pages indexed: {results['total_pages_indexed']}, " +
                      f"brand pages: {results['brand_pages_indexed']}, " +
                      f"successful crawls: {results['successful_crawls']}, " +
                      f"max depth used: {actual_max_depth}")

            return results

        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
            error_msg = f"Missing required libraries. Run: pip install requests beautifulsoup4"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            logger.error(f"Error during URL crawling: {str(e)}")
            return {"status": "error", "message": str(e)}

    def _extract_useful_content(self, soup) -> str:
        """
        Extract useful content from a BeautifulSoup object and filter out irrelevant information.

        Args:
            soup: BeautifulSoup object representing the HTML page

        Returns:
            Extracted and filtered content as a string
        """
        # Remove irrelevant elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()

        # Try to identify and extract main content
        main_content = None

        # Look for common content containers
        for container in ['main', 'article', 'div[role="main"]', '.content', '.main-content', '#content', '#main']:
            if container.startswith('.'):
                elements = soup.select(container)
            elif container.startswith('#'):
                elements = soup.select(container)
            elif '[' in container:
                tag, attr = container.split('[', 1)
                attr = attr.rstrip(']')
                key, value = attr.split('=')
                value = value.strip('"\'')
                elements = soup.find_all(tag, {key: value})
            else:
                elements = soup.find_all(container)

            if elements:
                main_content = elements[0]
                break

        # If no main content container found, use the body
        if not main_content:
            main_content = soup.body or soup

        # Extract text and clean it
        content = main_content.get_text(separator=' ', strip=True)

        # Remove irrelevant content using configured patterns
        for pattern in self.irrelevant_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # Clean up extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()

        return content

    def _discover_links(self, soup, current_url, domain, visited_urls, failed_urls) -> List[Tuple[str, str]]:
        """
        Discover and normalize links from HTML content.

        Args:
            soup: BeautifulSoup object of the page
            current_url: The current URL being processed
            domain: Base domain for the website
            visited_urls: Set of already visited URLs
            failed_urls: Set of URLs that failed to process

        Returns:
            List of tuples containing (url, link_text)
        """
        from urllib.parse import urlparse, urljoin, urldefrag

        discovered_links = []
        seen_urls = set()  # Track URLs within this page to avoid duplicates

        # Find all links
        links = soup.find_all('a', href=True)

        # Process nav menus and main content links first - they tend to be more valuable
        priority_containers = soup.select('nav, .navigation, .menu, main, article, .content')
        priority_links = []

        for container in priority_containers:
            for link in container.find_all('a', href=True):
                priority_links.append(link)

        # Process priority links first, then remaining links
        for link in priority_links + links:
            href = link['href'].strip()
            link_text = link.get_text().strip()

            # Skip empty links, anchors, javascript, or mailto links
            if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                continue

            # Convert relative URLs to absolute
            if not href.startswith(('http://', 'https://')):
                href = urljoin(current_url, href)

            # Remove URL fragments to avoid duplicate content
            href = urldefrag(href)[0]

            # Skip already processed URLs
            if href in visited_urls or href in failed_urls or href in seen_urls:
                continue

            # Only follow links to the same domain
            parsed_href = urlparse(href)
            href_domain = f"{parsed_href.scheme}://{parsed_href.netloc}"

            if href_domain == domain:
                seen_urls.add(href)
                discovered_links.append((href, link_text))

        return discovered_links

    def learn_from_urls(self, urls: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Learn from content at provided URLs and enhance the RAG model.

        Args:
            urls: List of URLs to learn from
            labels: Optional labels to categorize the URL sources

        Returns:
            Dictionary with learning statistics and status
        """
        # Start by crawling and indexing the URLs
        crawl_results = self.crawl_and_index_urls(urls, labels)

        if crawl_results["status"] != "success":
            return crawl_results

        # Return success with crawling statistics
        return {
            "status": "success",
            "message": f"Successfully learned from {crawl_results['successful_crawls']} URLs, "
                      f"indexed {crawl_results['total_pages_indexed']} pages",
            "crawl_stats": crawl_results
        }

    def crawl_brand_help_pages(self, brand_urls: List[str], labels: Optional[List[str]] = None, max_articles: int = 100) -> Dict[str, Any]:
        """
        Specialized crawler for brand help pages that ensures thorough crawling of brand articles
        and documentation, prioritizing deeper crawling within brand help centers.

        Args:
            brand_urls: List of brand help page URLs to crawl
            labels: Optional labels to associate with each URL for better organization
            max_articles: Maximum number of brand articles to index per brand help center

        Returns:
            Dictionary with crawling statistics and status
        """
        if not brand_urls:
            logger.warning("No brand URLs provided for crawling")
            return {"status": "error", "message": "No brand URLs provided"}

        try:
            from bs4 import BeautifulSoup
            from urllib.parse import urlparse, urljoin
            import time
            from collections import deque

            # Initialize results dictionary
            results = {
                "status": "success",
                "total_urls": len(brand_urls),
                "successful_crawls": 0,
                "failed_crawls": 0,
                "total_articles_indexed": 0,
                "brand_stats": {}
            }

            # Common patterns for brand help articles and important content
            article_indicators = [
                '/article/', '/help/', '/support/', '/knowledge/', '/faq/',
                '/guide/', '/tutorial/', '/doc/', '/documentation/', '/manual/',
                '/product/', '/brand/', '/collection/'
            ]

            # Brand-specific keywords to prioritize
            brand_keywords = [
                'brand', 'product', 'catalog', 'collection', 'model', 'series',
                'feature', 'specification', 'guide', 'manual', 'instruction'
            ]

            # Process each brand URL
            for i, url in enumerate(brand_urls):
                url_label = labels[i] if labels and i < len(labels) else f"brand_{i+1}"

                # Parse domain for site structure
                parsed_url = urlparse(url)
                domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

                try:
                    logger.info(f"Crawling brand help pages from: {url} with label: {url_label}")

                    # Check URL cache first
                    url_cache_key = hashlib.md5(f"{url}_brand_help".encode()).hexdigest()
                    url_cache_file = os.path.join(self.url_cache_dir, f"{url_cache_key}.json")

                    if os.path.exists(url_cache_file) and self.enable_cache:
                        with open(url_cache_file, 'r') as f:
                            cached_data = json.load(f)

                        # Check if cache is still valid
                        timestamp = cached_data.get('timestamp', 0)
                        if time.time() - timestamp <= self.url_cache_ttl:
                            logger.info(f"Using cached brand data for URL: {url}")

                            # Update statistics
                            results["successful_crawls"] += 1
                            articles_indexed = cached_data.get('articles_indexed', 0)
                            results["total_articles_indexed"] += articles_indexed
                            results["brand_stats"][url] = cached_data.get('stats', {})
                            continue

                    # Initialize brand-specific crawl state
                    visited_urls = set()
                    articles_to_visit = deque([(url, 0)])  # (url, depth)
                    articles_indexed = 0
                    article_inventory = []

                    # Track article types found
                    article_types = {
                        'product_info': 0,
                        'user_guide': 0,
                        'faq': 0,
                        'troubleshooting': 0,
                        'other': 0
                    }

                    # Respect robots.txt
                    try:
                        import urllib.robotparser
                        robot_parser = urllib.robotparser.RobotFileParser()
                        robot_parser.set_url(urljoin(domain, "/robots.txt"))
                        robot_parser.read()
                    except Exception as e:
                        robot_parser = None
                        logger.warning(f"Could not parse robots.txt for {domain}: {e}")

                    # Crawl brand help articles
                    logger.info(f"Starting deep crawl of brand help articles from {url}")
                    while articles_to_visit and articles_indexed < max_articles:
                        current_url, depth = articles_to_visit.popleft()

                        if current_url in visited_urls:
                            continue

                        # Process maximum depth of 5 for brand articles to ensure complete coverage
                        if depth > 5:  # Deeper crawl for brand help pages
                            continue

                        # Check if crawling is allowed by robots.txt
                        if robot_parser and not robot_parser.can_fetch("*", current_url):
                            logger.info(f"Skipping {current_url} - disallowed by robots.txt")
                            continue

                        # Add to visited set
                        visited_urls.add(current_url)

                        try:
                            # Respect crawl delay
                            time.sleep(self.crawl_delay)

                            # Fetch page content
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (compatible; TestAutomationBot/1.0; +https://example.com/bot)'
                            }

                            logger.info(f"Fetching brand article: {current_url} (depth: {depth})")
                            response = http_session.get(current_url, headers=headers, timeout=15)

                            if response.status_code != 200:
                                logger.warning(f"Failed to fetch brand article {current_url}: HTTP {response.status_code}")
                                continue

                            # Parse HTML
                            soup = BeautifulSoup(response.text, 'html.parser')

                            # Determine article type based on content and URL patterns
                            article_type = 'other'
                            page_content_lower = soup.get_text().lower()
                            page_title = soup.title.string.lower() if soup.title else ""

                            # Classify article type
                            if any(kw in current_url.lower() or kw in page_title for kw in ['product', 'specs', 'specification']):
                                article_type = 'product_info'
                            elif any(kw in current_url.lower() or kw in page_title for kw in ['guide', 'manual', 'instruction']):
                                article_type = 'user_guide'
                            elif any(kw in current_url.lower() or kw in page_title for kw in ['faq', 'question']):
                                article_type = 'faq'
                            elif any(kw in current_url.lower() or kw in page_title for kw in ['troubleshoot', 'problem', 'issue']):
                                article_type = 'troubleshooting'

                            article_types[article_type] += 1

                            # Extract useful content
                            content = self._extract_useful_content(soup)

                            if content:
                                # Extract key information for brand article
                                article_metadata = {
                                    'url': current_url,
                                    'domain': domain,
                                    'label': url_label,
                                    'title': soup.title.string if soup.title else current_url,
                                    'crawled_at': datetime.now().isoformat(),
                                    'depth': depth,
                                    'article_type': article_type,
                                    'is_brand_page': True
                                }

                                # Create document for indexing
                                document = {
                                    'content': content,
                                    'metadata': article_metadata
                                }

                                # Index the document
                                if self.index_document(document):
                                    articles_indexed += 1
                                    article_inventory.append(article_metadata)
                                    logger.info(f"Indexed brand article: {current_url} (type: {article_type})")

                            # Find more article links - specifically look for article containers
                            article_links = []

                            # Look for article containers first
                            article_containers = soup.select('.articles, .article-list, .faq-list, .knowledge-base, ul.list-articles, div.help-center')

                            if article_containers:
                                for container in article_containers:
                                    for link in container.find_all('a', href=True):
                                        href = link['href']
                                        if not href or href.startswith('#') or href.startswith('javascript:'):
                                            continue

                                        # Convert to absolute URL if relative
                                        if not href.startswith(('http://', 'https://')):
                                            href = urljoin(current_url, href)

                                        # Only add URLs from same domain
                                        if href.startswith(domain) and href not in visited_urls:
                                            article_links.append((href, depth + 1))

                            # If no dedicated article containers found, look for links that might be articles
                            if not article_links:
                                for link in soup.find_all('a', href=True):
                                    href = link['href']
                                    if not href or href.startswith('#') or href.startswith('javascript:'):
                                        continue

                                    # Convert to absolute URL if relative
                                    if not href.startswith(('http://', 'https://')):
                                        href = urljoin(current_url, href)

                                    # Only add URLs that look like articles
                                    if (href.startswith(domain) and
                                        href not in visited_urls and
                                        any(indicator in href for indicator in article_indicators)):
                                        article_links.append((href, depth + 1))

                            # Add discovered articles to queue
                            for article_url, article_depth in article_links:
                                articles_to_visit.append((article_url, article_depth))

                            # Log what we found
                            logger.info(f"Discovered {len(article_links)} more brand articles from {current_url}")

                        except Exception as e:
                            logger.warning(f"Error processing brand article {current_url}: {str(e)}")

                    # Cache the results
                    cache_data = {
                        'timestamp': time.time(),
                        'domain': domain,
                        'articles_indexed': articles_indexed,
                        'stats': {
                            'articles_by_type': article_types,
                            'total_visited': len(visited_urls)
                        }
                    }

                    with open(url_cache_file, 'w') as f:
                        json.dump(cache_data, f)

                    # Update results
                    results["successful_crawls"] += 1
                    results["total_articles_indexed"] += articles_indexed
                    results["brand_stats"][url] = {
                        'articles_indexed': articles_indexed,
                        'article_types': article_types,
                        'total_visited': len(visited_urls)
                    }

                    logger.info(f"Completed crawling brand help pages from {url}: " +
                               f"visited {len(visited_urls)} pages, indexed {articles_indexed} articles")

                except Exception as e:
                    logger.error(f"Failed to crawl brand help page {url}: {str(e)}")
                    results["failed_crawls"] += 1
                    results["brand_stats"][url] = {"status": "error", "message": str(e)}

            # Print final statistics
            logger.info(f"Brand help crawling complete. Total articles indexed: {results['total_articles_indexed']}, " +
                       f"successful crawls: {results['successful_crawls']}")

            return results

        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else str(e)
            error_msg = f"Missing required libraries. Run: pip install requests beautifulsoup4"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            logger.error(f"Error during brand help page crawling: {str(e)}")
            return {"status": "error", "message": str(e)}

