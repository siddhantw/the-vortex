import re
import openai
import os
import sys

# Add the parent directory to sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from azure_openai_client import AzureOpenAIClient
import nltk
import logging
import json
import uuid
import ssl
import time
from typing import Dict, List, Set, Tuple, Union, Optional, Any
from collections import defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("requirement_analyser.log"), logging.StreamHandler()]
)
logger = logging.getLogger("RequirementAnalyser")

# Fix SSL certificate verification issue for NLTK downloads on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    logger.info("SSL certificate verification disabled for NLTK downloads")

# Custom tokenisation functions to avoid punkt_tab dependency
def custom_sentence_tokenize(text: str) -> List[str]:
    """
    Custom sentence tokeniser that doesn't rely on NLTK's punkt_tab

    This tokeniser uses regex patterns to split text into sentences. It handles:
    1. Standard sentence boundaries (period, exclamation, question mark followed by space and capital letter)
    2. Paragraph breaks (double newlines)
    3. Bullet points and numbered list items

    Args:
        text: Input text to tokenise into sentences

    Returns:
        List of sentences
    """
    # Basic pattern for sentence boundaries: period/exclamation/question followed by space and capital letter
    basic_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Further split on double newlines (paragraph breaks)
    result = []
    for sentence in basic_sentences:
        parts = re.split(r'\n\s*\n', sentence)
        for part in parts:
            clean = part.strip()
            if clean:  # Only add non-empty sentences
                result.append(clean)

    # Handle single-line bullet points and numbered items
    final_result = []
    bullet_pattern = r'(?:^|\n)(?:\d+\.\s+|\*\s+|\-\s+)'

    for item in result:
        # Split on bullet points or numbered items
        if re.search(bullet_pattern, item):
            bullets = re.split(bullet_pattern, item)
            # Clean and add non-empty bullet points
            for bullet in bullets:
                clean_bullet = bullet.strip()
                if clean_bullet:
                    final_result.append(clean_bullet)
        else:
            final_result.append(item)

    return final_result

def custom_word_tokenize(text: str) -> List[str]:
    """
    Custom word tokeniser that doesn't rely on NLTK's punkt_tab

    This tokeniser uses regex to separate punctuation and words. It:
    1. Adds spaces around punctuation marks
    2. Normalises multiple spaces to single spaces
    3. Splits the text on spaces to get individual words

    Args:
        text: Input text to tokenise into words

    Returns:
        List of words and punctuation marks
    """
    # Simple but effective word tokenisation
    # First normalise spaces and punctuation with spaces
    text = re.sub(r'([.,;:!?()])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)

    # Then split on spaces
    words = text.split()

    return words

# Replace NLTK's tokenisers with our custom versions
sent_tokenize = custom_sentence_tokenize
word_tokenize = custom_word_tokenize

# Download necessary NLTK resources on first run (except punkt_tab, which is problematic)
NLTK_RESOURCES = [
    ('punkt', 'punkt'),
    ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
    ('corpora/words', 'words'),
    ('corpora/wordnet', 'wordnet'),
    ('sentiment/vader_lexicon', 'vader_lexicon')
]

# Ensure all NLTK resources are downloaded with robust error handling
for _, resource_name in NLTK_RESOURCES:
    try:
        nltk.download(resource_name, quiet=True)
        logger.info(f"NLTK resource checked: {resource_name}")
    except Exception as e:
        logger.warning(f"Failed to download NLTK resource {resource_name}: {e}")

# Import remaining modules after resource download
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.corpus import wordnet
except ImportError as e:
    logger.warning(f"Failed to import NLTK modules: {e}")
    SentimentIntensityAnalyzer = None

try:
    # Try loading spaCy model (install if not present)
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model: en_core_web_sm")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    logger.warning("spaCy not installed. Some features will be disabled.")
    spacy = None
    subprocess = None
    nlp = None

class Requirement:
    """
    Class representing a software requirement with analysis capabilities.

    Attributes:
        id (str): Unique identifier for the requirement
        text (str): The text content of the requirement
        category (str): The category or type of the requirement
        priority (str): The priority level of the requirement
        complexity (float): Calculated complexity score
        ambiguity (float): Calculated ambiguity score
        completeness (float): Calculated completeness score
        testability (float): Calculated testability score
        analysis_results (dict): Dictionary containing all analysis results
    """

    def __init__(self, req_id: str, text: str, category: str = "Functional", priority: str = "Medium"):
        """
        Initialise a requirement with basic attributes.

        Args:
            req_id (str): Unique identifier for the requirement
            text (str): The text content of the requirement
            category (str, optional): The category or type of the requirement. Defaults to "Functional".
            priority (str, optional): The priority level of the requirement. Defaults to "Medium".
        """
        self.id = req_id
        self.text = text
        self.category = category
        self.priority = priority
        self.complexity = 0.0
        self.ambiguity = 0.0
        self.completeness = 0.0
        self.testability = 0.0
        self.analysis_results = {
            "complexity": {},
            "ambiguity": {},
            "completeness": {},
            "testability": {},
            "entities": [],
            "keywords": [],
            "sentiment": {},
            "test_suggestions": []
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the requirement to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all requirement attributes and analysis results
        """
        return {
            "id": self.id,
            "text": self.text,
            "category": self.category,
            "priority": self.priority,
            "complexity": self.complexity,
            "ambiguity": self.ambiguity,
            "completeness": self.completeness,
            "testability": self.testability,
            "analysis_results": self.analysis_results
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update the requirement from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary containing requirement attributes and analysis results
        """
        self.id = data.get("id", self.id)
        self.text = data.get("text", self.text)
        self.category = data.get("category", self.category)
        self.priority = data.get("priority", self.priority)
        self.complexity = data.get("complexity", self.complexity)
        self.ambiguity = data.get("ambiguity", self.ambiguity)
        self.completeness = data.get("completeness", self.completeness)
        self.testability = data.get("testability", self.testability)
        self.analysis_results = data.get("analysis_results", self.analysis_results)

    def __str__(self) -> str:
        """
        String representation of the requirement.

        Returns:
            str: A formatted string representation of the requirement
        """
        return f"Requirement({self.id}): {self.text[:50]}... [Complexity: {self.complexity:.2f}, Ambiguity: {self.ambiguity:.2f}]"

class RequirementAnalyzer:
    """
    Enhanced class for analysing software requirements using NLP techniques.

    This analyser evaluates requirements for complexity, ambiguity,
    completeness, and testability using various NLP techniques with
    improved filtering and smart detection.
    """

    # Cache for syllable counts to improve performance
    _syllable_cache = {}

    # Enhanced filters for irrelevant content
    IRRELEVANT_PATTERNS = [
        # Personal names patterns
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last name patterns
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',  # Titles with names
        r'\bwritten by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Author attributions
        r'\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # General by attributions

        # Document metadata
        r'\bversion\s+\d+\.\d+\b',  # Version numbers
        r'\bdated?\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
        r'\bpage\s+\d+\s+of\s+\d+\b',  # Page numbers
        r'\bdocument\s+id\s*:?\s*\w+\b',  # Document IDs

        # Email and contact info
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers

        # Legal and copyright text
        r'\bcopyright\s+©?\s*\d{4}\b',  # Copyright notices
        r'\ball rights reserved\b',  # Rights text
        r'\bconfidential\b',  # Confidentiality notices

        # Template placeholders
        r'\b\[.*?\]\b',  # Bracketed placeholders
        r'\b<.*?>\b',  # Angle bracket placeholders
        r'\b\{.*?\}\b',  # Curly bracket placeholders
    ]

    # Enhanced requirement quality indicators
    HIGH_QUALITY_INDICATORS = {
        'specificity': [
            r'\b(?:exactly|precisely|specifically|explicitly)\s+\d+\b',
            r'\b(?:within|less than|more than|at least|maximum|minimum)\s+\d+\s*(?:seconds?|minutes?|hours?|days?|ms|%)\b',
            r'\b(?:shall|must|will)\s+(?:display|show|return|provide|execute|validate|verify)\b',
            r'\b(?:when|if|given|provided that|in case of)\s+.+\s+(?:then|should|shall|must|will)\b'
        ],
        'clarity': [
            r'\b(?:the system|application|platform|user interface|database|API)\s+(?:shall|must|will)\b',
            r'\b(?:shall|must|will)\s+(?:be able to|allow|enable|prevent|ensure|guarantee)\b',
            r'\b(?:input|output|format|field|parameter|value|result)\s+(?:shall|must|will|should)\b'
        ],
        'measurability': [
            r'\b\d+(?:\.\d+)?\s*(?:%|percent|percentage|MB|GB|KB|bytes?|users?|requests?|transactions?)\b',
            r'\b(?:response time|load time|processing time|uptime|availability)\s+(?:of|less than|within|maximum)\s+\d+\b',
            r'\b(?:success rate|error rate|accuracy|precision)\s+(?:of|at least|minimum)\s+\d+(?:\.\d+)?%\b'
        ],
        'actionability': [
            r'\b(?:create|update|delete|modify|configure|validate|verify|calculate|process|generate|display)\b',
            r'\b(?:login|logout|authenticate|authorize|register|submit|search|filter|sort|export)\b',
            r'\b(?:send|receive|store|retrieve|save|load|backup|restore|migrate|transfer)\b'
        ]
    }

    # Enhanced ambiguity patterns with context awareness
    AMBIGUITY_PATTERNS = {
        'weak_modal_verbs': [
            r'\b(?:may|might|could|would)\s+(?:be|have|do|perform|allow|enable)\b',
            r'\b(?:possibly|probably|likely|perhaps|maybe)\b',
            r'\b(?:should|ought to)\s+(?:consider|try|attempt)\b'
        ],
        'vague_quantifiers': [
            r'\b(?:several|many|few|some|numerous|various|multiple)\b(?!\s+\d)',
            r'\b(?:appropriate|suitable|adequate|sufficient|reasonable|optimal)\b',
            r'\b(?:fast|slow|quick|rapid|high|low|large|small|big|little)\b(?!\s+(?:\d|enough|than))'
        ],
        'subjective_terms': [
            r'\b(?:user-friendly|intuitive|easy|simple|complex|difficult|good|bad|better|worse|best|worst)\b',
            r'\b(?:efficient|effective|improved|enhanced|optimized|streamlined)\b',
            r'\b(?:flexible|robust|scalable|maintainable|reliable|secure)\b(?!\s+(?:by|through|via))'
        ],
        'unclear_scope': [
            r'\b(?:as needed|when necessary|if required|as appropriate|where applicable)\b',
            r'\b(?:to be determined|to be defined|to be specified|subject to|up to)\b',
            r'\b(?:and so on|etc|among others|such as|for example|or similar)\b'
        ]
    }

    # Completeness assessment criteria
    COMPLETENESS_CRITERIA = {
        'actors': [
            r'\b(?:user|customer|client|admin|administrator|operator|manager|guest|member)\b',
            r'\b(?:system|application|platform|service|component|module|interface)\b',
            r'\b(?:stakeholder|business user|end user|external system|third party)\b'
        ],
        'actions': [
            r'\b(?:shall|must|will|should|can|could|may|might)\s+(?:be|have|do|perform|allow|enable|prevent|ensure)\b',
            r'\b(?:create|read|update|delete|modify|configure|setup|install|deploy|remove)\b',
            r'\b(?:validate|verify|check|confirm|authenticate|authorize|approve|reject)\b'
        ],
        'objects': [
            r'\b(?:data|information|record|file|document|report|message|notification)\b',
            r'\b(?:account|profile|setting|configuration|preference|option|parameter)\b',
            r'\b(?:transaction|payment|order|request|response|result|output|input)\b'
        ],
        'constraints': [
            r'\b(?:within|less than|more than|at least|maximum|minimum|between|exactly)\s+\d+\b',
            r'\b(?:if|when|unless|until|after|before|during|while|in case of|provided that)\b',
            r'\b(?:only|except|excluding|including|limited to|restricted to|subject to)\b'
        ],
        'acceptance_criteria': [
            r'\b(?:verify that|ensure that|confirm that|validate that|check that)\b',
            r'\b(?:given|when|then|and|but)\s+.+(?:should|shall|must|will)\b',
            r'\b(?:success criteria|acceptance criteria|definition of done|exit criteria)\b'
        ]
    }

    # Business domain specific filters for web hosting/domain industry
    BUSINESS_RELEVANT_KEYWORDS = [
        'domain', 'hosting', 'website', 'ssl', 'certificate', 'dns', 'server',
        'email', 'backup', 'security', 'performance', 'uptime', 'bandwidth',
        'storage', 'database', 'wordpress', 'ecommerce', 'cpanel', 'ftp',
        'migration', 'transfer', 'registration', 'renewal', 'billing', 'payment',
        'support', 'ticket', 'dashboard', 'control panel', 'user', 'customer',
        'account', 'login', 'authentication', 'authorization', 'api', 'integration'
    ]

    # Personal/irrelevant entity types to filter out
    IRRELEVANT_ENTITY_TYPES = [
        'PERSON', 'GPE', 'ORG'  # Filter out person names, locations, organizations unless business relevant
    ]

    def __init__(self, model_name: str = "dev-chat-ai-gpt4.1-mini", azure_api_key: Optional[str] = None):
        """
        Initialise the enhanced RequirementAnalyser with model settings.

        Args:
            model_name (str, optional): Name of the AI model to use. Defaults to "gpt‑4.1‑mini".
            azure_api_key (str, optional): Azure OpenAI API key. If None, will use default or environment variable.
        """
        self.requirements: Dict[str, Requirement] = {}
        self.model_name = model_name

        # Initialize Azure OpenAI client with proper configuration
        self.has_ai_access = False
        try:
            # Use the custom Azure OpenAI client
            from azure_openai_client import AzureOpenAIClient

            self.azure_client = AzureOpenAIClient(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", "5e98b3558f5d4dcebe68f8ca8a3352b7"),
                api_version="2024-10-21",
                deployment_name=model_name
            )
            self.has_ai_access = True
            logger.info("Azure OpenAI client initialized successfully with enhanced configuration")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure OpenAI client: {e}. Advanced analysis features will be disabled.")
            self.azure_client = None

        # Initialise sentiment analyser if available
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        logger.info(f"Enhanced RequirementAnalyser initialised with Azure OpenAI model: {model_name}")

    def _is_business_relevant(self, text: str) -> bool:
        """
        Enhanced business relevance check with better filtering.

        Args:
            text (str): Text to check for business relevance

        Returns:
            bool: True if text appears to be business relevant
        """
        text_lower = text.lower()

        # Enhanced business keywords with more comprehensive coverage
        enhanced_business_keywords = [
            # Core business terms
            'requirement', 'feature', 'functionality', 'capability', 'service',
            'user', 'customer', 'client', 'system', 'application', 'platform',
            'interface', 'dashboard', 'portal', 'website', 'mobile', 'api',

            # Domain-specific terms
            'domain', 'hosting', 'website', 'ssl', 'certificate', 'dns', 'server',
            'email', 'backup', 'security', 'performance', 'uptime', 'bandwidth',
            'storage', 'database', 'wordpress', 'ecommerce', 'cpanel', 'ftp',
            'migration', 'transfer', 'registration', 'renewal', 'billing', 'payment',
            'support', 'ticket', 'dashboard', 'control panel', 'account', 'login',
            'authentication', 'authorization', 'integration',

            # Action/behavior keywords
            'shall', 'must', 'will', 'should', 'can', 'could', 'may', 'might',
            'verify', 'validate', 'check', 'ensure', 'allow', 'enable', 'disable',
            'create', 'update', 'delete', 'modify', 'configure', 'setup', 'install',
            'deploy', 'manage', 'monitor', 'track', 'report', 'display', 'show',

            # Quality attributes
            'secure', 'reliable', 'scalable', 'maintainable', 'usable', 'accessible',
            'compatible', 'responsive', 'fast', 'efficient', 'available',

            # Test-related terms
            'test', 'testing', 'validate', 'verification', 'acceptance', 'criteria'
        ]

        # Check for business keywords with improved scoring
        relevant_keywords_found = sum(1 for keyword in enhanced_business_keywords
                                    if keyword in text_lower)

        # More flexible threshold - need at least 1 strong business keyword
        has_business_context = relevant_keywords_found >= 1

        # Check for requirement patterns
        requirement_patterns = [
            r'\b(?:the\s+)?(?:system|application|platform|website|service)\s+(?:shall|must|will|should|can)\b',
            r'\b(?:user|customer|client|admin)\s+(?:shall|must|will|should|can)\b',
            r'\bwhen\s+.*\s+then\b',
            r'\bif\s+.*\s+then\b',
            r'\bgiven\s+.*\s+when\s+.*\s+then\b',
            r'\brequirement\s*:?\s*.+',
            r'\bfeature\s*:?\s*.+',
            r'\bfunctionality\s*:?\s*.+',
            r'\bas\s+a\s+.+\s+i\s+want\s+.+\s+so\s+that\b',  # User story pattern
        ]

        has_requirement_pattern = any(re.search(pattern, text_lower) for pattern in requirement_patterns)

        # Check for measurable criteria
        measurable_patterns = [
            r'\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?)\b',
            r'\b(?:less\s+than|more\s+than|at\s+least|maximum|minimum)\s+\d+\b',
            r'\b\d+\s*(?:%|percent|percentage)\b',
            r'\b\d+\s*(?:mb|gb|kb|bytes?)\b',
            r'\b\d+\s*(?:users?|customers?|requests?|transactions?)\b'
        ]

        has_measurable_criteria = any(re.search(pattern, text_lower) for pattern in measurable_patterns)

        # More inclusive relevance check
        is_relevant = (
            has_business_context or
            has_requirement_pattern or
            has_measurable_criteria or
            len(text.split()) >= 10  # Give longer texts benefit of the doubt
        )

        return is_relevant

    def _filter_irrelevant_content(self, text: str) -> str:
        """
        Filter out irrelevant content like personal names, metadata, etc.

        Args:
            text (str): Original text to filter

        Returns:
            str: Filtered text with irrelevant content removed
        """
        filtered_text = text

        # Apply regex patterns to remove irrelevant content
        for pattern in self.IRRELEVANT_PATTERNS:
            filtered_text = re.sub(pattern, '', filtered_text, flags=re.IGNORECASE)

        # Remove extra whitespace
        filtered_text = re.sub(r'\s+', ' ', filtered_text).strip()

        return filtered_text

    def _classify_requirement_type(self, text: str) -> str:
        """
        Classify the type of requirement based on content analysis.

        Args:
            text (str): Requirement text to classify

        Returns:
            str: Classification of the requirement
        """
        text_lower = text.lower()

        # Functional requirement indicators
        functional_patterns = [
            r'\b(?:user|system|application)\s+(?:shall|must|will|should|can)\b',
            r'\bwhen\s+.*\bthen\b',
            r'\bverify\s+that\b',
            r'\bensure\s+that\b',
            r'\ballow\s+.*\bto\b'
        ]

        # Non-functional requirement indicators
        non_functional_patterns = [
            r'\bperformance\b.*\b(?:seconds|minutes|response time)\b',
            r'\bsecurity\b.*\b(?:encryption|authentication|authorization)\b',
            r'\busability\b.*\b(?:user-friendly|intuitive|accessible)\b',
            r'\breliability\b.*\b(?:uptime|availability|fault tolerance)\b',
            r'\bscalability\b.*\b(?:concurrent|load|capacity)\b'
        ]

        # Business rule indicators
        business_rule_patterns = [
            r'\bbusiness\s+rule\b',
            r'\bpolicy\b.*\b(?:states|requires|mandates)\b',
            r'\bregulation\b.*\b(?:compliance|adherence)\b'
        ]

        # Check patterns
        if any(re.search(pattern, text_lower) for pattern in functional_patterns):
            return "Functional"
        elif any(re.search(pattern, text_lower) for pattern in non_functional_patterns):
            return "Non-Functional"
        elif any(re.search(pattern, text_lower) for pattern in business_rule_patterns):
            return "Business Rule"
        else:
            return "Functional"  # Default classification

    def add_requirements_from_text(self, text: str) -> List[Requirement]:
        """
        Enhanced method to parse and add requirements from a block of text with improved analysis.

        This method takes a block of text, intelligently filters content,
        splits it into individual requirements, and adds each as a separate requirement.

        Args:
            text (str): Block of text containing one or more requirements

        Returns:
            List[Requirement]: List of the added requirement objects
        """
        # Pre-filter the entire text to remove obvious irrelevant content
        filtered_text = self._filter_irrelevant_content(text)

        # Enhanced text splitting with multiple strategies
        potential_requirements = []

        # Strategy 1: Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', filtered_text.strip())
        potential_requirements.extend(paragraphs)

        # Strategy 2: Split by sentence boundaries for longer paragraphs
        for paragraph in paragraphs:
            if len(paragraph.split()) > 50:  # Long paragraphs
                sentences = sent_tokenize(paragraph)
                # Group sentences into logical requirements
                current_req = ""
                for sentence in sentences:
                    if len(current_req.split()) < 20:
                        current_req += " " + sentence if current_req else sentence
                    else:
                        if current_req.strip():
                            potential_requirements.append(current_req.strip())
                        current_req = sentence
                if current_req.strip():
                    potential_requirements.append(current_req.strip())

        # Strategy 3: Split by numbered lists or bullet points
        list_pattern = r'(?:^|\n)\s*(?:\d+\.|\*|\-|\•)\s*(.+?)(?=(?:\n\s*(?:\d+\.|\*|\-|\•))|$)'
        list_matches = re.findall(list_pattern, filtered_text, re.MULTILINE | re.DOTALL)
        potential_requirements.extend([match.strip() for match in list_matches if match.strip()])

        # Remove duplicates while preserving order
        seen = set()
        unique_requirements = []
        for req in potential_requirements:
            req_clean = req.strip()
            if req_clean and req_clean not in seen:
                seen.add(req_clean)
                unique_requirements.append(req_clean)

        requirements_added = []

        # Process each potential requirement
        for i, req_text in enumerate(unique_requirements):
            # Enhanced filtering at sentence level
            if len(req_text.split()) < 3:  # Too short
                continue

            # Check if the requirement is business relevant with more lenient criteria
            if not self._is_business_relevant(req_text):
                logger.debug(f"Skipping potentially irrelevant requirement: {req_text[:50]}...")
                continue

            # Generate a unique ID for the requirement
            req_id = f"REQ-{str(uuid.uuid4())[:8]}"

            # Enhanced requirement type classification
            req_type = self._classify_requirement_type(req_text)

            # Determine priority based on content analysis
            priority = self._determine_priority(req_text)

            # Add the requirement with enhanced classification
            self.add_requirement(req_id, req_text, req_type, priority)

            # Keep track of added requirements
            requirements_added.append(self.requirements[req_id])

        logger.info(f"Added {len(requirements_added)} relevant requirements from text (processed {len(unique_requirements)} potential requirements)")
        return requirements_added

    def _determine_priority(self, text: str) -> str:
        """
        Determine requirement priority based on content analysis.

        Args:
            text (str): Requirement text to analyze

        Returns:
            str: Priority level (High, Medium, Low)
        """
        text_lower = text.lower()

        # High priority indicators
        high_priority_terms = [
            'critical', 'essential', 'mandatory', 'required', 'must', 'crucial',
            'security', 'authentication', 'authorization', 'login', 'password',
            'payment', 'billing', 'transaction', 'data loss', 'corruption',
            'failure', 'error handling', 'backup', 'recovery'
        ]

        # Medium priority indicators
        medium_priority_terms = [
            'should', 'important', 'significant', 'performance', 'usability',
            'interface', 'user experience', 'efficiency', 'optimization'
        ]

        # Low priority indicators
        low_priority_terms = [
            'nice to have', 'optional', 'enhancement', 'improvement',
            'could', 'might', 'possibly', 'cosmetic', 'aesthetic'
        ]

        high_score = sum(1 for term in high_priority_terms if term in text_lower)
        medium_score = sum(1 for term in medium_priority_terms if term in text_lower)
        low_score = sum(1 for term in low_priority_terms if term in text_lower)

        if high_score > 0 or 'must' in text_lower or 'shall' in text_lower:
            return "High"
        elif low_score > 0:
            return "Low"
        else:
            return "Medium"

    def _generate_test_suggestions(self, req: Requirement) -> None:
        """
        Enhanced test suggestion generation using Azure OpenAI with better prompts and error handling.

        Args:
            req (Requirement): The requirement to analyse
        """
        if not self.has_ai_access or not self.azure_client:
            req.analysis_results["test_suggestions"] = []
            logger.warning(f"No AI access available for test suggestion generation for requirement {req.id}")
            return

        try:
            # Enhanced prompt with better structure and domain context
            prompt = f"""
            As an expert QA engineer specializing in web hosting, domain registration, and digital services, analyze the following software requirement and generate comprehensive, actionable test cases.

            **Requirement Details:**
            - ID: {req.id}
            - Text: {req.text}
            - Category: {req.category}
            - Priority: {req.priority}
            - Complexity Score: {req.complexity:.2f}
            - Testability Score: {req.testability:.2f}

            **Instructions:**
            Generate 3-5 detailed test cases that cover different testing scenarios. For each test case, provide:

            1. **test_case_id**: Format as TC_{req.id}_[TYPE]_[NUMBER] (e.g., TC_{req.id}_POS_001)
            2. **name**: Clear, descriptive test case name
            3. **description**: Brief description of what this test validates
            4. **type**: Test type (Positive, Negative, Boundary, Integration, Performance, Security, Usability)
            5. **priority**: High, Medium, or Low
            6. **preconditions**: Required system state before testing
            7. **test_steps**: Detailed, numbered steps to execute
            8. **test_data**: Specific data needed for testing
            9. **expected_result**: Clear, measurable expected outcome
            10. **postconditions**: Expected system state after testing

            **Focus Areas:**
            - Web hosting services (shared, VPS, dedicated hosting)
            - Domain registration and management
            - SSL certificates and security
            - Email services and configuration
            - Control panels (cPanel, Plesk, custom dashboards)
            - Billing and payment processing
            - User account management
            - API integrations
            - Performance and scalability
            - Security and compliance

            **Response Format:**
            Return a valid JSON array of test case objects. Ensure all JSON is properly formatted and escaped.

            Example structure:
            [
              {{
                "test_case_id": "TC_{req.id}_POS_001",
                "name": "Verify successful domain registration",
                "description": "Test that users can register an available domain successfully",
                "type": "Positive",
                "priority": "High",
                "preconditions": "User is logged in and has sufficient balance",
                "test_steps": [
                  "1. Navigate to domain registration page",
                  "2. Enter available domain name",
                  "3. Select domain extension",
                  "4. Click 'Search' button",
                  "5. Verify domain is available",
                  "6. Click 'Add to Cart'",
                  "7. Proceed to checkout",
                  "8. Complete payment process"
                ],
                "test_data": "Domain: testdomain123.com, User: valid_user@example.com",
                "expected_result": "Domain is successfully registered and appears in user's domain list",
                "postconditions": "Domain is registered and DNS is configured"
              }}
            ]
            """

            # Generate response using Azure OpenAI client
            response = self.azure_client.chat_completion_create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert QA engineer with deep knowledge of web hosting, domain registration, and digital services. Generate precise, actionable test cases in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,  # Lower temperature for more consistent results
                max_tokens=3000
            )

            if response and response.get("choices"):
                test_suggestions_text = response["choices"][0]["message"]["content"]

                # Enhanced JSON parsing with multiple fallback strategies
                test_suggestions = self._parse_test_suggestions_response(test_suggestions_text, req.id)

                req.analysis_results["test_suggestions"] = test_suggestions
                logger.info(f"Generated {len(test_suggestions)} enhanced test suggestions for {req.id}")
            else:
                logger.warning(f"No response received from Azure OpenAI for requirement {req.id}")
                req.analysis_results["test_suggestions"] = []

        except Exception as e:
            logger.error(f"Error generating enhanced test suggestions for {req.id}: {e}")
            req.analysis_results["test_suggestions"] = []

    def _parse_test_suggestions_response(self, response_text: str, req_id: str) -> List[Dict[str, Any]]:
        """
        Enhanced parsing of test suggestions response with multiple fallback strategies.

        Args:
            response_text (str): Raw response from AI
            req_id (str): Requirement ID for fallback generation

        Returns:
            List[Dict[str, Any]]: Parsed test suggestions
        """
        test_suggestions = []

        try:
            # Strategy 1: Direct JSON parsing
            clean_text = response_text.strip()

            # Remove markdown code blocks if present
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                parts = clean_text.split("```")
                if len(parts) >= 3:
                    clean_text = parts[1].strip()

            # Try to parse as JSON array
            if clean_text.startswith("[") and clean_text.endswith("]"):
                test_suggestions = json.loads(clean_text)
            elif clean_text.startswith("{") and clean_text.endswith("}"):
                # Single object, wrap in array
                test_obj = json.loads(clean_text)
                test_suggestions = [test_obj]
            else:
                # Strategy 2: Extract JSON objects using regex
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, clean_text, re.DOTALL)
                for match in json_matches:
                    try:
                        parsed = json.loads(match)
                        if isinstance(parsed, dict):
                            test_suggestions.append(parsed)
                    except json.JSONDecodeError:
                        continue

        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed for requirement {req_id}: {e}")

            # Strategy 3: Structured text parsing
            test_suggestions = self._parse_structured_text_response(response_text, req_id)

        # Strategy 4: Final fallback - create basic test cases
        if not test_suggestions:
            logger.warning(f"All parsing strategies failed for {req_id}, creating fallback test cases")
            test_suggestions = self._create_fallback_test_cases(req_id, response_text)

        # Validate and enhance test suggestions
        validated_suggestions = []
        for i, suggestion in enumerate(test_suggestions):
            if isinstance(suggestion, dict):
                enhanced_suggestion = self._validate_and_enhance_test_case(suggestion, req_id, i)
                validated_suggestions.append(enhanced_suggestion)

        return validated_suggestions[:5]  # Limit to 5 test cases

    def _parse_structured_text_response(self, response_text: str, req_id: str) -> List[Dict[str, Any]]:
        """
        Parse structured text response when JSON parsing fails.

        Args:
            response_text (str): Raw response text
            req_id (str): Requirement ID

        Returns:
            List[Dict[str, Any]]: Parsed test suggestions
        """
        test_suggestions = []

        # Split by test case indicators
        test_case_patterns = [
            r'(?:test case \d+|tc[_\s]*\d+|\d+\.|^\d+[\.\)]\s*)',
            r'(?:test[_\s]*case[_\s]*\d+)',
            r'(?:scenario \d+)'
        ]

        for pattern in test_case_patterns:
            matches = re.split(pattern, response_text, flags=re.IGNORECASE | re.MULTILINE)
            if len(matches) > 1:
                for i, match in enumerate(matches[1:], 1):  # Skip first empty match
                    test_case = self._extract_test_case_from_text(match, req_id, i)
                    if test_case:
                        test_suggestions.append(test_case)
                break

        return test_suggestions

    def _extract_test_case_from_text(self, text: str, req_id: str, index: int) -> Dict[str, Any]:
        """
        Extract test case information from structured text.

        Args:
            text (str): Text containing test case information
            req_id (str): Requirement ID
            index (int): Test case index

        Returns:
            Dict[str, Any]: Test case dictionary
        """
        test_case = {
            "test_case_id": f"TC_{req_id}_{index:03d}",
            "name": f"Test case {index} for {req_id}",
            "description": "Generated test case",
            "type": "Functional",
            "priority": "Medium",
            "preconditions": "System is accessible",
            "test_steps": [],
            "test_data": "Standard test data",
            "expected_result": "Test passes successfully",
            "postconditions": "System returns to stable state"
        }

        # Extract information using regex patterns
        patterns = {
            "name": r'(?:name|title):\s*(.+?)(?:\n|$)',
            "description": r'(?:description|desc):\s*(.+?)(?:\n|$)',
            "type": r'(?:type|category):\s*(.+?)(?:\n|$)',
            "priority": r'(?:priority|pri):\s*(.+?)(?:\n|$)',
            "preconditions": r'(?:preconditions?|prerequisites?):\s*(.+?)(?:\n|$)',
            "expected_result": r'(?:expected[_\s]*result|expected):\s*(.+?)(?:\n|$)',
            "test_data": r'(?:test[_\s]*data|data):\s*(.+?)(?:\n|$)'
        }

        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                test_case[field] = match.group(1).strip()

        # Extract test steps
        steps_pattern = r'(?:steps?|test[_\s]*steps?):\s*((?:\d+\..*?)*)(?:\n\n|\n(?=[A-Z])|$)'
        steps_match = re.search(steps_pattern, text, re.IGNORECASE | re.DOTALL)
        if steps_match:
            steps_text = steps_match.group(1)
            steps = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', steps_text, re.DOTALL)
            test_case["test_steps"] = [step.strip() for step in steps if step.strip()]

        return test_case

    def _create_fallback_test_cases(self, req_id: str, response_text: str) -> List[Dict[str, Any]]:
        """
        Create basic fallback test cases when all parsing fails.

        Args:
            req_id (str): Requirement ID
            response_text (str): Original response text

        Returns:
            List[Dict[str, Any]]: Basic test cases
        """
        return [
            {
                "test_case_id": f"TC_{req_id}_001",
                "name": f"Basic functionality test for {req_id}",
                "description": "Test basic functionality described in the requirement",
                "type": "Positive",
                "priority": "High",
                "preconditions": "System is accessible and user is authenticated",
                "test_steps": [
                    "1. Access the system",
                    "2. Navigate to the relevant feature",
                    "3. Execute the required functionality",
                    "4. Verify the result"
                ],
                "test_data": "Valid test data as per requirement",
                "expected_result": "Functionality works as described in the requirement",
                "postconditions": "System returns to stable state",
                "raw_response": response_text[:500]  # Include partial response for debugging
            }
        ]

    def _validate_and_enhance_test_case(self, test_case: Dict[str, Any], req_id: str, index: int) -> Dict[str, Any]:
        """
        Validate and enhance a test case with required fields.

        Args:
            test_case (Dict[str, Any]): Raw test case
            req_id (str): Requirement ID
            index (int): Test case index

        Returns:
            Dict[str, Any]: Enhanced test case
        """
        enhanced_test_case = {
            "test_case_id": test_case.get("test_case_id", f"TC_{req_id}_{index+1:03d}"),
            "name": test_case.get("name", test_case.get("test_case_name", f"Test case {index+1}")),
            "description": test_case.get("description", "Generated test case"),
            "type": test_case.get("type", test_case.get("test_type", "Functional")),
            "priority": test_case.get("priority", "Medium"),
            "preconditions": test_case.get("preconditions", test_case.get("prerequisites", "System is accessible")),
            "test_steps": test_case.get("test_steps", test_case.get("steps", [])),
            "test_data": test_case.get("test_data", test_case.get("data", "Standard test data")),
            "expected_result": test_case.get("expected_result", test_case.get("expected", "Test passes")),
            "postconditions": test_case.get("postconditions", "System state is preserved")
        }

        # Ensure test_steps is a list
        if isinstance(enhanced_test_case["test_steps"], str):
            # Split by numbers or newlines
            steps_text = enhanced_test_case["test_steps"]
            steps = re.split(r'(?:\d+\.|\n)', steps_text)
            enhanced_test_case["test_steps"] = [step.strip() for step in steps if step.strip()]

        # Validate required fields
        if not enhanced_test_case["test_steps"]:
            enhanced_test_case["test_steps"] = [
                "1. Execute the functionality described in the requirement",
                "2. Verify the expected behavior occurs",
                "3. Check for any error conditions"
            ]

        return enhanced_test_case

    def add_requirement(self, req_id: str, text: str, category: str = "Functional", priority: str = "Medium") -> None:
        """
        Add a new requirement to the analyser.

        Args:
            req_id (str): Unique identifier for the requirement
            text (str): The text content of the requirement
            category (str, optional): Category of the requirement. Defaults to "Functional".
            priority (str, optional): Priority of the requirement. Defaults to "Medium".
        """
        if req_id in self.requirements:
            logger.warning(f"Requirement with ID {req_id} already exists and will be overwritten.")

        self.requirements[req_id] = Requirement(req_id, text, category, priority)
        logger.info(f"Added requirement {req_id}")

    def get_requirement(self, req_id: str) -> Optional[Requirement]:
        """
        Get a requirement by its ID.

        Args:
            req_id (str): The ID of the requirement to retrieve

        Returns:
            Optional[Requirement]: The requirement object if found, None otherwise
        """
        return self.requirements.get(req_id)

    def get_all_requirements(self) -> Dict[str, Requirement]:
        """
        Get all requirements.

        Returns:
            Dict[str, Requirement]: Dictionary of all requirements with their IDs as keys
        """
        return self.requirements

    def analyze_requirement(self, req: Requirement) -> None:
        """
        Enhanced requirement analysis with comprehensive quality assessment.

        Args:
            req (Requirement): The requirement to analyse
        """
        logger.info(f"Analysing requirement {req.id}")

        # Run enhanced analysis for each metric
        self._analyze_complexity(req)
        self._analyze_ambiguity(req)
        self._analyze_completeness(req)
        self._analyze_testability(req)

        # NEW: Enhanced quality assessment
        quality_scores = self._assess_requirement_quality(req)
        req.analysis_results["quality_assessment"] = quality_scores

        # NEW: Structural analysis
        structure_analysis = self._analyze_requirement_structure(req)
        req.analysis_results["structure_analysis"] = structure_analysis

        # NEW: Improvement opportunities
        improvements = self._detect_improvement_opportunities(req)
        req.analysis_results["improvement_opportunities"] = improvements

        # Extract entities and keywords
        self._extract_entities(req)
        self._extract_keywords(req)

        # Analyse sentiment
        if self.sentiment_analyzer:
            self._analyze_sentiment(req)

        # Generate test suggestions using Azure OpenAI if available
        if self.has_ai_access:
            self._generate_test_suggestions(req)

        logger.info(f"Completed enhanced analysis for requirement {req.id}")

    def _assess_requirement_quality(self, req: Requirement) -> Dict[str, float]:
        """
        Comprehensive quality assessment using enhanced patterns and weighted scoring.

        Args:
            req (Requirement): The requirement to assess

        Returns:
            Dict[str, float]: Quality scores for different dimensions
        """
        text = req.text.lower()
        quality_scores = {}

        # Assess specificity (how precise and detailed the requirement is)
        specificity_matches = 0
        for pattern in self.HIGH_QUALITY_INDICATORS['specificity']:
            specificity_matches += len(re.findall(pattern, text, re.IGNORECASE))
        quality_scores['specificity'] = min(1.0, specificity_matches / 3.0)

        # Assess clarity (how clear and unambiguous the requirement is)
        clarity_matches = 0
        for pattern in self.HIGH_QUALITY_INDICATORS['clarity']:
            clarity_matches += len(re.findall(pattern, text, re.IGNORECASE))
        quality_scores['clarity'] = min(1.0, clarity_matches / 2.0)

        # Assess measurability (presence of quantifiable criteria)
        measurability_matches = 0
        for pattern in self.HIGH_QUALITY_INDICATORS['measurability']:
            measurability_matches += len(re.findall(pattern, text, re.IGNORECASE))
        quality_scores['measurability'] = min(1.0, measurability_matches / 2.0)

        # Assess actionability (presence of clear actions to be taken)
        actionability_matches = 0
        for pattern in self.HIGH_QUALITY_INDICATORS['actionability']:
            actionability_matches += len(re.findall(pattern, text, re.IGNORECASE))
        quality_scores['actionability'] = min(1.0, actionability_matches / 3.0)

        # Calculate overall quality score with weighted average
        weights = {'specificity': 0.3, 'clarity': 0.3, 'measurability': 0.2, 'actionability': 0.2}
        overall_quality = sum(quality_scores[dimension] * weights[dimension]
                            for dimension in weights.keys())
        quality_scores['overall'] = overall_quality

        return quality_scores

    def _analyze_requirement_structure(self, req: Requirement) -> Dict[str, Any]:
        """
        Analyze the structural quality of a requirement using linguistic patterns.

        Args:
            req (Requirement): The requirement to analyze

        Returns:
            Dict[str, Any]: Structural analysis results
        """
        text = req.text
        structure_analysis = {}

        # Check for proper requirement structure (Actor + Action + Object + Constraint)
        has_actor = any(re.search(pattern, text, re.IGNORECASE)
                       for pattern in self.COMPLETENESS_CRITERIA['actors'])
        has_action = any(re.search(pattern, text, re.IGNORECASE)
                        for pattern in self.COMPLETENESS_CRITERIA['actions'])
        has_object = any(re.search(pattern, text, re.IGNORECASE)
                        for pattern in self.COMPLETENESS_CRITERIA['objects'])
        has_constraint = any(re.search(pattern, text, re.IGNORECASE)
                           for pattern in self.COMPLETENESS_CRITERIA['constraints'])

        # Analyze sentence structure
        sentences = sent_tokenize(text)
        structure_analysis.update({
            'has_actor': has_actor,
            'has_action': has_action,
            'has_object': has_object,
            'has_constraint': has_constraint,
            'sentence_count': len(sentences),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'structure_completeness': sum([has_actor, has_action, has_object, has_constraint]) / 4.0
        })

        # Check for conditional logic patterns
        conditional_patterns = [
            r'\bif\s+.+\s+then\b',
            r'\bwhen\s+.+\s+(?:then|should|shall|must)\b',
            r'\bgiven\s+.+\s+when\s+.+\s+then\b'
        ]
        has_conditional_logic = any(re.search(pattern, text, re.IGNORECASE)
                                  for pattern in conditional_patterns)
        structure_analysis['has_conditional_logic'] = has_conditional_logic

        return structure_analysis

    def _detect_improvement_opportunities(self, req: Requirement) -> List[Dict[str, str]]:
        """
        Identify specific improvement opportunities for a requirement.

        Args:
            req (Requirement): The requirement to analyze

        Returns:
            List[Dict[str, str]]: List of improvement suggestions
        """
        text = req.text
        improvements = []

        # Check for ambiguous terms
        for category, patterns in self.AMBIGUITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    improvements.append({
                        'type': 'ambiguity',
                        'category': category,
                        'issue': f"Contains ambiguous {category.replace('_', ' ')}: {', '.join(set(matches))}",
                        'suggestion': f"Replace {category.replace('_', ' ')} with specific, measurable terms",
                        'severity': 'medium' if len(matches) <= 2 else 'high'
                    })

        # Check for missing elements
        structure = self._analyze_requirement_structure(req)
        if not structure['has_actor']:
            improvements.append({
                'type': 'completeness',
                'category': 'missing_actor',
                'issue': "No clear actor (who) identified",
                'suggestion': "Specify who will perform the action or be affected (user, system, admin, etc.)",
                'severity': 'high'
            })

        if not structure['has_action']:
            improvements.append({
                'type': 'completeness',
                'category': 'missing_action',
                'issue': "No clear action (what) specified",
                'suggestion': "Use clear action verbs with modal verbs (shall, must, will)",
                'severity': 'high'
            })

        if not structure['has_object']:
            improvements.append({
                'type': 'completeness',
                'category': 'missing_object',
                'issue': "No clear object (what is affected) identified",
                'suggestion': "Specify what data, interface, or system component is involved",
                'severity': 'medium'
            })

        # Check for measurability issues
        has_measurable_criteria = any(re.search(pattern, text, re.IGNORECASE)
                                    for pattern in self.HIGH_QUALITY_INDICATORS['measurability'])
        if not has_measurable_criteria and req.category == 'Non-Functional':
            improvements.append({
                'type': 'measurability',
                'category': 'missing_metrics',
                'issue': "Non-functional requirement lacks measurable criteria",
                'suggestion': "Add specific metrics, thresholds, or acceptance criteria",
                'severity': 'high'
            })

        # Check sentence length and complexity
        if structure['avg_sentence_length'] > 25:
            improvements.append({
                'type': 'readability',
                'category': 'sentence_length',
                'issue': f"Average sentence length is {structure['avg_sentence_length']:.1f} words (too long)",
                'suggestion': "Break down into shorter, clearer sentences (aim for 15-20 words per sentence)",
                'severity': 'medium'
            })

        return improvements

    def get_improvement_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive improvement report for all requirements.

        Returns:
            Dict[str, Any]: Detailed improvement recommendations and statistics
        """
        if not self.requirements:
            return {"error": "No requirements available for analysis"}

        requirements_list = list(self.requirements.values())
        total_requirements = len(requirements_list)

        # Aggregate improvement opportunities by type and severity
        improvement_summary = {
            'high_severity': [],
            'medium_severity': [],
            'low_severity': [],
            'by_category': {},
            'common_issues': {},
            'quality_distribution': {}
        }

        for req in requirements_list:
            improvements = req.analysis_results.get("improvement_opportunities", [])
            quality_scores = req.analysis_results.get("quality_assessment", {})

            # Categorize by severity
            for improvement in improvements:
                severity = improvement.get('severity', 'medium')
                if severity == 'high':
                    improvement_summary['high_severity'].append({
                        'req_id': req.id,
                        'issue': improvement['issue'],
                        'suggestion': improvement['suggestion'],
                        'category': improvement['category']
                    })
                elif severity == 'medium':
                    improvement_summary['medium_severity'].append({
                        'req_id': req.id,
                        'issue': improvement['issue'],
                        'suggestion': improvement['suggestion'],
                        'category': improvement['category']
                    })
                else:
                    improvement_summary['low_severity'].append({
                        'req_id': req.id,
                        'issue': improvement['issue'],
                        'suggestion': improvement['suggestion'],
                        'category': improvement['category']
                    })

                # Track common issues
                issue_type = improvement['category']
                if issue_type not in improvement_summary['common_issues']:
                    improvement_summary['common_issues'][issue_type] = 0
                improvement_summary['common_issues'][issue_type] += 1

            # Track quality score distribution
            overall_quality = quality_scores.get('overall', 0.0)
            quality_bucket = 'high' if overall_quality >= 0.7 else 'medium' if overall_quality >= 0.4 else 'low'
            if quality_bucket not in improvement_summary['quality_distribution']:
                improvement_summary['quality_distribution'][quality_bucket] = 0
            improvement_summary['quality_distribution'][quality_bucket] += 1

        # Calculate overall statistics
        total_high_issues = len(improvement_summary['high_severity'])
        total_medium_issues = len(improvement_summary['medium_severity'])
        total_low_issues = len(improvement_summary['low_severity'])

        # Generate prioritized recommendations
        prioritized_recommendations = []

        # High priority recommendations
        if total_high_issues > 0:
            prioritized_recommendations.append({
                'priority': 'Critical',
                'title': f'{total_high_issues} Critical Issues Requiring Immediate Attention',
                'description': 'These issues significantly impact requirement quality and should be addressed first.',
                'actions': [
                    'Review requirements with missing actors or actions',
                    'Add specific measurable criteria to non-functional requirements',
                    'Replace ambiguous terms with precise specifications'
                ]
            })

        # Medium priority recommendations
        if total_medium_issues > 0:
            prioritized_recommendations.append({
                'priority': 'Important',
                'title': f'{total_medium_issues} Important Issues for Quality Improvement',
                'description': 'These issues affect requirement clarity and completeness.',
                'actions': [
                    'Improve sentence structure and readability',
                    'Add missing constraint information',
                    'Clarify subjective terms with objective criteria'
                ]
            })

        # Best practices recommendations
        prioritized_recommendations.append({
            'priority': 'Best Practice',
            'title': 'General Quality Enhancement Recommendations',
            'description': 'Apply these practices to maintain high requirement quality.',
            'actions': [
                'Use consistent terminology throughout requirements',
                'Include acceptance criteria for all functional requirements',
                'Ensure all requirements follow Actor-Action-Object-Constraint pattern',
                'Add measurable performance criteria where applicable'
            ]
        })

        return {
            'summary': {
                'total_requirements_analyzed': total_requirements,
                'total_high_severity_issues': total_high_issues,
                'total_medium_severity_issues': total_medium_issues,
                'total_low_severity_issues': total_low_issues,
                'avg_issues_per_requirement': round((total_high_issues + total_medium_issues + total_low_issues) / total_requirements, 2)
            },
            'quality_distribution': improvement_summary['quality_distribution'],
            'common_issues': dict(sorted(improvement_summary['common_issues'].items(), key=lambda x: x[1], reverse=True)),
            'prioritized_recommendations': prioritized_recommendations,
            'detailed_issues': {
                'critical': improvement_summary['high_severity'][:10],  # Top 10 critical issues
                'important': improvement_summary['medium_severity'][:15],  # Top 15 important issues
            }
        }

    def generate_stakeholder_summary(self) -> Dict[str, Any]:
        """
        Generate a clear, concise summary for stakeholders with actionable insights.

        Returns:
            Dict[str, Any]: Executive summary with key metrics and recommendations
        """
        if not self.requirements:
            return {"error": "No requirements available for analysis"}

        requirements_list = list(self.requirements.values())
        total_requirements = len(requirements_list)

        # Calculate overall quality metrics
        total_quality_score = 0
        total_complexity = 0
        total_ambiguity = 0
        total_completeness = 0
        total_testability = 0

        high_quality_count = 0
        needs_improvement_count = 0
        critical_issues_count = 0

        for req in requirements_list:
            quality_scores = req.analysis_results.get("quality_assessment", {})
            improvements = req.analysis_results.get("improvement_opportunities", [])

            overall_quality = quality_scores.get('overall', 0.0)
            total_quality_score += overall_quality
            total_complexity += req.complexity
            total_ambiguity += req.ambiguity
            total_completeness += req.completeness
            total_testability += req.testability

            if overall_quality >= 0.7:
                high_quality_count += 1
            elif overall_quality < 0.4:
                needs_improvement_count += 1

            # Count critical issues
            critical_issues = [imp for imp in improvements if imp.get('severity') == 'high']
            if critical_issues:
                critical_issues_count += 1

        avg_quality = total_quality_score / total_requirements
        avg_complexity = total_complexity / total_requirements
        avg_ambiguity = total_ambiguity / total_requirements
        avg_completeness = total_completeness / total_requirements
        avg_testability = total_testability / total_requirements

        # Generate clear, actionable statements
        key_findings = []

        if avg_quality >= 0.7:
            key_findings.append("✅ Overall requirement quality is good with clear, well-structured specifications.")
        elif avg_quality >= 0.4:
            key_findings.append("⚠️  Requirement quality is moderate - improvements needed for clarity and completeness.")
        else:
            key_findings.append("🔴 Requirement quality requires significant improvement to ensure project success.")

        if avg_ambiguity > 0.6:
            key_findings.append("🔴 High ambiguity detected - requirements contain too many vague terms that could lead to misinterpretation.")
        elif avg_ambiguity > 0.3:
            key_findings.append("⚠️  Moderate ambiguity - some requirements need clearer, more specific language.")

        if avg_completeness < 0.4:
            key_findings.append("🔴 Many requirements are incomplete - missing essential elements like actors, actions, or constraints.")
        elif avg_completeness < 0.7:
            key_findings.append("⚠️  Some requirements lack complete information needed for implementation.")

        if avg_testability < 0.4:
            key_findings.append("🔴 Requirements are difficult to test - need measurable criteria and clear expected outcomes.")

        # Immediate actions needed
        immediate_actions = []
        if critical_issues_count > 0:
            immediate_actions.append(f"Review {critical_issues_count} requirements with critical quality issues")
        if needs_improvement_count > total_requirements * 0.3:
            immediate_actions.append("Conduct requirement quality review workshop with stakeholders")
        if avg_ambiguity > 0.5:
            immediate_actions.append("Replace vague terms with specific, measurable criteria")
        if avg_completeness < 0.5:
            immediate_actions.append("Add missing requirement elements (who, what, when, constraints)")

        return {
            'executive_summary': {
                'total_requirements': total_requirements,
                'overall_quality_score': round(avg_quality, 2),
                'high_quality_requirements': high_quality_count,
                'requirements_needing_improvement': needs_improvement_count,
                'requirements_with_critical_issues': critical_issues_count
            },
            'key_findings': key_findings,
            'quality_metrics': {
                'clarity_score': round(1 - avg_ambiguity, 2),
                'completeness_score': round(avg_completeness, 2),
                'testability_score': round(avg_testability, 2),
                'complexity_score': round(avg_complexity, 2)
            },
            'immediate_actions_required': immediate_actions,
            'success_indicators': [
                f"Target: {high_quality_count}/{total_requirements} requirements meet quality standards",
                f"Goal: Reduce ambiguous requirements by {max(0, needs_improvement_count - int(total_requirements * 0.1))}",
                "Objective: All requirements should have clear acceptance criteria"
            ]
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the analyzed requirements with enhanced quality metrics.

        Returns:
            Dict[str, Any]: Dictionary containing various statistics about the requirements
        """
        if not self.requirements:
            return {
                "total_requirements": 0,
                "avg_complexity": 0.0,
                "avg_ambiguity": 0.0,
                "avg_completeness": 0.0,
                "avg_testability": 0.0,
                "category_distribution": {},
                "priority_distribution": {},
                "complexity_distribution": {},
                "ambiguity_distribution": {},
                "completeness_distribution": {},
                "testability_distribution": {},
                "total_test_cases": 0,
                "quality_metrics": {},
                "enhancement_summary": {}
            }

        requirements_list = list(self.requirements.values())

        # Basic counts
        total_requirements = len(requirements_list)

        # Calculate averages for traditional metrics
        complexities = [req.complexity for req in requirements_list]
        ambiguities = [req.ambiguity for req in requirements_list]
        completenesses = [req.completeness for req in requirements_list]
        testabilities = [req.testability for req in requirements_list]

        avg_complexity = sum(complexities) / total_requirements if complexities else 0.0
        avg_ambiguity = sum(ambiguities) / total_requirements if ambiguities else 0.0
        avg_completeness = sum(completenesses) / total_requirements if completenesses else 0.0
        avg_testability = sum(testabilities) / total_requirements if testabilities else 0.0

        # NEW: Calculate enhanced quality metrics
        quality_scores = []
        structure_scores = []
        improvement_counts = {'high': 0, 'medium': 0, 'low': 0}

        for req in requirements_list:
            quality_assessment = req.analysis_results.get("quality_assessment", {})
            structure_analysis = req.analysis_results.get("structure_analysis", {})
            improvements = req.analysis_results.get("improvement_opportunities", [])

            if quality_assessment:
                quality_scores.append(quality_assessment.get('overall', 0.0))
            if structure_analysis:
                structure_scores.append(structure_analysis.get('structure_completeness', 0.0))

            # Count improvements by severity
            for improvement in improvements:
                severity = improvement.get('severity', 'medium')
                improvement_counts[severity] += 1

        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_structure_score = sum(structure_scores) / len(structure_scores) if structure_scores else 0.0

        # Category distribution
        category_dist = {}
        for req in requirements_list:
            category = req.category
            category_dist[category] = category_dist.get(category, 0) + 1

        # Priority distribution
        priority_dist = {}
        for req in requirements_list:
            priority = req.priority
            priority_dist[priority] = priority_dist.get(priority, 0) + 1

        # Score distributions (buckets: Low 0-0.33, Medium 0.33-0.66, High 0.66-1.0)
        def get_score_bucket(score):
            if score < 0.33:
                return "Low"
            elif score < 0.66:
                return "Medium"
            else:
                return "High"

        complexity_dist = {"Low": 0, "Medium": 0, "High": 0}
        ambiguity_dist = {"Low": 0, "Medium": 0, "High": 0}
        completeness_dist = {"Low": 0, "Medium": 0, "High": 0}
        testability_dist = {"Low": 0, "Medium": 0, "High": 0}
        quality_dist = {"Low": 0, "Medium": 0, "High": 0}

        for req in requirements_list:
            complexity_dist[get_score_bucket(req.complexity)] += 1
            ambiguity_dist[get_score_bucket(req.ambiguity)] += 1
            completeness_dist[get_score_bucket(req.completeness)] += 1
            testability_dist[get_score_bucket(req.testability)] += 1

            # NEW: Quality distribution
            quality_assessment = req.analysis_results.get("quality_assessment", {})
            overall_quality = quality_assessment.get('overall', 0.0)
            quality_dist[get_score_bucket(overall_quality)] += 1

        # Count total test cases generated
        total_test_cases = 0
        for req in requirements_list:
            test_suggestions = req.analysis_results.get("test_suggestions", [])
            total_test_cases += len(test_suggestions)

        # Enhanced quality metrics
        high_complexity_count = complexity_dist["High"]
        high_ambiguity_count = ambiguity_dist["High"]
        low_completeness_count = completeness_dist["Low"]
        low_testability_count = testability_dist["Low"]
        low_quality_count = quality_dist["Low"]

        quality_issues = high_complexity_count + high_ambiguity_count + low_completeness_count + low_testability_count
        quality_score = max(0.0, 1.0 - (quality_issues / (total_requirements * 4)))

        # Identify problematic requirements with enhanced criteria
        problematic_reqs = []
        for req in requirements_list:
            issues = []
            if req.complexity > 0.66:
                issues.append("High Complexity")
            if req.ambiguity > 0.66:
                issues.append("High Ambiguity")
            if req.completeness < 0.33:
                issues.append("Low Completeness")
            if req.testability < 0.33:
                issues.append("Low Testability")

            # NEW: Add quality-based issues
            quality_assessment = req.analysis_results.get("quality_assessment", {})
            if quality_assessment.get('overall', 0.0) < 0.4:
                issues.append("Poor Overall Quality")

            if issues:
                problematic_reqs.append({
                    "id": req.id,
                    "issues": issues,
                    "text_preview": req.text[:100] + "..." if len(req.text) > 100 else req.text
                })

        return {
            "total_requirements": total_requirements,
            "avg_complexity": round(avg_complexity, 3),
            "avg_ambiguity": round(avg_ambiguity, 3),
            "avg_completeness": round(avg_completeness, 3),
            "avg_testability": round(avg_testability, 3),
            "avg_quality_score": round(avg_quality_score, 3),
            "avg_structure_score": round(avg_structure_score, 3),
            "category_distribution": category_dist,
            "priority_distribution": priority_dist,
            "complexity_distribution": complexity_dist,
            "ambiguity_distribution": ambiguity_dist,
            "completeness_distribution": completeness_dist,
            "testability_distribution": testability_dist,
            "quality_distribution": quality_dist,
            "total_test_cases": total_test_cases,
            "avg_test_cases_per_requirement": round(total_test_cases / total_requirements, 2) if total_requirements > 0 else 0,
            "quality_metrics": {
                "overall_quality_score": round(quality_score, 3),
                "high_complexity_requirements": high_complexity_count,
                "high_ambiguity_requirements": high_ambiguity_count,
                "low_completeness_requirements": low_completeness_count,
                "low_testability_requirements": low_testability_count,
                "low_quality_requirements": low_quality_count,
                "problematic_requirements": problematic_reqs[:10]  # Limit to top 10
            },
            "enhancement_summary": {
                "total_improvement_opportunities": sum(improvement_counts.values()),
                "critical_issues": improvement_counts['high'],
                "important_issues": improvement_counts['medium'],
                "minor_issues": improvement_counts['low'],
                "requirements_needing_attention": len([req for req in requirements_list
                                                    if req.analysis_results.get("improvement_opportunities", [])])
            },
            "text_analysis": {
                "total_words": sum(req.analysis_results.get("complexity", {}).get("word_count", 0) for req in requirements_list),
                "total_sentences": sum(req.analysis_results.get("complexity", {}).get("sentence_count", 0) for req in requirements_list),
                "avg_words_per_requirement": round(sum(req.analysis_results.get("complexity", {}).get("word_count", 0) for req in requirements_list) / total_requirements, 2) if total_requirements > 0 else 0
            }
        }

    def analyze_all_requirements(self) -> None:
        """
        Analyse all requirements in the collection.
        """
        logger.info(f"Analysing {len(self.requirements)} requirements")
        for req_id, req in self.requirements.items():
            self.analyze_requirement(req)
        logger.info("Completed analysis of all requirements")

    def save_analysis_results(self, file_path: str) -> None:
        """
        Save analysis results to a JSON file.

        Args:
            file_path (str): Path to save the JSON file
        """
        results = {req_id: req.to_dict() for req_id, req in self.requirements.items()}

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved analysis results to {file_path}")

    def load_analysis_results(self, file_path: str) -> None:
        """
        Load analysis results from a JSON file.

        Args:
            file_path (str): Path to the JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.requirements = {}
            for req_id, req_data in data.items():
                req = Requirement(req_id, req_data.get("text", ""))
                req.from_dict(req_data)
                self.requirements[req_id] = req

            logger.info(f"Loaded {len(self.requirements)} requirements from {file_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading analysis results: {e}")

    def _analyze_complexity(self, req: Requirement) -> None:
        """
        Analyse the complexity of a requirement.

        Args:
            req (Requirement): The requirement to analyse
        """
        text = req.text

        # Basic metrics
        num_chars = len(text)
        words = word_tokenize(text)
        num_words = len(words)
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)

        # Average word length
        avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0

        # Average sentence length
        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

        # Count complex words (words with 3+ syllables)
        complex_words = self._count_complex_words(words)

        # Calculate readability scores
        flesch_reading_ease = self._calculate_flesch_reading_ease(num_words, num_sentences, complex_words)
        fog_index = self._calculate_fog_index(avg_sentence_length, complex_words, num_words)

        # Store complexity results
        req.analysis_results["complexity"] = {
            "char_count": num_chars,
            "word_count": num_words,
            "sentence_count": num_sentences,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "complex_word_count": complex_words,
            "flesch_reading_ease": flesch_reading_ease,
            "fog_index": fog_index
        }

        # Set complexity score (normalised)
        # Lower readability means higher complexity
        req.complexity = min(1.0, max(0.0, (100 - flesch_reading_ease) / 100))

        logger.info(f"Complexity analysis completed for {req.id}. Score: {req.complexity:.2f}")

    def _count_complex_words(self, words: List[str]) -> int:
        """
        Count the number of complex words (3+ syllables).

        Args:
            words (List[str]): List of words to analyse

        Returns:
            int: Count of complex words
        """
        complex_word_count = 0
        for word in words:
            if self._count_syllables(word) >= 3:
                complex_word_count += 1
        return complex_word_count

    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word using a simple heuristic.

        Args:
            word (str): Word to count syllables for

        Returns:
            int: Estimated syllable count
        """
        word = word.lower()
        # Remove common endings for better syllable estimation
        if word.endswith('es') or word.endswith('ed'):
            word = word[:-2]
        elif word.endswith('e'):
            word = word[:-1]

        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_is_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel

        # Ensure at least one syllable
        return max(1, count)

    def _calculate_flesch_reading_ease(self, num_words: int, num_sentences: int, complex_words: int) -> float:
        """
        Calculate Flesch Reading Ease score.

        Args:
            num_words (int): Number of words
            num_sentences (int): Number of sentences
            complex_words (int): Number of complex words

        Returns:
            float: Flesch Reading Ease score (0-100)
        """
        if num_words == 0 or num_sentences == 0:
            return 100.0

        # Approximate syllables from complex words
        syllables = num_words + complex_words * 2

        # Flesch Reading Ease = 206.835 - (1.015 × ASL) - (84.6 × ASW)
        # ASL = average sentence length (words/sentence)
        # ASW = average syllables per word
        asl = num_words / num_sentences
        asw = syllables / num_words

        score = 206.835 - (1.015 * asl) - (84.6 * asw)

        # Clamp to 0-100 range
        return min(100.0, max(0.0, score))

    def _calculate_fog_index(self, avg_sentence_length: float, complex_words: int, num_words: int) -> float:
        """
        Calculate Gunning Fog Index.

        Args:
            avg_sentence_length (float): Average sentence length
            complex_words (int): Number of complex words
            num_words (int): Total number of words

        Returns:
            float: Gunning Fog Index
        """
        if num_words == 0:
            return 0.0

        percent_complex = (complex_words / num_words) * 100 if num_words > 0 else 0

        # Gunning Fox Index = 0.4 × (ASL + PCW)
        # ASL = average sentence length
        # PCW = percentage of complex words
        return 0.4 * (avg_sentence_length + percent_complex)

    def _analyze_ambiguity(self, req: Requirement) -> None:
        """
        Enhanced ambiguity analysis with better detection of vague and unclear content.

        Args:
            req (Requirement): The requirement to analyse
        """
        text = req.text
        words = word_tokenize(text.lower())

        # Enhanced lists of potentially ambiguous words and phrases
        ambiguous_terms = [
            'may', 'might', 'could', 'should', 'would', 'can',
            'possibly', 'probably', 'likely', 'unlikely', 'perhaps',
            'several', 'many', 'few', 'some', 'numerous', 'appropriate',
            'fast', 'slow', 'big', 'small', 'better', 'worse', 'best',
            'worst', 'good', 'bad', 'normal', 'regular', 'custom',
            'easy', 'hard', 'simple', 'complex', 'flexible', 'efficient',
            'user-friendly', 'effective', 'adequate', 'significant',
            'clear', 'easy to use', 'improved', 'enhanced', 'reasonable',
            'optimal', 'minimal', 'maximum', 'sufficient', 'acceptable'
        ]

        vague_phrases = [
            'as appropriate', 'if appropriate', 'as required', 'if required',
            'as necessary', 'if necessary', 'to be determined', 'to be defined',
            'to be specified', 'subject to', 'up to', 'and so on', 'etc',
            'and so forth', 'as soon as possible', 'as much as possible',
            'where applicable', 'if applicable', 'as needed', 'when needed',
            'among others', 'such as', 'for example', 'or similar'
        ]

        # Count occurrences of ambiguous terms and phrases
        ambiguous_term_counts = {}
        for term in ambiguous_terms:
            count = sum(1 for word in words if word == term)
            if count > 0:
                ambiguous_term_counts[term] = count

        vague_phrase_counts = {}
        for phrase in vague_phrases:
            count = text.lower().count(phrase)
            if count > 0:
                vague_phrase_counts[phrase] = count

        # Enhanced passive voice detection
        passive_voice_count = 0
        if nlp:
            doc = nlp(text)
            for sent in doc.sents:
                # Look for passive voice patterns
                for token in sent:
                    if token.dep_ == "auxpass" or token.dep_ == "nsubjpass":
                        passive_voice_count += 1
                    # Additional pattern: "be" + past participle
                    if (token.lemma_ == "be" and
                        token.i + 1 < len(doc) and
                        doc[token.i + 1].tag_ in ["VBN", "VBD"]):
                        passive_voice_count += 1

        # Check for measurement ambiguity
        measurement_ambiguity = 0
        measurement_patterns = [
            r'\b(?:fast|slow|quick|long|short)\b(?!\s+(?:seconds|minutes|hours|ms|response time))',
            r'\b(?:large|small|big|little)\b(?!\s+(?:MB|GB|KB|bytes|files))',
            r'\b(?:many|few|several)\b(?!\s+\d)',
            r'\b(?:high|low)\b(?!\s+(?:priority|severity|\d))'
        ]

        for pattern in measurement_patterns:
            measurement_ambiguity += len(re.findall(pattern, text, re.IGNORECASE))

        # Calculate total ambiguity indicators with weighted scoring
        total_ambiguous_terms = sum(ambiguous_term_counts.values())
        total_vague_phrases = sum(vague_phrase_counts.values())
        total_indicators = (
            total_ambiguous_terms * 1.0 +  # Normal weight
            total_vague_phrases * 1.5 +    # Higher weight for vague phrases
            passive_voice_count * 0.5 +    # Lower weight for passive voice
            measurement_ambiguity * 2.0    # Highest weight for measurement ambiguity
        )

        # Normalise score based on text length (per 100 words) with improved scaling
        word_count = len(words)
        if word_count > 0:
            ambiguity_score = min(1.0, total_indicators / (word_count / 50 + 1) / 5)
        else:
            ambiguity_score = 0.0

        # Store enhanced results
        req.analysis_results["ambiguity"] = {
            "ambiguous_terms": ambiguous_term_counts,
            "vague_phrases": vague_phrase_counts,
            "passive_voice_count": passive_voice_count,
            "measurement_ambiguity": measurement_ambiguity,
            "total_indicators": total_indicators,
            "word_count": word_count
        }

        req.ambiguity = ambiguity_score
        logger.info(f"Enhanced ambiguity analysis completed for {req.id}. Score: {req.ambiguity:.2f}")

    def _analyze_completeness(self, req: Requirement) -> None:
        """
        Analyse the completeness of a requirement.

        Args:
            req (Requirement): The requirement to analyse
        """
        text = req.text

        # Check for important sections that should be present
        has_actor = bool(re.search(r'\b(user|system|admin|customer|client|operator)\b', text, re.IGNORECASE))
        has_action = bool(re.search(r'\b(shall|must|will|should|can|could|may|might)\b', text, re.IGNORECASE))
        has_object = bool(re.search(r'\b(system|application|software|platform|database|interface|screen|page|form|report)\b', text, re.IGNORECASE))
        has_condition = bool(re.search(r'\b(if|when|after|before|during|while|unless|until)\b', text, re.IGNORECASE))

        # Check for acceptance criteria or measurable outcomes
        has_measurement = bool(re.search(r'\b(within|less than|more than|at least|maximum|minimum|equal to|seconds|minutes|hours|days)\b', text, re.IGNORECASE))

        # Search for data specifications
        has_data_spec = bool(re.search(r'\b(format|field|value|input|output|type|string|number|date|boolean|integer|float)\b', text, re.IGNORECASE))

        # Check basic length requirements
        words = word_tokenize(text)
        word_count = len(words)
        is_adequate_length = word_count >= 10

        # Calculate completeness score (simple average of indicators)
        indicators = [has_actor, has_action, has_object, has_condition, has_measurement, has_data_spec, is_adequate_length]
        completeness_score = sum(1 for i in indicators if i) / len(indicators)

        # Store results
        req.analysis_results["completeness"] = {
            "has_actor": has_actor,
            "has_action": has_action,
            "has_object": has_object,
            "has_condition": has_condition,
            "has_measurement": has_measurement,
            "has_data_spec": has_data_spec,
            "word_count": word_count,
            "is_adequate_length": is_adequate_length
        }

        req.completeness = completeness_score
        logger.info(f"Completeness analysis completed for {req.id}. Score: {req.completeness:.2f}")

    def _analyze_testability(self, req: Requirement) -> None:
        """
        Analyse the testability of a requirement.

        Args:
            req (Requirement): The requirement to analyse
        """
        text = req.text

        # Check for measurable criteria
        has_measurable = bool(re.search(r'\b(equal to|greater than|less than|maximum|minimum|within|between)\b', text, re.IGNORECASE))

        # Check for clear conditions
        has_clear_conditions = bool(re.search(r'\b(if|when|after|before|during|while)\b', text, re.IGNORECASE))

        # Check for specific outputs or results
        has_specific_results = bool(re.search(r'\b(return|display|show|produce|generate|create|update|delete|remove)\b', text, re.IGNORECASE))

        # Check if the requirement uses specific, unambiguous language
        ambiguity_score = req.ambiguity if hasattr(req, 'ambiguity') and req.ambiguity is not None else 1.0
        has_unambiguous_language = ambiguity_score < 0.5

        # Check for testable attributes
        has_testable_attributes = bool(re.search(r'\b(response time|throughput|accuracy|precision|error rate|reliability|availability)\b', text, re.IGNORECASE))

        # Check if the requirement seems functional (easier to test than non-functional)
        is_functional = bool(re.search(r'\b(shall|must|will)\s+\b(do|perform|execute|calculate|process|validate|verify)\b', text, re.IGNORECASE))

        # Calculate testability score
        indicators = [has_measurable, has_clear_conditions, has_specific_results,
                    has_unambiguous_language, has_testable_attributes, is_functional]
        testability_score = sum(1 for i in indicators if i) / len(indicators)

        # Store results
        req.analysis_results["testability"] = {
            "has_measurable": has_measurable,
            "has_clear_conditions": has_clear_conditions,
            "has_specific_results": has_specific_results,
            "has_unambiguous_language": has_unambiguous_language,
            "has_testable_attributes": has_testable_attributes,
            "is_functional": is_functional
        }

        req.testability = testability_score
        logger.info(f"Testability analysis completed for {req.id}. Score: {req.testability:.2f}")

    def _extract_entities(self, req: Requirement) -> None:
        """
        Extract entities from requirement text using spaCy.

        Args:
            req (Requirement): The requirement to analyse
        """
        entities = []

        if nlp:
            doc = nlp(req.text)
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        req.analysis_results["entities"] = entities

    def _extract_keywords(self, req: Requirement) -> None:
        """
        Extract important keywords from the requirement.

        Args:
            req (Requirement): The requirement to analyse
        """
        # Use spaCy for keyword extraction if available
        keywords = []

        if nlp:
            doc = nlp(req.text)

            # Extract nouns, verbs, and proper nouns as keywords
            for token in doc:
                if token.pos_ in ["NOUN", "VERB", "PROPN"] and not token.is_stop and len(token.text) > 2:
                    keywords.append({
                        "text": token.text,
                        "pos": token.pos_,
                        "relevance": token.prob  # Log probability (lower means more relevant/unique)
                    })
        else:
            # Fallback to simple word frequency if spaCy is not available
            words = word_tokenize(req.text.lower())
            word_freq = {}

            # Simple stopwords list for filtering
            stopwords = {"the", "a", "an", "and", "or", "but", "if", "then", "that", "this", "to", "in", "on", "at", "by"}

            for word in words:
                if word.isalnum() and word not in stopwords and len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1

            # Convert frequencies to keywords list
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                keywords.append({
                    "text": word,
                    "pos": "UNKNOWN",
                    "frequency": freq
                })

        req.analysis_results["keywords"] = keywords

    def _analyze_sentiment(self, req: Requirement) -> None:
        """
        Analyse the sentiment of the requirement text.

        Args:
            req (Requirement): The requirement to analyse
        """
        sentiment = {"compound": 0, "pos": 0, "neu": 0, "neg": 0}

        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.polarity_scores(req.text)

        req.analysis_results["sentiment"] = sentiment

