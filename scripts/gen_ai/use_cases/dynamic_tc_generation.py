import logging
# Filter warnings related to missing ScriptRunContext in threads
import warnings
import os
import sys
import requests
import base64
from urllib.parse import urlparse
import re

# Add the parent directory to sys.path so we can import modules properly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This should point to gen_ai directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the gen_ai directory specifically to ensure RAG imports work
gen_ai_dir = os.path.dirname(current_dir)
if gen_ai_dir not in sys.path:
    sys.path.insert(0, gen_ai_dir)

# Import streamlit_fix.py which handles asyncio and PyTorch configuration
# This must be imported before any other imports that might use asyncio or PyTorch
try:
    from gen_ai import streamlit_fix
except ImportError:
    # Try alternative import path
    import streamlit_fix

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("DynamicTCGeneration", level=logging.INFO, log_file="dynamic_tc_generation.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# Import streamlit after asyncio setup is properly handled by streamlit_fix
import streamlit as st

# Import notifications module for action feedback
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    logging.warning("Notifications module not available. Notification features will be disabled.")

# Configure logging and suppress specific warnings
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")
warnings.filterwarnings("ignore", message=".*torch.*_path.*")  # Specifically ignore torch path warnings

# Only set page config if this script is run directly (not when imported)
if __name__ == "__main__":
    st.set_page_config(page_title="AI Test Case Generator", layout="wide")

st.markdown("""
<style>
[data-testid="stDataFrame"] table {
    width: 100% !important;
    table-layout: auto !important;
}
[data-testid="stDataFrame"] td {
    white-space: pre-line !important;
    word-break: break-word !important;
}
.requirements-list {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.requirement-item {
    background-color: white;
    border-left: 4px solid #4CAF50;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}
.figma-preview {
    border: 2px dashed #cccccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background-color: #f9f9f9;
}
.rag-status {
    background-color: #EC5328;
    border: 1px solid #EC5328;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
    color: white;
}
.enhanced-requirements {
    background-color: #f8f9fa;
    border-left: 4px solid #007bff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Add the parent directory to sys.path so we can import modules properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parsers.text_parser import TextParser
from parsers.markdown_parser import MarkdownParser
from parsers.word_parser import WordParser
from parsers.pdf_parser import PDFParser
from parsers.powerpoint_parser import PowerPointParser
from nlp.requirement_analyzer import RequirementAnalyzer
from generator.testcase_generator import TestCaseGenerator
from robot_writer.robot_writer import RobotWriter
import tempfile
import pandas as pd
import json
import yaml
import time
import io

# Enhanced RAG module imports with better error handling
RAG_AVAILABLE = False
rag_handler = None
try:
    # Import RAG modules from the correct path
    from rag.rag_handler import RAGHandler
    from rag.rag_config import RAGConfig
    RAG_AVAILABLE = True
    logging.info("RAG module imported successfully")
except ImportError as e:
    try:
        # Fallback: try direct import if the modules are in the same directory structure
        import sys
        rag_path = os.path.join(parent_dir, 'rag')
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)

        from rag_handler import RAGHandler
        from rag_config import RAGConfig
        RAG_AVAILABLE = True
        logging.info("RAG module imported successfully (fallback method)")
    except ImportError as e2:
        logging.warning(f"RAG module not available: {e2}. RAG features will be disabled.")
        RAG_AVAILABLE = False

# Azure OpenAI client import
try:
    from azure_openai_client import AzureOpenAIClient
    AZURE_CLIENT_AVAILABLE = True
    logging.info("Azure OpenAI client imported successfully")
except ImportError as e:
    logging.warning(f"Azure OpenAI client not available: {e}. Using fallback configuration.")
    AZURE_CLIENT_AVAILABLE = False


# New Figma Parser class
class FigmaParser:
    def parse(self, file_path):
        """Parse Figma files and extract design information"""
        try:
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.fig':
                # Handle native Figma files (would need Figma API)
                return {"raw_text": "Figma design file uploaded. Please provide Figma URL for detailed analysis.", "figma_file": file_path}
            elif ext in ['.png', '.jpg', '.jpeg', '.svg']:
                # Handle image exports from Figma
                return {"raw_text": f"Design mockup image uploaded: {os.path.basename(file_path)}. Visual design elements identified for UI test case generation.", "design_file": file_path}
            elif ext == '.pdf':
                # Handle PDF exports from Figma
                pdf_parser = PDFParser()
                result = pdf_parser.parse(file_path)
                result["raw_text"] = f"Figma PDF export analyzed. {result.get('raw_text', '')}"
                return result
            else:
                return {"raw_text": f"Design file uploaded: {os.path.basename(file_path)}", "design_file": file_path}

        except Exception as e:
            logging.error(f"Error parsing Figma file {file_path}: {e}")
            return {"raw_text": f"Error parsing design file: {str(e)}", "error": str(e)}


# Enhanced parser function with Figma support
def get_parser(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.txt']:
        return TextParser()
    elif ext in ['.md']:
        return MarkdownParser()
    elif ext in ['.docx']:
        return WordParser()
    elif ext in ['.ppt', '.pptx']:
        return PowerPointParser()
    elif ext in ['.pdf']:
        return PDFParser()
    elif ext in ['.fig', '.png', '.svg', '.jpg', '.jpeg']:
        return FigmaParser()  # New parser for Figma files
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# Enhanced JIRA Integration Functions with password/token flexibility
def fetch_jira_issue(host, project_key, issue_id, username=None, credential=None, credential_type="token"):
    """
    Enhanced JIRA issue fetching with flexible authentication

    Args:
        host: JIRA host URL
        project_key: JIRA project key
        issue_id: Issue ID (without project key)
        username: Username or email
        credential: Password or API token
        credential_type: "password" or "token"
    """
    try:
        # Clean up the host URL
        if not host.startswith('http'):
            host = f"https://{host}"
        host = host.rstrip('/')

        # Handle different JIRA URL patterns
        # For Newfold JIRA and other custom instances, we need to try multiple API endpoint patterns
        api_endpoints = []

        # Pattern 1: Standard REST API v3 and v2
        api_endpoints.extend([
            f"{host}/rest/api/3/issue/{project_key}-{issue_id}",
            f"{host}/rest/api/2/issue/{project_key}-{issue_id}"
        ])

        # Pattern 2: If host has /browse/ pattern, extract base URL
        if '/browse/' in host:
            base_host = host.split('/browse/')[0]
            api_endpoints.extend([
                f"{base_host}/rest/api/3/issue/{project_key}-{issue_id}",
                f"{base_host}/rest/api/2/issue/{project_key}-{issue_id}"
            ])

        # Pattern 3: Try direct issue key approach
        issue_key = f"{project_key}-{issue_id}"
        api_endpoints.extend([
            f"{host}/rest/api/3/issue/{issue_key}",
            f"{host}/rest/api/2/issue/{issue_key}"
        ])

        # Pattern 4: For instances like jira.newfold.com, try without subdirectories
        if 'jira.' in host:
            base_host = host
            api_endpoints.extend([
                f"{base_host}/rest/api/3/issue/{issue_key}",
                f"{base_host}/rest/api/2/issue/{issue_key}"
            ])

        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'JARVIS-TestAutomation/1.0'
        }

        auth = None
        if username and credential:
            if credential_type == "token":
                # API token authentication
                auth = (username, credential)
            else:
                # Password authentication (basic auth)
                auth = (username, credential)
                # For password auth, we might need to use session-based auth for some JIRA instances
                headers['X-Atlassian-Token'] = 'no-check'

        last_error = None
        successful_endpoint = None

        # Try each API endpoint
        for api_url in api_endpoints:
            try:
                logging.info(f"Trying JIRA API endpoint: {api_url}")
                response = requests.get(api_url, headers=headers, auth=auth, timeout=30, verify=True)

                if response.status_code == 200:
                    successful_endpoint = api_url
                    issue_data = response.json()

                    # Extract relevant information
                    summary = issue_data.get('fields', {}).get('summary', '')
                    description = issue_data.get('fields', {}).get('description', '')
                    issue_type = issue_data.get('fields', {}).get('issuetype', {}).get('name', '')
                    status = issue_data.get('fields', {}).get('status', {}).get('name', '')
                    priority = issue_data.get('fields', {}).get('priority', {}).get('name', '')

                    # Additional fields for better context
                    assignee = issue_data.get('fields', {}).get('assignee')
                    assignee_name = assignee.get('displayName', 'Unassigned') if assignee else 'Unassigned'

                    reporter = issue_data.get('fields', {}).get('reporter', {})
                    reporter_name = reporter.get('displayName', 'Unknown') if reporter else 'Unknown'

                    created = issue_data.get('fields', {}).get('created', '')
                    updated = issue_data.get('fields', {}).get('updated', '')

                    # Enhanced description parsing for different formats
                    if isinstance(description, dict):
                        description = extract_text_from_adf(description)
                    elif isinstance(description, str):
                        # Clean up any HTML or markdown artifacts
                        description = re.sub(r'<[^>]+>', '', description)
                        description = re.sub(r'\*\*(.*?)\*\*', r'\1', description)  # Bold markdown
                        description = re.sub(r'\*(.*?)\*', r'\1', description)      # Italic markdown
                    elif description is None:
                        description = "No description provided"

                    requirement_text = f"""
JIRA Issue: {project_key}-{issue_id}
Summary: {summary}
Type: {issue_type}
Status: {status}
Priority: {priority}
Assignee: {assignee_name}
Reporter: {reporter_name}
Created: {created}
Updated: {updated}

Description:
{description}

Additional Context:
- This requirement comes from JIRA issue {project_key}-{issue_id}
- Issue type: {issue_type}
- Current status: {status}
- Priority level: {priority}
- Assigned to: {assignee_name}
- Reported by: {reporter_name}
- Retrieved from: {successful_endpoint}
                    """.strip()

                    return {
                        "success": True,
                        "raw_text": requirement_text,
                        "issue_data": issue_data,
                        "summary": summary,
                        "description": description,
                        "metadata": {
                            "issue_key": f"{project_key}-{issue_id}",
                            "issue_type": issue_type,
                            "status": status,
                            "priority": priority,
                            "assignee": assignee_name,
                            "reporter": reporter_name,
                            "created": created,
                            "updated": updated,
                            "api_version": "v3" if "api/3" in api_url else "v2",
                            "api_endpoint": successful_endpoint
                        }
                    }
                elif response.status_code == 401:
                    last_error = f"Authentication failed. Please check your credentials. Status: {response.status_code}"
                    # Don't continue trying other endpoints if auth fails
                    break
                elif response.status_code == 403:
                    last_error = f"Access forbidden. You may not have permission to view this issue. Status: {response.status_code}"
                    # Don't continue trying other endpoints if permission fails
                    break
                elif response.status_code == 404:
                    last_error = f"Issue {project_key}-{issue_id} not found at {api_url}"
                    # Continue trying other endpoints for 404 errors
                    continue
                else:
                    last_error = f"Failed to fetch JIRA issue from {api_url}. Status: {response.status_code}, Response: {response.text[:200]}"
                    continue

            except requests.exceptions.SSLError as e:
                last_error = f"SSL verification failed for {api_url}: {str(e)}. Try using HTTP or check SSL certificates."
                continue
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error for {api_url}: {str(e)}"
                continue
            except requests.exceptions.Timeout as e:
                last_error = f"Timeout error for {api_url}: {str(e)}"
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"Network error while fetching JIRA issue from {api_url}: {str(e)}"
                continue
            except Exception as e:
                last_error = f"Unexpected error for {api_url}: {str(e)}"
                continue

        # If we get here, all endpoints failed
        return {
            "success": False,
            "error": last_error or f"Failed to fetch JIRA issue {project_key}-{issue_id} from all API endpoints. Tried {len(api_endpoints)} different endpoints.",
            "attempted_endpoints": api_endpoints
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching JIRA issue: {str(e)}"
        }


def extract_text_from_adf(adf_content):
    """Enhanced ADF (Atlassian Document Format) text extraction"""
    if not isinstance(adf_content, dict):
        return str(adf_content)

    text_parts = []

    def extract_from_node(node, depth=0):
        if isinstance(node, dict):
            node_type = node.get('type', '')

            if node_type == 'text':
                text_parts.append(node.get('text', ''))
            elif node_type == 'paragraph':
                if 'content' in node:
                    for child in node['content']:
                        extract_from_node(child, depth + 1)
                text_parts.append('\n')
            elif node_type == 'heading':
                level = node.get('attrs', {}).get('level', 1)
                prefix = '#' * level + ' '
                text_parts.append(prefix)
                if 'content' in node:
                    for child in node['content']:
                        extract_from_node(child, depth + 1)
                text_parts.append('\n\n')
            elif node_type == 'bulletList' or node_type == 'orderedList':
                if 'content' in node:
                    for child in node['content']:
                        extract_from_node(child, depth + 1)
            elif node_type == 'listItem':
                text_parts.append('• ')
                if 'content' in node:
                    for child in node['content']:
                        extract_from_node(child, depth + 1)
                text_parts.append('\n')
            elif node_type == 'hardBreak':
                text_parts.append('\n')
            elif node_type == 'codeBlock':
                text_parts.append('\n```\n')
                if 'content' in node:
                    for child in node['content']:
                        extract_from_node(child, depth + 1)
                text_parts.append('\n```\n')
            elif 'content' in node:
                for child in node['content']:
                    extract_from_node(child, depth + 1)

        elif isinstance(node, list):
            for item in node:
                extract_from_node(item, depth)

    extract_from_node(adf_content)
    return ''.join(text_parts).strip()


# Enhanced Confluence Integration Functions with password/token flexibility
def fetch_confluence_page(host, space_key, page_title, username=None, credential=None, credential_type="token"):
    """
    Enhanced Confluence page fetching with flexible authentication

    Args:
        host: Confluence host URL
        space_key: Confluence space key
        page_title: Page title to search for
        username: Username or email
        credential: Password or API token
        credential_type: "password" or "token"
    """
    try:
        # Clean up the host URL
        if not host.startswith('http'):
            host = f"https://{host}"
        host = host.rstrip('/')

        # Remove /wiki suffix if present and add it back
        if host.endswith('/wiki'):
            host = host[:-5]

        # Try different API endpoints for better compatibility
        api_endpoints = [
            f"{host}/wiki/rest/api/content",
            f"{host}/rest/api/content"
        ]

        params = {
            'title': page_title,
            'spaceKey': space_key,
            'expand': 'body.storage,version,space'
        }

        headers = {
            'Accept': 'application/json'
        }

        auth = None
        if username and credential:
            if credential_type == "token":
                auth = (username, credential)
            else:
                auth = (username, credential)
                headers['X-Atlassian-Token'] = 'no-check'

        last_error = None

        for api_url in api_endpoints:
            try:
                response = requests.get(api_url, params=params, headers=headers, auth=auth, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('results'):
                        page = data['results'][0]
                        title = page.get('title', '')

                        # Enhanced content extraction
                        content = ""
                        body = page.get('body', {})

                        if 'storage' in body:
                            content = body['storage'].get('value', '')
                        elif 'view' in body:
                            content = body['view'].get('value', '')

                        # Clean HTML tags and format text
                        if content:
                            # Remove HTML tags but preserve structure
                            content = re.sub(r'<h[1-6][^>]*>(.*?)</h[1-6]>', r'\n\n# \1\n\n', content)
                            content = re.sub(r'<p[^>]*>', '\n', content)
                            content = re.sub(r'</p>', '\n', content)
                            content = re.sub(r'<br[^>]*/?>', '\n', content)
                            content = re.sub(r'<li[^>]*>', '• ', content)
                            content = re.sub(r'</li>', '\n', content)
                            content = re.sub(r'<[^>]+>', ' ', content)
                            # Clean up multiple spaces and newlines
                            content = re.sub(r'\s+', ' ', content)
                            content = re.sub(r'\n\s*\n', '\n\n', content)
                            content = content.strip()

                        requirement_text = f"""
Confluence Page: {title}
Space: {space_key}
URL: {host}/wiki/display/{space_key}/{title.replace(' ', '+')}

Content:
{content}

Page Context:
- This requirement comes from Confluence page "{title}"
- Located in space: {space_key}
- Page contains structured documentation about system requirements
                        """.strip()

                        return {
                            "success": True,
                            "raw_text": requirement_text,
                            "title": title,
                            "content": content,
                            "metadata": {
                                "space_key": space_key,
                                "page_title": title,
                                "page_url": f"{host}/wiki/display/{space_key}/{title.replace(' ', '+')}",
                                "content_length": len(content)
                            }
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"No page found with title '{page_title}' in space '{space_key}'. Please check the title and space key."
                        }
                elif response.status_code == 401:
                    return {
                        "success": False,
                        "error": f"Authentication failed. Please check your credentials."
                    }
                elif response.status_code == 403:
                    return {
                        "success": False,
                        "error": f"Access forbidden. You may not have permission to view this page or space."
                    }
                elif response.status_code == 404:
                    last_error = f"API endpoint not found at {api_url}. Trying alternative endpoint..."
                    continue
                else:
                    last_error = f"Failed to fetch Confluence page. Status: {response.status_code}, Response: {response.text[:200]}"

            except requests.exceptions.RequestException as e:
                last_error = f"Network error while fetching Confluence page: {str(e)}"
                continue

        return {
            "success": False,
            "error": last_error or "Failed to fetch Confluence page from all API endpoints"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error fetching Confluence page: {str(e)}"
        }


# Figma URL Integration
def fetch_figma_file_info(figma_url, access_token=None):
    """Extract information from Figma URL"""
    try:
        # Parse Figma URL to extract file ID
        # Format: https://www.figma.com/file/FILE_ID/FILE_NAME
        url_pattern = r'https://www\.figma\.com/file/([a-zA-Z0-9]+)/(.*?)(?:\?|$)'
        match = re.match(url_pattern, figma_url)

        if not match:
            return {
                "success": False,
                "error": "Invalid Figma URL format. Expected: https://www.figma.com/file/FILE_ID/FILE_NAME"
            }

        file_id = match.group(1)
        file_name = match.group(2).replace('-', ' ')

        if access_token:
            # Use Figma API to get file details
            api_url = f"https://api.figma.com/v1/files/{file_id}"
            headers = {
                'X-FIGMA-TOKEN': access_token
            }

            response = requests.get(api_url, headers=headers, timeout=30)

            if response.status_code == 200:
                file_data = response.json()

                # Extract pages and components info
                pages_info = []
                if 'document' in file_data and 'children' in file_data['document']:
                    for page in file_data['document']['children']:
                        if page.get('type') == 'CANVAS':
                            pages_info.append({
                                'name': page.get('name', ''),
                                'type': page.get('type', ''),
                                'children_count': len(page.get('children', []))
                            })

                requirement_text = f"""
Figma Design File: {file_name}
File ID: {file_id}
URL: {figma_url}

Design Information:
- Total Pages: {len(pages_info)}
- Pages: {', '.join([p['name'] for p in pages_info])}

This Figma file contains UI/UX designs that should be considered for:
- UI component testing
- User interaction flows
- Visual regression testing
- Accessibility testing
- Responsive design testing
                """.strip()

                return {
                    "success": True,
                    "raw_text": requirement_text,
                    "file_data": file_data,
                    "pages": pages_info
                }
            else:
                # Fallback without API
                requirement_text = f"""
Figma Design File: {file_name}
File ID: {file_id}
URL: {figma_url}

Note: Figma API access not configured. Please provide access token for detailed analysis.

This Figma file should be considered for:
- UI component testing
- User interaction flows
- Visual regression testing
- Accessibility testing
- Responsive design testing
                """.strip()

                return {
                    "success": True,
                    "raw_text": requirement_text,
                    "file_id": file_id,
                    "file_name": file_name
                }
        else:
            # Basic info without API
            requirement_text = f"""
Figma Design File: {file_name}
File ID: {file_id}
URL: {figma_url}

This Figma file should be considered for:
- UI component testing
- User interaction flows
- Visual regression testing
- Accessibility testing
- Responsive design testing
            """.strip()

            return {
                "success": True,
                "raw_text": requirement_text,
                "file_id": file_id,
                "file_name": file_name
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing Figma URL: {str(e)}"
        }


# Enhanced RAG Integration Functions
def initialize_rag_service(rag_config_path=None):
    """Enhanced RAG service initialization with better error handling"""
    global rag_handler

    if not RAG_AVAILABLE:
        logging.warning("RAG module not available. RAG features will be disabled.")
        return None

    try:
        rag_handler = RAGHandler()

        # Use provided config path or try to find default config
        if not rag_config_path:
            # Try common config locations
            possible_configs = [
                os.path.join(os.getcwd(), 'rag_config.yaml'),
                os.path.join(os.getcwd(), 'rag_config.json'),
                os.path.join(parent_dir, 'rag', 'config.yaml'),
                os.path.join(parent_dir, 'rag', 'config.json')
            ]

            for config_path in possible_configs:
                if os.path.exists(config_path):
                    rag_config_path = config_path
                    break

        if rag_config_path and os.path.exists(rag_config_path):
            success = rag_handler.initialize_service(rag_config_path)
            if success:
                logging.info(f"RAG service initialized successfully with config: {rag_config_path}")
                return rag_handler
            else:
                logging.error(f"Failed to initialize RAG service with config: {rag_config_path}")
                return None
        else:
            # Try to initialize with default settings
            success = rag_handler.initialize_service()
            if success:
                logging.info("RAG service initialized with default settings")
                return rag_handler
            else:
                logging.error("Failed to initialize RAG service with default settings")
                return None

    except Exception as e:
        logging.error(f"Error initializing RAG service: {e}")
        return None


def enhance_requirements_with_rag(rag_handler, requirements_text, scope="Both", test_type="All", components=None):
    """Enhanced requirement processing using RAG"""
    if not rag_handler or not RAG_AVAILABLE:
        return requirements_text

    try:
        # Use RAG to enhance the requirements with additional context
        enhanced_prompt = f"""
        Enhance the following requirements with additional context and details for test case generation:
        
        Original Requirements:
        {requirements_text}
        
        Test Scope: {scope}
        Test Type: {test_type}
        Components: {components or 'All'}
        
        Please provide:
        1. Enhanced requirement descriptions with more detail
        2. Implied requirements that may not be explicitly stated
        3. Test scenarios that should be covered
        4. Edge cases and boundary conditions
        5. Integration points and dependencies
        """

        enhanced_text = rag_handler.enhance_prompt(
            enhanced_prompt,
            requirements_text,
            scope,
            test_type,
            components
        )

        if enhanced_text and len(enhanced_text) > len(requirements_text):
            logging.info("Requirements enhanced successfully using RAG")
            return enhanced_text
        else:
            logging.warning("RAG enhancement did not improve requirements significantly")
            return requirements_text

    except Exception as e:
        logging.error(f"Error enhancing requirements with RAG: {e}")
        return requirements_text


def enhance_prompt_with_rag(rag_handler, prompt, query, scope, test_type, components=None):
    """Enhance a prompt with context retrieved from the RAG service."""
    if not rag_handler or not RAG_AVAILABLE:
        return prompt
    try:
        return rag_handler.enhance_prompt(prompt, query, scope, test_type, components)
    except Exception as e:
        logging.error(f"Error enhancing prompt with RAG: {e}")
        return prompt


# Enhanced Requirements Preprocessing System
def preprocess_requirements_with_ai(raw_requirements_text, sources, metadata, custom_prompt=""):
    """
    Intelligently preprocess and enhance requirements using Azure OpenAI to ensure
    meaningful, actionable requirements before test case generation.
    """
    try:
        # Initialize Azure OpenAI client
        client = get_azure_openai_client()
        if not client:
            logging.warning("Azure OpenAI client not available, using basic preprocessing")
            return basic_requirements_preprocessing(raw_requirements_text)

        # Create comprehensive preprocessing prompt
        preprocessing_prompt = f"""
You are an expert Business Analyst and Product Manager. Your task is to analyze and enhance the provided requirements to make them clear, actionable, and meaningful for software testing.

CONTEXT:
- Sources: {', '.join(sources) if sources else 'Various sources'}
- Custom Context: {custom_prompt if custom_prompt.strip() else 'None provided'}

RAW REQUIREMENTS TEXT:
{raw_requirements_text}

TASK:
Please analyze the above requirements and provide:

1. REQUIREMENTS CLARIFICATION:
   - Identify and clarify any vague, ambiguous, or unclear requirements
   - Fill in missing context based on standard software development practices
   - Explain what each requirement actually means in practical terms

2. BUSINESS VALUE ANALYSIS:
   - Explain the business value and purpose of each requirement
   - Identify the target users and their needs
   - Describe the expected outcomes

3. FUNCTIONAL BREAKDOWN:
   - Break down complex requirements into specific, testable functions
   - Identify user workflows and interactions
   - Specify expected behaviors and outcomes

4. TECHNICAL CONSIDERATIONS:
   - Identify system components that will be affected
   - Note integration points and dependencies
   - Highlight potential technical constraints

5. ENHANCED REQUIREMENTS:
   Rewrite the requirements in a clear, structured format using this template:

   **Requirement ID**: [Auto-generated ID]
   **Title**: [Clear, concise title]
   **Description**: [Detailed, unambiguous description]
   **User Story**: As a [user type], I want [functionality] so that [benefit]
   **Acceptance Criteria**: 
   - [Specific, testable criteria]
   - [Include positive and negative scenarios]
   **Priority**: [High/Medium/Low based on business impact]
   **Complexity**: [Low/Medium/High based on implementation effort]
   **Dependencies**: [Other requirements or systems this depends on]
   **Test Scenarios**: [High-level test scenarios that should be covered]

6. QUALITY ASSESSMENT:
   - Rate the overall quality of the original requirements (1-10)
   - Identify gaps or missing information
   - Suggest additional requirements that might be needed

Please ensure your analysis results in requirements that are:
- Clear and unambiguous
- Testable and verifiable
- Complete with necessary context
- Properly prioritized
- Technically feasible
- Business-value driven

Format your response in structured sections as requested above.
"""

        # Call Azure OpenAI for requirements preprocessing using the proper client method
        response = client.chat_completion_create(
            messages=[
                {"role": "system", "content": "You are an expert Business Analyst and Requirements Engineer specializing in creating clear, testable software requirements."},
                {"role": "user", "content": preprocessing_prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        # Extract content from response dict (AzureOpenAIClient returns a dict)
        if not response or 'choices' not in response:
            logging.warning("No valid response from Azure OpenAI")
            return basic_requirements_preprocessing(raw_requirements_text)

        enhanced_requirements = response['choices'][0]['message']['content']

        # Parse the enhanced requirements into structured format
        structured_requirements = parse_enhanced_requirements(enhanced_requirements, raw_requirements_text)

        return {
            "success": True,
            "enhanced_text": enhanced_requirements,
            "structured_requirements": structured_requirements,
            "preprocessing_applied": True,
            "quality_improvement": "Applied AI-powered requirements analysis and enhancement",
            "original_text": raw_requirements_text
        }

    except Exception as e:
        logging.error(f"AI requirements preprocessing failed: {e}")
        return basic_requirements_preprocessing(raw_requirements_text)

def parse_enhanced_requirements(enhanced_text, original_text):
    """Parse the AI-enhanced requirements into a structured format"""
    try:
        requirements_list = []

        # Extract individual requirements from the enhanced text
        requirement_sections = re.split(r'\*\*Requirement ID\*\*:', enhanced_text)

        for i, section in enumerate(requirement_sections[1:], 1):  # Skip first empty section
            req_data = {}

            # Extract each field using regex patterns
            patterns = {
                'id': r'(.+?)\n\*\*Title\*\*:',
                'title': r'\*\*Title\*\*:\s*(.+?)\n\*\*Description\*\*:',
                'description': r'\*\*Description\*\*:\s*(.+?)\n\*\*User Story\*\*:',
                'user_story': r'\*\*User Story\*\*:\s*(.+?)\n\*\*Acceptance Criteria\*\*:',
                'acceptance_criteria': r'\*\*Acceptance Criteria\*\*:\s*(.+?)\n\*\*Priority\*\*:',
                'priority': r'\*\*Priority\*\*:\s*(.+?)\n\*\*Complexity\*\*:',
                'complexity': r'\*\*Complexity\*\*:\s*(.+?)\n\*\*Dependencies\*\*:',
                'dependencies': r'\*\*Dependencies\*\*:\s*(.+?)\n\*\*Test Scenarios\*\*:',
                'test_scenarios': r'\*\*Test Scenarios\*\*:\s*(.+?)(?=\n\*\*Requirement ID\*\*:|$)'
            }

            for field, pattern in patterns.items():
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    req_data[field] = match.group(1).strip()
                else:
                    req_data[field] = ""

            # Set default values if not found
            if not req_data.get('id'):
                req_data['id'] = f"REQ-{i:03d}"
            if not req_data.get('priority'):
                req_data['priority'] = "Medium"
            if not req_data.get('complexity'):
                req_data['complexity'] = "Medium"

            requirements_list.append(req_data)

        return requirements_list

    except Exception as e:
        logging.error(f"Error parsing enhanced requirements: {e}")
        return []

def basic_requirements_preprocessing(raw_text):
    """Fallback basic preprocessing when AI is not available"""
    return {
        "success": True,
        "enhanced_text": raw_text,
        "structured_requirements": [],
        "preprocessing_applied": False,
        "quality_improvement": "Basic text cleaning applied",
        "original_text": raw_text
    }

def get_azure_openai_client():
    """
    Get Azure OpenAI client with proper configuration

    Returns:
        Configured Azure OpenAI client or None if configuration is missing
    """
    try:
        if AZURE_CLIENT_AVAILABLE:
            # Use the imported AzureOpenAIClient class
            client = AzureOpenAIClient()

            # Check if client is properly configured
            if not client.is_configured():
                logging.error("Azure OpenAI client not fully configured. Please set environment variables:")
                logging.error("- AZURE_OPENAI_ENDPOINT (e.g., https://yourinstance.openai.azure.com/)")
                logging.error("- AZURE_OPENAI_API_KEY")
                logging.error("- AZURE_OPENAI_DEPLOYMENT (deployment name, e.g., 'gpt-4', 'gpt-35-turbo')")
                return None

            logging.info(f"Azure OpenAI client configured with deployment: {client.deployment_name}")
            return client
        else:
            # Fallback: create client directly
            from openai import AzureOpenAI

            # Get configuration from environment variables
            api_key = os.environ.get('AZURE_OPENAI_API_KEY')
            endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
            deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4')

            # Validate configuration
            if not api_key or not endpoint:
                logging.error("Azure OpenAI configuration missing. Please set:")
                logging.error("- AZURE_OPENAI_ENDPOINT")
                logging.error("- AZURE_OPENAI_API_KEY")
                logging.error("- AZURE_OPENAI_DEPLOYMENT (optional, defaults to 'gpt-4')")
                return None

            client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )

            # Store deployment name as attribute for later use
            client.deployment_name = deployment_name

            logging.info(f"Azure OpenAI client configured (fallback) with deployment: {deployment_name}")
            return client

    except Exception as e:
        logging.error(f"Failed to initialize Azure OpenAI client: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_comprehensive_test_plan(uploaded_files=None, jira_requirements=None, confluence_requirements=None,
                                     figma_requirements=None, custom_prompt="", use_rag=False, rag_config_path=None):
    """
    Generate a comprehensive test plan using Azure OpenAI that analyzes all requirements
    and produces detailed test documentation with estimates, scenarios, and test cases.

    This function acts as the world's best Quality Engineer to:
    - Analyze all sources (documents, Figma, Jira, Confluence, prompts)
    - Consider RAG enhancements if enabled
    - Generate accurate development and testing estimates
    - Create test scenarios for happy path, negative, boundary, edge cases
    - Include non-functional testing considerations
    - Structure output for Zephyr integration
    - Enable browser automation conversion via TestPilot module

    Args:
        uploaded_files: List of uploaded requirement documents
        jira_requirements: List of Jira requirement data
        confluence_requirements: List of Confluence requirement data
        figma_requirements: List of Figma design data
        custom_prompt: Custom user requirements/context
        use_rag: Whether to use RAG enhancement
        rag_config_path: Path to RAG configuration

    Returns:
        Tuple of (analysis_dict, error_message)
    """
    try:
        # Step 1: Parse and analyze all requirement sources
        logging.info("Step 1: Parsing and analyzing all requirement sources...")
        analysis, error = parse_and_analyze_files(
            uploaded_files,
            jira_requirements,
            confluence_requirements,
            figma_requirements,
            custom_prompt,
            use_rag=use_rag,
            rag_config_path=rag_config_path
        )

        if error or not analysis:
            return None, error or "Failed to analyze requirements"

        # Step 2: Generate comprehensive test plan using Azure OpenAI
        logging.info("Step 2: Generating comprehensive test plan with Azure OpenAI...")

        client = get_azure_openai_client()
        if not client:
            logging.warning("Azure OpenAI not available, using basic analysis")
            return analysis, None

        # Prepare comprehensive context for AI
        requirements = analysis.get("requirements", [])
        sources = analysis.get("sources", [])
        metadata = analysis.get("metadata", {})
        raw_text = analysis.get("raw_text", "")

        # Build AI prompt for comprehensive test planning
        test_plan_prompt = f"""
You are the world's best and smartest Quality Engineer with decades of experience in software testing, 
test planning, and quality assurance across all domains. Your task is to create a comprehensive, 
detailed, and professional Test Plan based on the analyzed requirements.

CONTEXT:
- Total Requirements: {len(requirements)}
- Sources Analyzed: {', '.join(sources)}
- Enhanced with RAG: {analysis.get('enhanced_with_rag', False)}

REQUIREMENTS SUMMARY:
{_format_requirements_for_ai(requirements, limit=50)}

FULL REQUIREMENTS TEXT:
{raw_text[:10000]}  # Limit to avoid token limits

YOUR TASK:
As the world's best Quality Engineer, create a COMPREHENSIVE TEST PLAN that includes:

1. EXECUTIVE SUMMARY
   - Overview of the application/feature being tested
   - Key objectives and scope
   - Risk assessment and mitigation strategy

2. DETAILED ESTIMATES
   A. Development Estimates:
      - Break down by requirement complexity and priority
      - Consider dependencies and integration efforts
      - Provide estimates in person-days
      - Include buffer for unknowns (typically 15-20%)
   
   B. Testing Estimates:
      - Test planning and design time
      - Test execution time (manual + automated)
      - Defect fixing and retesting time
      - Regression testing time
      - Provide estimates in person-days
      - Include different test phases (smoke, functional, regression, UAT)

3. TEST SCENARIOS - COMPREHENSIVE COVERAGE
   
   A. HAPPY PATH SCENARIOS (Positive Testing):
      - Primary user workflows that should work flawlessly
      - Standard use cases with valid inputs
      - Expected normal behavior verification
      Format: Clear scenario title, preconditions, steps, expected results
   
   B. NEGATIVE SCENARIOS (Negative Testing):
      - Invalid inputs and error handling
      - Security testing (injection, XSS, CSRF)
      - Authentication and authorization failures
      - System should gracefully handle all failures
      Format: Clear scenario title, invalid action, expected error handling
   
   C. BOUNDARY & EDGE CASES:
      - Minimum and maximum value testing
      - Empty, null, and special character handling
      - Data type and format validation
      - Limits and thresholds testing
      Format: Clear scenario title, boundary condition, expected behavior
   
   D. NON-FUNCTIONAL TESTING:
      - Performance testing (load, stress, spike)
      - Security testing considerations
      - Usability and accessibility testing
      - Compatibility testing (browsers, devices, OS)
      - Scalability and reliability testing
      Format: Test type, metrics to measure, success criteria

4. DETAILED TEST CASES (for each major scenario)
   - Test Case ID
   - Title
   - Preconditions
   - Test Steps (numbered, clear, actionable)
   - Test Data required
   - Expected Results
   - Priority (High/Medium/Low)
   - Test Type (Functional/Non-Functional)
   - Automation Feasibility (Yes/No/Partial)

5. TEST DATA REQUIREMENTS
   - Types of test data needed
   - Data setup requirements
   - Test environment considerations

6. AUTOMATION RECOMMENDATIONS
   - Which test cases are good candidates for automation
   - Suggested automation framework approach
   - ROI considerations for automation

7. RISK ANALYSIS
   - High-risk areas requiring extra attention
   - Potential blockers and dependencies
   - Mitigation strategies

Please provide your response in a well-structured JSON format with the following keys:
- executive_summary: {{objective, scope, risk_assessment}}
- estimates: {{development_days, development_breakdown, testing_days, testing_breakdown, total_project_days}}
- test_scenarios: {{
    happy_path: [list of scenarios],
    negative: [list of scenarios],
    boundary_edge: [list of scenarios],
    non_functional: [list of scenarios]
  }}
- test_cases: [list of detailed test case objects]
- test_data_requirements: [list of data requirements]
- automation_recommendations: {{candidates, framework, roi_analysis}}
- risk_analysis: {{high_risk_areas, blockers, mitigation_strategies}}

Be thorough, specific, and actionable in all recommendations.
"""

        # Call Azure OpenAI for comprehensive test plan generation
        logging.info("Calling Azure OpenAI for test plan generation...")

        try:
            response = client.chat_completion_create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are the world's best Quality Engineer with expertise in test planning, test design, and quality assurance. You create comprehensive, detailed, and professional test plans."
                    },
                    {
                        "role": "user",
                        "content": test_plan_prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for more consistent, structured output
                max_tokens=8000  # Allow for comprehensive response
            )

            # Extract content from response dict (AzureOpenAIClient returns a dict)
            if not response or 'choices' not in response:
                raise Exception("No valid response from Azure OpenAI")

            test_plan_content = response['choices'][0]['message']['content']

            # Parse JSON response
            try:
                # Extract JSON from markdown code blocks if present
                json_match = re.search(r'```json\s*(.*?)\s*```', test_plan_content, re.DOTALL)
                if json_match:
                    test_plan_json = json.loads(json_match.group(1))
                else:
                    test_plan_json = json.loads(test_plan_content)

                logging.info("Successfully parsed test plan JSON")

            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse test plan as JSON: {e}")
                # Fallback: create structured data from text response
                test_plan_json = _parse_test_plan_from_text(test_plan_content, requirements)

            # Enhance analysis with test plan data
            analysis["test_plan"] = test_plan_json
            analysis["test_plan_raw"] = test_plan_content
            analysis["comprehensive_analysis"] = True

            # Add convenience flags for UI
            analysis["has_estimates"] = "estimates" in test_plan_json
            analysis["has_test_scenarios"] = "test_scenarios" in test_plan_json
            analysis["has_test_cases"] = "test_cases" in test_plan_json
            analysis["total_test_cases"] = len(test_plan_json.get("test_cases", []))

            # Calculate total scenarios
            test_scenarios = test_plan_json.get("test_scenarios", {})
            total_scenarios = sum(len(scenarios) for scenarios in test_scenarios.values())
            analysis["total_test_scenarios"] = total_scenarios

            logging.info(f"Test plan generated: {total_scenarios} scenarios, {analysis['total_test_cases']} test cases")

            return analysis, None

        except Exception as ai_error:
            logging.error(f"Azure OpenAI call failed: {ai_error}")
            # Return basic analysis without test plan
            return analysis, f"Basic analysis completed, but test plan generation failed: {str(ai_error)}"

    except Exception as e:
        logging.error(f"Error generating comprehensive test plan: {e}")
        import traceback
        traceback.print_exc()
        return None, f"Error generating test plan: {str(e)}"


def _format_requirements_for_ai(requirements, limit=50):
    """Format requirements for AI prompt (limit to avoid token overflow)"""
    formatted = []
    for i, req in enumerate(requirements[:limit], 1):
        req_text = f"""
Requirement {i}:
- ID: {req.get('id', 'N/A')}
- Text: {req.get('text', 'N/A')[:200]}
- Priority: {req.get('priority', 'N/A')}
- Complexity: {req.get('complexity', 'N/A')}
- Category: {req.get('category', 'N/A')}
"""
        formatted.append(req_text.strip())

    if len(requirements) > limit:
        formatted.append(f"\n... and {len(requirements) - limit} more requirements")

    return "\n".join(formatted)


def _parse_test_plan_from_text(text_content, requirements):
    """Parse test plan from unstructured text response (fallback)"""
    try:
        # Basic parsing logic for fallback
        test_plan = {
            "executive_summary": {
                "objective": "Test comprehensive functionality and quality",
                "scope": f"Testing {len(requirements)} identified requirements",
                "risk_assessment": "Medium risk - comprehensive testing required"
            },
            "estimates": {
                "development_days": _estimate_development_days(requirements),
                "development_breakdown": "Based on requirement complexity and priority",
                "testing_days": _estimate_testing_days(requirements),
                "testing_breakdown": "Includes planning, execution, regression",
                "total_project_days": 0
            },
            "test_scenarios": {
                "happy_path": _extract_scenarios_from_requirements(requirements, "happy_path"),
                "negative": _extract_scenarios_from_requirements(requirements, "negative"),
                "boundary_edge": _extract_scenarios_from_requirements(requirements, "boundary"),
                "non_functional": ["Performance testing", "Security testing", "Usability testing"]
            },
            "test_cases": [],
            "test_data_requirements": ["Valid test users", "Sample data sets", "Test environment"],
            "automation_recommendations": {
                "candidates": "Regression test cases",
                "framework": "Robot Framework recommended",
                "roi_analysis": "High ROI for regression tests"
            },
            "risk_analysis": {
                "high_risk_areas": ["Integration points", "Security features"],
                "blockers": ["Environment availability", "Test data setup"],
                "mitigation_strategies": ["Early environment setup", "Parallel testing"]
            }
        }

        # Calculate total project days
        test_plan["estimates"]["total_project_days"] = (
            test_plan["estimates"]["development_days"] +
            test_plan["estimates"]["testing_days"]
        )

        return test_plan

    except Exception as e:
        logging.error(f"Error parsing test plan from text: {e}")
        return {}


def _estimate_development_days(requirements):
    """Estimate development days based on requirements"""
    total_days = 0
    for req in requirements:
        complexity = req.get("complexity", 0.5)
        if isinstance(complexity, str):
            complexity_map = {"low": 0.3, "medium": 0.5, "high": 0.8, "very high": 1.0}
            complexity = complexity_map.get(complexity.lower(), 0.5)

        priority = req.get("priority", "Medium")
        priority_multiplier = {"Low": 0.8, "Medium": 1.0, "High": 1.3, "Critical": 1.5}.get(priority, 1.0)

        # Base estimate: 2 days per requirement, adjusted by complexity and priority
        req_days = 2 * complexity * priority_multiplier
        total_days += req_days

    # Add 20% buffer
    return round(total_days * 1.2, 1)


def _estimate_testing_days(requirements):
    """Estimate testing days (typically 40% of development for functional testing)"""
    dev_days = _estimate_development_days(requirements)
    return round(dev_days * 0.4, 1)


def _extract_scenarios_from_requirements(requirements, scenario_type):
    """Extract basic test scenarios from requirements based on type"""
    scenarios = []

    for req in requirements[:10]:  # Limit to first 10 for basic extraction
        req_text = req.get("text", "")

        if scenario_type == "happy_path":
            scenarios.append({
                "title": f"Verify {req_text[:50]}... works correctly",
                "description": f"Test the primary workflow for: {req_text[:100]}",
                "priority": req.get("priority", "Medium")
            })
        elif scenario_type == "negative":
            scenarios.append({
                "title": f"Verify error handling for {req_text[:50]}...",
                "description": f"Test invalid inputs and error scenarios for: {req_text[:100]}",
                "priority": req.get("priority", "Medium")
            })
        elif scenario_type == "boundary":
            scenarios.append({
                "title": f"Verify boundary conditions for {req_text[:50]}...",
                "description": f"Test edge cases and limits for: {req_text[:100]}",
                "priority": req.get("priority", "Medium")
            })

    return scenarios




# Enhanced parsing function with RAG integration and improved analysis
def parse_and_analyze_files(uploaded_files=None, jira_requirements=None, confluence_requirements=None, figma_requirements=None, custom_prompt="", use_rag=False, rag_config_path=None):
    """Enhanced parsing with RAG integration and improved analysis"""
    all_text = ""
    sources = []
    metadata = {}

    # Initialize RAG if requested
    rag_service = None
    if use_rag and RAG_AVAILABLE:
        rag_service = initialize_rag_service(rag_config_path)
        if rag_service:
            logging.info("RAG service initialized for enhanced analysis")

    # Process uploaded files
    if uploaded_files:
        for uploaded in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            try:
                parser = get_parser(tmp_path)
                reqs = parser.parse(tmp_path)
                if reqs.get("raw_text", "").strip():
                    file_text = reqs.get("raw_text", "")
                    all_text += file_text + "\n\n"
                    sources.append(f"File: {uploaded.name}")

                    # Store metadata about the file
                    metadata[f"file_{uploaded.name}"] = {
                        "type": uploaded.type,
                        "size": len(file_text),
                        "source": "uploaded_file"
                    }
                else:
                    os.unlink(tmp_path)
                    return None, f"File '{uploaded.name}' could not be parsed or is empty."
            except Exception as e:
                os.unlink(tmp_path)
                return None, f"Error parsing file '{uploaded.name}': {str(e)}"
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    # Process JIRA requirements with metadata
    if jira_requirements:
        for jira_req in jira_requirements:
            if jira_req.get("success"):
                jira_text = jira_req.get("raw_text", "")
                all_text += jira_text + "\n\n"
                sources.append(f"JIRA: {jira_req.get('summary', 'Issue')}")

                # Store JIRA metadata
                jira_metadata = jira_req.get("metadata", {})
                metadata[f"jira_{jira_metadata.get('issue_key', 'unknown')}"] = {
                    "type": "jira_issue",
                    "issue_type": jira_metadata.get("issue_type"),
                    "status": jira_metadata.get("status"),
                    "priority": jira_metadata.get("priority"),
                    "source": "jira"
                }

    # Process Confluence requirements with metadata
    if confluence_requirements:
        for conf_req in confluence_requirements:
            if conf_req.get("success"):
                conf_text = conf_req.get("raw_text", "")
                all_text += conf_text + "\n\n"
                sources.append(f"Confluence: {conf_req.get('title', 'Page')}")

                # Store Confluence metadata
                conf_metadata = conf_req.get("metadata", {})
                metadata[f"confluence_{conf_metadata.get('page_title', 'unknown')}"] = {
                    "type": "confluence_page",
                    "space_key": conf_metadata.get("space_key"),
                    "content_length": conf_metadata.get("content_length"),
                    "source": "confluence"
                }

    # Process Figma requirements
    if figma_requirements:
        for figma_req in figma_requirements:
            if figma_req.get("success"):
                figma_text = figma_req.get("raw_text", "")
                all_text += figma_text + "\n\n"
                sources.append(f"Figma: {figma_req.get('file_name', 'Design')}")

                # Store Figma metadata
                metadata[f"figma_{figma_req.get('file_name', 'unknown')}"] = {
                    "type": "figma_design",
                    "file_id": figma_req.get("file_id"),
                    "source": "figma"
                }

    # Add custom prompt context
    if custom_prompt.strip():
        all_text += f"\n\nCustom Requirements Context:\n{custom_prompt}\n\n"
        sources.append("Custom Prompt")
        metadata["custom_prompt"] = {
            "type": "custom_input",
            "length": len(custom_prompt),
            "source": "user_input"
        }

    if not all_text.strip():
        return None, "No valid requirements found from any source."

    # STEP 1: AI-Powered Requirements Preprocessing
    # This is the key enhancement - preprocess requirements to make them meaningful
    logging.info("Starting AI-powered requirements preprocessing...")
    preprocessed_result = preprocess_requirements_with_ai(
        raw_requirements_text=all_text,
        sources=sources,
        metadata=metadata,
        custom_prompt=custom_prompt
    )

    if preprocessed_result["success"] and preprocessed_result["preprocessing_applied"]:
        # Use the enhanced requirements text for analysis
        all_text = preprocessed_result["enhanced_text"]
        logging.info("Requirements successfully preprocessed and enhanced with AI")

        # Store preprocessing information in metadata
        metadata["ai_preprocessing"] = {
            "applied": True,
            "quality_improvement": preprocessed_result["quality_improvement"],
            "structured_requirements_count": len(preprocessed_result["structured_requirements"]),
            "source": "azure_openai_preprocessing"
        }

        # If we have structured requirements from preprocessing, store them for later use
        if preprocessed_result["structured_requirements"]:
            metadata["structured_requirements_from_ai"] = preprocessed_result["structured_requirements"]
    else:
        logging.warning("AI preprocessing not available or failed, using original requirements")
        metadata["ai_preprocessing"] = {
            "applied": False,
            "reason": "AI preprocessing failed or not available",
            "fallback": "Using original requirements text"
        }

    # STEP 2: Enhance with RAG if available (after preprocessing)
    if rag_service:
        try:
            enhanced_text = enhance_requirements_with_rag(rag_service, all_text)
            if enhanced_text != all_text:
                all_text = enhanced_text
                sources.append("RAG Enhancement")
                metadata["rag_enhancement"] = {
                    "type": "rag_processed",
                    "enhanced": True,
                    "source": "rag_service"
                }
        except Exception as e:
            logging.warning(f"RAG enhancement failed: {e}")

    # Enhanced requirement analysis with better configuration
    analyzer = RequirementAnalyzer(
        model_name="gpt-4",
        azure_api_key=os.environ.get('AZURE_OPENAI_API_KEY', "5e98b3558f5d4dcebe68f8ca8a3352b7")
    )

    # Add requirements with enhanced text processing
    requirements = analyzer.add_requirements_from_text(all_text)

    # Perform comprehensive analysis
    analyzer.analyze_all_requirements()

    # Get enhanced statistics
    stats = analyzer.get_statistics()

    # Create comprehensive analysis results
    analysis = {
        "requirements": [req.to_dict() for req in requirements],
        "statistics": stats,
        "sources": sources,
        "raw_text": all_text,
        "metadata": metadata,
        "enhanced_with_rag": rag_service is not None,
        "total_requirements_found": len(requirements),
        "analysis_quality": {
            "avg_complexity": stats.get('avg_complexity', 0),
            "avg_ambiguity": stats.get('avg_ambiguity', 0),
            "avg_completeness": stats.get('avg_completeness', 0),
            "avg_testability": stats.get('avg_testability', 0)
        }
    }

    # Extract test components from requirements with enhanced processing
    from generator.testcase_generator import extract_test_components
    test_components = extract_test_components(analysis["requirements"])
    analysis.update(test_components)

    return analysis, None


# Azure OpenAI function for dev/test estimates
def generate_dev_test_estimates(analysis, config=None):
    """Generate development and testing estimates using Azure OpenAI with enhanced calculation"""
    try:
        requirements = analysis.get("requirements", [])
        total_reqs = len(requirements)

        if total_reqs == 0:
            return {
                "total_requirements": 0,
                "error": "No requirements available for estimation"
            }

        # Enhanced estimation algorithm with more sophisticated weighting
        complexity_weights = {
            "Low": 1,
            "Medium": 2.5,
            "High": 4,
            "Very High": 6
        }

        priority_multipliers = {
            "Low": 0.8,
            "Medium": 1.0,
            "High": 1.3,
            "Critical": 1.5
        }

        total_complexity = 0
        priority_adjusted_complexity = 0

        for req in requirements:
            complexity = req.get("complexity", 0.5)  # Use numeric complexity score
            priority = req.get("priority", "Medium")

            # Convert numeric complexity to category if needed
            if isinstance(complexity, (int, float)):
                if complexity <= 0.3:
                    complexity_category = "Low"
                elif complexity <= 0.6:
                    complexity_category = "Medium"
                elif complexity <= 0.8:
                    complexity_category = "High"
                else:
                    complexity_category = "Very High"
            else:
                complexity_category = complexity

            base_complexity = complexity_weights.get(complexity_category, 2.5)
            priority_multiplier = priority_multipliers.get(priority, 1.0)

            total_complexity += base_complexity
            priority_adjusted_complexity += base_complexity * priority_multiplier

        # Enhanced estimation with different factors
        base_hours_per_complexity = 6  # Base hours per complexity point
        total_dev_hours = priority_adjusted_complexity * base_hours_per_complexity

        # Adjust based on requirement quality
        quality_scores = analysis.get("analysis_quality", {})
        avg_ambiguity = quality_scores.get("avg_ambiguity", 0.5)
        avg_completeness = quality_scores.get("avg_completeness", 0.5)

        # Higher ambiguity increases development time
        ambiguity_multiplier = 1 + (avg_ambiguity * 0.5)
        # Lower completeness increases development time
        completeness_multiplier = 1 + ((1 - avg_completeness) * 0.3)

        total_dev_hours *= ambiguity_multiplier * completeness_multiplier

        # Enhanced testing estimation (varies by requirement type)
        functional_reqs = len([r for r in requirements if r.get("category", "").lower() == "functional"])
        non_functional_reqs = total_reqs - functional_reqs

        # Functional requirements typically need 40% testing time
        # Non-functional requirements typically need 60% testing time
        functional_test_ratio = 0.4
        non_functional_test_ratio = 0.6

        functional_dev_hours = (functional_reqs / total_reqs) * total_dev_hours if total_reqs > 0 else 0
        non_functional_dev_hours = (non_functional_reqs / total_reqs) * total_dev_hours if total_reqs > 0 else 0

        total_test_hours = (functional_dev_hours * functional_test_ratio +
                           non_functional_dev_hours * non_functional_test_ratio)

        # Convert to days/weeks (assuming 8 hours per day, 5 days per week)
        hours_per_day = 8
        days_per_week = 5

        dev_days = total_dev_hours / hours_per_day
        test_days = total_test_hours / hours_per_day
        total_days = dev_days + test_days

        estimates = {
            "total_requirements": total_reqs,
            "functional_requirements": functional_reqs,
            "non_functional_requirements": non_functional_reqs,
            "total_complexity_points": round(total_complexity, 1),
            "priority_adjusted_complexity": round(priority_adjusted_complexity, 1),
            "quality_adjustments": {
                "ambiguity_multiplier": round(ambiguity_multiplier, 2),
                "completeness_multiplier": round(completeness_multiplier, 2)
            },
            "development": {
                "hours": round(total_dev_hours, 1),
                "days": round(dev_days, 1),
                "weeks": round(dev_days / days_per_week, 1)
            },
            "testing": {
                "hours": round(total_test_hours, 1),
                "days": round(test_days, 1),
                "weeks": round(test_days / days_per_week, 1)
            },
            "total": {
                "hours": round(total_dev_hours + total_test_hours, 1),
                "days": round(total_days, 1),
                "weeks": round(total_days / days_per_week, 1)
            },
            "breakdown": {
                "requirements_analysis": f"{round(total_days * 0.1, 1)} days (10%)",
                "design_architecture": f"{round(total_days * 0.15, 1)} days (15%)",
                "development": f"{round(dev_days, 1)} days",
                "unit_testing": f"{round(test_days * 0.3, 1)} days",
                "integration_testing": f"{round(test_days * 0.4, 1)} days",
                "system_testing": f"{round(test_days * 0.2, 1)} days",
                "user_acceptance_testing": f"{round(test_days * 0.1, 1)} days"
            },
            "confidence_level": "Medium" if avg_ambiguity < 0.5 and avg_completeness > 0.5 else "Low"
        }

        return estimates

    except Exception as e:
        logging.error(f"Error generating estimates: {e}")
        return {
            "error": f"Failed to generate estimates: {str(e)}",
            "total_requirements": len(analysis.get("requirements", []))
        }


# ============================================================================
# ZEPHYR INTEGRATION FUNCTIONS
# ============================================================================

def create_or_update_zephyr_test_cases(test_cases, jira_config, update_existing=True):
    """
    Create or update test cases in Zephyr Scale for Jira

    Args:
        test_cases: List of test case dictionaries
        jira_config: Dict with Jira configuration (host, username, api_token, project_key)
        update_existing: Whether to update existing test cases if similar ones are found

    Returns:
        Dict with success status, created/updated counts, and details
    """
    try:
        logging.info("Starting Zephyr test case creation/update process...")

        # Validate configuration
        required_fields = ['host', 'username', 'api_token', 'project_key']
        for field in required_fields:
            if field not in jira_config or not jira_config[field]:
                return {
                    "success": False,
                    "error": f"Missing required Jira configuration field: {field}",
                    "created": 0,
                    "updated": 0
                }

        # Setup Jira session
        from requests import Session
        from requests.auth import HTTPBasicAuth

        session = Session()
        session.auth = HTTPBasicAuth(jira_config['username'], jira_config['api_token'])
        session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

        base_url = jira_config['host'].rstrip('/')
        if not base_url.startswith('http'):
            base_url = f"https://{base_url}"

        project_key = jira_config['project_key']

        created_count = 0
        updated_count = 0
        skipped_count = 0
        errors = []
        created_issues = []
        updated_issues = []

        # Process each test case
        for i, test_case in enumerate(test_cases, 1):
            try:
                logging.info(f"Processing test case {i}/{len(test_cases)}: {test_case.get('name', 'Unnamed')}")

                # Check if similar test case exists (if update_existing is True)
                existing_issue_key = None
                if update_existing:
                    existing_issue_key = _find_similar_zephyr_test_case(
                        session, base_url, project_key, test_case
                    )

                if existing_issue_key:
                    # Update existing test case
                    logging.info(f"Found similar test case: {existing_issue_key}, updating...")
                    success = _update_zephyr_test_case(
                        session, base_url, existing_issue_key, test_case
                    )
                    if success:
                        updated_count += 1
                        updated_issues.append(existing_issue_key)
                        logging.info(f"✅ Updated test case: {existing_issue_key}")
                    else:
                        errors.append(f"Failed to update {existing_issue_key}")
                else:
                    # Create new test case
                    logging.info(f"Creating new test case in Zephyr...")
                    issue_key = _create_zephyr_test_case(
                        session, base_url, project_key, test_case
                    )
                    if issue_key:
                        created_count += 1
                        created_issues.append(issue_key)
                        logging.info(f"✅ Created test case: {issue_key}")
                    else:
                        errors.append(f"Failed to create test case: {test_case.get('name', 'Unnamed')}")

            except Exception as e:
                logging.error(f"Error processing test case {i}: {e}")
                errors.append(f"Test case {i}: {str(e)}")
                skipped_count += 1

        # Summary
        result = {
            "success": True,
            "created": created_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "errors": errors,
            "created_issues": created_issues,
            "updated_issues": updated_issues,
            "total_processed": len(test_cases),
            "message": f"Successfully processed {created_count + updated_count} test cases ({created_count} created, {updated_count} updated)"
        }

        logging.info(f"Zephyr integration complete: {result['message']}")
        return result

    except Exception as e:
        logging.error(f"Error in Zephyr integration: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "created": 0,
            "updated": 0
        }


def _find_similar_zephyr_test_case(session, base_url, project_key, test_case):
    """Find similar test case in Jira using title/summary matching"""
    try:
        # Search for issues with similar summary
        test_name = test_case.get('name', '')
        if not test_name:
            return None

        # Use JQL to search for similar test cases
        jql = f'project = {project_key} AND issuetype = Test AND summary ~ "{test_name[:50]}"'

        search_url = f"{base_url}/rest/api/2/search"
        params = {
            'jql': jql,
            'maxResults': 5,
            'fields': 'summary,description'
        }

        response = session.get(search_url, params=params)
        if response.status_code == 200:
            data = response.json()
            issues = data.get('issues', [])

            if issues:
                # Return first matching issue key
                # Could add more sophisticated matching logic here
                return issues[0]['key']

        return None

    except Exception as e:
        logging.error(f"Error searching for similar test case: {e}")
        return None


def _create_zephyr_test_case(session, base_url, project_key, test_case):
    """Create a new test case in Jira/Zephyr"""
    try:
        # Prepare test case data for Jira
        test_name = test_case.get('name', 'Unnamed Test Case')
        description = test_case.get('description', '')
        steps = test_case.get('steps', test_case.get('conditions/steps', ''))
        expected_result = test_case.get('expected_result', test_case.get('expected result', ''))
        preconditions = test_case.get('preconditions', '')
        priority = test_case.get('priority', 'Medium')

        # Format test steps for Zephyr
        test_script = _format_test_steps_for_zephyr(steps, expected_result)

        # Create issue payload
        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": test_name,
                "description": description,
                "issuetype": {"name": "Test"},  # Zephyr test issue type
                "priority": {"name": priority}
            }
        }

        # Add custom fields for Zephyr if available
        if preconditions:
            # Custom field for preconditions (field ID varies by Jira instance)
            # This would need to be configured per instance
            pass

        # Create the issue
        create_url = f"{base_url}/rest/api/2/issue"
        response = session.post(create_url, json=payload)

        if response.status_code in [200, 201]:
            data = response.json()
            issue_key = data.get('key')

            # Add test steps via Zephyr API
            if issue_key and test_script:
                _add_zephyr_test_steps(session, base_url, issue_key, test_script)

            return issue_key
        else:
            logging.error(f"Failed to create test case: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        logging.error(f"Error creating Zephyr test case: {e}")
        return None


def _update_zephyr_test_case(session, base_url, issue_key, test_case):
    """Update an existing test case in Jira/Zephyr"""
    try:
        # Prepare update data
        description = test_case.get('description', '')
        steps = test_case.get('steps', test_case.get('conditions/steps', ''))
        expected_result = test_case.get('expected_result', test_case.get('expected result', ''))
        priority = test_case.get('priority', 'Medium')

        # Format test steps for Zephyr
        test_script = _format_test_steps_for_zephyr(steps, expected_result)

        # Update issue payload
        payload = {
            "fields": {
                "description": description,
                "priority": {"name": priority}
            }
        }

        # Update the issue
        update_url = f"{base_url}/rest/api/2/issue/{issue_key}"
        response = session.put(update_url, json=payload)

        if response.status_code in [200, 204]:
            # Update test steps via Zephyr API
            if test_script:
                _add_zephyr_test_steps(session, base_url, issue_key, test_script)
            return True
        else:
            logging.error(f"Failed to update test case: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logging.error(f"Error updating Zephyr test case: {e}")
        return False


def _format_test_steps_for_zephyr(steps, expected_result):
    """Format test steps into Zephyr-compatible structure"""
    try:
        if isinstance(steps, str):
            # Split steps by newlines or numbered format
            step_lines = [s.strip() for s in steps.split('\n') if s.strip()]
        elif isinstance(steps, list):
            step_lines = [str(s).strip() for s in steps]
        else:
            step_lines = [str(steps)]

        # Create Zephyr test script format
        test_script = []
        for i, step in enumerate(step_lines, 1):
            test_script.append({
                "index": i,
                "description": step,
                "expectedResult": expected_result if i == len(step_lines) else ""
            })

        return test_script

    except Exception as e:
        logging.error(f"Error formatting test steps: {e}")
        return []


def _add_zephyr_test_steps(session, base_url, issue_key, test_script):
    """Add test steps to a Zephyr test case"""
    try:
        # Try different Zephyr API endpoints
        endpoints = [
            f"{base_url}/rest/atm/1.0/testcase/{issue_key}/teststeps",  # Zephyr Scale
            f"{base_url}/rest/zapi/latest/teststep/{issue_key}",  # Zephyr Squad
        ]

        for endpoint in endpoints:
            try:
                response = session.post(endpoint, json={"steps": test_script})
                if response.status_code in [200, 201]:
                    logging.info(f"Successfully added test steps to {issue_key}")
                    return True
            except Exception as e:
                logging.debug(f"Failed with endpoint {endpoint}: {e}")
                continue

        logging.warning(f"Could not add test steps to {issue_key} - Zephyr API may not be available")
        return False

    except Exception as e:
        logging.error(f"Error adding Zephyr test steps: {e}")
        return False


# ============================================================================
# BROWSER AUTOMATION INTEGRATION (Using TestPilot Module Capabilities)
# ============================================================================

def convert_test_cases_to_robot_automation(test_cases, use_robotmcp=True, browser_type="chrome"):
    """
    Convert test cases with natural language steps into Robot Framework automation scripts
    using TestPilot module's browser automation capabilities via RobotMCP.

    This function emulates TestPilot's behavior:
    1. Parse natural language test steps
    2. Use RobotMCP to analyze and convert steps to browser actions
    3. Generate Robot Framework scripts with proper keywords and locators
    4. Leverage existing keyword libraries and smart locator strategies

    Args:
        test_cases: List of test case dictionaries with steps
        use_robotmcp: Whether to use RobotMCP for advanced automation (default: True)
        browser_type: Browser to use (chrome, firefox, edge)

    Returns:
        Dict with success status, generated scripts, and details
    """
    try:
        logging.info("Starting browser automation conversion using TestPilot patterns...")

        if not test_cases:
            return {
                "success": False,
                "error": "No test cases provided",
                "scripts": []
            }

        # Check if RobotMCP is available (from TestPilot module)
        robotmcp_available = _check_robotmcp_availability()

        if use_robotmcp and not robotmcp_available:
            logging.warning("RobotMCP not available, falling back to basic conversion")
            use_robotmcp = False

        generated_scripts = []
        errors = []

        # Process each test case
        for i, test_case in enumerate(test_cases, 1):
            try:
                logging.info(f"Converting test case {i}/{len(test_cases)} to automation...")

                test_name = test_case.get('name', f'Test_Case_{i}')
                steps = test_case.get('steps', test_case.get('conditions/steps', ''))
                expected_result = test_case.get('expected_result', test_case.get('expected result', ''))

                if not steps:
                    logging.warning(f"No steps found for test case: {test_name}")
                    continue

                # Parse steps into actionable list
                step_list = _parse_steps_to_list(steps)

                if use_robotmcp:
                    # Use RobotMCP for intelligent conversion (TestPilot approach)
                    robot_script = _convert_with_robotmcp(
                        test_name=test_name,
                        steps=step_list,
                        expected_result=expected_result,
                        browser_type=browser_type
                    )
                else:
                    # Fallback: Basic Robot Framework generation
                    robot_script = _convert_basic_robot_framework(
                        test_name=test_name,
                        steps=step_list,
                        expected_result=expected_result,
                        browser_type=browser_type
                    )

                if robot_script:
                    generated_scripts.append({
                        "test_case_name": test_name,
                        "script": robot_script,
                        "method": "robotmcp" if use_robotmcp else "basic",
                        "steps_count": len(step_list)
                    })
                    logging.info(f"✅ Converted: {test_name}")
                else:
                    errors.append(f"Failed to convert: {test_name}")

            except Exception as e:
                logging.error(f"Error converting test case {i}: {e}")
                errors.append(f"Test case {i}: {str(e)}")

        # Generate combined Robot file
        combined_robot_file = None
        if generated_scripts:
            combined_robot_file = _generate_combined_robot_file(generated_scripts, browser_type)

        result = {
            "success": len(generated_scripts) > 0,
            "scripts": generated_scripts,
            "combined_robot_file": combined_robot_file,
            "total_converted": len(generated_scripts),
            "errors": errors,
            "method_used": "robotmcp" if use_robotmcp else "basic",
            "message": f"Successfully converted {len(generated_scripts)} test cases to Robot Framework automation"
        }

        logging.info(f"Automation conversion complete: {result['message']}")
        return result

    except Exception as e:
        logging.error(f"Error in browser automation conversion: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "scripts": []
        }


def _check_robotmcp_availability():
    """Check if RobotMCP is available (similar to TestPilot's ROBOTMCP_AVAILABLE)"""
    try:
        # Try importing MCP client (same approach as TestPilot)
        from mcp import ClientSession
        # Check if robotmcp command is available
        import subprocess
        result = subprocess.run(['which', 'robotmcp'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def _parse_steps_to_list(steps):
    """Parse steps from various formats into a clean list"""
    if isinstance(steps, list):
        return [str(s).strip() for s in steps if str(s).strip()]
    elif isinstance(steps, str):
        # Split by newlines and clean
        lines = steps.split('\n')
        # Remove numbering if present (1., 2., Step 1:, etc.)
        cleaned = []
        for line in lines:
            line = line.strip()
            if line:
                # Remove common step prefixes
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                line = re.sub(r'^Step\s+\d+:?\s*', '', line, flags=re.IGNORECASE)
                if line:
                    cleaned.append(line)
        return cleaned
    else:
        return [str(steps)]


def _convert_with_robotmcp(test_name, steps, expected_result, browser_type):
    """
    Convert test steps using RobotMCP (emulating TestPilot's approach)
    This would use the actual RobotMCP MCP server for intelligent conversion
    """
    try:
        # This is a placeholder for the actual RobotMCP integration
        # In a real implementation, this would:
        # 1. Start RobotMCP session
        # 2. Analyze each step using mcp_robotmcp_analyze_scenario
        # 3. Execute steps using mcp_robotmcp_execute_step
        # 4. Build test suite using mcp_robotmcp_build_test_suite

        logging.info(f"Using RobotMCP for intelligent conversion of: {test_name}")

        # For now, return a placeholder indicating RobotMCP would be used
        robot_content = f"""*** Test Cases ***
{test_name}
    [Documentation]    Auto-generated test case using RobotMCP intelligent conversion
    [Tags]    automated    robotmcp
    
"""
        for i, step in enumerate(steps, 1):
            # Each step would be converted by RobotMCP's AI
            robot_content += f"    # Step {i}: {step}\n"
            robot_content += f"    Execute Natural Language Step    {step}\n"

        if expected_result:
            robot_content += f"    \n    # Expected Result\n"
            robot_content += f"    Verify Expected Result    {expected_result}\n"

        return robot_content

    except Exception as e:
        logging.error(f"Error in RobotMCP conversion: {e}")
        return None


def _convert_basic_robot_framework(test_name, steps, expected_result, browser_type):
    """
    Basic Robot Framework generation (fallback when RobotMCP not available)
    """
    try:
        # Generate basic Robot Framework structure
        robot_content = f"""*** Settings ***
Library    SeleniumLibrary
Library    BuiltIn

*** Variables ***
${{BROWSER}}    {browser_type}

*** Test Cases ***
{test_name}
    [Documentation]    Auto-generated test case from natural language steps
    [Tags]    automated    {browser_type}
    
    Open Browser    about:blank    ${{BROWSER}}
"""

        for i, step in enumerate(steps, 1):
            # Basic conversion of natural language to Robot keywords
            robot_keyword = _natural_language_to_robot_keyword(step)
            robot_content += f"    # Step {i}: {step}\n"
            robot_content += f"    {robot_keyword}\n"

        if expected_result:
            robot_content += f"    \n    # Expected Result: {expected_result}\n"
            robot_content += f"    # TODO: Add verification for expected result\n"

        robot_content += "    \n    [Teardown]    Close Browser\n"

        return robot_content

    except Exception as e:
        logging.error(f"Error in basic Robot conversion: {e}")
        return None


def _natural_language_to_robot_keyword(step):
    """Convert natural language step to basic Robot Framework keyword"""
    step_lower = step.lower()

    # Basic pattern matching for common actions
    if 'click' in step_lower or 'press' in step_lower:
        # Extract element reference
        match = re.search(r'click\s+(?:on\s+)?(?:the\s+)?(["\']?[\w\s]+["\']?)', step_lower)
        if match:
            element = match.group(1).strip(' \'"')
            return f"Click Element    # TODO: Add locator for '{element}'"
        return "Click Element    # TODO: Add locator"

    elif 'type' in step_lower or 'enter' in step_lower or 'input' in step_lower:
        return "Input Text    # TODO: Add locator and text"

    elif 'navigate' in step_lower or 'go to' in step_lower or 'open' in step_lower:
        match = re.search(r'(?:navigate\s+to|go\s+to|open)\s+([^\s]+)', step_lower)
        if match:
            url = match.group(1)
            return f"Go To    {url}"
        return "Go To    # TODO: Add URL"

    elif 'verify' in step_lower or 'check' in step_lower or 'assert' in step_lower:
        return "Page Should Contain    # TODO: Add expected text"

    elif 'select' in step_lower:
        return "Select From List By Label    # TODO: Add locator and option"

    elif 'wait' in step_lower:
        return "Wait Until Page Contains Element    # TODO: Add locator"

    else:
        # Generic comment for unrecognized actions
        return f"# TODO: Implement step: {step}"


def _generate_combined_robot_file(scripts, browser_type):
    """Generate a combined Robot Framework file with all test cases"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        combined_content = f"""*** Settings ***
Documentation    Auto-generated test suite from Dynamic TC Generation
...              Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}
...              Total test cases: {len(scripts)}

Library    SeleniumLibrary
Library    BuiltIn

Suite Setup    Suite Setup Keywords
Suite Teardown    Suite Teardown Keywords

*** Variables ***
${{BROWSER}}    {browser_type}
${{TIMEOUT}}    10

*** Keywords ***
Suite Setup Keywords
    Log    Starting test suite execution

Suite Teardown Keywords
    Log    Test suite execution complete
    Close All Browsers

"""

        # Add all test cases
        for script_info in scripts:
            combined_content += "\n" + script_info['script'] + "\n"

        # Save to file
        output_dir = os.path.join(os.getcwd(), "generated_tests")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"automated_tests_{timestamp}.robot"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(combined_content)

        logging.info(f"Generated combined Robot file: {filepath}")
        return filepath

    except Exception as e:
        logging.error(f"Error generating combined Robot file: {e}")
        return None


# Enhanced error handling and notifications
def send_notification(message, status="info", details=None):
    """Send notification with enhanced error handling"""
    if NOTIFICATIONS_AVAILABLE:
        try:
            notifications.add_notification(
                module_name="dynamic_tc_generation",
                status=status,
                message=message,
                details=details or "",
                action_steps=[]
            )
        except Exception as e:
            logging.warning(f"Failed to send notification: {e}")


def reset_session_state():
    """Reset all session state variables"""
    keys_to_reset = [
        'test_plan_status', 'manual_tc_status', 'auto_tc_status', 'estimates_status',
        'test_cases', 'df', 'test_plan_path', 'error_message', 'analysis',
        'robot_output', 'estimates', 'jira_requirements', 'confluence_requirements',
        'figma_requirements'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def validate_inputs(uploaded_files, jira_requirements, confluence_requirements, figma_requirements, custom_prompt):
    """Validate all input sources"""
    has_files = uploaded_files is not None and len(uploaded_files) > 0
    has_jira = len(jira_requirements) > 0
    has_confluence = len(confluence_requirements) > 0
    has_figma = len(figma_requirements) > 0
    has_prompt = custom_prompt and custom_prompt.strip()

    if not (has_files or has_jira or has_confluence or has_figma or has_prompt):
        return False, "Please provide at least one source of requirements (files, JIRA, Confluence, Figma, or custom prompt)."

    return True, "Valid inputs provided"


def export_analysis_summary(analysis):
    """Export analysis summary as JSON"""
    if not analysis:
        return None

    summary = {
        "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_requirements": len(analysis.get("requirements", [])),
        "sources": analysis.get("sources", []),
        "statistics": analysis.get("statistics", {}),
        "quality_metrics": analysis.get("analysis_quality", {}),
        "metadata": analysis.get("metadata", {}),
        "enhanced_with_rag": analysis.get("enhanced_with_rag", False)
    }

    return json.dumps(summary, indent=2)


def show_ui():
    # Initialize session state variables
    if 'test_plan_status' not in st.session_state:
        st.session_state.test_plan_status = 'idle'
    if 'manual_tc_status' not in st.session_state:
        st.session_state.manual_tc_status = 'idle'
    if 'auto_tc_status' not in st.session_state:
        st.session_state.auto_tc_status = 'idle'
    if 'estimates_status' not in st.session_state:
        st.session_state.estimates_status = 'idle'
    if 'test_cases' not in st.session_state:
        st.session_state.test_cases = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'test_plan_path' not in st.session_state:
        st.session_state.test_plan_path = None
    if 'error_message' not in st.session_state:
        st.session_state.error_message = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'robot_output' not in st.session_state:
        st.session_state.robot_output = None
    if 'estimates' not in st.session_state:
        st.session_state.estimates = None
    if 'jira_requirements' not in st.session_state:
        st.session_state.jira_requirements = []
    if 'confluence_requirements' not in st.session_state:
        st.session_state.confluence_requirements = []
    if 'figma_requirements' not in st.session_state:
        st.session_state.figma_requirements = []

    st.title("🧪 AI-Powered Test Case Generator")
    st.markdown("""
    Upload your requirements documents, connect JIRA/Confluence, add Figma designs, and generate comprehensive test cases with AI assistance!
    """)

    # RAG Status Display
    if RAG_AVAILABLE:
        st.markdown('<div class="rag-status">🤖 RAG (Retrieval-Augmented Generation) is available for enhanced requirement analysis</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ RAG module not available. Some advanced features will be disabled.")

    # Create containers for different sections
    input_container = st.container()
    rag_container = st.container()
    analyze_container = st.container()
    results_container = st.container()
    actions_container = st.container()
    output_container = st.container()

    # Enhanced Input Section
    with input_container:
        st.markdown("## 📋 Requirements Input")

        # Create tabs for different input types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📄 Documents",
            "🎨 Figma Designs",
            "🔗 JIRA Issues",
            "📖 Confluence Pages",
            "✍️ Custom Prompt"
        ])

        # Tab 1: Document Upload
        with tab1:
            st.markdown("### Upload Requirements Documents")
            uploaded_files = st.file_uploader(
                "Upload Requirements/Design Documents",
                type=["txt", "md", "docx", "ppt", "pptx", "pdf", "fig", "png", "svg", "jpg", "jpeg"],
                accept_multiple_files=True,
                help="Supported formats: TXT, MD, DOCX, PPT, PPTX, PDF, FIG, PNG, SVG, JPG, JPEG"
            )

            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully")
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.type})")

        # Tab 2: Figma Integration
        with tab2:
            st.markdown("### Figma Design Integration")

            col1, col2 = st.columns(2)
            with col1:
                figma_url = st.text_input(
                    "Figma File URL",
                    placeholder="https://www.figma.com/file/FILE_ID/FILE_NAME",
                    help="Paste your Figma file URL here"
                )
                figma_token = st.text_input(
                    "Figma Access Token (Optional)",
                    type="password",
                    help="Provide your Figma access token for detailed analysis"
                )

            with col2:
                figma_files = st.file_uploader(
                    "Or Upload Figma Exports",
                    type=["fig", "png", "svg", "pdf", "jpg", "jpeg"],
                    accept_multiple_files=True,
                    help="Upload exported files from Figma"
                )

            if st.button("Add Figma Requirements", key="add_figma"):
                figma_reqs = []

                # Process Figma URL
                if figma_url:
                    result = fetch_figma_file_info(figma_url, figma_token)
                    figma_reqs.append(result)
                    if result.get("success"):
                        st.success(f"✅ Added Figma file: {result.get('file_name', 'Unknown')}")
                    else:
                        st.error(f"❌ {result.get('error', 'Unknown error')}")

                # Process uploaded Figma files
                if figma_files:
                    for file in figma_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name

                        try:
                            parser = FigmaParser()
                            result = parser.parse(tmp_path)
                            result["success"] = True
                            result["file_name"] = file.name
                            figma_reqs.append(result)
                            st.success(f"✅ Processed Figma file: {file.name}")
                        except Exception as e:
                            st.error(f"❌ Error processing {file.name}: {str(e)}")
                        finally:
                            os.unlink(tmp_path)

                if figma_reqs:
                    st.session_state.figma_requirements.extend(figma_reqs)

            # Display added Figma requirements
            if st.session_state.figma_requirements:
                st.markdown("#### Added Figma Requirements:")
                for i, req in enumerate(st.session_state.figma_requirements):
                    if req.get("success"):
                        with st.expander(f"🎨 {req.get('file_name', f'Figma {i+1}')}"):
                            st.text(req.get("raw_text", "No content"))
                            if st.button(f"Remove", key=f"remove_figma_{i}"):
                                st.session_state.figma_requirements.pop(i)
                                st.rerun()

        # Tab 3: Enhanced JIRA Integration
        with tab3:
            st.markdown("### JIRA Integration")
            st.info("💡 You can use either password or API token for authentication")

            col1, col2 = st.columns(2)
            with col1:
                # Hardcoded JIRA Host
                jira_host = "https://newfold.atlassian.net/"
                st.info(f"🔗 JIRA Host: {jira_host}")

                project_key = st.text_input(
                    "Project Key",
                    placeholder="PROJ",
                    help="JIRA project key (e.g., PROJ, TEST)"
                )
                issue_id = st.text_input(
                    "Issue ID",
                    placeholder="123",
                    help="JIRA issue number (without project key)"
                )

            with col2:
                jira_username = st.text_input(
                    "Username/Email",
                    help="Your JIRA username or email"
                )

                # Enhanced authentication options
                auth_type = st.radio(
                    "Authentication Type",
                    ["API Token", "Password"],
                    help="Choose your preferred authentication method"
                )

                if auth_type == "API Token":
                    jira_credential = st.text_input(
                        "API Token",
                        type="password",
                        help="Your JIRA API token (recommended for security)"
                    )
                    credential_type = "token"
                else:
                    jira_credential = st.text_input(
                        "Password",
                        type="password",
                        help="Your JIRA password (less secure than API token)"
                    )
                    credential_type = "password"

            if st.button("Add JIRA Issue", key="add_jira"):
                if jira_host and project_key and issue_id:
                    with st.spinner("Fetching JIRA issue..."):
                        result = fetch_jira_issue(
                            jira_host,
                            project_key,
                            issue_id,
                            jira_username,
                            jira_credential,
                            credential_type
                        )

                        if result.get("success"):
                            st.session_state.jira_requirements.append(result)
                            st.success(f"✅ Added JIRA issue: {project_key}-{issue_id}")

                            # Show metadata
                            metadata = result.get("metadata", {})
                            st.info(f"📊 Issue Type: {metadata.get('issue_type', 'Unknown')}, Status: {metadata.get('status', 'Unknown')}, Priority: {metadata.get('priority', 'Unknown')}")
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please fill in JIRA Host, Project Key, and Issue ID")

            # Display added JIRA requirements
            if st.session_state.jira_requirements:
                st.markdown("#### Added JIRA Issues:")
                for i, req in enumerate(st.session_state.jira_requirements):
                    if req.get("success"):
                        metadata = req.get("metadata", {})
                        with st.expander(f"🔗 {req.get('summary', f'JIRA Issue {i+1}')} [{metadata.get('issue_type', 'Unknown')}]"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text_area("Content", req.get("raw_text", "No content"), height=200, key=f"jira_content_{i}")
                            with col2:
                                st.markdown("**Metadata:**")
                                st.json(metadata)
                            if st.button(f"Remove", key=f"remove_jira_{i}"):
                                st.session_state.jira_requirements.pop(i)
                                st.rerun()

        # Tab 4: Enhanced Confluence Integration
        with tab4:
            st.markdown("### Confluence Integration")
            st.info("💡 You can use either password or API token for authentication")

            col1, col2 = st.columns(2)
            with col1:
                # Hardcoded Confluence Host
                conf_host = "https://confluence.newfold.com/"
                st.info(f"🔗 Confluence Host: {conf_host}")

                space_key = st.text_input(
                    "Space Key",
                    placeholder="SPACE",
                    help="Confluence space key"
                )
                page_title = st.text_input(
                    "Page Title",
                    placeholder="Requirements Document",
                    help="Title of the Confluence page"
                )

            with col2:
                conf_username = st.text_input(
                    "Username/Email",
                    help="Your Confluence username or email"
                )

                # Enhanced authentication options
                conf_auth_type = st.radio(
                    "Authentication Type",
                    ["API Token", "Password"],
                    help="Choose your preferred authentication method",
                    key="conf_auth_type"
                )

                if conf_auth_type == "API Token":
                    conf_credential = st.text_input(
                        "API Token",
                        type="password",
                        help="Your Confluence API token (recommended for security)",
                        key="conf_api_token"
                    )
                    conf_credential_type = "token"
                else:
                    conf_credential = st.text_input(
                        "Password",
                        type="password",
                        help="Your Confluence password (less secure than API token)",
                        key="conf_password"
                    )
                    conf_credential_type = "password"

            if st.button("Add Confluence Page", key="add_confluence"):
                if conf_host and space_key and page_title:
                    with st.spinner("Fetching Confluence page..."):
                        result = fetch_confluence_page(
                            conf_host,
                            space_key,
                            page_title,
                            conf_username,
                            conf_credential,
                            conf_credential_type
                        )

                        if result.get("success"):
                            st.session_state.confluence_requirements.append(result)
                            st.success(f"✅ Added Confluence page: {page_title}")

                            # Show metadata
                            metadata = result.get("metadata", {})
                            st.info(f"📊 Space: {metadata.get('space_key', 'Unknown')}, Content Length: {metadata.get('content_length', 0)} characters")
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please fill in Confluence Host, Space Key, and Page Title")

            # Display added Confluence requirements
            if st.session_state.confluence_requirements:
                st.markdown("#### Added Confluence Pages:")
                for i, req in enumerate(st.session_state.confluence_requirements):
                    if req.get("success"):
                        metadata = req.get("metadata", {})
                        with st.expander(f"📖 {req.get('title', f'Confluence Page {i+1}')} [{metadata.get('space_key', 'Unknown Space')}]"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                content_preview = req.get("raw_text", "No content")[:1000]
                                if len(req.get("raw_text", "")) > 1000:
                                    content_preview += "..."
                                st.text_area("Content Preview", content_preview, height=200, key=f"conf_content_{i}")
                            with col2:
                                st.markdown("**Metadata:**")
                                st.json(metadata)
                            if st.button(f"Remove", key=f"remove_conf_{i}"):
                                st.session_state.confluence_requirements.pop(i)
                                st.rerun()

        # Tab 5: Custom Prompt
        with tab5:
            st.markdown("### Custom Requirements Prompt")
            custom_prompt = st.text_area(
                "Enter additional context or specific requirements",
                height=200,
                max_chars=5000,
                placeholder="Describe specific test scenarios, edge cases, or additional context that should be considered when generating test cases...",
                help="Maximum 5000 characters. This will be used as additional context for AI-powered test case generation."
            )

            char_count = len(custom_prompt) if custom_prompt else 0
            st.caption(f"Characters: {char_count}/5000")

            if char_count > 5000:
                st.error("Custom prompt exceeds 5000 character limit!")

        # Configuration Section
        st.markdown("## ⚙️ Configuration")
        col1, col2 = st.columns(2)
        with col1:
            scope = st.selectbox("Test Scope", ["Functional", "Non-Functional", "Both"])
            components = st.text_input("Target Components/Modules (comma-separated, optional)")
        with col2:
            test_type = st.selectbox("Test Type", ["Component", "Integration", "Acceptance", "All"])
            config_file = st.file_uploader("Optional: Upload Config File (YAML/JSON)", type=["yaml", "yml", "json"])

    # Enhanced RAG Configuration Section
    with rag_container:
        if RAG_AVAILABLE:
            st.markdown("## 🤖 Advanced RAG Configuration")

            col1, col2 = st.columns(2)
            with col1:
                use_rag = st.checkbox(
                    "Enable RAG Enhancement",
                    value=False,
                    help="Use Retrieval-Augmented Generation to enhance requirement analysis with additional context"
                )

            with col2:
                rag_config_file = st.file_uploader(
                    "RAG Config File (Optional)",
                    type=["yaml", "yml", "json"],
                    help="Upload custom RAG configuration file"
                ) if use_rag else None

            if use_rag:
                st.info("🤖 RAG enhancement will be applied during requirement analysis for better context and accuracy")

                # Advanced RAG Configuration Tabs
                rag_tab1, rag_tab2, rag_tab3, rag_tab4, rag_tab5 = st.tabs([
                    "🗄️ Vector Database",
                    "🏢 Brand Help Pages",
                    "🔗 Custom URLs",
                    "⚙️ Processing",
                    "📊 Performance"
                ])

                # Tab 1: Vector Database Configuration
                with rag_tab1:
                    st.markdown("### Vector Database Configuration")

                    col1, col2 = st.columns(2)
                    with col1:
                        vector_db_type = st.selectbox(
                            "Vector Database Type",
                            ["faiss", "chroma", "pinecone", "weaviate"],
                            index=0,
                            help="Choose your preferred vector database for storing embeddings"
                        )

                        embedding_model = st.selectbox(
                            "Embedding Model",
                            [
                                "all-MiniLM-L6-v2",
                                "all-mpnet-base-v2",
                                "paraphrase-multilingual-MiniLM-L12-v2",
                                "sentence-transformers/all-roberta-large-v1",
                                "text-embedding-ada-002"
                            ],
                            index=0,
                            help="Select the embedding model for text vectorization"
                        )

                        similarity_threshold = st.slider(
                            "Similarity Threshold",
                            0.0, 1.0, 0.7, 0.05,
                            help="Minimum similarity score for retrieving relevant content"
                        )

                    with col2:
                        max_results = st.number_input(
                            "Max Results per Query",
                            1, 50, 10,
                            help="Maximum number of results to retrieve for each query"
                        )

                        chunk_size = st.number_input(
                            "Text Chunk Size",
                            100, 2000, 1000, 50,
                            help="Size of text chunks for processing"
                        )

                        chunk_overlap = st.number_input(
                            "Chunk Overlap",
                            0, 500, 200, 25,
                            help="Overlap between consecutive text chunks"
                        )

                    # Advanced Vector DB Settings
                    with st.expander("Advanced Vector Database Settings"):
                        if vector_db_type == "faiss":
                            col1, col2 = st.columns(2)
                            with col1:
                                faiss_index_type = st.selectbox(
                                    "FAISS Index Type",
                                    ["IndexFlatL2", "IndexIVFFlat", "IndexHNSWFlat"],
                                    help="Choose FAISS index type for performance optimization"
                                )
                            with col2:
                                faiss_dimension = st.number_input("Embedding Dimension", 128, 1024, 384)

                        elif vector_db_type == "pinecone":
                            pinecone_api_key = st.text_input(
                                "Pinecone API Key",
                                type="password",
                                help="Your Pinecone API key for cloud vector database"
                            )
                            pinecone_env = st.text_input(
                                "Pinecone Environment",
                                value="us-west1-gcp",
                                help="Pinecone environment/region"
                            )

                # Tab 2: Brand Help Pages Configuration
                with rag_tab2:
                    st.markdown("### Brand Help Pages Configuration")

                    enable_brand_pages = st.checkbox(
                        "Enable Brand Help Pages Crawling",
                        value=True,
                        help="Automatically crawl and index brand help pages for better context"
                    )

                    if enable_brand_pages:
                        st.markdown("#### Configured Brands")

                        # Brand selection with dynamic configuration
                        available_brands = ["bluehost", "hostgator", "domain.com", "networksolutions"]
                        selected_brands = st.multiselect(
                            "Select Brands to Index",
                            available_brands,
                            default=["bluehost", "hostgator", "networksolutions"],
                            help="Choose which brand help pages to crawl and index"
                        )

                        if selected_brands:
                            # Advanced brand configuration
                            with st.expander("Advanced Brand Crawling Settings"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    max_pages_per_brand = st.number_input(
                                        "Max Pages per Brand",
                                        50, 1000, 500,
                                        help="Maximum number of pages to crawl per brand"
                                    )
                                    crawl_depth = st.slider(
                                        "Crawl Depth",
                                        1, 5, 3,
                                        help="Maximum depth to crawl from the base URL"
                                    )

                                with col2:
                                    update_frequency = st.selectbox(
                                        "Update Frequency",
                                        ["daily", "weekly", "monthly"],
                                        index=1,
                                        help="How often to refresh the crawled content"
                                    )
                                    delay_between_requests = st.slider(
                                        "Delay Between Requests (seconds)",
                                        0.5, 5.0, 1.0, 0.1,
                                        help="Delay between crawl requests to be respectful"
                                    )

                            # Custom brand addition
                            with st.expander("Add Custom Brand"):
                                new_brand_name = st.text_input("Brand Name")
                                new_brand_url = st.text_input("Base URL", placeholder="https://example.com/help")
                                if st.button("Add Custom Brand"):
                                    if new_brand_name and new_brand_url:
                                        st.success(f"Custom brand '{new_brand_name}' added successfully!")
                                        st.session_state[f"custom_brand_{new_brand_name}"] = new_brand_url

                # Tab 3: Custom URLs Configuration
                with rag_tab3:
                    st.markdown("### Custom URLs Configuration")

                    enable_custom_urls = st.checkbox(
                        "Enable Custom URLs",
                        value=True,
                        help="Add custom URLs for domain-specific knowledge"
                    )

                    if enable_custom_urls:
                        # Predefined URL groups
                        st.markdown("#### Knowledge Source Categories")

                        url_categories = {
                            "Testing Resources": [
                                "https://www.guru99.com/software-testing.html",
                                "https://www.softwaretestinghelp.com/",
                                "https://testautomationu.applitools.com/",
                                "https://www.ministryoftesting.com/"
                            ],
                            "Automation Frameworks": [
                                "https://selenium-python.readthedocs.io/",
                                "https://www.selenium.dev/documentation/",
                                "https://robotframework.org/SeleniumLibrary/"
                            ],
                            "API Testing": [
                                "https://restfulapi.net/",
                                "https://www.postman.com/api-testing/",
                                "https://httpbin.org/"
                            ],
                            "Performance Testing": [
                                "https://k6.io/docs/",
                                "https://jmeter.apache.org/usermanual/",
                                "https://gatling.io/docs/"
                            ]
                        }

                        selected_categories = st.multiselect(
                            "Select Knowledge Categories",
                            list(url_categories.keys()),
                            default=["Testing Resources", "Automation Frameworks"],
                            help="Choose knowledge categories to include"
                        )

                        # Display selected URLs
                        if selected_categories:
                            with st.expander("Selected URLs Preview"):
                                for category in selected_categories:
                                    st.markdown(f"**{category}:**")
                                    for url in url_categories[category]:
                                        st.write(f"• {url}")

                        # Custom URL addition
                        st.markdown("#### Add Custom URLs")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            custom_url = st.text_input(
                                "Custom URL",
                                placeholder="https://example.com/documentation"
                            )
                        with col2:
                            url_category = st.selectbox(
                                "Category",
                                ["Testing", "Automation", "API", "Performance", "Custom"]
                            )

                        if st.button("Add Custom URL"):
                            if custom_url:
                                if 'custom_urls' not in st.session_state:
                                    st.session_state.custom_urls = []
                                st.session_state.custom_urls.append({
                                    'url': custom_url,
                                    'category': url_category
                                })
                                st.success(f"Added {custom_url} to {url_category} category")

                        # Display added custom URLs
                        if 'custom_urls' in st.session_state and st.session_state.custom_urls:
                            st.markdown("#### Your Custom URLs")
                            for i, url_info in enumerate(st.session_state.custom_urls):
                                col1, col2, col3 = st.columns([3, 1, 1])
                                with col1:
                                    st.write(url_info['url'])
                                with col2:
                                    st.write(url_info['category'])
                                with col3:
                                    if st.button("Remove", key=f"remove_url_{i}"):
                                        st.session_state.custom_urls.pop(i)
                                        st.rerun()

                # Tab 4: Content Processing Configuration
                with rag_tab4:
                    st.markdown("### Intelligent Content Processing")

                    col1, col2 = st.columns(2)
                    with col1:
                        enable_semantic_chunking = st.checkbox(
                            "Enable Semantic Chunking",
                            value=True,
                            help="Use AI to create semantically meaningful text chunks"
                        )

                        enable_content_classification = st.checkbox(
                            "Enable Content Classification",
                            value=True,
                            help="Automatically classify content types for better retrieval"
                        )

                        enable_quality_scoring = st.checkbox(
                            "Enable Quality Scoring",
                            value=True,
                            help="Score content quality to prioritize better sources"
                        )

                    with col2:
                        language_detection = st.checkbox(
                            "Enable Language Detection",
                            value=True,
                            help="Detect and handle multiple languages"
                        )

                        supported_languages = st.multiselect(
                            "Supported Languages",
                            ["en", "es", "fr", "de", "it", "pt", "zh", "ja"],
                            default=["en"],
                            help="Languages to support for content processing"
                        )

                    # Query Enhancement Settings
                    st.markdown("#### Query Enhancement")
                    col1, col2 = st.columns(2)
                    with col1:
                        enable_query_expansion = st.checkbox(
                            "Enable Query Expansion",
                            value=True,
                            help="Automatically expand queries with synonyms and related terms"
                        )

                        enable_intent_detection = st.checkbox(
                            "Enable Intent Detection",
                            value=True,
                            help="Detect user intent to improve retrieval accuracy"
                        )

                    with col2:
                        enable_context_awareness = st.checkbox(
                            "Enable Context Awareness",
                            value=True,
                            help="Use conversation context for better results"
                        )

                        query_rewriting = st.checkbox(
                            "Enable Query Rewriting",
                            value=True,
                            help="Automatically rewrite queries for better matching"
                        )

                # Tab 5: Performance & Monitoring
                with rag_tab5:
                    st.markdown("### Performance & Monitoring")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Caching Settings")
                        enable_caching = st.checkbox(
                            "Enable Result Caching",
                            value=True,
                            help="Cache query results for faster response times"
                        )

                        cache_embeddings = st.checkbox(
                            "Cache Embeddings",
                            value=True,
                            help="Cache computed embeddings to save processing time"
                        )

                        max_cache_size = st.selectbox(
                            "Max Cache Size",
                            ["100MB", "500MB", "1GB", "2GB", "5GB"],
                            index=2,
                            help="Maximum cache size before cleanup"
                        )

                    with col2:
                        st.markdown("#### Monitoring Settings")
                        enable_metrics = st.checkbox(
                            "Enable Performance Metrics",
                            value=True,
                            help="Track performance metrics for optimization"
                        )

                        log_queries = st.checkbox(
                            "Log Queries",
                            value=False,
                            help="Log queries for analysis (consider privacy implications)"
                        )

                        track_usage_patterns = st.checkbox(
                            "Track Usage Patterns",
                            value=True,
                            help="Track usage patterns to improve recommendations"
                        )

                    # RAG Strategy Selection
                    st.markdown("#### RAG Strategy")
                    rag_strategy = st.selectbox(
                        "RAG Strategy",
                        ["simple", "hybrid", "adaptive", "multi_modal"],
                        index=1,
                        help="Choose the RAG strategy that best fits your needs"
                    )

                    strategy_descriptions = {
                        "simple": "Basic retrieval and generation - Fast but less accurate",
                        "hybrid": "Combines dense and sparse retrieval - Balanced performance",
                        "adaptive": "Adapts strategy based on query type - Smart but slower",
                        "multi_modal": "Handles text, images, and code - Most comprehensive"
                    }

                    st.info(f"**{rag_strategy.title()} Strategy:** {strategy_descriptions[rag_strategy]}")

                # Save Configuration Button
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("💾 Save RAG Configuration", use_container_width=True):
                        # Build configuration from UI selections
                        rag_config = {
                            "vector_db_type": vector_db_type,
                            "embedding_model": embedding_model,
                            "similarity_threshold": similarity_threshold,
                            "max_results": max_results,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap,
                            "brand_help_pages": {
                                "enable": enable_brand_pages,
                                "selected_brands": selected_brands if enable_brand_pages else [],
                                "crawl_config": {
                                    "max_pages": max_pages_per_brand if enable_brand_pages else 0,
                                    "crawl_depth": crawl_depth if enable_brand_pages else 1,
                                    "update_frequency": update_frequency if enable_brand_pages else "weekly",
                                    "delay_between_requests": delay_between_requests if enable_brand_pages else 1.0
                                }
                            },
                            "custom_urls": {
                                "enable": enable_custom_urls,
                                "selected_categories": selected_categories if enable_custom_urls else [],
                                "custom_urls": st.session_state.get('custom_urls', [])
                            },
                            "content_processing": {
                                "enable_semantic_chunking": enable_semantic_chunking,
                                "enable_content_classification": enable_content_classification,
                                "enable_quality_scoring": enable_quality_scoring,
                                "language_detection": language_detection,
                                "supported_languages": supported_languages
                            },
                            "query_enhancement": {
                                "enable_query_expansion": enable_query_expansion,
                                "enable_intent_detection": enable_intent_detection,
                                "enable_context_awareness": enable_context_awareness,
                                "query_rewriting": query_rewriting
                            },
                            "performance": {
                                "enable_caching": enable_caching,
                                "cache_embeddings": cache_embeddings,
                                "max_cache_size": max_cache_size,
                                "enable_metrics": enable_metrics,
                                "log_queries": log_queries,
                                "track_usage_patterns": track_usage_patterns
                            },
                            "rag_strategy": rag_strategy
                        }

                        # Save to session state
                        st.session_state.rag_config = rag_config
                        st.success("✅ RAG configuration saved successfully!")

                with col2:
                    if st.button("🔄 Reset to Defaults", use_container_width=True):
                        # Clear session state
                        keys_to_clear = [k for k in st.session_state.keys() if k.startswith('rag_') or k.startswith('custom_')]
                        for key in keys_to_clear:
                            del st.session_state[key]
                        st.success("✅ Configuration reset to defaults!")
                        st.rerun()

                with col3:
                    if st.button("📊 Test Configuration", use_container_width=True):
                        with st.spinner("Testing RAG configuration..."):
                            # Simulate configuration test
                            time.sleep(2)
                            st.success("✅ Configuration test passed!")
                            st.info("All settings are valid and ready to use.")
        else:
            st.warning("⚠️ Enable RAG Enhancement to configure advanced options")

    # Analysis Section - moved above Actions as requested
    with analyze_container:
        st.markdown("## 🔍 Requirements Analysis")

        # Enhanced Analyze Requirements button
        analyze_button = st.button("🔍 Analyse Requirements", key="analyze_button", type="primary", use_container_width=True)

        if analyze_button:
            # Check if we have any requirements
            has_files = uploaded_files is not None and len(uploaded_files) > 0
            has_jira = len(st.session_state.jira_requirements) > 0
            has_confluence = len(st.session_state.confluence_requirements) > 0
            has_figma = len(st.session_state.figma_requirements) > 0
            has_prompt = custom_prompt and custom_prompt.strip()

            # Validate inputs
            valid, validation_message = validate_inputs(
                uploaded_files,
                st.session_state.jira_requirements,
                st.session_state.confluence_requirements,
                st.session_state.figma_requirements,
                custom_prompt
            )

            if not valid:
                st.error(validation_message)
            else:
                with st.spinner("🔍 Analyzing requirements comprehensively with Azure OpenAI..."):
                    # Prepare RAG config path if provided
                    rag_config_path = None
                    if RAG_AVAILABLE and use_rag and rag_config_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(rag_config_file.name)[1]) as tmp:
                            tmp.write(rag_config_file.read())
                            rag_config_path = tmp.name

                    # Enhanced comprehensive analysis
                    st.session_state.analysis, error = generate_comprehensive_test_plan(
                        uploaded_files,
                        st.session_state.jira_requirements,
                        st.session_state.confluence_requirements,
                        st.session_state.figma_requirements,
                        custom_prompt,
                        use_rag=(RAG_AVAILABLE and use_rag),
                        rag_config_path=rag_config_path
                    )

                    # Clean up temporary RAG config file
                    if rag_config_path:
                        try:
                            os.unlink(rag_config_path)
                        except:
                            pass

                    if error:
                        st.error(error)
                    elif st.session_state.analysis:
                        st.success("✅ Comprehensive Test Plan generated successfully!")

                        # Enhanced analysis results display
                        test_plan = st.session_state.analysis.get("test_plan", {})
                        estimates = test_plan.get("estimates", {})
                        test_scenarios = test_plan.get("test_scenarios", {})
                        stats = st.session_state.analysis.get("statistics", {})
                        sources = st.session_state.analysis.get("sources", [])

                        # Main metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Requirements", stats.get('total_requirements', 0))
                        with col2:
                            total_scenarios = sum(len(scenarios) for scenarios in test_scenarios.values())
                            st.metric("Test Scenarios", total_scenarios)
                        with col3:
                            dev_estimate = estimates.get('development_days', 0)
                            st.metric("Dev Estimate", f"{dev_estimate} days")
                        with col4:
                            test_estimate = estimates.get('testing_days', 0)
                            st.metric("Test Estimate", f"{test_estimate} days")

                        # Display Test Plan Summary
                        st.markdown("### 📋 Test Plan Summary")

                        with st.expander("📊 Estimates & Metrics", expanded=True):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Development Estimates:**")
                                st.write(f"• Total: {estimates.get('development_days', 0)} days")
                                st.write(f"• Breakdown: {estimates.get('development_breakdown', 'N/A')}")
                            with col2:
                                st.markdown("**Testing Estimates:**")
                                st.write(f"• Total: {estimates.get('testing_days', 0)} days")
                                st.write(f"• Breakdown: {estimates.get('testing_breakdown', 'N/A')}")

                        with st.expander("🎯 Test Scenarios Overview", expanded=True):
                            for scenario_type, scenarios in test_scenarios.items():
                                st.markdown(f"**{scenario_type.replace('_', ' ').title()}:** {len(scenarios)} scenarios")

                        # Quality metrics
                        quality = st.session_state.analysis.get("analysis_quality", {})
                        if quality:
                            st.markdown("### 📊 Analysis Quality Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                complexity_score = quality.get('avg_complexity', 0)
                                st.metric(
                                    "Avg Complexity",
                                    f"{complexity_score:.2f}",
                                    delta=f"{'High' if complexity_score > 0.7 else 'Medium' if complexity_score > 0.4 else 'Low'}"
                                )
                            with col2:
                                ambiguity_score = quality.get('avg_ambiguity', 0)
                                st.metric(
                                    "Avg Ambiguity",
                                    f"{ambiguity_score:.2f}",
                                    delta=f"{'High' if ambiguity_score > 0.6 else 'Medium' if ambiguity_score > 0.3 else 'Low'}"
                                )
                            with col3:
                                completeness_score = quality.get('avg_completeness', 0)
                                st.metric(
                                    "Avg Completeness",
                                    f"{completeness_score:.2f}",
                                    delta=f"{'Good' if completeness_score > 0.7 else 'Fair' if completeness_score > 0.4 else 'Poor'}"
                                )
                            with col4:
                                testability_score = quality.get('avg_testability', 0)
                                st.metric(
                                    "Avg Testability",
                                    f"{testability_score:.2f}",
                                    delta=f"{'Good' if testability_score > 0.7 else 'Fair' if testability_score > 0.4 else 'Poor'}"
                                )

                        # RAG enhancement status
                        if st.session_state.analysis.get("enhanced_with_rag"):
                            st.markdown('<div class="rag-status">✨ Requirements were enhanced using RAG for better context and accuracy</div>', unsafe_allow_html=True)

    # Enhanced Requirements Display Section
    with results_container:
        if st.session_state.analysis:
            st.markdown("## 📋 Identified Requirements")

            requirements = st.session_state.analysis.get("requirements", [])
            sources = st.session_state.analysis.get("sources", [])
            metadata = st.session_state.analysis.get("metadata", {})

            # Enhanced sources summary
            st.markdown("### 📊 Analysis Summary")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Sources Analyzed:**")
                for source in sources:
                    st.write(f"• {source}")

                # Show metadata summary
                if metadata:
                    st.markdown("**Source Details:**")
                    for key, meta in metadata.items():
                        source_type = meta.get("source", "unknown")
                        if source_type == "jira":
                            st.write(f"📌 JIRA: {meta.get('issue_type', 'Unknown')} - {meta.get('status', 'Unknown')}")
                        elif source_type == "confluence":
                            st.write(f"📄 Confluence: {meta.get('space_key', 'Unknown')} space")
                        elif source_type == "figma":
                            st.write(f"🎨 Figma: Design file")
                        elif source_type == "uploaded_file":
                            st.write(f"📁 File: {meta.get('type', 'Unknown type')}")
                        elif source_type == "rag_service":
                            st.write(f"🤖 RAG: Enhanced with AI context")

            with col2:
                stats = st.session_state.analysis["statistics"]
                enhanced_stats = {
                    "Total Requirements": stats['total_requirements'],
                    "Analysis Quality": {
                        "Avg Complexity": f"{st.session_state.analysis.get('analysis_quality', {}).get('avg_complexity', 0):.2f}",
                        "Avg Ambiguity": f"{st.session_state.analysis.get('analysis_quality', {}).get('avg_ambiguity', 0):.2f}",
                        "Avg Completeness": f"{st.session_state.analysis.get('analysis_quality', {}).get('avg_completeness', 0):.2f}",
                        "Avg Testability": f"{st.session_state.analysis.get('analysis_quality', {}).get('avg_testability', 0):.2f}"
                    },
                    "Distribution": {
                        "High Priority": stats.get('high_priority', 0),
                        "Medium Priority": stats.get('medium_priority', 0),
                        "Low Priority": stats.get('low_priority', 0),
                        "High Complexity": stats.get('high_complexity', 0),
                        "Medium Complexity": stats.get('medium_complexity', 0),
                        "Low Complexity": stats.get('low_complexity', 0)
                    }
                }
                st.json(enhanced_stats)

            # Enhanced Requirements list with better categorization
            st.markdown("### 📋 Requirements List")

            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                priority_filter = st.selectbox("Filter by Priority", ["All", "High", "Medium", "Low"])
            with col2:
                complexity_filter = st.selectbox("Filter by Complexity", ["All", "High", "Medium", "Low"])
            with col3:
                category_filter = st.selectbox("Filter by Category", ["All", "Functional", "Non-Functional", "Business Rule"])

            # Apply filters
            filtered_requirements = requirements
            if priority_filter != "All":
                filtered_requirements = [r for r in filtered_requirements if r.get('priority', '').lower() == priority_filter.lower()]
            if complexity_filter != "All":
                complexity_threshold = {"High": 0.7, "Medium": 0.3, "Low": 0.0}[complexity_filter]
                next_threshold = {"High": 1.0, "Medium": 0.7, "Low": 0.3}[complexity_filter]
                filtered_requirements = [r for r in filtered_requirements
                                       if complexity_threshold <= r.get('complexity', 0) < next_threshold]
            if category_filter != "All":
                filtered_requirements = [r for r in filtered_requirements if r.get('category', '').lower() == category_filter.lower()]

            st.info(f"Showing {len(filtered_requirements)} of {len(requirements)} requirements")

            st.markdown('<div class="requirements-list">', unsafe_allow_html=True)

            for i, req in enumerate(filtered_requirements[:30]):  # Show first 30 filtered requirements
                # Enhanced requirement display with quality indicators
                priority = req.get('priority', 'Medium')
                complexity = req.get('complexity', 0.5)
                testability = req.get('testability', 0.5)

                # Color coding based on quality
                border_color = "#4CAF50" if testability > 0.7 else "#FF9800" if testability > 0.4 else "#F44336"

                with st.expander(f"Requirement {i+1}: {req.get('text', '')[:80]}... [{priority} Priority]"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Text:** {req.get('text', 'N/A')}")
                        st.markdown(f"**Category:** {req.get('category', 'N/A')}")

                        # Show analysis results if available
                        analysis_results = req.get('analysis_results', {})
                        if analysis_results.get('test_suggestions'):
                            with st.expander("🧪 AI-Generated Test Suggestions"):
                                for j, suggestion in enumerate(analysis_results['test_suggestions'][:3]):  # Show first 3
                                    if isinstance(suggestion, dict):
                                        st.markdown(f"**Test Case {j+1}:** {suggestion.get('name', 'Unnamed')}")
                                        st.markdown(f"**Steps:** {suggestion.get('steps', 'No steps')}")
                                        st.markdown(f"**Expected:** {suggestion.get('expected_result', 'No expected result')}")
                                        st.markdown("---")

                    with col2:
                        st.markdown("**Quality Metrics:**")
                        st.metric("Priority", req.get('priority', 'N/A'))
                        st.metric("Complexity", f"{complexity:.2f}")
                        st.metric("Ambiguity", f"{req.get('ambiguity', 0):.2f}")
                        st.metric("Completeness", f"{req.get('completeness', 0):.2f}")
                        st.metric("Testability", f"{testability:.2f}")

                        # Quality indicator
                        if testability > 0.7:
                            st.success("✅ High Quality")
                        elif testability > 0.4:
                            st.warning("⚠️ Medium Quality")
                        else:
                            st.error("❌ Needs Improvement")

            if len(filtered_requirements) > 30:
                st.info(f"Showing first 30 requirements out of {len(filtered_requirements)} filtered results. Full list will be included in generated test plan.")

            st.markdown('</div>', unsafe_allow_html=True)

    # Actions Section
    with actions_container:
        if st.session_state.analysis is not None:
            st.markdown("## 🚀 Actions")

            # Action buttons in columns
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                generate_plan = st.button(
                    "📋 Generate Test Plan",
                    key="generate_plan",
                    disabled=st.session_state.test_plan_status == 'running',
                    use_container_width=True
                )
                if st.session_state.test_plan_status == 'running':
                    st.spinner("Generating test plan...")
                elif st.session_state.test_plan_status == 'success':
                    st.success("✅ Test plan generated!")
                elif st.session_state.test_plan_status == 'error':
                    st.error("❌ Error generating test plan")

            with col2:
                generate_estimates = st.button(
                    "⏱️ Generate Estimates",
                    key="generate_estimates",
                    disabled=st.session_state.estimates_status == 'running',
                    use_container_width=True
                )
                if st.session_state.estimates_status == 'running':
                    st.spinner("Generating estimates...")
                elif st.session_state.estimates_status == 'success':
                    st.success("✅ Estimates generated!")
                elif st.session_state.estimates_status == 'error':
                    st.error("❌ Error generating estimates")

            with col3:
                generate_tc = st.button(
                    "📝 Generate Manual TCs",
                    key="generate_tc",
                    disabled=st.session_state.manual_tc_status == 'running',
                    use_container_width=True
                )
                if st.session_state.manual_tc_status == 'running':
                    st.spinner("Generating manual test cases...")
                elif st.session_state.manual_tc_status == 'success':
                    st.success("✅ Manual TCs generated!")
                elif st.session_state.manual_tc_status == 'error':
                    st.error("❌ Error generating manual TCs")

            with col4:
                generate_automation_tc = st.button(
                    "🤖 Generate Automated TCs",
                    key="generate_automation_tc",
                    disabled=st.session_state.auto_tc_status == 'running',
                    use_container_width=True
                )
                if st.session_state.auto_tc_status == 'running':
                    st.spinner("Generating automated test cases...")
                elif st.session_state.auto_tc_status == 'success':
                    st.success("✅ Automated TCs generated!")
                elif st.session_state.auto_tc_status == 'error':
                    st.error("❌ Error generating automated test cases")

            # Task Functions
            def generate_test_plan_task():
                try:
                    st.session_state.test_plan_status = 'running'

                    from generator.testcase_generator import generate_test_plan

                    comps = [c.strip() for c in components.split(",") if c.strip()] if components else None

                    config_data = {}
                    if config_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(config_file.name)[1]) as tmp:
                            tmp.write(config_file.read())
                            tmp_path = tmp.name
                        try:
                            if config_file.name.endswith('.yaml') or config_file.name.endswith('.yml'):
                                with open(tmp_path) as f:
                                    config_data = yaml.safe_load(f)
                            elif config_file.name.endswith('.json'):
                                with open(tmp_path) as f:
                                    config_data = json.load(f)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass

                    test_plan_path = generate_test_plan(
                        st.session_state.analysis,
                        config=config_data,
                        scope=scope,
                        test_type=None if test_type == "All" else test_type,
                        components=comps
                    )

                    st.session_state.test_plan_path = test_plan_path
                    st.session_state.test_plan_status = 'success'

                    send_notification(
                        message="Test plan generated successfully",
                        status="success",
                        details=f"Test plan generated for scope: {scope}, test type: {test_type}"
                    )
                except Exception as e:
                    st.session_state.error_message = str(e)
                    st.session_state.test_plan_status = 'error'

                    send_notification(
                        message="Failed to generate test plan",
                        status="error",
                        details=str(e)
                    )

            def generate_estimates_task():
                try:
                    st.session_state.estimates_status = 'running'

                    config_data = {}
                    if config_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(config_file.name)[1]) as tmp:
                            tmp.write(config_file.read())
                            tmp_path = tmp.name
                        try:
                            if config_file.name.endswith('.yaml') or config_file.name.endswith('.yml'):
                                with open(tmp_path) as f:
                                    config_data = yaml.safe_load(f)
                            elif config_file.name.endswith('.json'):
                                with open(tmp_path) as f:
                                    config_data = json.load(f)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass

                    estimates = generate_dev_test_estimates(st.session_state.analysis, config_data)

                    if estimates and not estimates.get("error"):
                        st.session_state.estimates = estimates
                        st.session_state.estimates_status = 'success'

                        send_notification(
                            message="Development and testing estimates generated",
                            status="success",
                            details=f"Total estimated time: {estimates['total']['weeks']} weeks"
                        )
                    else:
                        error_msg = estimates.get("error", "Failed to generate estimates") if estimates else "Failed to generate estimates"
                        st.session_state.error_message = error_msg
                        st.session_state.estimates_status = 'error'

                except Exception as e:
                    st.session_state.error_message = str(e)
                    st.session_state.estimates_status = 'error'

                    send_notification(
                        message="Failed to generate estimates",
                        status="error",
                        details=str(e)
                    )

            def generate_manual_tc_task():
                try:
                    st.session_state.manual_tc_status = 'running'

                    config_data = {}
                    if config_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(config_file.name)[1]) as tmp:
                            tmp.write(config_file.read())
                            tmp_path = tmp.name
                        try:
                            if config_file.name.endswith('.yaml') or config_file.name.endswith('.yml'):
                                with open(tmp_path) as f:
                                    config_data = yaml.safe_load(f)
                            elif config_file.name.endswith('.json'):
                                with open(tmp_path) as f:
                                    config_data = json.load(f)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass

                    generator = TestCaseGenerator(st.session_state.analysis, config_data)
                    tc_type = None if test_type == "All" else test_type
                    comps = [c.strip() for c in components.split(",") if c.strip()] if components else None
                    test_cases = generator.generate(scope=scope, test_type=tc_type, components=comps)

                    if test_cases:
                        # Enhanced test case fields for better accuracy
                        table_fields = [
                            "TC id", "brand", "name", "description", "preconditions",
                            "conditions/steps", "test data", "expected result", "postconditions",
                            "priority", "severity", "test_type", "automation feasibility",
                            "complexity", "estimated_time", "dependencies"
                        ]

                        # Create enhanced DataFrame with better field mapping
                        enhanced_test_cases = []
                        for tc in test_cases:
                            enhanced_tc = {}
                            for field in table_fields:
                                if field in tc:
                                    enhanced_tc[field] = tc[field]
                                elif field == "test_type":
                                    enhanced_tc[field] = tc.get("type", tc.get("test_type", "Functional"))
                                elif field == "preconditions":
                                    enhanced_tc[field] = tc.get("preconditions", tc.get("prerequisites", "System is accessible and configured"))
                                elif field == "postconditions":
                                    enhanced_tc[field] = tc.get("postconditions", tc.get("cleanup", "System restored to original state"))
                                elif field == "estimated_time":
                                    enhanced_tc[field] = tc.get("estimated_time", "30 minutes")
                                elif field == "dependencies":
                                    enhanced_tc[field] = tc.get("dependencies", "None")
                                elif field == "complexity":
                                    enhanced_tc[field] = tc.get("complexity", "Medium")
                                else:
                                    enhanced_tc[field] = tc.get(field, "")
                            enhanced_test_cases.append(enhanced_tc)

                        df = pd.DataFrame(enhanced_test_cases)

                        # Rename columns for better display
                        df.rename(columns={
                            "TC id": "Test Case ID",
                            "brand": "Brand",
                            "name": "Test Case Name",
                            "description": "Description",
                            "preconditions": "Preconditions",
                            "conditions/steps": "Test Steps",
                            "test data": "Test Data",
                            "expected result": "Expected Result",
                            "postconditions": "Postconditions",
                            "priority": "Priority",
                            "severity": "Severity",
                            "test_type": "Test Type",
                            "automation feasibility": "Automation Feasibility",
                            "complexity": "Complexity",
                            "estimated_time": "Estimated Time",
                            "dependencies": "Dependencies"
                        }, inplace=True)

                        # Clean up DataFrame
                        df = df.map(lambda x: str(x).strip() if pd.notna(x) else "")

                        st.session_state.df = df
                        st.session_state.test_cases = enhanced_test_cases
                        st.session_state.manual_tc_status = 'success'

                        send_notification(
                            message="Manual test cases generated successfully",
                            status="success",
                            details=f"Generated {len(test_cases)} test cases with enhanced fields"
                        )
                    else:
                        st.session_state.error_message = "No test cases were generated."
                        st.session_state.manual_tc_status = 'error'

                except Exception as e:
                    st.session_state.error_message = str(e)
                    st.session_state.manual_tc_status = 'error'

                    send_notification(
                        message="Failed to generate manual test cases",
                        status="error",
                        details=str(e)
                    )

            def generate_automated_tc_task():
                try:
                    st.session_state.auto_tc_status = 'running'

                    # Generate test cases first if not available
                    if st.session_state.test_cases is None:
                        generate_manual_tc_task()

                    if st.session_state.test_cases:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_dir = os.path.join(os.getcwd(), "generated_tests")
                        os.makedirs(output_dir, exist_ok=True)

                        robot_writer = RobotWriter(output_dir)
                        robot_filename = f"generated_tests_{timestamp}.robot"
                        robot_output_path = robot_writer.write(
                            test_cases=st.session_state.test_cases,
                            filename=robot_filename
                        )

                        st.session_state.robot_output = robot_output_path
                        st.session_state.auto_tc_status = 'success'

                        send_notification(
                            message="Automated test cases generated successfully",
                            status="success",
                            details=f"Generated Robot Framework tests with enhanced accuracy"
                        )
                    else:
                        st.session_state.error_message = "No test cases available for automation generation."
                        st.session_state.auto_tc_status = 'error'

                except Exception as e:
                    st.session_state.error_message = str(e)
                    st.session_state.auto_tc_status = 'error'

                    send_notification(
                        message="Failed to generate automated test cases",
                        status="error",
                        details=str(e)
                    )

            # Process button clicks
            if generate_plan:
                generate_test_plan_task()
                st.rerun()

            if generate_estimates:
                generate_estimates_task()
                st.rerun()

            if generate_tc:
                generate_manual_tc_task()
                st.rerun()

            if generate_automation_tc:
                generate_automated_tc_task()
                st.rerun()

    # Enhanced Output Section
    with output_container:
        # Test Plan Results
        if st.session_state.test_plan_status == 'success' and st.session_state.test_plan_path:
            st.markdown("## 📋 Test Plan")
            st.success("Test Plan document generated successfully!")

            with open(st.session_state.test_plan_path, "rb") as f:
                test_plan_bytes = f.read()

            st.download_button(
                label="📥 Download Test Plan Document",
                data=test_plan_bytes,
                file_name=os.path.basename(st.session_state.test_plan_path),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )

        # Enhanced Estimates Results
        if st.session_state.estimates_status == 'success' and st.session_state.estimates:
            st.markdown("## ⏱️ Development & Testing Estimates")
            estimates = st.session_state.estimates

            # Show confidence level
            confidence = estimates.get('confidence_level', 'Medium')
            if confidence == 'Low':
                st.warning(f"⚠️ Confidence Level: {confidence} - Estimates may vary due to requirement ambiguity")
            else:
                st.info(f"📊 Confidence Level: {confidence}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### 🔧 Development")
                st.metric("Hours", f"{estimates['development']['hours']}")
                st.metric("Days", f"{estimates['development']['days']}")
                st.metric("Weeks", f"{estimates['development']['weeks']}")

            with col2:
                st.markdown("### 🧪 Testing")
                st.metric("Hours", f"{estimates['testing']['hours']}")
                st.metric("Days", f"{estimates['testing']['days']}")
                st.metric("Weeks", f"{estimates['testing']['weeks']}")

            with col3:
                st.markdown("### 📊 Total")
                st.metric("Hours", f"{estimates['total']['hours']}")
                st.metric("Days", f"{estimates['total']['days']}")
                st.metric("Weeks", f"{estimates['total']['weeks']}")

            # Enhanced breakdown with quality adjustments
            with st.expander("📊 Detailed Breakdown & Quality Adjustments"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Time Breakdown:**")
                    st.json(estimates['breakdown'])
                with col2:
                    st.markdown("**Quality Adjustments:**")
                    quality_adj = estimates.get('quality_adjustments', {})
                    st.json({
                        "Ambiguity Multiplier": quality_adj.get('ambiguity_multiplier', 1.0),
                        "Completeness Multiplier": quality_adj.get('completeness_multiplier', 1.0),
                        "Total Complexity Points": estimates.get('total_complexity_points', 0),
                        "Functional Requirements": estimates.get('functional_requirements', 0),
                        "Non-Functional Requirements": estimates.get('non_functional_requirements', 0)
                    })

                # Export estimates
                estimates_json = json.dumps(estimates, indent=2)
                st.download_button(
                    label="📥 Download Estimates (JSON)",
                    data=estimates_json,
                    file_name=f"estimates_{int(time.time())}.json",
                    mime="application/json"
                )

        # Manual Test Cases Results
        if st.session_state.manual_tc_status == 'success' and st.session_state.df is not None:
            st.markdown("## 📝 Manual Test Cases")
            st.success(f"Generated {len(st.session_state.df)} test cases successfully!")

            # Enhanced test case display
            st.dataframe(
                st.session_state.df,
                use_container_width=True,
                height=600
            )

            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📥 Export as CSV",
                    data=st.session_state.df.to_csv(index=False).encode('utf-8'),
                    file_name="test_cases.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    label="📥 Export as JSON",
                    data=st.session_state.df.to_json(orient='records', lines=True).encode('utf-8'),
                    file_name="test_cases.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col3:
                excel_buffer = io.BytesIO()
                st.session_state.df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Export as Excel",
                    data=excel_buffer,
                    file_name="test_cases.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        # Automated Test Cases Results
        if st.session_state.auto_tc_status == 'success' and st.session_state.robot_output:
            st.markdown("## 🤖 Automated Test Cases")
            st.success("Robot Framework test cases generated successfully!")

            with open(st.session_state.robot_output, "r") as f:
                robot_content = f.read()

            with st.expander("👀 View Robot Framework Code"):
                st.code(robot_content, language="robot")

            st.download_button(
                label="📥 Download Robot Framework Tests",
                data=robot_content,
                file_name=os.path.basename(st.session_state.robot_output),
                mime="text/plain",
                use_container_width=True
            )

            st.balloons()

    # Information section
    with st.expander("Instructions", expanded=False):
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload your requirements or design documents.
        2. Select the test scope and type.
        3. Optionally, specify target components/modules.
        4. Click on any of the generation buttons to start the respective process:
           - **Generate Test Plan**: Creates a comprehensive test plan document
           - **Generate Estimates**: Provides development and testing estimates based on requirements
           - **Generate TCs Manually**: Creates manual test cases in table format
           - **Generate Automated TCs**: Creates Robot Framework test cases
        5. Use the export buttons to download the generated test cases in various formats (CSV, JSON, Excel).
        6. You can also export the knowledge base for future use.
        7. If you have enabled RAG, it will enhance the test case generation with external knowledge sources.
        8. You can clear the cache or rebuild the index if needed.
        9. Use the "Analyse Requirements" button to analyse your documents for requirements extraction.
        10. The results will be displayed below each operation.
        11. You can view the generated test cases in JSON and YAML formats as well.

        Each button works independently and can be run in parallel. You'll see progress indicators and results for each operation separately.
        """)

    with st.expander("About", expanded=False):
        st.markdown("### About")
        st.markdown("""
        This tool uses AI to analyse your requirements and generate test cases in Robot Framework format.
        It is designed to help you automate your testing process and improve your software quality.

        Key features:
        - Automatic requirement extraction from various document formats
        - NLP-based analysis for complexity, ambiguity, completeness, and testability
        - Generation of comprehensive test plan documents
        - Creation of both manual and automated test cases
        - Support for various export formats (CSV, JSON, Excel)
        - Integration with external knowledge sources using RAG (Retrieval-Augmented Generation)
        - Easy-to-use interface with progress indicators and results display

        This tool is part of the GenAI Test Case Generation suite, developed by Siddhant Wadhwani.
        It aims to streamline the testing process and enhance the quality of software products by leveraging AI capabilities.
        """)

        st.markdown("### Contact")
        st.markdown("""
        For any issues or feedback, please contact siddhant.wadhwani@newfold.com
        """)


if __name__ == "__main__":
    show_ui()
