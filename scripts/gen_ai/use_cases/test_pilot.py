"""
TestPilot - AI-Powered Intelligent Test Automation Assistant

This module provides an intelligent, AI-powered automation assistant that can:
- Fetch test cases from Jira or Zephyr by ticket ID
- Read and interpret all step information in test cases
- Convert them into meaningful natural language and generate Robot Framework scripts
- Reuse existing keywords, variables, and locators as defined in the architecture
- Generate new code only when required

Features:
1. Enter steps manually in natural language (line-by-line)
2. Fetch test steps from Jira/Zephyr by ticket ID
3. Upload a recording JSON file (interprets actions and converts them to scripts)
4. Enable Record & Playback (real-time recording and conversion)
"""

import logging
import os
import sys
import json
import time
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import base64
import atexit

# Ensure streamlit compatibility
try:
    from gen_ai import streamlit_fix
except ImportError:
    try:
        import streamlit_fix
    except ImportError:
        pass

import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add parent directories to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_AI_DIR = os.path.dirname(CURRENT_DIR)
SCRIPTS_DIR = os.path.dirname(GEN_AI_DIR)
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)

for dir_path in [GEN_AI_DIR, SCRIPTS_DIR, ROOT_DIR]:
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)

# Import required modules
AZURE_AVAILABLE = False
try:
    from azure_openai_client import AzureOpenAIClient
    AZURE_AVAILABLE = True
except ImportError:
    try:
        from gen_ai.azure_openai_client import AzureOpenAIClient
        AZURE_AVAILABLE = True
    except ImportError:
        pass

ROBOT_WRITER_AVAILABLE = False
try:
    from robot_writer.robot_writer import RobotWriter
    ROBOT_WRITER_AVAILABLE = True
except ImportError:
    try:
        from gen_ai.robot_writer.robot_writer import RobotWriter
        ROBOT_WRITER_AVAILABLE = True
    except ImportError:
        pass

NOTIFICATIONS_AVAILABLE = False
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    pass

# Configure logging FIRST before using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestPilot")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function.*")
warnings.filterwarnings("ignore", message=".*Fallback loading failed.*")
warnings.filterwarnings("ignore", message=".*No module named.*")
warnings.filterwarnings("ignore", message=".*frozenset.*")
warnings.filterwarnings("ignore", message=".*async generator ignored GeneratorExit.*")

# Suppress Robot Framework library loading warnings
import logging as robot_logging
robot_logging.getLogger('robot').setLevel(logging.ERROR)
robot_logging.getLogger('robotmcp').setLevel(logging.ERROR)

# Suppress asyncio RuntimeError for async generators during cleanup
import sys
if sys.version_info >= (3, 8):
    import asyncio
    # Suppress the specific RuntimeError from async generators
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


# Environment Configuration
class EnvironmentConfig:
    """Environment configuration for different test environments"""

    ENVIRONMENTS = {
        'prod': {
            'name': 'Production',
            'proxy': None,
            'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'requires_proxy': False,
            'mode': 'direct'
        },
        'qamain': {
            'name': 'QA Main',
            'proxy': 'http://10.201.16.27:8080',  # zproxy.qamain.netsol.com
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 aem_env=qamain',
            'requires_proxy': True,
            'mode': 'proxy'
        },
        'stage': {
            'name': 'Stage',
            'proxy': 'http://zproxy.stg.netsol.com:8080',
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 aem_env=stage',
            'requires_proxy': True,
            'mode': 'proxy'
        },
        'jarvisqa1': {
            'name': 'Jarvis QA1',
            'proxy': None,  # No proxy - user agent mode only
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 jarvis_env=jarvisqa1 aem_env=jarvisqa1',
            'requires_proxy': False,
            'mode': 'user_agent'
        },
        'jarvisqa2': {
            'name': 'Jarvis QA2',
            'proxy': None,  # No proxy - user agent mode only
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 jarvis_env=jarvisqa2 aem_env=jarvisqa2',
            'requires_proxy': False,
            'mode': 'user_agent'
        }
    }

    @classmethod
    def get_config(cls, environment: str) -> dict:
        """Get configuration for specified environment"""
        return cls.ENVIRONMENTS.get(environment, cls.ENVIRONMENTS['prod'])

    @classmethod
    def get_available_environments(cls) -> list:
        """Get list of available environments"""
        return list(cls.ENVIRONMENTS.keys())

    @classmethod
    def format_environment_display(cls, env: str) -> str:
        """Format environment name for display"""
        config = cls.get_config(env)
        if config['mode'] == 'direct':
            mode_str = "🌐 Direct Access"
        elif config['mode'] == 'proxy':
            mode_str = "🔒 Proxy Mode"
        else:  # user_agent
            mode_str = "🏷️ User Agent Mode"
        return f"{config['name']} ({env}) - {mode_str}"

# RobotMCP availability check - for advanced automation (after logger is configured)
ROBOTMCP_AVAILABLE = False
ROBOTMCP_CLIENT = None
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    # Only import robotmcp if we need to verify it's installed
    # Don't import robotmcp.server which has issues - we only need the MCP client
    try:
        import robotmcp
        # Test if robotmcp tools are accessible
        ROBOTMCP_AVAILABLE = True
        logger.info("✅ RobotMCP available for advanced automation")
    except Exception as robotmcp_error:
        # robotmcp module has issues, but MCP client is available
        # We can still use it via the robotmcp command
        logger.warning(f"⚠️ RobotMCP module import warning: {robotmcp_error}")
        logger.info("ℹ️ Will attempt to use robotmcp via command-line interface")
        ROBOTMCP_AVAILABLE = True  # Still try to use it via CLI
except ImportError as e:
    logger.info(f"⚠️ RobotMCP not available - install with: pip install robotmcp (Error: {e})")
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None

# Global cleanup registry for RobotMCP connections
_robotmcp_instances = []

def _cleanup_robotmcp_connections():
    """Cleanup all RobotMCP connections on exit"""
    for instance in _robotmcp_instances:
        try:
            if hasattr(instance, 'shutdown'):
                instance.shutdown()
        except Exception as e:
            # Suppress errors during cleanup
            pass

# Register cleanup handler
atexit.register(_cleanup_robotmcp_connections)

@dataclass
class TestStep:
    """Represents a single test step"""
    step_number: int
    description: str
    action: str = ""  # click, input, navigate, verify, etc.
    target: str = ""  # element identifier
    value: str = ""  # input value or expected value
    keyword: str = ""  # Robot Framework keyword
    arguments: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class TestCase:
    """Represents a complete test case"""
    id: str
    title: str
    description: str = ""
    steps: List[TestStep] = field(default_factory=list)
    preconditions: str = ""
    expected_result: str = ""
    priority: str = "Medium"
    tags: List[str] = field(default_factory=list)
    source: str = "manual"  # manual, jira, zephyr, recording
    metadata: Dict[str, Any] = field(default_factory=dict)


class JiraZephyrIntegration:
    """Handles integration with Jira and Zephyr for fetching test cases"""

    def __init__(self):
        # Configure session with proper connection pooling
        self.session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST", "PUT"]
        )

        # Configure HTTP adapter with increased pool connections
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,  # Increased from default 1
            pool_maxsize=10,      # Increased from default 1
            pool_block=False      # Don't block when pool is full
        )

        # Mount adapter for both http and https
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.base_url = ""
        self.authenticated = False

    def authenticate(self, host: str, username: str = None, api_token: str = None,
                    credential_type: str = "token") -> Tuple[bool, str]:
        """
        Authenticate with Jira/Zephyr

        Args:
            host: Jira host URL (e.g., https://jira.example.com)
            username: Username (for basic auth) or email
            api_token: API token or password
            credential_type: 'token' or 'password'

        Returns:
            Tuple of (success, message)
        """
        try:
            # Clean up host URL
            if not host.startswith(('http://', 'https://')):
                host = 'https://' + host

            self.base_url = host.rstrip('/')

            # Set up authentication
            if credential_type == "token" and username and api_token:
                # Atlassian API token authentication
                auth_string = f"{username}:{api_token}"
                b64_auth = base64.b64encode(auth_string.encode()).decode()
                self.session.headers.update({
                    'Authorization': f'Basic {b64_auth}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                })
            elif username and api_token:
                # Basic authentication
                self.session.auth = (username, api_token)
                self.session.headers.update({
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                })
            else:
                return False, "Missing credentials"

            # Test authentication
            test_url = f"{self.base_url}/rest/api/2/myself"
            response = self.session.get(test_url, timeout=10)

            if response.status_code == 200:
                self.authenticated = True
                user_info = response.json()
                return True, f"Successfully authenticated as {user_info.get('displayName', username)}"
            else:
                return False, f"Authentication failed: {response.status_code} - {response.text}"

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, f"Authentication error: {str(e)}"

    def fetch_issue(self, issue_key: str) -> Tuple[bool, Dict[str, Any], str]:
        """
        Fetch issue from Jira

        Args:
            issue_key: Jira issue key (e.g., PROJ-123)

        Returns:
            Tuple of (success, issue_data, message)
        """
        if not self.authenticated:
            return False, {}, "Not authenticated. Please authenticate first."

        try:
            # Fetch issue
            url = f"{self.base_url}/rest/api/2/issue/{issue_key}"
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                issue_data = response.json()
                return True, issue_data, "Issue fetched successfully"
            else:
                return False, {}, f"Failed to fetch issue: {response.status_code} - {response.text}"

        except Exception as e:
            logger.error(f"Error fetching issue: {str(e)}")
            return False, {}, f"Error: {str(e)}"

    def fetch_zephyr_test_case(self, issue_key: str) -> Tuple[bool, TestCase, str]:
        """
        Fetch test case from Zephyr with test steps, expected results, and test data

        Priority Order:
        1. Test Execution Details (Test Steps, Test Data, Expected/Actual Results) - HIGHEST PRIORITY
        2. Description converted to actionable steps via AI - FALLBACK

        Args:
            issue_key: Jira issue key for the test case

        Returns:
            Tuple of (success, TestCase object, message)
        """
        success, issue_data, message = self.fetch_issue(issue_key)

        if not success:
            return False, TestCase(id="", title="", description=""), message

        try:
            # Parse issue data into TestCase
            fields = issue_data.get('fields', {})

            test_case = TestCase(
                id=issue_key,
                title=fields.get('summary', ''),
                description=fields.get('description', '') or '',
                priority=fields.get('priority', {}).get('name', 'Medium'),
                source='zephyr',
                metadata={
                    'issue_type': fields.get('issuetype', {}).get('name', ''),
                    'status': fields.get('status', {}).get('name', ''),
                    'created': fields.get('created', ''),
                    'updated': fields.get('updated', ''),
                    'reporter': fields.get('reporter', {}).get('displayName', ''),
                    'assignee': fields.get('assignee', {}).get('displayName', '') if fields.get('assignee') else '',
                    'brand': 'generated'  # Default brand for generated tests
                }
            )

            # PRIORITY 1: Extract test execution details from Zephyr custom fields
            # Zephyr Scale (formerly TM4J) uses different custom fields
            # Common field IDs: customfield_10014, customfield_10015, customfield_10016
            steps_data = None
            test_script_field = None
            has_test_execution_details = False

            logger.info(f"🔍 Searching for test execution details in {issue_key}...")

            # Try multiple common custom field patterns for Zephyr
            for field_id in fields.keys():
                field_value = fields[field_id]
                field_id_lower = str(field_id).lower()

                # Check for Zephyr test script/steps fields
                if field_value and isinstance(field_value, (str, list, dict)):
                    # Zephyr Scale test script field
                    if 'script' in field_id_lower or 'teststeps' in field_id_lower or 'steps' in field_id_lower:
                        test_script_field = field_value
                        logger.info(f"✅ Found test execution details in field: {field_id}")
                        has_test_execution_details = True
                        break
                    # Old TM4J format
                    elif 'customfield' in field_id_lower and field_value:
                        if not steps_data:  # Take the first non-empty custom field as fallback
                            steps_data = field_value

            # Parse steps from the found field
            if test_script_field:
                test_case.steps = self._parse_zephyr_steps(test_script_field)
                if test_case.steps:
                    logger.info(f"✅ PRIORITY 1: Parsed {len(test_case.steps)} steps from test execution details")
                    has_test_execution_details = True
            elif steps_data:
                test_case.steps = self._parse_steps(steps_data)
                if test_case.steps:
                    logger.info(f"✅ PRIORITY 1: Parsed {len(test_case.steps)} steps from custom field")
                    has_test_execution_details = True

            # PRIORITY 2: If no test execution details found, use AI to convert description to actionable steps
            if not has_test_execution_details or not test_case.steps:
                logger.info(f"⚠️  No test execution details found in {issue_key}")

                if test_case.description:
                    logger.info(f"🤖 PRIORITY 2: Using Azure OpenAI to convert description to actionable steps...")

                    # Use AI to convert description to actionable test steps
                    ai_steps = self._convert_description_to_steps_with_ai(test_case.description, test_case.title)

                    if ai_steps:
                        test_case.steps = ai_steps
                        logger.info(f"✅ AI converted description to {len(ai_steps)} actionable steps")
                    else:
                        # Fallback: parse from description using regex
                        logger.info(f"⚠️  AI conversion failed, using regex parsing as fallback")
                        test_case.steps = self._parse_steps_from_text(test_case.description)
                        logger.info(f"Parsed {len(test_case.steps)} steps from description")
                else:
                    logger.warning(f"⚠️  No description available for {issue_key}")

            # If still no steps, create at least one step
            if not test_case.steps:
                test_case.steps = [TestStep(
                    step_number=1,
                    description=test_case.title or "Execute test case"
                )]
                logger.info(f"ℹ️  Created default step from title")

            # Extract labels as tags
            test_case.tags = fields.get('labels', [])

            # Extract preconditions if available
            for field_id, field_value in fields.items():
                if 'precondition' in str(field_id).lower() and field_value:
                    test_case.preconditions = str(field_value)
                    break

            # Create detailed message
            details_source = "test execution details" if has_test_execution_details else "AI-converted description"
            return True, test_case, f"✅ Test case fetched successfully with {len(test_case.steps)} steps from {details_source}"

        except Exception as e:
            logger.error(f"Error parsing test case: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, TestCase(id="", title="", description=""), f"Error parsing test case: {str(e)}"

    def _convert_description_to_steps_with_ai(self, description: str, title: str) -> List[TestStep]:
        """
        Use Azure OpenAI to convert Jira ticket description into actionable test steps

        Args:
            description: Jira ticket description
            title: Jira ticket title

        Returns:
            List of TestStep objects with actionable steps
        """
        if not AZURE_AVAILABLE:
            logger.warning("Azure OpenAI not available for AI conversion")
            return []

        try:
            # Use the globally imported AzureOpenAIClient
            azure_client = AzureOpenAIClient()
            if not azure_client.is_configured():
                logger.warning("Azure OpenAI not configured")
                return []

            # Build intelligent prompt
            prompt = f"""You are an expert QA engineer. Convert the following Jira ticket description into clear, actionable test steps that can be automated in a browser.

📋 TICKET INFORMATION:
Title: {title}
Description:
{description}

🎯 YOUR TASK:
Convert this description into a numbered list of actionable test steps that can be executed in a browser. Each step should be:
1. Clear and specific (e.g., "Click the 'Login' button", "Enter 'test@example.com' in the email field")
2. Actionable (can be performed by an automation tool)
3. Sequential (in the order they should be executed)
4. Include test data where needed (e.g., email addresses, passwords, search terms)
5. Include verification steps (e.g., "Verify the page title is 'Dashboard'")

📝 FORMAT REQUIREMENTS:
Return ONLY a JSON array of step objects. Each step object must have:
- "step_number": integer (1, 2, 3, ...)
- "description": string (the actionable step description)
- "test_data": string (optional, any test data needed for this step)
- "expected_result": string (optional, what should happen after this step)

🎨 EXAMPLE OUTPUT:
[
  {{
    "step_number": 1,
    "description": "Navigate to https://www.example.com/",
    "test_data": "",
    "expected_result": "Home page loads successfully"
  }},
  {{
    "step_number": 2,
    "description": "Click on the 'Products' menu",
    "test_data": "",
    "expected_result": "Products dropdown menu appears"
  }},
  {{
    "step_number": 3,
    "description": "Enter 'laptop' in the search box",
    "test_data": "laptop",
    "expected_result": "Search suggestions appear"
  }}
]

🚨 CRITICAL:
- Return ONLY valid JSON array, no explanations or markdown
- Each step must be browser-automatable
- Include specific URLs, button names, field names when mentioned
- Break complex tasks into multiple simple steps
- Add verification steps after important actions

Generate the actionable test steps now:"""

            # Call Azure OpenAI
            response = azure_client.completion_create(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3  # Lower temperature for more consistent output
            )

            if not response or 'choices' not in response:
                logger.warning("No response from Azure OpenAI")
                return []

            # Extract JSON from response
            ai_response = response['choices'][0]['message']['content'].strip()

            # Clean up response - remove markdown code blocks if present
            if ai_response.startswith('```'):
                ai_response = ai_response.split('```')[1]
                if ai_response.startswith('json'):
                    ai_response = ai_response[4:]
                ai_response = ai_response.strip()

            # Parse JSON
            try:
                steps_data = json.loads(ai_response)
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse AI response as JSON: {str(je)}")
                logger.error(f"Response was: {ai_response[:500]}")
                return []

            # Convert to TestStep objects
            steps = []
            for step_data in steps_data:
                if isinstance(step_data, dict):
                    step = TestStep(
                        step_number=step_data.get('step_number', len(steps) + 1),
                        description=step_data.get('description', ''),
                        value=step_data.get('test_data', ''),
                        notes=step_data.get('expected_result', '')
                    )
                    steps.append(step)

            logger.info(f"✅ AI successfully converted description to {len(steps)} actionable steps")
            return steps

        except Exception as e:
            logger.error(f"Error in AI conversion: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _parse_zephyr_steps(self, test_script_data: Any) -> List[TestStep]:
        """
        Parse Zephyr Scale test script format

        Zephyr Scale format:
        {
            "steps": [
                {
                    "index": 1,
                    "description": "Step description",
                    "testData": "Test data",
                    "expectedResult": "Expected result"
                }
            ]
        }
        """
        steps = []

        try:
            # Handle string JSON
            if isinstance(test_script_data, str):
                try:
                    test_script_data = json.loads(test_script_data)
                except json.JSONDecodeError:
                    # If not JSON, try parsing as plain text
                    return self._parse_steps_from_text(test_script_data)

            # Handle Zephyr Scale format
            if isinstance(test_script_data, dict):
                steps_list = test_script_data.get('steps', [])
                if not steps_list and test_script_data.get('text'):
                    # Sometimes it's in 'text' field
                    return self._parse_steps_from_text(test_script_data['text'])
            elif isinstance(test_script_data, list):
                steps_list = test_script_data
            else:
                return []

            for i, step_data in enumerate(steps_list):
                if isinstance(step_data, dict):
                    # Zephyr Scale format
                    step_num = step_data.get('index', i + 1)
                    description = step_data.get('description', step_data.get('step', ''))
                    test_data = step_data.get('testData', step_data.get('data', ''))
                    expected = step_data.get('expectedResult', step_data.get('result', step_data.get('expected', '')))
                    actual = step_data.get('actualResult', step_data.get('actual', ''))

                    step = TestStep(
                        step_number=step_num,
                        description=description,
                        value=test_data if test_data else '',
                        notes=expected if expected else '',
                        action=actual if actual else ''  # Store actual result in action field for now
                    )
                    steps.append(step)
                elif isinstance(step_data, str):
                    # Plain text step
                    step = TestStep(
                        step_number=i + 1,
                        description=step_data
                    )
                    steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"Error parsing Zephyr steps: {str(e)}")
            return []

    def _parse_steps(self, steps_data: Any) -> List[TestStep]:
        """Parse steps from Zephyr custom field data"""
        steps = []

        try:
            # Zephyr steps can be in various formats
            if isinstance(steps_data, str):
                # Try JSON first
                try:
                    data = json.loads(steps_data)
                    return self._parse_zephyr_steps(data)
                except json.JSONDecodeError:
                    # Parse from text
                    return self._parse_steps_from_text(steps_data)
            elif isinstance(steps_data, list):
                # Try Zephyr format first
                zephyr_steps = self._parse_zephyr_steps(steps_data)
                if zephyr_steps:
                    return zephyr_steps

                # Fallback to simple list parsing
                for i, step_data in enumerate(steps_data, 1):
                    if isinstance(step_data, dict):
                        step = TestStep(
                            step_number=i,
                            description=step_data.get('step', step_data.get('description', step_data.get('action', ''))),
                            value=step_data.get('data', step_data.get('testData', '')),
                            notes=step_data.get('result', step_data.get('expectedResult', step_data.get('expected', '')))
                        )
                    else:
                        step = TestStep(
                            step_number=i,
                            description=str(step_data)
                        )
                    steps.append(step)
            elif isinstance(steps_data, dict):
                # Some formats have numbered steps
                for key, value in sorted(steps_data.items()):
                    step_num = len(steps) + 1
                    step = TestStep(
                        step_number=step_num,
                        description=value if isinstance(value, str) else str(value)
                    )
                    steps.append(step)
        except Exception as e:
            logger.error(f"Error parsing steps: {str(e)}")

        return steps

    def _parse_steps_from_text(self, text: str) -> List[TestStep]:
        """Parse steps from plain text description"""
        steps = []

        # Split by common delimiters
        lines = text.split('\n')
        step_num = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with step indicator
            if any(line.lower().startswith(prefix) for prefix in ['step', 'test step', '-', '*', '•', str(step_num + 1)]):
                step_num += 1
                # Clean up the step text
                description = line
                for prefix in ['step', 'test step', '-', '*', '•']:
                    if line.lower().startswith(prefix):
                        description = line[len(prefix):].strip()
                        # Remove numbering like "1.", "2:", etc.
                        if description and description[0].isdigit():
                            description = description.lstrip('0123456789.:) ').strip()
                        break

                step = TestStep(
                    step_number=step_num,
                    description=description
                )
                steps.append(step)
            elif step_num > 0 and steps:
                # Continuation of previous step
                steps[-1].description += " " + line

        # If no structured steps found, create single step from entire text
        if not steps:
            steps.append(TestStep(
                step_number=1,
                description=text
            ))

        return steps

    def create_bug_ticket(self, bug_data: Dict[str, Any], project_key: str, issue_type: str = "Bug") -> Tuple[bool, str, str]:
        """
        Create a Jira bug ticket from bug report data

        Args:
            bug_data: Dictionary containing bug information
            project_key: Jira project key (e.g., 'TEST', 'QA')
            issue_type: Jira issue type (default: "Bug")

        Returns:
            Tuple of (success, ticket_key, message)
        """
        if not self.authenticated:
            return False, "", "Not authenticated. Please authenticate first."

        try:
            # Build bug description from bug_data
            description = self._format_bug_description(bug_data)

            # Determine priority based on severity
            severity = bug_data.get('severity', 'medium').lower()
            priority = {
                'critical': 'Highest',
                'high': 'High',
                'medium': 'Medium',
                'low': 'Low'
            }.get(severity, 'Medium')

            # Create issue payload
            issue_payload = {
                "fields": {
                    "project": {
                        "key": project_key
                    },
                    "summary": bug_data.get('summary', 'Bug detected during automated testing'),
                    "description": description,
                    "issuetype": {
                        "name": issue_type  # Use parameter instead of hardcoded "Bug"
                    },
                    "priority": {
                        "name": priority
                    }
                }
            }

            # Add labels if provided
            if bug_data.get('labels'):
                issue_payload["fields"]["labels"] = bug_data['labels']
            else:
                issue_payload["fields"]["labels"] = ["automated-testing", "test-pilot"]

            # Create the ticket
            url = f"{self.base_url}/rest/api/2/issue"
            response = self.session.post(url, json=issue_payload, timeout=30)

            if response.status_code in [200, 201]:
                result = response.json()
                ticket_key = result.get('key', '')
                logger.info(f"✅ Created Jira ticket: {ticket_key}")
                return True, ticket_key, f"Successfully created Jira ticket: {ticket_key}"
            else:
                error_msg = f"Failed to create ticket: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return False, "", error_msg

        except Exception as e:
            error_msg = f"Error creating Jira ticket: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg

    def _enhance_bug_description_with_ai(self, bug_data: Dict[str, Any], azure_client) -> str:
        """
        Use AI to enhance bug description with detailed steps to reproduce and analysis

        Args:
            bug_data: Bug information dictionary
            azure_client: Azure OpenAI client

        Returns:
            Enhanced bug description or None if AI enhancement fails
        """
        try:
            if not azure_client or not azure_client.is_configured():
                return None

            # Build context for AI
            bug_context = {
                'summary': bug_data.get('summary', ''),
                'type': bug_data.get('type', ''),
                'severity': bug_data.get('severity', ''),
                'description': bug_data.get('description', ''),
                'field_name': bug_data.get('field_name', ''),
                'step': bug_data.get('step', ''),
                'recommendation': bug_data.get('recommendation', ''),
                'element': bug_data.get('element', ''),
                'url': bug_data.get('url', ''),
                'error': bug_data.get('error', '')
            }

            # Create AI prompt
            prompt = f"""You are a QA engineer creating a Jira bug ticket. Enhance the following bug report with:
1. Clear, detailed description
2. Steps to reproduce
3. Expected vs Actual behavior
4. Impact assessment
5. Suggested fix (if applicable)

Bug Information:
- Summary: {bug_context['summary']}
- Type: {bug_context['type']}
- Severity: {bug_context['severity']}
- Description: {bug_context['description']}
- Field/Element: {bug_context['field_name'] or bug_context['element']}
- Step: {bug_context['step']}
- URL: {bug_context['url']}
- Error: {bug_context['error']}
- Recommendation: {bug_context['recommendation']}

Create a comprehensive bug description in Jira markdown format that includes:
- Clear problem statement
- Numbered steps to reproduce
- Expected vs Actual results
- Impact on users
- Suggested fix

Format using Jira markdown (h2, h3, *, numbered lists, etc.)"""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert QA engineer who writes clear, comprehensive bug reports for Jira. Use Jira markdown formatting."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = azure_client.chat_completion_create(
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )

            if response and 'choices' in response:
                enhanced_description = response['choices'][0]['message']['content']
                logger.info("✅ AI enhanced bug description")
                return enhanced_description

            return None

        except Exception as e:
            logger.warning(f"AI enhancement failed: {str(e)}")
            return None

    def _format_bug_description(self, bug_data: Dict[str, Any]) -> str:
        """Format bug data into Jira description"""
        description_parts = []

        # Add bug type and severity
        bug_type = bug_data.get('type', 'Unknown')
        severity = bug_data.get('severity', 'medium')
        description_parts.append(f"*Bug Type:* {bug_type}")
        description_parts.append(f"*Severity:* {severity.upper()}")
        description_parts.append("")

        # Add main description
        if bug_data.get('description'):
            description_parts.append("*Description:*")
            description_parts.append(bug_data['description'])
            description_parts.append("")

        # Add field information for validation issues
        if bug_data.get('field_name'):
            description_parts.append(f"*Affected Field:* {bug_data['field_name']}")
            if bug_data.get('field_type'):
                description_parts.append(f"*Field Type:* {bug_data['field_type']}")
            description_parts.append("")

        # Add step information
        if bug_data.get('step'):
            description_parts.append(f"*Detected at Step:* {bug_data['step']}")
            description_parts.append("")

        # Add recommendation
        if bug_data.get('recommendation'):
            description_parts.append("*Recommended Fix:*")
            description_parts.append(bug_data['recommendation'])
            description_parts.append("")

        # Add WCAG criterion for accessibility issues
        if bug_data.get('wcag_criterion'):
            description_parts.append(f"*WCAG Criterion:* {bug_data['wcag_criterion']}")
            description_parts.append("")

        # Add technical details
        if bug_data.get('url'):
            description_parts.append(f"*URL:* {bug_data['url']}")

        if bug_data.get('element'):
            description_parts.append(f"*Element:* {bug_data['element']}")

        # Add source information
        description_parts.append("")
        description_parts.append("_This bug was automatically detected by TestPilot during automated testing._")

        return "\n".join(description_parts)


class RobotMCPHelper:
    """
    Helper class for managing RobotMCP MCP client connections and tool calls

    Provides convenient methods for interacting with RobotMCP server via MCP protocol
    """

    def __init__(self):
        """Initialize RobotMCP helper"""
        self.session = None
        self.read_stream = None
        self.write_stream = None
        self.current_session_id = None
        self.is_connected = False
        self._stdio_ctx = None
        self._session_ctx = None
        self._cleanup_done = False

        # Register for cleanup on exit
        global _robotmcp_instances
        _robotmcp_instances.append(self)

    def __del__(self):
        """Destructor to ensure cleanup of async resources"""
        if not self._cleanup_done:
            try:
                self.shutdown()
            except Exception as e:
                # Suppress errors during cleanup to avoid issues in __del__
                pass

    async def connect(self) -> bool:
        """
        Connect to RobotMCP MCP server

        Returns:
            True if connection successful, False otherwise
        """
        if not ROBOTMCP_AVAILABLE:
            logger.warning("RobotMCP not available")
            return False

        try:
            import sys

            # Use Python to run the robotmcp server directly via its mcp object
            # The robotmcp.server module exports an 'mcp' FastMCP instance
            server_script = """
import sys
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Temporarily suppress stderr during imports to hide library loading warnings
original_stderr = sys.stderr
try:
    # Redirect stderr to devnull during imports
    sys.stderr = open(os.devnull, 'w')
    
    # Import the MCP server instance (this is where warnings occur)
    from robotmcp.server import mcp
    
finally:
    # Restore stderr for MCP protocol communication
    if sys.stderr != original_stderr:
        sys.stderr.close()
    sys.stderr = original_stderr

# Run the server
if __name__ == '__main__':
    try:
        mcp.run()
    except AttributeError:
        print("Error: mcp.run() not available", file=sys.stderr)
        sys.exit(1)
"""

            server_params = StdioServerParameters(
                command=sys.executable,
                args=["-c", server_script]
            )

            # Create stdio streams and client session
            stdio_ctx = stdio_client(server_params)
            self.read_stream, self.write_stream = await stdio_ctx.__aenter__()

            # Create ClientSession with the streams
            session_ctx = ClientSession(self.read_stream, self.write_stream)
            self.session = await session_ctx.__aenter__()

            # CRITICAL: Initialize the session (required by MCP protocol)
            await self.session.initialize()
            logger.debug("✅ MCP session initialized")

            # List available tools for debugging
            try:
                tools_result = await self.session.list_tools()
                tool_names = [tool.name for tool in tools_result.tools]
                logger.debug(f"Available RobotMCP tools: {tool_names[:10]}...")  # First 10
                self._available_tools = tool_names  # Store for reference
            except Exception as list_error:
                logger.debug(f"Could not list tools: {list_error}")
                self._available_tools = []

            # Store context managers for cleanup
            self._stdio_ctx = stdio_ctx
            self._session_ctx = session_ctx

            self.is_connected = True
            logger.info("✅ Connected to RobotMCP MCP server")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to RobotMCP: {str(e)}")
            logger.debug(f"Connection error details: {repr(e)}", exc_info=True)
            logger.info("💡 RobotMCP integration is optional. Tests will continue with fallback automation.")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from RobotMCP MCP server (async version)"""
        if self._cleanup_done:
            return

        if self.session:
            try:
                # Close session first
                if hasattr(self, '_session_ctx') and self._session_ctx:
                    await self._session_ctx.__aexit__(None, None, None)
                    self._session_ctx = None

                # Then close stdio streams
                if hasattr(self, '_stdio_ctx') and self._stdio_ctx:
                    await self._stdio_ctx.__aexit__(None, None, None)
                    self._stdio_ctx = None

                self.is_connected = False
                self.session = None
                self.read_stream = None
                self.write_stream = None
                self._cleanup_done = True
                logger.info("Disconnected from RobotMCP")
            except Exception as e:
                logger.error(f"Error disconnecting from RobotMCP: {str(e)}")
                self._cleanup_done = True

    def shutdown(self):
        """
        Synchronous shutdown for cleanup when event loop is not available
        Called during cleanup phase to avoid async issues
        """
        if self._cleanup_done:
            return

        try:
            self.is_connected = False

            # Try to close async generators properly if possible
            try:
                # Close the async context managers by calling their close methods
                if hasattr(self, '_stdio_ctx') and self._stdio_ctx is not None:
                    # Get the async generator and close it
                    if hasattr(self._stdio_ctx, 'aclose'):
                        try:
                            # Try to close the generator
                            self._stdio_ctx.aclose()
                        except:
                            pass
                    self._stdio_ctx = None

                if hasattr(self, '_session_ctx') and self._session_ctx is not None:
                    self._session_ctx = None
            except Exception as e:
                logger.debug(f"Error closing async contexts: {e}")

            # Clear references to prevent further use
            if hasattr(self, 'session'):
                self.session = None
            if hasattr(self, 'read_stream'):
                self.read_stream = None
            if hasattr(self, 'write_stream'):
                self.write_stream = None

            self._cleanup_done = True
            logger.debug("RobotMCP shutdown completed (synchronous)")
        except Exception as e:
            logger.debug(f"Error during synchronous shutdown: {e}")
            self._cleanup_done = True

    def _extract_result(self, result):
        """
        Extract actual data from MCP CallToolResult

        Args:
            result: CallToolResult object or dict

        Returns:
            Extracted data (dict, list, or str)
        """
        import json

        # Handle CallToolResult object
        if hasattr(result, 'content'):
            content = result.content
            if isinstance(content, list) and len(content) > 0:
                # MCP returns [{"type": "text", "text": "..."}]
                first_item = content[0]
                if hasattr(first_item, 'text'):
                    try:
                        return json.loads(first_item.text)
                    except json.JSONDecodeError:
                        return first_item.text
                elif isinstance(first_item, dict) and 'text' in first_item:
                    try:
                        return json.loads(first_item['text'])
                    except json.JSONDecodeError:
                        return first_item['text']
            return content

        # Handle dict response (legacy)
        if isinstance(result, dict):
            if 'content' in result:
                content = result['content']
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        try:
                            return json.loads(first_item['text'])
                        except json.JSONDecodeError:
                            return first_item['text']
                    return content
            return result

        # Return as-is if already the right type
        return result

    async def analyze_scenario(self, scenario: str, context: str = "web") -> Dict:
        """
        Analyze test scenario using RobotMCP

        Args:
            scenario: Test scenario description
            context: Context (web, mobile, api, etc.)

        Returns:
            Analysis results with test intent
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                logger.warning("Cannot analyze scenario - RobotMCP connection failed")
                return {"error": "Connection failed"}

        try:
            if not self.session:
                logger.warning("No active RobotMCP session")
                return {"error": "No session"}

            result = await self.session.call_tool(
                "analyze_scenario",  # No mcp_robotmcp_ prefix
                arguments={
                    "scenario": scenario,
                    "context": context
                }
            )

            # Extract content from CallToolResult
            result_data = self._extract_result(result)

            # Store session ID for subsequent calls
            if result_data and isinstance(result_data, dict):
                session_info = result_data.get("session_info", {})
                self.current_session_id = session_info.get("session_id", "default")

            return result_data

        except Exception as e:
            logger.error(f"Error analyzing scenario with RobotMCP: {str(e)}")
            return {"error": str(e)}

    async def discover_keywords(self, action_description: str, context: str = "web") -> List[Dict]:
        """
        Discover matching Robot Framework keywords

        Args:
            action_description: Description of the action
            context: Context for keyword discovery

        Returns:
            List of matching keywords with metadata
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                logger.warning("Cannot discover keywords - RobotMCP connection failed")
                return []

        try:
            if not self.session:
                logger.warning("No active RobotMCP session")
                return []

            # Use the correct tool name (no prefix)
            tool_name = "discover_keywords"

            # Log the call for debugging
            logger.debug(f"Calling {tool_name} with action='{action_description}', context='{context}'")

            result = await self.session.call_tool(
                tool_name,
                arguments={
                    "action_description": action_description,
                    "context": context
                }
            )

            # Extract data from CallToolResult
            result_data = self._extract_result(result)

            logger.debug(f"discover_keywords extracted result type: {type(result_data)}")

            # Handle different response formats
            if isinstance(result_data, dict):
                return result_data.get("keywords", result_data.get("result", []))
            elif isinstance(result_data, list):
                return result_data
            else:
                logger.warning(f"Unexpected result data type: {type(result_data)}")
                return []

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error discovering keywords: {error_msg}")

            # Provide helpful hints based on error type
            if "Invalid request parameters" in error_msg:
                logger.info("💡 Hint: Tool parameters may not match. Check available tools with list_tools()")
            elif "not found" in error_msg.lower():
                logger.info(f"💡 Hint: Tool might not exist. Available tools: {getattr(self, '_available_tools', 'unknown')[:5]}")

            logger.debug(f"Full error details: {repr(e)}", exc_info=True)
            logger.info("💡 Continuing without RobotMCP keyword discovery")
            return []

    async def execute_step(self, keyword: str, arguments: List[str] = None,
                          session_id: str = None, use_context: bool = True) -> Dict:
        """
        Execute a Robot Framework keyword step

        Args:
            keyword: Keyword to execute
            arguments: Keyword arguments
            session_id: Session ID (uses current if not provided)
            use_context: Whether to use RF context

        Returns:
            Execution result
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                logger.warning(f"Cannot execute step '{keyword}' - RobotMCP connection failed")
                return {"status": "FAIL", "error": "Connection failed"}

        session = session_id or self.current_session_id or "default"

        try:
            if not self.session:
                logger.warning("No active RobotMCP session")
                return {"status": "FAIL", "error": "No session"}

            result = await self.session.call_tool(
                "execute_step",  # No prefix
                arguments={
                    "keyword": keyword,
                    "arguments": arguments or [],
                    "session_id": session,
                    "use_context": use_context,
                    "detail_level": "minimal"
                }
            )

            # Extract data from CallToolResult
            result_data = self._extract_result(result)

            return result_data if isinstance(result_data, dict) else {"status": "PASS", "result": result_data}

        except Exception as e:
            logger.error(f"Error executing step '{keyword}': {str(e)}")
            return {"status": "FAIL", "error": str(e)}

    async def build_test_suite(self, test_name: str, session_id: str = None,
                               tags: List[str] = None, documentation: str = "") -> Dict:
        """
        Build Robot Framework test suite from executed steps

        Args:
            test_name: Name for the test case
            session_id: Session ID (uses current if not provided)
            tags: Test tags
            documentation: Test documentation

        Returns:
            Test suite generation result
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                logger.warning("Cannot build test suite - RobotMCP connection failed")
                return {"error": "Connection failed"}

        session = session_id or self.current_session_id or "default"

        try:
            if not self.session:
                logger.warning("No active RobotMCP session")
                return {"error": "No session"}

            result = await self.session.call_tool(
                "build_test_suite",  # No prefix
                arguments={
                    "test_name": test_name,
                    "session_id": session,
                    "tags": tags or [],
                    "documentation": documentation
                }
            )

            # Extract data from CallToolResult
            result_data = self._extract_result(result)

            return result_data if isinstance(result_data, dict) else {"result": result_data}

        except Exception as e:
            logger.error(f"Error building test suite: {str(e)}")
            return {"error": str(e)}

    async def get_page_source(self, session_id: str = None, filtered: bool = True) -> Dict:
        """
        Get page source from browser session

        Args:
            session_id: Session ID
            filtered: Return filtered page source

        Returns:
            Page source and metadata
        """
        if not self.is_connected:
            connected = await self.connect()
            if not connected:
                logger.warning("Cannot get page source - RobotMCP connection failed")
                return {"error": "Connection failed"}

        session = session_id or self.current_session_id or "default"

        try:
            if not self.session:
                logger.warning("No active RobotMCP session")
                return {"error": "No session"}

            result = await self.session.call_tool(
                "get_page_source",  # No prefix
                arguments={
                    "session_id": session,
                    "filtered": filtered,
                    "full_source": False
                }
            )

            # Extract data from CallToolResult
            result_data = self._extract_result(result)

            return result_data if isinstance(result_data, dict) else {"page_source": result_data}

        except Exception as e:
            logger.error(f"Error getting page source: {str(e)}")
            return {"error": str(e)}


class TestRecorder:
    """Records test actions for later playback and analysis"""
    pass  # Implementation placeholder


class BrowserAutomationManager:
    """
    Intelligent Browser Automation Manager for TestPilot

    Features:
    - Smart browser automation with step-by-step execution
    - Network log capture (XHR, Fetch, API calls)
    - Console error detection
    - DOM snapshot capture for locators and text
    - Screenshot capture at each step
    - Performance metrics tracking
    - AI-powered bug detection and analysis
    """

    def __init__(self, azure_client: Optional[AzureOpenAIClient] = None):
        self.azure_client = azure_client
        self.driver = None
        self.network_logs = []
        self.console_errors = []
        self.dom_snapshots = []
        self.screenshots = []
        self.performance_metrics = []
        self.captured_locators = {}
        self.captured_variables = {}
        self.filled_forms = set()  # Track forms that have been filled to prevent duplicates
        self.bug_report = {
            'functionality_issues': [],
            'ui_ux_issues': [],
            'performance_issues': [],
            'accessibility_issues': [],
            'security_issues': [],
            'validation_issues': [],
            'console_errors': [],
            'network_errors': []
        }

        # Initialize RobotMCP helper if available
        self.robotmcp_helper = RobotMCPHelper() if ROBOTMCP_AVAILABLE else None
        self.use_robotmcp = ROBOTMCP_AVAILABLE

    def initialize_browser(self, base_url: str, headless: bool = False, environment: str = 'prod') -> bool:
        """
        Initialize browser with logging capabilities and environment-specific configuration

        Args:
            base_url: Base URL to start from
            headless: Run in headless mode
            environment: Test environment (prod, qamain, stage, jarvisqa1, jarvisqa2)

        Returns:
            Success status
        """
        max_retries = 3
        retry_delay = 1  # Reduced from 2s to 1s for faster retries

        # Get environment configuration
        env_config = EnvironmentConfig.get_config(environment)
        logger.info(f"🌍 Initializing browser for environment: {env_config['name']} ({environment})")

        # Log configuration based on mode
        if env_config['mode'] == 'proxy':
            logger.info(f"   🔒 Proxy Mode: {env_config['proxy']}")
        elif env_config['mode'] == 'user_agent':
            logger.info(f"   🏷️  User Agent Mode (no proxy) - env tag in UA")
        else:
            logger.info(f"   🌐 Direct Access Mode")

        for attempt in range(1, max_retries + 1):
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                from selenium.webdriver.chrome.service import Service
                from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

                logger.info(f"🚀 Initializing browser (attempt {attempt}/{max_retries}) for: {base_url}")

                # Setup Chrome with advanced logging and stability options
                chrome_options = Options()

                # Basic arguments
                chrome_options.add_argument('--incognito')
                if headless:
                    chrome_options.add_argument('--headless=new')  # Use new headless mode
                    chrome_options.add_argument('--window-size=1920,1080')
                else:
                    chrome_options.add_argument('--start-maximized')

                # Stability and performance arguments
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--disable-software-rasterizer')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-infobars')
                chrome_options.add_argument('--disable-browser-side-navigation')
                chrome_options.add_argument('--disable-blink-features=AutomationControlled')
                chrome_options.add_argument('--disable-background-timer-throttling')
                chrome_options.add_argument('--disable-renderer-backgrounding')
                chrome_options.add_argument('--disable-backgrounding-occluded-windows')
                chrome_options.add_argument('--disable-features=TranslateUI,BlinkGenPropertyTrees')
                chrome_options.add_argument('--remote-debugging-port=0')  # Use random port

                # Environment-specific configuration
                # User agent with environment tag for non-prod
                chrome_options.add_argument(f'user-agent={env_config["user_agent"]}')
                logger.info(f"   🔧 User agent: {env_config['user_agent'][:80]}...")

                # Proxy configuration for non-prod environments
                if env_config['proxy']:
                    chrome_options.add_argument(f'--proxy-server={env_config["proxy"]}')
                    logger.info(f"   🔒 Proxy configured: {env_config['proxy']}")

                # Enable performance and network logging (only if needed)
                try:
                    chrome_options.set_capability('goog:loggingPrefs', {
                        'performance': 'ALL',
                        'browser': 'ALL'
                    })

                    # Enable Chrome DevTools Protocol
                    chrome_options.add_experimental_option('perfLoggingPrefs', {
                        'enableNetwork': True,
                        'enablePage': True,
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Could not set logging capabilities: {e}")

                # Exclude automation flags
                chrome_options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
                chrome_options.add_experimental_option('useAutomationExtension', False)

                # Initialize driver with explicit service and increased timeout
                try:
                    service = Service()
                    service.creation_flags = 0  # Prevent window creation on Windows
                    self.driver = webdriver.Chrome(service=service, options=chrome_options)
                    logger.info("   ✅ Chrome driver started with explicit service")
                except Exception as service_error:
                    logger.warning(f"   ⚠️ Service initialization failed: {service_error}, trying fallback...")
                    self.driver = webdriver.Chrome(options=chrome_options)
                    logger.info("   ✅ Chrome driver started with fallback method")

                # Set optimized timeouts
                self.driver.set_page_load_timeout(60)  # Keep 60s for slow pages
                self.driver.set_script_timeout(20)  # Reduced from 30s to 20s

                # Optimized implicit wait - reduced from 10s to 5s for faster failures
                # This affects all element location operations
                self.driver.implicitly_wait(5)

                # Try to maximize or set a large window size
                try:
                    if headless:
                        self.driver.set_window_size(1920, 1080)
                    else:
                        self.driver.maximize_window()
                except Exception as window_error:
                    logger.warning(f"   ⚠️ Window resize failed: {window_error}")
                    try:
                        self.driver.set_window_size(1920, 1080)
                    except Exception:
                        pass

                # Navigate to base URL with retry logic
                logger.info(f"   🌐 Navigating to: {base_url}")
                max_nav_retries = 2
                for nav_attempt in range(1, max_nav_retries + 1):
                    try:
                        self.driver.get(base_url)
                        self._wait_for_page_load(timeout=20)  # Reduced from 30s to 20s - fail fast
                        logger.info("   ✅ Page loaded successfully")
                        break
                    except Exception as nav_error:
                        if nav_attempt < max_nav_retries:
                            logger.warning(f"   ⚠️ Navigation attempt {nav_attempt} failed: {nav_error}, retrying...")
                            time.sleep(1)  # Reduced from 2s to 1s
                        else:
                            raise nav_error

                logger.info("✅ Browser initialized successfully")
                return True

            except ImportError as import_error:
                logger.error(f"❌ Selenium not installed. Install with: pip install selenium")
                return False

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Browser initialization failed (attempt {attempt}/{max_retries}): {error_msg}")

                # Cleanup driver if it was partially created
                try:
                    if hasattr(self, 'driver') and self.driver:
                        self.driver.quit()
                        self.driver = None
                        logger.info("   🧹 Cleaned up partial driver instance")
                except Exception as cleanup_error:
                    logger.warning(f"   ⚠️ Cleanup warning: {cleanup_error}")

                # If this was the last attempt, return False
                if attempt == max_retries:
                    logger.error(f"❌ All {max_retries} browser initialization attempts failed")
                    return False

                # Wait before retry with exponential backoff
                wait_time = retry_delay * attempt
                logger.info(f"   ⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        return False

    def _wait_for_page_load(self, timeout: int = 10):
        """Wait for page to be fully loaded with multiple readiness checks - OPTIMIZED"""
        try:
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Wait for document.readyState to be complete
            WebDriverWait(self.driver, timeout).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )

            # Quick check for jQuery if present (non-blocking)
            try:
                jquery_ready = self.driver.execute_script(
                    'return typeof jQuery != "undefined" && jQuery.active == 0'
                )
                # If jQuery is active, wait briefly for it to finish
                if not jquery_ready:
                    WebDriverWait(self.driver, 2).until(
                        lambda d: d.execute_script('return jQuery.active == 0')
                    )
            except:
                pass  # jQuery not present or already settled

            # Optimized wait for dynamic content - reduced from 1s to 0.3s
            # Most modern sites load fast, this is just a stability buffer
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"⚠️ Page load wait timeout after {timeout}s: {str(e)}")
            # Continue anyway - page might be partially loaded

    def capture_network_logs(self):
        """Capture network activity from performance logs"""
        try:
            logs = self.driver.get_log('performance')
            for entry in logs:
                try:
                    log_entry = json.loads(entry['message'])
                    message = log_entry.get('message', {})
                    method = message.get('method', '')

                    # Capture network requests and responses
                    if 'Network.' in method:
                        params = message.get('params', {})

                        if method == 'Network.requestWillBeSent':
                            request = params.get('request', {})
                            self.network_logs.append({
                                'type': 'request',
                                'url': request.get('url', ''),
                                'method': request.get('method', ''),
                                'headers': request.get('headers', {}),
                                'timestamp': entry['timestamp']
                            })

                        elif method == 'Network.responseReceived':
                            response = params.get('response', {})
                            status = response.get('status', 0)

                            log_entry = {
                                'type': 'response',
                                'url': response.get('url', ''),
                                'status': status,
                                'headers': response.get('headers', {}),
                                'timestamp': entry['timestamp']
                            }

                            # Flag errors
                            if status >= 400:
                                log_entry['is_error'] = True
                                self.bug_report['network_errors'].append({
                                    'url': response.get('url', ''),
                                    'status': status,
                                    'statusText': response.get('statusText', '')
                                })

                            self.network_logs.append(log_entry)

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error parsing network log: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"⚠️ Could not capture network logs: {str(e)}")

    def capture_console_errors(self):
        """Capture console errors and warnings"""
        try:
            logs = self.driver.get_log('browser')
            for entry in logs:
                level = entry.get('level', '')
                message = entry.get('message', '')

                if level in ['SEVERE', 'ERROR', 'WARNING']:
                    error_entry = {
                        'level': level,
                        'message': message,
                        'timestamp': entry.get('timestamp', '')
                    }
                    self.console_errors.append(error_entry)

                    if level in ['SEVERE', 'ERROR']:
                        self.bug_report['console_errors'].append(error_entry)

        except Exception as e:
            logger.warning(f"⚠️ Could not capture console logs: {str(e)}")

    def capture_dom_snapshot(self, step_description: str):
        """
        Capture DOM snapshot including locators, text, and structure

        Args:
            step_description: Description of current step
        """
        try:
            from selenium.webdriver.common.by import By

            snapshot = {
                'step': step_description,
                'timestamp': datetime.now().isoformat(),
                'url': self.driver.current_url,
                'title': self.driver.title,
                'locators': {},
                'text_content': {},
                'interactive_elements': []
            }

            # Capture common interactive elements
            elements_to_capture = [
                ('buttons', By.TAG_NAME, 'button'),
                ('links', By.TAG_NAME, 'a'),
                ('inputs', By.TAG_NAME, 'input'),
                ('selects', By.TAG_NAME, 'select'),
                ('textareas', By.TAG_NAME, 'textarea')
            ]

            for element_type, by_type, tag_name in elements_to_capture:
                try:
                    elements = self.driver.find_elements(by_type, tag_name)
                    for elem in elements[:30]:  # Limit to avoid massive snapshots
                        try:
                            if not elem.is_displayed():
                                continue

                            elem_id = elem.get_attribute('id')
                            elem_name = elem.get_attribute('name')
                            elem_class = elem.get_attribute('class')
                            elem_text = elem.text.strip()[:100] if elem.text else ''

                            # Build locator
                            locator = None
                            if elem_id:
                                locator = f"id:{elem_id}"
                                locator_name = f"{elem_id}_locator"
                            elif elem_name:
                                locator = f"name:{elem_name}"
                                locator_name = f"{elem_name}_locator"
                            elif elem_text and len(elem_text) < 50:
                                if tag_name == 'a':
                                    locator = f"link:{elem_text}"
                                else:
                                    locator = f"xpath://{tag_name}[contains(text(), '{elem_text[:30]}')]"
                                locator_name = f"{elem_text.lower().replace(' ', '_')[:30]}_locator"
                            elif elem_class:
                                classes = elem_class.split()
                                if classes:
                                    locator = f"css:.{classes[0]}"
                                    locator_name = f"{classes[0]}_locator"

                            if locator:
                                snapshot['locators'][locator_name] = locator
                                self.captured_locators[locator_name] = locator

                                # Capture text content
                                if elem_text:
                                    snapshot['text_content'][locator_name] = elem_text

                                # Track interactive elements
                                snapshot['interactive_elements'].append({
                                    'type': element_type,
                                    'locator': locator,
                                    'text': elem_text,
                                    'visible': elem.is_displayed(),
                                    'enabled': elem.is_enabled()
                                })

                        except Exception:
                            continue

                except Exception as e:
                    logger.debug(f"Could not capture {element_type}: {str(e)}")

            self.dom_snapshots.append(snapshot)
            logger.info(f"📸 Captured DOM snapshot: {len(snapshot['locators'])} locators found")

        except Exception as e:
            logger.warning(f"⚠️ Could not capture DOM snapshot: {str(e)}")

    def capture_screenshot(self, step_description: str) -> str:
        """
        Capture screenshot of current page

        Args:
            step_description: Description for filename

        Returns:
            Path to screenshot file
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_desc = ''.join(c if c.isalnum() else '_' for c in step_description.lower()[:30])
            filename = f"step_{timestamp}_{safe_desc}.png"

            screenshot_dir = os.path.join(ROOT_DIR, "screenshots", "test_pilot_automation")
            os.makedirs(screenshot_dir, exist_ok=True)

            filepath = os.path.join(screenshot_dir, filename)
            self.driver.save_screenshot(filepath)

            self.screenshots.append({
                'step': step_description,
                'path': filepath,
                'timestamp': timestamp
            })

            logger.info(f"📷 Screenshot saved: {filename}")
            return filepath

        except Exception as e:
            logger.warning(f"⚠️ Could not capture screenshot: {str(e)}")
            return ""

    def capture_performance_metrics(self):
        """Capture performance metrics"""
        try:
            # Get navigation timing
            navigation_timing = self.driver.execute_script(
                "return window.performance.timing"
            )

            # Calculate key metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'url': self.driver.current_url,
                'load_time': navigation_timing.get('loadEventEnd', 0) - navigation_timing.get('navigationStart', 0),
                'dom_ready': navigation_timing.get('domContentLoadedEventEnd', 0) - navigation_timing.get('navigationStart', 0),
                'first_paint': navigation_timing.get('responseStart', 0) - navigation_timing.get('navigationStart', 0)
            }

            # Flag slow pages
            if metrics['load_time'] > 5000:  # > 5 seconds
                self.bug_report['performance_issues'].append({
                    'url': metrics['url'],
                    'load_time': metrics['load_time'],
                    'severity': 'high' if metrics['load_time'] > 10000 else 'medium'
                })

            self.performance_metrics.append(metrics)

        except Exception as e:
            logger.debug(f"Could not capture performance metrics: {str(e)}")

    async def _execute_step_with_robotmcp(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """
        Execute a test step using RobotMCP keyword discovery and execution

        Args:
            step: TestStep to execute
            test_case: Parent TestCase for context

        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.robotmcp_helper or not self.robotmcp_helper.is_connected:
                await self.robotmcp_helper.connect()

            # Discover matching keywords for this step
            keywords = await self.robotmcp_helper.discover_keywords(
                action_description=step.description,
                context="web"
            )

            if not keywords:
                return False, "No matching keywords found"

            # Use the best matching keyword
            best_keyword = keywords[0]
            keyword_name = best_keyword.get("name", "")
            keyword_library = best_keyword.get("library", "")
            keyword_args = best_keyword.get("args", [])

            logger.info(f"🤖 RobotMCP: Using {keyword_library}.{keyword_name}")

            # Parse arguments from step description (simple heuristic)
            args = []
            description_lower = step.description.lower()

            # Extract arguments based on action type
            if "click" in keyword_name.lower():
                # Extract element locator
                # Look for quoted text or specific keywords
                import re
                quoted = re.findall(r'"([^"]*)"', step.description)
                if quoted:
                    args.append(quoted[0])
                elif "button" in description_lower:
                    button_match = re.search(r'button.*?(["\']([^"\']+)["\']|(\w+))', description_lower)
                    if button_match:
                        args.append(button_match.group(2) or button_match.group(3))

            elif "input" in keyword_name.lower() or "type" in keyword_name.lower():
                # Extract locator and value
                parts = step.description.split(" in " if " in " in step.description else " into ")
                if len(parts) >= 2:
                    args.append(parts[1].strip())  # locator
                    args.append(parts[0].replace("Enter", "").replace("Type", "").strip())  # value

            # Execute the keyword via RobotMCP
            result = await self.robotmcp_helper.execute_step(
                keyword=keyword_name,
                arguments=args,
                use_context=True
            )

            if result.get("status") == "PASS":
                return True, f"Executed {keyword_library}.{keyword_name}"
            else:
                error_msg = result.get("error", "Unknown error")
                return False, f"Keyword execution failed: {error_msg}"

        except Exception as e:
            logger.debug(f"RobotMCP execution error: {str(e)}")
            return False, str(e)

    async def execute_step_smartly(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """
        Execute a test step intelligently with AI assistance

        Args:
            step: TestStep to execute
            test_case: Parent TestCase for context

        Returns:
            Tuple of (success, message)
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException, NoSuchElementException

            logger.info(f"🎯 Executing Step {step.step_number}: {step.description}")

            # Try RobotMCP first if available and enabled
            if self.use_robotmcp and self.robotmcp_helper:
                try:
                    robotmcp_success, message = await self._execute_step_with_robotmcp(step, test_case)
                    if robotmcp_success:
                        logger.info(f"✅ Step executed via RobotMCP: {message}")
                        return True, message
                    else:
                        logger.debug(f"RobotMCP execution not applicable, using standard Selenium")
                except Exception as e:
                    logger.debug(f"RobotMCP execution error (falling back): {str(e)}")

            # Standard Selenium execution (fallback or default)
            # Capture state before action
            self.capture_network_logs()
            self.capture_console_errors()
            self.capture_dom_snapshot(step.description)
            self.capture_performance_metrics()

            description_lower = step.description.lower()
            success = False
            message = ""

            # Smart action detection and execution
            if any(word in description_lower for word in ['navigate', 'open', 'go to', 'visit']):
                # Navigation
                url_match = re.search(r'https?://[^\s\)]+', step.description)
                if url_match:
                    url = url_match.group(0).rstrip('/').rstrip(',').rstrip('.')
                    self.driver.get(url)
                    self._wait_for_page_load()
                    success = True
                    message = f"Navigated to {url}"
                else:
                    message = "No URL found in navigation step"

            elif any(word in description_lower for word in ['click', 'press', 'select', 'choose']):
                # Click action - smart locator finding
                success, message = await self._smart_click(step, test_case)

            elif any(word in description_lower for word in ['hover', 'mouse over', 'mouseover']):
                # Hover action - use ActionChains
                success, message = await self._smart_hover(step, test_case)

            elif any(word in description_lower for word in ['enter', 'input', 'type', 'fill']):
                # Input action - smart field finding
                success, message = await self._smart_input(step, test_case)

            elif any(word in description_lower for word in ['verify', 'check', 'confirm', 'validate']):
                # Verification action
                success, message = self._smart_verify(step, test_case)

            else:
                # Default: try to find and click
                success, message = await self._smart_click(step, test_case)

            # OPTIMIZED: Capture state after action in parallel for faster execution
            import asyncio
            await asyncio.gather(
                asyncio.to_thread(self.capture_screenshot, step.description),
                asyncio.to_thread(self.capture_network_logs),
                asyncio.to_thread(self.capture_console_errors),
                return_exceptions=True
            )

            # Analyze for issues
            self._analyze_step_for_issues(step, success, message)

            return success, message

        except Exception as e:
            logger.error(f"❌ Error executing step: {str(e)}")
            return False, f"Error: {str(e)}"

    def _infer_locator_name(self, description: str) -> str:
        """
        Infer locator variable name from step description

        CRITICAL: This method generates the locator name that will be used for both:
        1. Capturing locators during browser automation
        2. Requesting locators during file generation

        Uses the same logic to ensure names match!
        """
        # Use consistent logic: clean description, filter short words, take first 5
        clean_desc = ''.join(c if c.isalnum() or c == ' ' else '_' for c in description.lower())
        words = [w for w in clean_desc.split() if len(w) > 2][:5]  # Filter small words (<=2 chars), take first 5

        if not words:
            return "element_locator"

        # Build locator name - consistent format
        locator_name = '_'.join(words) + '_locator'
        return locator_name

    def _capture_element_locator(self, element, step: TestStep, action_type: str, test_case: TestCase = None):
        """
        Capture the ACTUAL locator for an element that was successfully interacted with.
        This fixes the "NEED_TO_UPDATE" issue by capturing real, working locators.

        Args:
            element: Selenium WebElement that was successfully used
            step: Test step being executed
            action_type: Type of action (click, input, select, etc.)
            test_case: Parent test case (optional)
        """
        # Import at method level to ensure it's available for exception handling
        from selenium.common.exceptions import StaleElementReferenceException

        try:
            # Use the centralized method to generate locator name for consistency
            locator_base = self._infer_locator_name(step.description)

            # Fallback if the method returns a generic name
            if locator_base == "element_locator":
                locator_base = f'step_{step.step_number}_locator'

            # Try to get the best locator in priority order
            locators_found = []

            # Helper function to safely get element attribute with retry
            def safe_get_attribute(attr_name, max_retries=2):
                for attempt in range(max_retries):
                    try:
                        return element.get_attribute(attr_name)
                    except StaleElementReferenceException:
                        if attempt == max_retries - 1:
                            logger.debug(f"Element became stale while getting attribute '{attr_name}'")
                            return None
                        time.sleep(0.1)
                return None

            # Helper function to safely get element text with retry
            def safe_get_text(max_retries=2):
                for attempt in range(max_retries):
                    try:
                        return element.text.strip()
                    except StaleElementReferenceException:
                        if attempt == max_retries - 1:
                            logger.debug("Element became stale while getting text")
                            return ""
                        time.sleep(0.1)
                return ""

            # Helper function to safely get tag name with retry
            def safe_get_tag_name(max_retries=2):
                for attempt in range(max_retries):
                    try:
                        return element.tag_name.lower()
                    except StaleElementReferenceException:
                        if attempt == max_retries - 1:
                            logger.debug("Element became stale while getting tag name")
                            return "unknown"
                        time.sleep(0.1)
                return "unknown"

            # Priority 1: ID (fastest and most reliable)
            element_id = safe_get_attribute('id')
            if element_id and len(element_id) > 0:
                locators_found.append(('id', f"id:{element_id}", 1))

            # Priority 2: Name attribute
            element_name = safe_get_attribute('name')
            if element_name and len(element_name) > 0:
                locators_found.append(('name', f"name:{element_name}", 2))

            # Priority 3: Data attributes (test-friendly)
            for attr in ['data-testid', 'data-test', 'data-qa', 'data-cy', 'data-test-id', 'data-element-label']:
                data_value = safe_get_attribute(attr)
                if data_value:
                    locators_found.append((attr, f"css:[{attr}='{data_value}']", 3))
                    break

            # Priority 4: Aria-label (accessibility)
            aria_label = safe_get_attribute('aria-label')
            if aria_label and len(aria_label) < 100:
                locators_found.append(('aria-label', f"css:[aria-label='{aria_label}']", 4))

            # Priority 5: Text content (for links and buttons)
            element_text = safe_get_text()
            if element_text and len(element_text) < 50 and len(element_text) > 0:
                tag_name = safe_get_tag_name()
                if tag_name in ['a', 'button', 'span', 'label']:
                    locators_found.append(('text', f"link:{element_text}", 5))

            # Priority 6: CSS class (if not too generic)
            element_class = safe_get_attribute('class')
            if element_class:
                classes = element_class.split()
                # Filter out generic/framework classes
                specific_classes = [c for c in classes if len(c) > 3 and
                                  not c.startswith('btn-') and
                                  not c.startswith('mat-') and
                                  not c.startswith('ng-') and
                                  not c in ['active', 'disabled', 'selected', 'hidden', 'visible']]
                if specific_classes:
                    locators_found.append(('class', f"css:.{specific_classes[0]}", 6))

            # Priority 7: XPath (last resort - most reliable fallback)
            try:
                xpath = self.driver.execute_script("""
                    function getXPath(element) {
                        if (element.id !== '') {
                            return '//*[@id="' + element.id + '"]';
                        }
                        if (element === document.body) {
                            return '/html/body';
                        }
                        var ix = 0;
                        var siblings = element.parentNode ? element.parentNode.childNodes : [];
                        for (var i = 0; i < siblings.length; i++) {
                            var sibling = siblings[i];
                            if (sibling === element) {
                                var path = element.parentNode ? getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() : '';
                                if (ix > 0) path += '[' + (ix + 1) + ']';
                                return path;
                            }
                            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                                ix++;
                            }
                        }
                        return '';
                    }
                    return getXPath(arguments[0]);
                """, element)
                if xpath and len(xpath) > 0:
                    locators_found.append(('xpath', f"xpath:{xpath}", 7))
            except (StaleElementReferenceException, Exception) as xpath_error:
                logger.debug(f"Could not generate XPath: {xpath_error}")

            # Store the best locator found
            if locators_found:
                # Sort by priority (lower number is better)
                locators_found.sort(key=lambda x: x[2])
                best_locator_type, best_locator_value, priority = locators_found[0]

                # Store in captured_locators for use in file generation
                self.captured_locators[locator_base] = best_locator_value

                # Also store in test case metadata if available
                if test_case:
                    # Ensure metadata dict exists and is properly initialized
                    if not hasattr(test_case, 'metadata'):
                        test_case.metadata = {}
                    elif test_case.metadata is None:
                        test_case.metadata = {}

                    if 'captured_locators' not in test_case.metadata:
                        test_case.metadata['captured_locators'] = {}

                    # Store the captured locator
                    test_case.metadata['captured_locators'][locator_base] = best_locator_value

                    logger.debug(f"      💾 Stored in test_case.metadata: test_case.metadata['captured_locators']['{locator_base}'] = '{best_locator_value}'")

                # Get tag name safely for metadata
                tag_name = safe_get_tag_name()

                # Store additional metadata about the element
                element_info = {
                    'locator': best_locator_value,
                    'type': best_locator_type,
                    'priority': priority,
                    'element_tag': tag_name,
                    'element_text': element_text[:50] if element_text else '',
                    'step_number': step.step_number,
                    'action_type': action_type
                }

                # If it's an input field, capture the value too
                if action_type == 'input':
                    element_value = safe_get_attribute('value')
                    if element_value:
                        var_name = locator_base.replace('_locator', '_variable')
                        self.captured_variables[var_name] = element_value
                        element_info['captured_value'] = element_value

                logger.info(f"   ✅ CAPTURED: {locator_base} = '{best_locator_value}' (priority {priority}: {best_locator_type})")

                # DEBUG: Log capture details
                logger.info(f"      📝 Step #{step.step_number}: {step.description[:50]}...")
                logger.info(f"      🎯 Locator name: {locator_base}")
                logger.info(f"      💾 Stored in self.captured_locators: {locator_base in self.captured_locators}")
                if test_case:
                    logger.info(f"      💾 Stored in test_case.metadata: {locator_base in test_case.metadata.get('captured_locators', {})}")

                # Store step metadata
                if not hasattr(step, 'metadata') or step.metadata is None:
                    step.metadata = {}
                step.metadata.update(element_info)

                return element_info
            else:
                logger.warning(f"   ⚠️  Could not extract any locator for element in step {step.step_number}")
                return None

        except StaleElementReferenceException as e:
            logger.warning(f"   ⚠️  Element became stale during locator capture: {str(e)[:100]}")
            return None
        except Exception as e:
            logger.warning(f"   ⚠️  Error capturing locator: {str(e)[:100]}")
            return None

    async def _smart_click(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """Smart click with AI-powered element finding"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Extract target text from description
            description = step.description.lower()

            # Try multiple strategies
            strategies = []

            # Strategy 1: Look for quoted text
            quoted = re.findall(r'"([^"]+)"', step.description)
            if quoted:
                strategies.append(('link', quoted[0]))
                strategies.append(('xpath', f"//button[contains(text(), '{quoted[0]}')]"))
                strategies.append(('xpath', f"//*[contains(text(), '{quoted[0]}')]"))

            # Strategy 2: Extract key words
            key_words = ['button', 'link', 'menu', 'explore', 'continue', 'submit', 'checkout', 'plan', 'select']
            for word in key_words:
                if word in description:
                    strategies.append(('xpath', f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{word}')]"))
                    strategies.append(('xpath', f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{word}')]"))

            # Strategy 3: Use captured locators from DOM snapshot
            for locator_name, locator_value in self.captured_locators.items():
                if any(word in locator_name.lower() for word in description.split()):
                    strategy, value = locator_value.split(':', 1)
                    by_type = self._get_by_type(strategy)
                    if by_type:
                        strategies.append((by_type, value))

            # Try each strategy
            for by_type, value in strategies:
                try:
                    if isinstance(by_type, str):
                        by_type = self._get_by_type(by_type)

                    element = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((by_type, value))
                    )
                    element.click()

                    logger.info(f"✅ Clicked element using: {by_type}={value}")

                    # CAPTURE THE ACTUAL LOCATOR that worked
                    self._capture_element_locator(element, step, "click", test_case)

                    return True, f"Successfully clicked: {value}"

                except Exception:
                    continue

            return False, f"Could not find clickable element for: {step.description}"

        except Exception as e:
            return False, f"Click error: {str(e)}"

    async def _smart_hover(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """Smart hover with AI-powered element finding using ActionChains"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.action_chains import ActionChains

            # Extract target text from description
            description = step.description.lower()

            # Try multiple strategies to find hover target
            strategies = []

            # Strategy 1: Look for quoted text
            quoted = re.findall(r'"([^"]+)"', step.description)
            if quoted:
                target_text = quoted[0]
                # Try link text
                strategies.append(('link_text', target_text))
                # Try partial link text
                strategies.append(('partial_link_text', target_text))
                # Try XPath with text contains
                strategies.append(('xpath', f"//*[contains(text(), '{target_text}')]"))
                # Try nav/menu specific elements
                strategies.append(('xpath', f"//nav//*[contains(text(), '{target_text}')]"))
                strategies.append(('xpath', f"//ul//*[contains(text(), '{target_text}')]"))
                strategies.append(('xpath', f"//*[@role='menuitem' and contains(text(), '{target_text}')]"))
                # Try anchor tags
                strategies.append(('xpath', f"//a[contains(text(), '{target_text}')]"))

            # Strategy 2: Look for 'navbar' or 'menu' context
            if 'navbar' in description or 'menu' in description or 'header' in description:
                if quoted:
                    target_text = quoted[0]
                    strategies.insert(0, ('xpath', f"//nav//a[contains(text(), '{target_text}')]"))
                    strategies.insert(0, ('xpath', f"//header//a[contains(text(), '{target_text}')]"))
                    strategies.insert(0, ('xpath', f"//*[contains(@class, 'nav')]//a[contains(text(), '{target_text}')]"))

            # Try each strategy
            for by_type, value in strategies:
                try:
                    if isinstance(by_type, str):
                        by_type = self._get_by_type(by_type)

                    element = WebDriverWait(self.driver, 5).until(
                        EC.presence_of_element_located((by_type, value))
                    )

                    # Use ActionChains to hover
                    actions = ActionChains(self.driver)
                    actions.move_to_element(element).perform()

                    # Small pause to let hover effects activate
                    import time
                    time.sleep(0.5)

                    logger.info(f"✅ Hovered element using: {by_type}={value}")

                    # CAPTURE THE ACTUAL LOCATOR that worked
                    self._capture_element_locator(element, step, "hover", test_case)

                    return True, f"Successfully hovered: {value}"

                except Exception as e:
                    logger.debug(f"   Hover strategy failed ({by_type}={value}): {str(e)[:50]}")
                    continue

            return False, f"Could not find element to hover for: {step.description}"

        except Exception as e:
            return False, f"Hover error: {str(e)}"

    async def _smart_input(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """Smart input with AI-powered field finding and auto-fill for forms"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.support.ui import Select

            description_lower = step.description.lower()

            # Check if this is a form-filling step (multiple fields)
            is_form_fill = any(keyword in description_lower for keyword in [
                'random valid data', 'fill form', 'enter billing', 'complete form',
                'input data', 'provide information', 'fill all fields'
            ])

            if is_form_fill:
                logger.info("   📋 Detected FORM FILLING step - will auto-fill all fields")
                return await self._smart_form_fill(step, test_case)

            # Single field input - extract value and field
            value_match = re.search(r'"([^"]+)"', step.description)
            input_value = value_match.group(1) if value_match else "test_input"

            # Try to find input field
            strategies = [
                (By.NAME, 'search'),
                (By.NAME, 'domain'),
                (By.NAME, 'email'),
                (By.NAME, 'username'),
                (By.ID, 'search'),
                (By.CSS_SELECTOR, 'input[type="text"]'),
                (By.CSS_SELECTOR, 'input[type="search"]'),
                (By.CSS_SELECTOR, 'input[placeholder]')
            ]

            # Add captured input locators
            for locator_name, locator_value in self.captured_locators.items():
                if 'input' in locator_name.lower() or 'field' in locator_name.lower():
                    strategy, value = locator_value.split(':', 1)
                    by_type = self._get_by_type(strategy)
                    if by_type:
                        strategies.insert(0, (by_type, value))

            for by_type, value in strategies:
                try:
                    element = WebDriverWait(self.driver, 3).until(
                        EC.presence_of_element_located((by_type, value))
                    )
                    element.clear()
                    element.send_keys(input_value)

                    # Check if this is an address field and if a dropdown appeared
                    field_name = element.get_attribute('name') or element.get_attribute('id') or ''
                    is_address_field = any(keyword in field_name.lower() for keyword in [
                        'address', 'street', 'addr'
                    ])

                    if is_address_field:
                        # Optimized wait for autocomplete dropdown - reduced from 1s to 0.3s
                        time.sleep(0.3)

                        # Try to find and select from address dropdown
                        dropdown_selected = await self._handle_address_dropdown(element)
                        if dropdown_selected:
                            logger.info(f"✅ Selected address from dropdown for field: {field_name}")

                    logger.info(f"✅ Input text using: {by_type}={value}")

                    # CAPTURE THE ACTUAL LOCATOR that worked
                    self._capture_element_locator(element, step, "input", test_case)

                    return True, f"Successfully entered: {input_value}"

                except Exception:
                    continue

            return False, f"Could not find input field for: {step.description}"

        except Exception as e:
            return False, f"Input error: {str(e)}"

    async def _handle_address_dropdown(self, input_element) -> bool:
        """
        Handle address autocomplete dropdown selection

        Args:
            input_element: The address input field element

        Returns:
            True if dropdown was found and option selected, False otherwise
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.keys import Keys

            logger.info("      🔍 Searching for address dropdown...")

            # Optimized dropdown wait - reduced from 2s to 0.5s
            time.sleep(0.5)

            # Common selectors for address autocomplete dropdowns
            dropdown_selectors = [
                '[role="listbox"]',
                '.autocomplete-dropdown',
                '.pac-container',  # Google Places autocomplete
                '.address-suggestions',
                '[class*="dropdown"]',
                '[class*="suggestion"]',
                '[class*="autocomplete"]',
                'ul.suggestions',
                '.dropdown-menu',
                '[id*="dropdown"]',
                '[id*="suggestions"]',
                '[id*="autocomplete"]'
            ]

            # Try to find visible dropdown with multiple attempts
            for attempt in range(3):
                logger.debug(f"      Attempt {attempt + 1} to find dropdown...")

                for selector in dropdown_selectors:
                    try:
                        dropdowns = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        logger.debug(f"      Found {len(dropdowns)} elements matching '{selector}'")

                        visible_dropdown = None

                        for dropdown in dropdowns:
                            try:
                                if dropdown.is_displayed():
                                    visible_dropdown = dropdown
                                    logger.info(f"      ✓ Found visible dropdown using selector: {selector}")
                                    break
                            except:
                                continue

                        if visible_dropdown:
                            # Find selectable options within the dropdown
                            option_selectors = [
                                '[role="option"]',
                                'li',
                                '.suggestion-item',
                                '.pac-item',
                                '[class*="option"]',
                                '[class*="item"]',
                                'a',
                                'div[tabindex]'
                            ]

                            for opt_selector in option_selectors:
                                try:
                                    options = visible_dropdown.find_elements(By.CSS_SELECTOR, opt_selector)
                                    clickable_options = [opt for opt in options if opt.is_displayed()]

                                    logger.debug(f"      Found {len(clickable_options)} clickable options")

                                    if clickable_options:
                                        # Select the first valid option
                                        first_option = clickable_options[0]
                                        option_text = first_option.text[:50] if first_option.text else "unknown"

                                        logger.info(f"      🎯 Attempting to select: '{option_text}'")

                                        # Try multiple click methods
                                        try:
                                            # Method 1: Direct click
                                            first_option.click()
                                            logger.info(f"      ✅ Selected via direct click")
                                            time.sleep(0.2)  # Reduced from 0.5s
                                            return True
                                        except:
                                            pass

                                        try:
                                            # Method 2: JavaScript click
                                            self.driver.execute_script("arguments[0].click();", first_option)
                                            logger.info(f"      ✅ Selected via JavaScript click")
                                            time.sleep(0.2)  # Reduced from 0.5s
                                            return True
                                        except:
                                            pass

                                        try:
                                            # Method 3: Move to element and click
                                            from selenium.webdriver.common.action_chains import ActionChains
                                            actions = ActionChains(self.driver)
                                            actions.move_to_element(first_option).click().perform()
                                            logger.info(f"      ✅ Selected via ActionChains")
                                            time.sleep(0.2)  # Reduced from 0.5s
                                            return True
                                        except:
                                            pass

                                except Exception as opt_error:
                                    logger.debug(f"      Option selector '{opt_selector}' failed: {str(opt_error)[:30]}")
                                    continue

                    except Exception as sel_error:
                        logger.debug(f"      Selector '{selector}' failed: {str(sel_error)[:30]}")
                        continue

                # Optimized retry delay - reduced from 0.5s to 0.2s
                if attempt < 2:
                    time.sleep(0.2)

            # If no dropdown found, try keyboard navigation (arrow down + enter)
            logger.info("      🎹 Trying keyboard navigation...")
            try:
                input_element.send_keys(Keys.ARROW_DOWN)
                time.sleep(0.2)  # Reduced from 0.5s
                input_element.send_keys(Keys.ENTER)
                logger.info("      ✅ Selected address using keyboard navigation")
                time.sleep(0.2)  # Reduced from 0.5s
                return True
            except Exception as kb_error:
                logger.debug(f"      Keyboard navigation failed: {str(kb_error)[:50]}")

            logger.info("      ⚠️  No address dropdown found after all attempts")
            return False

        except Exception as e:
            logger.debug(f"Address dropdown handling error: {str(e)}")
            return False

    async def _generate_ai_test_data(self, field_info: Dict[str, str], page_context: str = "") -> str:
        """
        Use Azure OpenAI to generate realistic, context-aware test data

        Args:
            field_info: Dict with field_type, name, id, placeholder, label
            page_context: Context about the page (title, URL, form purpose)

        Returns:
            Intelligent test data string
        """
        if not self.azure_client or not self.azure_client.is_configured():
            # Fallback to rule-based generation
            return self._generate_test_data_for_field(
                field_info.get('type', 'text'),
                field_info.get('name', ''),
                field_info.get('id', ''),
                field_info.get('placeholder', ''),
                field_info.get('label', '')
            )

        try:
            # Build context for AI
            field_type = field_info.get('type', 'text')
            field_name = field_info.get('name', '')
            field_id = field_info.get('id', '')
            placeholder = field_info.get('placeholder', '')
            label = field_info.get('label', '')
            validation_pattern = field_info.get('pattern', '')
            required = field_info.get('required', False)
            maxlength = field_info.get('maxlength', '')
            minlength = field_info.get('minlength', '')

            # Extract business context from all available clues
            all_context = f"{field_type} {field_name} {field_id} {placeholder} {label}".lower()

            # Determine domain/industry context
            domain_hints = []
            if any(k in page_context.lower() for k in ['ecommerce', 'shop', 'cart', 'checkout', 'product', 'order']):
                domain_hints.append("E-commerce/Shopping")
            if any(k in page_context.lower() for k in ['bank', 'finance', 'payment', 'credit', 'account']):
                domain_hints.append("Banking/Finance")
            if any(k in page_context.lower() for k in ['medical', 'health', 'patient', 'clinic', 'doctor']):
                domain_hints.append("Healthcare")
            if any(k in page_context.lower() for k in ['register', 'signup', 'create account', 'join']):
                domain_hints.append("User Registration")
            if any(k in page_context.lower() for k in ['login', 'signin', 'authenticate']):
                domain_hints.append("User Login")
            if any(k in page_context.lower() for k in ['contact', 'inquiry', 'feedback', 'support']):
                domain_hints.append("Contact/Support Form")

            domain_context = ", ".join(domain_hints) if domain_hints else "General Web Form"

            # Build intelligent prompt with comprehensive context
            prompt = f"""You are an expert QA test data generator. Generate REALISTIC, INTELLIGENT, and CONTEXT-AWARE test data for this specific form field.

🎯 FIELD IDENTIFICATION:
- Field Type: {field_type}
- Field Name: {field_name}
- Field ID: {field_id}
- Label Text: {label}
- Placeholder: {placeholder}

🌐 BUSINESS CONTEXT:
- Page/Application: {page_context}
- Domain/Industry: {domain_context}
- Form Purpose: {self._infer_form_purpose(all_context)}

📋 VALIDATION CONSTRAINTS:
- Required: {required}
- Pattern: {validation_pattern if validation_pattern else 'None specified'}
- Min Length: {minlength if minlength else 'None'}
- Max Length: {maxlength if maxlength else 'None'}

🧠 INTELLIGENCE RULES:
1. **Realism**: Use data that looks authentic, not generic "test123" values
2. **Context Awareness**: Consider the business domain and form purpose
3. **Consistency**: If this is a name field on a checkout form, use a realistic customer name
4. **Validation Compliance**: Respect all validation patterns and constraints
5. **Format Precision**: Match exact format requirements (dates, phones, emails, etc.)
6. **Business Logic**: For shipping forms use real addresses, for payment use test-safe card numbers
7. **Regional Appropriateness**: Use region-appropriate formats based on context
8. **Data Relationships**: Consider how this field relates to others (e.g., city matches state)

🎨 FORMAT EXAMPLES BY TYPE:
- Email: firstname.lastname@domain.com (use realistic names)
- Phone: (555) 123-4567 or +1-555-123-4567 or 555.123.4567 (match regional format)
- Name: Use culturally appropriate realistic names (not "Test User")
- Address: Use realistic street addresses with proper formatting
- City: Use real city names appropriate to the region
- State/Province: Use proper 2-letter codes or full names based on field format
- Zip Code: Use realistic formats (US: 12345 or 12345-6789, UK: SW1A 1AA, etc.)
- Credit Card: Use valid test card numbers (4532 1111 1111 1111 for Visa test)
- CVV: 3-4 digits based on card type
- Date: Use appropriate format (MM/DD/YYYY, YYYY-MM-DD, DD/MM/YYYY based on region)
- Password: Strong passwords with mix of upper, lower, numbers, special chars
- Username: Realistic usernames based on context (not "testuser123")
- Company: Use realistic company names for business context

🚨 CRITICAL REQUIREMENTS:
- Return ONLY the raw value to be entered in the field
- NO explanations, NO quotes, NO additional text
- Must be copy-paste ready for direct field input
- Must pass field validation if validation rules are specified

Generate the test data value now:"""

            response = self.azure_client.completion_create(
                prompt=prompt,
                max_tokens=150,
                temperature=0.8  # Higher temperature for more variety
            )

            if response and 'choices' in response and len(response['choices']) > 0:
                generated_value = response['choices'][0]['message']['content'].strip()
                # Clean up the response
                generated_value = generated_value.strip('"').strip("'").strip()
                # Remove any explanatory text that might have been added
                if '\n' in generated_value:
                    generated_value = generated_value.split('\n')[0].strip()
                logger.info(f"      🤖 AI generated data for '{field_name or field_id}': {generated_value if field_type != 'password' else '***'}")
                return generated_value
            else:
                # Fallback to rule-based
                return self._generate_test_data_for_field(field_type, field_name, field_id, placeholder, label)

        except Exception as e:
            logger.warning(f"      ⚠️  AI generation failed: {str(e)[:50]}, using fallback")
            # Fallback to rule-based generation
            return self._generate_test_data_for_field(field_type, field_name, field_id, placeholder, label)

    def _infer_form_purpose(self, context: str) -> str:
        """Infer the purpose of the form from context clues"""
        if any(k in context for k in ['login', 'signin', 'sign in']):
            return "User Authentication/Login"
        elif any(k in context for k in ['register', 'signup', 'sign up', 'create account']):
            return "User Registration/Account Creation"
        elif any(k in context for k in ['checkout', 'billing', 'shipping', 'payment']):
            return "E-commerce Checkout/Payment"
        elif any(k in context for k in ['contact', 'inquiry', 'feedback', 'message']):
            return "Contact/Communication Form"
        elif any(k in context for k in ['search', 'query', 'find']):
            return "Search/Query Form"
        elif any(k in context for k in ['subscribe', 'newsletter', 'email']):
            return "Newsletter/Subscription"
        elif any(k in context for k in ['profile', 'account', 'settings']):
            return "Profile/Account Management"
        elif any(k in context for k in ['order', 'purchase', 'buy']):
            return "Order/Purchase Form"
        else:
            return "Data Entry Form"

    async def _select_ai_dropdown_option(self, field_name: str, options: list, page_context: str = "") -> Optional[str]:
        """
        Use Azure OpenAI to intelligently select the most appropriate dropdown option

        Args:
            field_name: Name of the dropdown field
            options: List of available option texts
            page_context: Context about the page

        Returns:
            Selected option text or None
        """
        if not options or len(options) == 0:
            return None

        # Remove empty/placeholder options
        valid_options = [opt for opt in options if opt.strip() and not opt.lower().startswith('select') and opt.lower() not in ['', '--', '---', 'choose', 'pick']]

        if not valid_options:
            return None

        # If only one option, select it
        if len(valid_options) == 1:
            return valid_options[0]

        # Smart fallback selection based on field context
        def smart_fallback_selection(field_name: str, options: list) -> str:
            """Intelligent fallback when AI is not available"""
            import random
            field_lower = field_name.lower()

            # State/Province selection - prefer common states
            if 'state' in field_lower or 'province' in field_lower:
                preferred = ['California', 'CA', 'New York', 'NY', 'Texas', 'TX', 'Florida', 'FL']
                for pref in preferred:
                    for opt in options:
                        if pref.lower() in opt.lower():
                            return opt

            # Country selection - prefer US
            elif 'country' in field_lower:
                preferred = ['United States', 'USA', 'US', 'America']
                for pref in preferred:
                    for opt in options:
                        if pref.lower() in opt.lower():
                            return opt

            # Title/Salutation - prefer common titles
            elif 'title' in field_lower or 'salutation' in field_lower:
                preferred = ['Mr', 'Mrs', 'Ms', 'Dr']
                for pref in preferred:
                    for opt in options:
                        if opt.strip() == pref or opt.strip() == pref + '.':
                            return opt

            # Gender - random but realistic
            elif 'gender' in field_lower or 'sex' in field_lower:
                preferred = ['Male', 'Female']
                for pref in preferred:
                    for opt in options:
                        if pref.lower() == opt.lower():
                            return opt

            # Quantity - prefer middle range
            elif 'quantity' in field_lower or 'qty' in field_lower or 'amount' in field_lower:
                # Try to find numeric options
                numeric_opts = []
                for opt in options:
                    try:
                        val = int(opt.strip())
                        if 1 <= val <= 10:
                            numeric_opts.append(opt)
                    except:
                        pass
                if numeric_opts:
                    return random.choice(numeric_opts[:5])  # Prefer first 5

            # Month - prefer current or next month
            elif 'month' in field_lower:
                from datetime import datetime
                current_month = datetime.now().strftime('%B')
                for opt in options:
                    if current_month.lower() in opt.lower():
                        return opt

            # Year - prefer current or recent year
            elif 'year' in field_lower:
                from datetime import datetime
                current_year = str(datetime.now().year)
                for opt in options:
                    if current_year in opt:
                        return opt

            # Payment method - prefer credit card
            elif 'payment' in field_lower or 'method' in field_lower:
                preferred = ['Credit Card', 'Visa', 'Mastercard', 'Card']
                for pref in preferred:
                    for opt in options:
                        if pref.lower() in opt.lower():
                            return opt

            # Shipping method - prefer standard
            elif 'shipping' in field_lower or 'delivery' in field_lower:
                preferred = ['Standard', 'Regular', 'Ground', 'Normal']
                for pref in preferred:
                    for opt in options:
                        if pref.lower() in opt.lower():
                            return opt

            # Default: return first non-placeholder option or random
            return options[0] if options else None

        if not self.azure_client or not self.azure_client.is_configured():
            # Fallback to smart selection
            selected = smart_fallback_selection(field_name, valid_options)
            logger.info(f"      🎯 Smart selected dropdown '{field_name}': {selected}")
            return selected

        try:
            # Infer field purpose and business context
            field_lower = field_name.lower()
            field_purpose = ""

            if 'country' in field_lower:
                field_purpose = "Country selection - prefer United States for testing"
            elif 'state' in field_lower or 'province' in field_lower:
                field_purpose = "State/Province selection - prefer common states like California, New York, Texas"
            elif 'payment' in field_lower or 'method' in field_lower:
                field_purpose = "Payment method - prefer Credit Card or similar"
            elif 'shipping' in field_lower or 'delivery' in field_lower:
                field_purpose = "Shipping/Delivery method - prefer Standard or Regular shipping"
            elif 'title' in field_lower or 'salutation' in field_lower:
                field_purpose = "Personal title - prefer Mr, Mrs, Ms, or Dr"
            elif 'gender' in field_lower or 'sex' in field_lower:
                field_purpose = "Gender selection - select a realistic value"
            else:
                field_purpose = "General dropdown - select the most common, realistic option"

            # Use AI to select most appropriate option
            prompt = f"""You are an expert at selecting realistic test data for automated testing. Choose the MOST REALISTIC and COMMONLY USED option from this dropdown.

📋 FIELD INFORMATION:
- Field Name: {field_name}
- Field Purpose: {field_purpose}

🌐 PAGE CONTEXT: {page_context}

📝 AVAILABLE OPTIONS (select ONE):
{chr(10).join([f"{i+1}. {opt}" for i, opt in enumerate(valid_options[:30])])}

🎯 SELECTION CRITERIA:
1. **Realism**: Choose what a real user would typically select
2. **Common Choice**: Prefer frequently used options (e.g., "United States" for country, "California" or "New York" for state)
3. **Test Safety**: Avoid options that might trigger real actions or charges
4. **Context Awareness**: Consider the page context and business domain
5. **Avoid Placeholders**: Never select "Select...", "Choose...", or similar

🚨 CRITICAL REQUIREMENT:
Return ONLY the option text EXACTLY as it appears in the list above (without the number prefix).
No explanations, no quotes, just the exact option text.

Your selection:"""

            response = self.azure_client.completion_create(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3  # Lower temperature for more consistent selection
            )

            if response and 'choices' in response and len(response['choices']) > 0:
                selected = response['choices'][0]['message']['content'].strip()
                # Clean up
                selected = selected.strip('"').strip("'").strip().strip('-').strip()
                # Remove any numbering that might have been included
                import re
                selected = re.sub(r'^\d+\.\s*', '', selected)

                # Find best match in available options
                for opt in valid_options:
                    if opt.lower() == selected.lower() or selected.lower() in opt.lower() or opt.lower() in selected.lower():
                        logger.info(f"      🤖 AI selected dropdown '{field_name}': {opt}")
                        return opt

                # If no exact match, use smart fallback
                logger.info(f"      ⚠️  AI selection '{selected}' not found in options")
                fallback = smart_fallback_selection(field_name, valid_options)
                logger.info(f"      🎯 Using smart fallback: {fallback}")
                return fallback
            else:
                # Fallback to smart selection
                return smart_fallback_selection(field_name, valid_options)

        except Exception as e:
            logger.warning(f"      ⚠️  AI dropdown selection failed: {str(e)[:50]}, using random")
            import random
            return random.choice(valid_options) if valid_options else None

    async def _smart_form_fill(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """
        Intelligently detect and fill all form fields with appropriate test data
        Handles: text inputs, email, password, dropdowns, checkboxes, etc.
        OPTIMIZED: Prevents duplicate form filling
        """
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import Select
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import random
            import string
            import time

            # DUPLICATE PREVENTION: Generate unique form identifier
            # This prevents filling the same form multiple times in a test
            current_url = self.driver.current_url
            page_title = self.driver.title
            form_id = f"{current_url}#{page_title}#{step.step_number}"

            # Check if this form has already been filled
            if form_id in self.filled_forms:
                logger.info("   ✅ Form already filled in this test run - skipping to avoid duplicates")
                return True, "Form already completed - no duplicate filling needed"

            # Mark this form as being processed
            self.filled_forms.add(form_id)

            logger.info("   🔍 Scanning page for form fields...")

            # Wait for page to be ready
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete'
                )
                logger.info("   ⏱️  Page ready state: complete")
            except Exception as e:
                logger.warning(f"   ⚠️  Page ready state timeout: {str(e)}")

            # Optimized wait for dynamic content - reduced from 2s to 0.5s
            time.sleep(0.5)
            logger.info("   ⏱️  Waited 0.5s for dynamic content (optimized)")

            # Get page info for debugging (already captured above for form_id)
            logger.info(f"   📄 Current page: {page_title} ({current_url})")

            # Check for iframes first - payment forms are often in iframes
            iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
            logger.info(f"   🖼️  Found {len(iframes)} iframes on page")

            # Collect all input fields from main page AND iframes
            all_inputs = []
            all_selects = []
            all_textareas = []
            iframe_contexts = []  # Track which iframe each field belongs to

            # First, get fields from main page
            main_inputs = self.driver.find_elements(By.TAG_NAME, 'input')
            main_selects = self.driver.find_elements(By.TAG_NAME, 'select')
            main_textareas = self.driver.find_elements(By.TAG_NAME, 'textarea')

            all_inputs.extend(main_inputs)
            all_selects.extend(main_selects)
            all_textareas.extend(main_textareas)
            iframe_contexts.extend([None] * (len(main_inputs) + len(main_selects) + len(main_textareas)))

            # Now check inside each iframe for payment fields
            if iframes:
                logger.info(f"   🔍 Scanning {len(iframes)} iframe(s) for payment fields...")
                for iframe_idx, iframe in enumerate(iframes):
                    try:
                        # Get iframe info for logging
                        iframe_id = iframe.get_attribute('id') or f'iframe_{iframe_idx}'
                        iframe_name = iframe.get_attribute('name') or ''
                        iframe_src = iframe.get_attribute('src') or ''

                        logger.info(f"      🖼️  Scanning iframe #{iframe_idx}: id='{iframe_id}', name='{iframe_name}'")

                        # Switch to iframe
                        self.driver.switch_to.frame(iframe)

                        # Find all fields in this iframe
                        iframe_inputs = self.driver.find_elements(By.TAG_NAME, 'input')
                        iframe_selects = self.driver.find_elements(By.TAG_NAME, 'select')
                        iframe_textareas = self.driver.find_elements(By.TAG_NAME, 'textarea')

                        if iframe_inputs or iframe_selects or iframe_textareas:
                            logger.info(f"      ✅ Found {len(iframe_inputs)} inputs, {len(iframe_selects)} selects in iframe")
                            all_inputs.extend(iframe_inputs)
                            all_selects.extend(iframe_selects)
                            all_textareas.extend(iframe_textareas)
                            # Track which iframe these fields belong to
                            iframe_contexts.extend([iframe] * (len(iframe_inputs) + len(iframe_selects) + len(iframe_textareas)))

                        # Switch back to main content
                        self.driver.switch_to.default_content()

                    except Exception as e:
                        logger.warning(f"      ⚠️ Could not scan iframe #{iframe_idx}: {str(e)[:100]}")
                        # Make sure we're back in main content
                        try:
                            self.driver.switch_to.default_content()
                        except:
                            pass

            logger.info(f"   📊 Found on page: {len(all_inputs)} inputs, {len(all_selects)} selects, {len(all_textareas)} textareas")

            # Debug: Show all input types found
            input_types_found = {}
            payment_fields_found = []
            for inp in all_inputs:
                try:
                    inp_type = (inp.get_attribute('type') or 'text').lower()
                    inp_name = inp.get_attribute('name') or inp.get_attribute('id') or 'unnamed'
                    is_visible = inp.is_displayed()
                    is_enabled = inp.is_enabled()
                    input_types_found[inp_type] = input_types_found.get(inp_type, 0) + 1

                    # Track payment fields specifically
                    if any(keyword in inp_name.lower() for keyword in ['card', 'cvv', 'cvc', 'expiry', 'expiration', 'exp']):
                        payment_fields_found.append({
                            'name': inp_name,
                            'type': inp_type,
                            'visible': is_visible,
                            'enabled': is_enabled
                        })

                    logger.debug(f"      - {inp_type} field '{inp_name}': visible={is_visible}, enabled={is_enabled}")
                except:
                    pass

            if input_types_found:
                logger.info(f"   📋 Input types: {input_types_found}")

            if payment_fields_found:
                logger.info(f"   💳 Payment fields detected: {len(payment_fields_found)}")
                for pf in payment_fields_found:
                    logger.info(f"      - {pf['name']} ({pf['type']}): visible={pf['visible']}, enabled={pf['enabled']}")

            fields_filled = 0
            fields_skipped = 0
            field_details = []

            # Keep track of current iframe context
            current_iframe = None

            # Process INPUT fields (with iframe context tracking)
            for idx, input_elem in enumerate(all_inputs):
                try:
                    # Check if we need to switch iframe context
                    field_iframe = iframe_contexts[idx] if idx < len(iframe_contexts) else None

                    # Switch to the appropriate context if needed
                    if field_iframe != current_iframe:
                        # Switch back to main content first
                        self.driver.switch_to.default_content()

                        # Then switch to target iframe if needed
                        if field_iframe is not None:
                            try:
                                self.driver.switch_to.frame(field_iframe)
                                logger.debug(f"      🖼️  Switched to iframe for field processing")
                            except Exception as e:
                                logger.warning(f"      ⚠️ Could not switch to iframe: {str(e)[:50]}")
                                fields_skipped += 1
                                continue

                        current_iframe = field_iframe

                    # Get element info first for debugging
                    input_type = (input_elem.get_attribute('type') or 'text').lower()
                    input_name = input_elem.get_attribute('name') or ''
                    input_id = input_elem.get_attribute('id') or ''
                    placeholder = input_elem.get_attribute('placeholder') or ''

                    # Skip buttons and submits first
                    if input_type in ['submit', 'button', 'image', 'reset']:
                        logger.debug(f"      ⊘ Skipping {input_type} button '{input_name or input_id}'")
                        continue

                    # Build comprehensive context for payment field detection
                    field_context = f"{input_name} {input_id} {placeholder}".lower()

                    # Detect specific payment field types
                    is_card_number = any(keyword in field_context for keyword in
                                        ['cardnumber', 'card-number', 'card_number', 'ccnumber', 'cc-number',
                                         'cc_number', 'creditcard', 'credit-card', 'debit', 'pan', 'accountnumber'])

                    is_cvv = any(keyword in field_context for keyword in
                                ['cvv', 'cvc', 'cvv2', 'cid', 'security', 'securitycode', 'security-code',
                                 'security_code', 'verification', 'card-code'])

                    is_expiry = any(keyword in field_context for keyword in
                                   ['expiry', 'expiration', 'exp', 'expirydate', 'expdate', 'expmm', 'expyy',
                                    'exp-month', 'exp-year', 'exp_month', 'exp_year', 'mm/yy', 'mmyy',
                                    'valid', 'validthru', 'expiration-date'])

                    # General payment field check
                    is_payment_field = is_card_number or is_cvv or is_expiry or any(keyword in field_context
                                          for keyword in ['card', 'payment', 'billing'])

                    # Log payment field detection
                    if is_payment_field:
                        field_type_str = []
                        if is_card_number: field_type_str.append("CARD_NUMBER")
                        if is_cvv: field_type_str.append("CVV")
                        if is_expiry: field_type_str.append("EXPIRY")
                        if not field_type_str: field_type_str.append("PAYMENT")
                        logger.info(f"      💳 Detected payment field: '{input_name or input_id}' ({', '.join(field_type_str)})")

                    # Also check if it looks like a name field to avoid re-entering
                    is_name_field = any(keyword in field_context
                                       for keyword in ['firstname', 'first-name', 'first_name', 'fname',
                                                      'lastname', 'last-name', 'last_name', 'lname',
                                                      'fullname', 'full-name', 'full_name'])

                    is_displayed = False
                    is_enabled = False
                    try:
                        is_displayed = input_elem.is_displayed()
                        is_enabled = input_elem.is_enabled()
                    except Exception as e:
                        logger.debug(f"      ⚠ Cannot check visibility for '{input_name or input_id}': {str(e)[:50]}")
                        # For payment fields, try to fill anyway
                        if not is_payment_field:
                            fields_skipped += 1
                            continue

                    # Be more lenient with payment fields - they might be in iframes or dynamically loaded
                    if not is_displayed and not is_payment_field:
                        logger.debug(f"      ⊘ Skipping hidden {input_type} field '{input_name or input_id}'")
                        fields_skipped += 1
                        continue

                    if not is_enabled and not is_payment_field:
                        logger.debug(f"      ⊘ Skipping disabled {input_type} field '{input_name or input_id}'")
                        fields_skipped += 1
                        continue

                    # Special handling for payment fields that appear hidden but are actually in iframes
                    if is_payment_field and not is_displayed:
                        logger.info(f"      💳 Payment field '{input_name or input_id}' appears hidden - trying to fill anyway")
                        # Try to scroll into view first
                        try:
                            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", input_elem)
                            time.sleep(0.5)
                            is_displayed = input_elem.is_displayed()
                        except:
                            pass

                    # Generate appropriate test data using AI (with fallback to rule-based)
                    # Build page context for AI
                    page_context = f"Page: {self.driver.title}, URL: {self.driver.current_url}"

                    # Try to get label text for better context
                    label_text = ""
                    try:
                        if input_id:
                            label_elem = self.driver.find_element(By.CSS_SELECTOR, f"label[for='{input_id}']")
                            label_text = label_elem.text.strip() if label_elem else ""
                    except:
                        pass

                    # Get validation attributes for intelligent data generation
                    pattern = input_elem.get_attribute('pattern') or ''
                    required = input_elem.get_attribute('required') is not None
                    maxlength = input_elem.get_attribute('maxlength') or ''
                    minlength = input_elem.get_attribute('minlength') or ''
                    min_val = input_elem.get_attribute('min') or ''
                    max_val = input_elem.get_attribute('max') or ''

                    field_info = {
                        'type': input_type,
                        'name': input_name,
                        'id': input_id,
                        'placeholder': placeholder,
                        'label': label_text,
                        'pattern': pattern,
                        'required': required,
                        'maxlength': maxlength,
                        'minlength': minlength,
                        'min': min_val,
                        'max': max_val
                    }

                    # Use AI-powered generation if available
                    test_value = await self._generate_ai_test_data(field_info, page_context)

                    # CRITICAL: Ensure payment fields always have fallback values with smart detection
                    if is_payment_field and not test_value:
                        logger.warning(f"      ⚠️ Payment field '{input_name or input_id}' has no test value - using smart fallback")

                        # Card Number field
                        if is_card_number:
                            # Visa test card number (passes Luhn check)
                            test_value = "4532123456789012"
                            logger.info(f"      💳 Generated card number: ************{test_value[-4:]}")

                        # CVV/CVC field
                        elif is_cvv:
                            # Check maxlength to determine if 3 or 4 digits
                            if maxlength == '4':
                                test_value = "1234"
                            else:
                                test_value = "123"
                            logger.info(f"      💳 Generated CVV: ***")

                        # Expiry field
                        elif is_expiry:
                            import datetime
                            current_year = datetime.datetime.now().year
                            future_year = current_year + 2

                            # Detect if this is month or year field
                            if any(keyword in field_context for keyword in ['month', 'mm', 'mon']):
                                test_value = "12"
                                logger.info(f"      💳 Generated expiry month: {test_value}")
                            elif any(keyword in field_context for keyword in ['year', 'yy', 'yyyy']):
                                # Determine 2-digit or 4-digit based on maxlength or pattern
                                if maxlength == '4' or 'yyyy' in field_context:
                                    test_value = str(future_year)
                                else:
                                    test_value = str(future_year)[-2:]
                                logger.info(f"      💳 Generated expiry year: {test_value}")
                            else:
                                # Combined expiry field (MM/YY or MM/YYYY format)
                                if 'yyyy' in field_context or maxlength == '7':  # MM/YYYY
                                    test_value = f"12/{future_year}"
                                else:  # MM/YY
                                    test_value = f"12/{str(future_year)[-2:]}"
                                logger.info(f"      💳 Generated expiry: {test_value}")

                        # Generic payment field
                        else:
                            # Check if it might be a cardholder name
                            if 'name' in field_context and 'card' in field_context:
                                test_value = "John Smith"
                                logger.info(f"      💳 Generated cardholder name: {test_value}")
                            else:
                                # Default to card number
                                test_value = "4532123456789012"
                                logger.info(f"      💳 Generated default payment value")

                    # Log what was generated for payment fields (with masking)
                    elif is_payment_field and test_value:
                        if is_cvv:
                            logger.info(f"      💳 Using value for CVV field '{input_name or input_id}': ***")
                        elif is_card_number:
                            logger.info(f"      💳 Using value for card number field '{input_name or input_id}': ************{test_value[-4:] if len(test_value) >= 4 else '****'}")
                        else:
                            logger.info(f"      💳 Using value for payment field '{input_name or input_id}': {test_value if not is_cvv else '***'}")

                    # Check if field already has a value (to avoid re-entering)
                    # This prevents duplicate form filling
                    current_value = input_elem.get_attribute('value') or ''
                    if current_value and input_type not in ['checkbox', 'radio', 'hidden']:
                        # Skip if field already has a non-empty value
                        # Exception: Re-fill if it's clearly wrong (like default placeholder values)
                        if len(current_value) > 2:  # Meaningful value, not just a placeholder
                            logger.info(f"      ⊘ Skipping '{input_name or input_id}' - already has value: {current_value[:30]}")
                            fields_skipped += 1
                            continue

                    if test_value and input_type not in ['checkbox', 'radio']:
                        try:
                            # Scroll element into view
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", input_elem)
                            time.sleep(0.3)

                            # For payment fields, try multiple filling methods
                            fill_success = False

                            if is_payment_field:
                                logger.info(f"      💳 Attempting to fill payment field '{input_name or input_id}' with: {test_value if 'cvv' not in input_name.lower() and 'cvc' not in input_name.lower() else '***'}")

                                # Method 1: Standard clear and send_keys
                                try:
                                    input_elem.clear()
                                    input_elem.send_keys(test_value)
                                    fill_success = True
                                    logger.info(f"      ✅ Method 1 (clear+send_keys) succeeded for '{input_name or input_id}'")
                                except Exception as e1:
                                    logger.debug(f"      Method 1 failed: {str(e1)[:50]}")

                                # Method 2: JavaScript value setting if Method 1 failed
                                if not fill_success:
                                    try:
                                        self.driver.execute_script(f"arguments[0].value = '{test_value}';", input_elem)
                                        # Trigger events
                                        self.driver.execute_script("""
                                            arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
                                            arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
                                        """, input_elem)
                                        fill_success = True
                                        logger.info(f"      ✅ Method 2 (JavaScript) succeeded for '{input_name or input_id}'")
                                    except Exception as e2:
                                        logger.debug(f"      Method 2 failed: {str(e2)[:50]}")

                                # Method 3: Click then send_keys without clear
                                if not fill_success:
                                    try:
                                        input_elem.click()
                                        time.sleep(0.2)
                                        input_elem.send_keys(test_value)
                                        fill_success = True
                                        logger.info(f"      ✅ Method 3 (click+send_keys) succeeded for '{input_name or input_id}'")
                                    except Exception as e3:
                                        logger.debug(f"      Method 3 failed: {str(e3)[:50]}")

                                # Method 4: Focus, select all, then send_keys
                                if not fill_success:
                                    try:
                                        from selenium.webdriver.common.keys import Keys
                                        input_elem.click()
                                        input_elem.send_keys(Keys.CONTROL + "a")
                                        input_elem.send_keys(test_value)
                                        fill_success = True
                                        logger.info(f"      ✅ Method 4 (select+replace) succeeded for '{input_name or input_id}'")
                                    except Exception as e4:
                                        logger.debug(f"      Method 4 failed: {str(e4)[:50]}")

                                if fill_success:
                                    fields_filled += 1
                                    field_details.append({
                                        'type': input_type,
                                        'name': input_name or input_id,
                                        'value': test_value if 'cvv' not in input_name.lower() and 'cvc' not in input_name.lower() else '***',
                                        'is_payment': True
                                    })
                                    # CAPTURE THE LOCATOR for this successfully filled field
                                    try:
                                        self._capture_element_locator(input_elem, step, "input", test_case)
                                    except Exception as capture_error:
                                        logger.debug(f"      ⚠️ Could not capture locator: {str(capture_error)[:50]}")
                                else:
                                    logger.warning(f"      ❌ All methods failed for payment field '{input_name or input_id}'")
                                    fields_skipped += 1
                            else:
                                # Standard filling for non-payment fields
                                input_elem.clear()
                                input_elem.send_keys(test_value)
                                fields_filled += 1
                                field_details.append({
                                    'type': input_type,
                                    'name': input_name or input_id,
                                    'value': test_value if input_type != 'password' else '***'
                                })
                                logger.info(f"      ✓ Filled {input_type} field '{input_name or input_id}': {test_value if input_type != 'password' else '***'}")

                                # CAPTURE THE LOCATOR for this successfully filled field
                                try:
                                    self._capture_element_locator(input_elem, step, "input", test_case)
                                except Exception as capture_error:
                                    logger.debug(f"      ⚠️ Could not capture locator: {str(capture_error)[:50]}")

                                # Check if this is an address field and handle dropdown
                                is_address_field = any(keyword in (input_name or '').lower() or keyword in (input_id or '').lower()
                                                      for keyword in ['address', 'street', 'addr'])
                                if is_address_field:
                                    logger.info(f"      🏠 Address field detected, checking for dropdown...")
                                    time.sleep(0.4)  # Optimized from 1.5s to 0.4s for faster form filling
                                    dropdown_selected = await self._handle_address_dropdown(input_elem)
                                    if dropdown_selected:
                                        logger.info(f"      ✅ Selected address from dropdown for '{input_name or input_id}'")
                                    else:
                                        logger.debug(f"      ℹ️  No address dropdown appeared for '{input_name or input_id}'")
                        except Exception as fill_error:
                            logger.warning(f"      ✗ Failed to fill '{input_name or input_id}': {str(fill_error)[:50]}")
                            fields_skipped += 1

                    elif input_type == 'checkbox':
                        try:
                            if not input_elem.is_selected():
                                self.driver.execute_script("arguments[0].scrollIntoView(true);", input_elem)
                                time.sleep(0.3)
                                input_elem.click()
                                fields_filled += 1
                                logger.info(f"      ✓ Checked checkbox '{input_name or input_id}'")
                                # CAPTURE THE LOCATOR
                                try:
                                    self._capture_element_locator(input_elem, step, "click", test_case)
                                except Exception as capture_error:
                                    logger.debug(f"      ⚠️ Could not capture locator: {str(capture_error)[:50]}")
                        except Exception as check_error:
                            logger.warning(f"      ✗ Failed to check '{input_name or input_id}': {str(check_error)[:50]}")
                            fields_skipped += 1

                    elif input_type == 'radio':
                        try:
                            if not input_elem.is_selected():
                                self.driver.execute_script("arguments[0].scrollIntoView(true);", input_elem)
                                time.sleep(0.3)
                                input_elem.click()
                                fields_filled += 1
                                logger.info(f"      ✓ Selected radio '{input_name or input_id}'")
                                # CAPTURE THE LOCATOR
                                try:
                                    self._capture_element_locator(input_elem, step, "click", test_case)
                                except Exception as capture_error:
                                    logger.debug(f"      ⚠️ Could not capture locator: {str(capture_error)[:50]}")
                        except Exception as radio_error:
                            logger.warning(f"      ✗ Failed to select '{input_name or input_id}': {str(radio_error)[:50]}")
                            fields_skipped += 1

                except Exception as e:
                    logger.debug(f"      ⚠ Error processing input field: {str(e)[:50]}")
                    fields_skipped += 1
                    continue

            # Process SELECT dropdowns (with iframe context tracking)
            select_start_idx = len(all_inputs)
            for idx, select_elem in enumerate(all_selects):
                try:
                    # Check if we need to switch iframe context
                    field_iframe = iframe_contexts[select_start_idx + idx] if (select_start_idx + idx) < len(iframe_contexts) else None

                    # Switch to the appropriate context if needed
                    if field_iframe != current_iframe:
                        # Switch back to main content first
                        self.driver.switch_to.default_content()

                        # Then switch to target iframe if needed
                        if field_iframe is not None:
                            try:
                                self.driver.switch_to.frame(field_iframe)
                            except Exception as e:
                                logger.warning(f"      ⚠️ Could not switch to iframe for select: {str(e)[:50]}")
                                fields_skipped += 1
                                continue

                        current_iframe = field_iframe

                    select_name = select_elem.get_attribute('name') or select_elem.get_attribute('id') or 'dropdown'

                    if not select_elem.is_displayed() or not select_elem.is_enabled():
                        logger.debug(f"      ⊘ Skipping hidden/disabled dropdown '{select_name}'")
                        fields_skipped += 1
                        continue

                    select_obj = Select(select_elem)
                    options = select_obj.options

                    # Skip first option (usually placeholder) and select intelligent option
                    if len(options) > 1:
                        # Try to select a meaningful option (not empty, not "Select...")
                        valid_options = [opt for opt in options[1:] if opt.text.strip() and not opt.text.startswith('Select')]

                        if valid_options:
                            # Use AI to select most appropriate option
                            page_context = f"Page: {self.driver.title}, URL: {self.driver.current_url}"
                            option_texts = [opt.text for opt in valid_options]
                            selected_text = await self._select_ai_dropdown_option(select_name, option_texts, page_context)

                            if selected_text:
                                self.driver.execute_script("arguments[0].scrollIntoView(true);", select_elem)
                                time.sleep(0.3)
                                select_obj.select_by_visible_text(selected_text)
                                fields_filled += 1
                                field_details.append({
                                    'type': 'select',
                                    'name': select_name,
                                    'value': selected_text
                                })
                                logger.info(f"      ✓ Selected dropdown '{select_name}': {selected_text}")
                                # CAPTURE THE LOCATOR
                                try:
                                    self._capture_element_locator(select_elem, step, "select", test_case)
                                except Exception as capture_error:
                                    logger.debug(f"      ⚠️ Could not capture locator: {str(capture_error)[:50]}")
                        else:
                            logger.debug(f"      ⊘ No valid options for dropdown '{select_name}'")
                            fields_skipped += 1

                except Exception as e:
                    logger.debug(f"      ⚠ Error processing select field: {str(e)[:50]}")
                    fields_skipped += 1
                    continue

            # Process TEXTAREA fields (with iframe context tracking)
            textarea_start_idx = len(all_inputs) + len(all_selects)
            for idx, textarea_elem in enumerate(all_textareas):
                try:
                    # Check if we need to switch iframe context
                    field_iframe = iframe_contexts[textarea_start_idx + idx] if (textarea_start_idx + idx) < len(iframe_contexts) else None

                    # Switch to the appropriate context if needed
                    if field_iframe != current_iframe:
                        # Switch back to main content first
                        self.driver.switch_to.default_content()

                        # Then switch to target iframe if needed
                        if field_iframe is not None:
                            try:
                                self.driver.switch_to.frame(field_iframe)
                            except Exception as e:
                                logger.warning(f"      ⚠️ Could not switch to iframe for textarea: {str(e)[:50]}")
                                fields_skipped += 1
                                continue

                        current_iframe = field_iframe

                    textarea_name = textarea_elem.get_attribute('name') or textarea_elem.get_attribute('id') or 'textarea'

                    if not textarea_elem.is_displayed() or not textarea_elem.is_enabled():
                        logger.debug(f"      ⊘ Skipping hidden/disabled textarea '{textarea_name}'")
                        fields_skipped += 1
                        continue

                    test_value = "This is test content for textarea field. Lorem ipsum dolor sit amet."

                    self.driver.execute_script("arguments[0].scrollIntoView(true);", textarea_elem)
                    time.sleep(0.3)
                    textarea_elem.clear()
                    textarea_elem.send_keys(test_value)
                    fields_filled += 1
                    logger.info(f"      ✓ Filled textarea '{textarea_name}'")

                except Exception as e:
                    logger.debug(f"      ⚠ Error processing textarea: {str(e)[:50]}")
                    fields_skipped += 1
                    continue

            # Ensure we're back in the main content after processing all fields
            try:
                self.driver.switch_to.default_content()
                logger.debug("   🔄 Switched back to main content after field processing")
            except Exception as e:
                logger.debug(f"   ⚠ Could not switch to default content: {str(e)[:50]}")

            # Store captured data for variables file
            if field_details:
                for field in field_details:
                    var_name = f"{field['name']}_test_data"
                    self.captured_variables[var_name] = field['value']

            # Enhanced reporting
            total_fields = len(all_inputs) + len(all_selects) + len(all_textareas)
            logger.info(f"   📊 Summary: Total fields={total_fields}, Filled={fields_filled}, Skipped={fields_skipped}")

            if fields_filled > 0:
                summary = f"Successfully filled {fields_filled} form fields"
                if fields_skipped > 0:
                    summary += f" (skipped {fields_skipped} hidden/disabled fields)"
                logger.info(f"   ✅ {summary}")
                return True, summary
            else:
                error_msg = f"No fillable form fields found on page. Total fields detected: {total_fields} (all were hidden, disabled, or buttons)"
                logger.error(f"   ❌ {error_msg}")
                logger.error(f"   💡 Possible issues:")
                logger.error(f"      1. Fields may be inside an iframe")
                logger.error(f"      2. Page may not have finished loading")
                logger.error(f"      3. Fields may be hidden with CSS")
                logger.error(f"      4. Page may require interaction before showing form")
                return False, error_msg

        except Exception as e:
            logger.error(f"   ❌ Form fill error: {str(e)}")
            import traceback
            logger.error(f"   📋 Traceback: {traceback.format_exc()}")
            return False, f"Form fill error: {str(e)}"

    def _generate_test_data_for_field(self, field_type: str, name: str, field_id: str, placeholder: str, label: str = "") -> str:
        """
        Generate appropriate test data based on field type and context
        ENHANCED: Now with realistic, intelligent, context-aware data generation

        Args:
            field_type: Type of input field (text, email, password, etc.)
            name: Name attribute of field
            field_id: ID attribute of field
            placeholder: Placeholder text
            label: Label text for additional context

        Returns:
            Generated test data string
        """
        import random
        import string
        from datetime import datetime, timedelta

        # Combine all context for pattern matching
        context = f"{field_type} {name} {field_id} {placeholder} {label}".lower()

        # Realistic data sets for intelligent generation
        REALISTIC_FIRST_NAMES = [
            'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
            'William', 'Barbara', 'David', 'Elizabeth', 'Richard', 'Susan', 'Joseph', 'Jessica',
            'Thomas', 'Sarah', 'Christopher', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Lisa',
            'Anthony', 'Betty', 'Mark', 'Margaret', 'Donald', 'Sandra', 'Steven', 'Ashley',
            'Andrew', 'Kimberly', 'Paul', 'Emily', 'Joshua', 'Donna', 'Kenneth', 'Michelle'
        ]

        REALISTIC_LAST_NAMES = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas',
            'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White',
            'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker', 'Young',
            'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores'
        ]

        REALISTIC_COMPANIES = [
            'Acme Corporation', 'Global Tech Solutions', 'Innovation Dynamics Inc',
            'Pacific Industries', 'Summit Enterprises', 'Nexus Technologies',
            'Quantum Systems LLC', 'Horizon Business Group', 'Vertex Solutions',
            'Pinnacle Services', 'Atlas Corporation', 'Stellar Innovations',
            'Fusion Technologies', 'Beacon Consulting', 'Meridian Group',
            'Catalyst Ventures', 'Synergy Solutions', 'Paramount Industries'
        ]

        REALISTIC_CITIES = [
            'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
            'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
            'Fort Worth', 'Columbus', 'Charlotte', 'Indianapolis', 'San Francisco', 'Seattle',
            'Denver', 'Boston', 'El Paso', 'Nashville', 'Detroit', 'Portland',
            'Las Vegas', 'Memphis', 'Louisville', 'Baltimore', 'Milwaukee', 'Albuquerque'
        ]

        US_STATES = [
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        ]

        EMAIL_DOMAINS = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com',
            'protonmail.com', 'aol.com', 'mail.com', 'zoho.com', 'fastmail.com'
        ]

        STREET_TYPES = ['Street', 'Avenue', 'Boulevard', 'Drive', 'Lane', 'Road', 'Way', 'Court', 'Place', 'Terrace']
        STREET_NAMES = ['Main', 'Oak', 'Maple', 'Pine', 'Cedar', 'Elm', 'Washington', 'Park', 'Lake', 'Hill',
                       'Forest', 'River', 'Spring', 'Valley', 'Highland', 'Sunset', 'Meadow', 'Ridge', 'Garden']

        # EMAIL fields - realistic with varied domains
        if field_type == 'email' or any(keyword in context for keyword in ['email', 'e-mail', 'mail']):
            first_name = random.choice(REALISTIC_FIRST_NAMES).lower()
            last_name = random.choice(REALISTIC_LAST_NAMES).lower()
            domain = random.choice(EMAIL_DOMAINS)
            # Vary email format for realism
            formats = [
                f"{first_name}.{last_name}@{domain}",
                f"{first_name}{last_name[0]}@{domain}",
                f"{first_name[0]}{last_name}@{domain}",
                f"{first_name}_{last_name}@{domain}"
            ]
            return random.choice(formats)

        # PASSWORD fields - strong, varied passwords
        elif field_type == 'password' or 'password' in context or 'pwd' in context:
            # Generate strong, realistic passwords
            password_patterns = [
                lambda: f"{random.choice(['Pass', 'Secure', 'Key', 'Auth'])}{random.randint(100,999)}!{random.choice(string.ascii_uppercase)}",
                lambda: f"{random.choice(REALISTIC_FIRST_NAMES)}{random.randint(1990,2005)}!{random.choice(['#','@','$'])}",
                lambda: f"{random.choice(['Test','Demo','Auto'])}_{random.choice(string.ascii_uppercase)}{random.randint(10,99)}!",
            ]
            return random.choice(password_patterns)()

        # PHONE/TEL fields - varied realistic formats
        elif field_type == 'tel' or any(keyword in context for keyword in ['phone', 'telephone', 'mobile', 'tel', 'cell']):
            area_code = random.randint(200, 999)
            exchange = random.randint(200, 999)
            subscriber = random.randint(1000, 9999)
            # Vary phone formats
            formats = [
                f"({area_code}) {exchange}-{subscriber}",
                f"{area_code}-{exchange}-{subscriber}",
                f"{area_code}.{exchange}.{subscriber}",
                f"+1-{area_code}-{exchange}-{subscriber}"
            ]
            return random.choice(formats)

        # ZIP/POSTAL CODE fields
        elif any(keyword in context for keyword in ['zip', 'postal', 'postcode', 'postalcode']):
            # US ZIP codes - with optional +4 extension
            if random.random() > 0.5:
                return f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
            return f"{random.randint(10000, 99999)}"

        # DATE fields - context-aware dates
        elif field_type == 'date' or 'date' in context:
            if 'birth' in context or 'dob' in context:
                # Birth dates: 18-70 years ago
                random_date = datetime.now() - timedelta(days=random.randint(365*18, 365*70))
            elif 'start' in context or 'from' in context:
                # Start dates: within last 5 years
                random_date = datetime.now() - timedelta(days=random.randint(0, 365*5))
            elif 'end' in context or 'to' in context or 'expir' in context:
                # End/expiry dates: future dates
                random_date = datetime.now() + timedelta(days=random.randint(30, 365*3))
            else:
                # Generic dates: within reasonable past
                random_date = datetime.now() - timedelta(days=random.randint(0, 365*2))

            # Detect format from placeholder
            if 'mm/dd/yyyy' in context or 'mm-dd-yyyy' in context:
                return random_date.strftime('%m/%d/%Y')
            elif 'dd/mm/yyyy' in context or 'dd-mm-yyyy' in context:
                return random_date.strftime('%d/%m/%Y')
            else:
                return random_date.strftime('%Y-%m-%d')

        # NUMBER fields - context-aware numbers
        elif field_type == 'number':
            if 'age' in context:
                return str(random.randint(18, 75))
            elif 'quantity' in context or 'qty' in context or 'amount' in context:
                return str(random.randint(1, 10))
            elif 'year' in context:
                return str(random.randint(1950, datetime.now().year))
            elif 'price' in context or 'cost' in context:
                return str(random.randint(10, 1000))
            else:
                return str(random.randint(1, 100))

        # URL fields
        elif field_type == 'url' or 'url' in context or 'website' in context:
            domains = ['example.com', 'testsite.com', 'demo-website.com', 'mycompany.com']
            return f"https://www.{random.choice(domains)}"

        # FIRST NAME fields
        elif any(keyword in context for keyword in ['firstname', 'first_name', 'fname', 'given', 'first name']):
            return random.choice(REALISTIC_FIRST_NAMES)

        # LAST NAME fields
        elif any(keyword in context for keyword in ['lastname', 'last_name', 'lname', 'surname', 'family', 'last name']):
            return random.choice(REALISTIC_LAST_NAMES)

        # FULL NAME fields
        elif any(keyword in context for keyword in ['fullname', 'full_name', 'full name', 'name']) and 'user' not in context and 'company' not in context and 'file' not in context:
            return f"{random.choice(REALISTIC_FIRST_NAMES)} {random.choice(REALISTIC_LAST_NAMES)}"

        # COMPANY/ORGANIZATION fields
        elif any(keyword in context for keyword in ['company', 'organization', 'business', 'employer', 'firm']):
            return random.choice(REALISTIC_COMPANIES)

        # ADDRESS fields - realistic street addresses
        elif any(keyword in context for keyword in ['address', 'street', 'addr']):
            if 'address2' in context or 'address_2' in context or 'line2' in context or 'apt' in context or 'suite' in context:
                # Secondary address line
                return random.choice([
                    f"Apt {random.randint(1, 999)}",
                    f"Suite {random.randint(100, 999)}",
                    f"Unit {random.randint(1, 99)}",
                    f"#{ random.randint(1, 999)}"
                ])
            else:
                # Primary address
                street_number = random.randint(100, 9999)
                street_name = random.choice(STREET_NAMES)
                street_type = random.choice(STREET_TYPES)
                return f"{street_number} {street_name} {street_type}"

        # CITY fields
        elif 'city' in context:
            return random.choice(REALISTIC_CITIES)

        # STATE fields
        elif 'state' in context or 'province' in context:
            return random.choice(US_STATES)

        # COUNTRY fields
        elif 'country' in context:
            countries = ['United States', 'USA', 'US']
            return random.choice(countries)

        # CVV/CVC fields (check first, more specific)
        elif any(keyword in context for keyword in ['cvv', 'cvc', 'cvv2', 'cid', 'security', 'securitycode', 'security-code', 'verification', 'card-code']):
            # Default to 3 digits (most common), check context for 4-digit CVV (Amex)
            if 'amex' in context or 'american' in context:
                value = str(random.randint(1000, 9999))
            else:
                value = str(random.randint(100, 999))
            logger.debug(f"Generated CVV: ***")
            return value

        # CARD NUMBER fields
        elif any(keyword in context for keyword in ['cardnumber', 'card-number', 'card_number', 'ccnumber', 'cc-number', 'creditcard', 'credit-card', 'debitcard', 'pan']):
            # Test credit card numbers (Visa test format - passes Luhn check)
            # Using 4532 prefix for test Visa cards
            value = "4532123456789012"  # Valid test card
            logger.debug(f"Generated card number: ************{value[-4:]}")
            return value

        # Cardholder name
        elif 'name' in context and any(keyword in context for keyword in ['card', 'holder', 'cardholder']):
            value = f"{random.choice(REALISTIC_FIRST_NAMES)} {random.choice(REALISTIC_LAST_NAMES)}"
            logger.debug(f"Generated cardholder name: {value}")
            return value

        # EXPIRY DATE fields (check for specific patterns)
        elif any(keyword in context for keyword in ['expiry', 'expiration', 'exp', 'valid', 'expirydate', 'expdate', 'expmm', 'expyy']):
            # Handle month field specifically
            if any(keyword in context for keyword in ['month', 'mm', 'mon']) and not any(keyword in context for keyword in ['yy', 'yyyy', 'year']):
                value = f"{random.randint(1, 12):02d}"
                logger.debug(f"Generated expiry month: {value}")
                return value
            # Handle year field specifically
            elif any(keyword in context for keyword in ['year', 'yy', 'yyyy']) and not any(keyword in context for keyword in ['mm', 'month', 'mon']):
                future_year = datetime.now().year + random.randint(1, 5)
                # Return 2-digit or 4-digit year based on context
                if 'yyyy' in context:
                    value = str(future_year)
                else:
                    value = str(future_year)[-2:]
                logger.debug(f"Generated expiry year: {value}")
                return value
            # Combined expiry field (MM/YY or MM/YYYY format)
            else:
                future_month = random.randint(1, 12)
                future_year = datetime.now().year + random.randint(1, 5)
                # Detect format from placeholder
                if 'yyyy' in context:  # MM/YYYY
                    value = f"{future_month:02d}/{future_year}"
                elif '-' in placeholder or '-' in label:
                    value = f"{future_month:02d}-{str(future_year)[-2:]}"
                else:  # Default MM/YY
                    value = f"{future_month:02d}/{str(future_year)[-2:]}"
                logger.debug(f"Generated expiry date: {value}")
                return value
                logger.debug(f"Generated expiry date: {value}")
                return value

        # USERNAME fields
        elif any(keyword in context for keyword in ['username', 'user_name', 'login', 'userid', 'user id']):
            first = random.choice(REALISTIC_FIRST_NAMES).lower()
            last = random.choice(REALISTIC_LAST_NAMES).lower()
            patterns = [
                f"{first}{last[0]}{random.randint(10,99)}",
                f"{first}.{last}",
                f"{first}_{last}",
                f"{first}{random.randint(100,999)}"
            ]
            return random.choice(patterns)

        # SEARCH/QUERY fields
        elif 'search' in context or 'query' in context or 'find' in context:
            search_terms = [
                'laptop computers', 'wireless headphones', 'office supplies',
                'running shoes', 'smartphone accessories', 'home decor',
                'kitchen appliances', 'fitness equipment', 'garden tools'
            ]
            return random.choice(search_terms)

        # TITLE/POSITION fields
        elif 'title' in context or 'position' in context or 'role' in context:
            titles = [
                'Software Engineer', 'Product Manager', 'Data Analyst', 'Project Manager',
                'Marketing Specialist', 'Sales Representative', 'Operations Manager',
                'Business Analyst', 'Quality Assurance Engineer', 'Customer Success Manager'
            ]
            return random.choice(titles)

        # MESSAGE/COMMENT/DESCRIPTION fields
        elif any(keyword in context for keyword in ['message', 'comment', 'description', 'notes', 'feedback', 'details']):
            messages = [
                'This is a test message for automated testing purposes.',
                'Automated test entry to verify form functionality.',
                'Testing the form submission process with sample data.',
                'Quality assurance test case execution in progress.'
            ]
            return random.choice(messages)

        # DEFAULT: Context-aware generic text
        else:
            # Try to infer from label/placeholder
            if label or placeholder:
                context_hint = (label or placeholder).lower()
                if any(k in context_hint for k in ['description', 'detail', 'note', 'comment']):
                    return f"Automated test entry for {label or placeholder}"
                elif any(k in context_hint for k in ['code', 'id', 'number']):
                    return f"TEST{random.randint(1000, 9999)}"

            # Generic fallback with variety
            return random.choice([
                f"Test Data {random.randint(100, 999)}",
                f"Automated Entry {random.randint(100, 999)}",
                f"QA Test {random.randint(100, 999)}"
            ])

    def _smart_verify(self, step: TestStep, test_case: TestCase) -> Tuple[bool, str]:
        """Smart verification"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            # Simple verification - check if page loaded
            self._wait_for_page_load()

            current_url = self.driver.current_url
            page_title = self.driver.title

            logger.info(f"✅ Verification: URL={current_url}, Title={page_title}")
            return True, f"Page verified: {page_title}"

        except Exception as e:
            return False, f"Verification error: {str(e)}"

    def _get_by_type(self, strategy: str):
        """Convert strategy string to Selenium By type"""
        from selenium.webdriver.common.by import By

        strategy_map = {
            'id': By.ID,
            'name': By.NAME,
            'xpath': By.XPATH,
            'css': By.CSS_SELECTOR,
            'link': By.LINK_TEXT,
            'partial_link': By.PARTIAL_LINK_TEXT,
            'tag': By.TAG_NAME,
            'class': By.CLASS_NAME
        }

        return strategy_map.get(strategy.lower())

    def _analyze_step_for_issues(self, step: TestStep, success: bool, message: str):
        """Analyze step execution for potential issues"""
        try:
            # Check for failures
            if not success:
                self.bug_report['functionality_issues'].append({
                    'step': step.step_number,
                    'description': step.description,
                    'error': message,
                    'severity': 'high'
                })

            # Check for console errors during this step
            recent_errors = [e for e in self.console_errors[-5:] if e['level'] in ['SEVERE', 'ERROR']]
            if recent_errors:
                self.bug_report['functionality_issues'].append({
                    'step': step.step_number,
                    'description': f"Console errors detected during: {step.description}",
                    'errors': recent_errors,
                    'severity': 'medium'
                })

            # Check for network errors
            recent_network_errors = [n for n in self.network_logs[-10:] if n.get('is_error')]
            if recent_network_errors:
                self.bug_report['network_errors'].extend(recent_network_errors)

            # Check form validation issues
            self._check_form_validation_issues(step)

            # Check accessibility issues
            self._check_accessibility_issues(step)

            # Check security vulnerabilities
            self._check_security_vulnerabilities(step)

        except Exception as e:
            logger.debug(f"Could not analyze step for issues: {str(e)}")

    def _check_form_validation_issues(self, step: TestStep):
        """Check for empty required fields and validation issues"""
        try:
            from selenium.webdriver.common.by import By

            # Find all form fields on the page
            try:
                forms = self.driver.find_elements(By.TAG_NAME, 'form')

                for form in forms:
                    # Check for empty required fields
                    required_inputs = form.find_elements(By.CSS_SELECTOR, 'input[required], select[required], textarea[required]')

                    for field in required_inputs:
                        try:
                            if not field.is_displayed():
                                continue

                            field_type = field.get_attribute('type') or 'text'
                            field_name = field.get_attribute('name') or field.get_attribute('id') or 'unnamed'
                            field_value = field.get_attribute('value') or ''

                            # Check if field is empty
                            if not field_value.strip():
                                # Check specifically for payment fields (card number, expiry, CVV)
                                is_payment_field = any(keyword in field_name.lower() for keyword in [
                                    'card', 'cvv', 'cvc', 'expiry', 'expiration', 'security'
                                ])

                                # Check for address field
                                is_address_field = any(keyword in field_name.lower() for keyword in [
                                    'address', 'street', 'addr'
                                ])

                                severity = 'critical' if is_payment_field else 'high'

                                self.bug_report['validation_issues'].append({
                                    'step': step.step_number,
                                    'type': 'empty_required_field',
                                    'field_name': field_name,
                                    'field_type': field_type,
                                    'is_payment_field': is_payment_field,
                                    'is_address_field': is_address_field,
                                    'description': f"Required field '{field_name}' is empty",
                                    'severity': severity,
                                    'recommendation': f"Ensure '{field_name}' field is filled before form submission"
                                })

                                logger.warning(f"⚠️  Validation Issue: Required field '{field_name}' is empty")

                            # Check for address dropdowns that need selection
                            if is_address_field and field.tag_name.lower() == 'input':
                                # Check if there's an address dropdown/autocomplete
                                try:
                                    dropdown = self.driver.find_elements(By.CSS_SELECTOR,
                                        '[role="listbox"], .autocomplete-dropdown, .address-suggestions, [class*="dropdown"]')

                                    if dropdown and any(d.is_displayed() for d in dropdown):
                                        self.bug_report['validation_issues'].append({
                                            'step': step.step_number,
                                            'type': 'address_dropdown_not_selected',
                                            'field_name': field_name,
                                            'description': f"Address dropdown appeared for '{field_name}' but no option was selected",
                                            'severity': 'high',
                                            'recommendation': 'Select an address from the dropdown before proceeding'
                                        })
                                        logger.warning(f"⚠️  Validation Issue: Address dropdown visible but not selected for '{field_name}'")
                                except:
                                    pass

                        except Exception as field_error:
                            logger.debug(f"Could not check field validation: {str(field_error)}")

            except Exception as form_error:
                logger.debug(f"Could not check form validation: {str(form_error)}")

        except Exception as e:
            logger.debug(f"Form validation check error: {str(e)}")

    def _check_accessibility_issues(self, step: TestStep):
        """Check for accessibility violations using basic WCAG principles"""
        try:
            from selenium.webdriver.common.by import By

            # Check for images without alt text
            try:
                images = self.driver.find_elements(By.TAG_NAME, 'img')
                for img in images:
                    if img.is_displayed():
                        alt_text = img.get_attribute('alt')
                        src = img.get_attribute('src') or 'unknown'

                        if not alt_text:
                            self.bug_report['accessibility_issues'].append({
                                'step': step.step_number,
                                'type': 'missing_alt_text',
                                'element': 'img',
                                'src': src[:100],
                                'description': 'Image missing alt text for screen readers',
                                'severity': 'medium',
                                'wcag_criterion': '1.1.1 Non-text Content',
                                'recommendation': 'Add descriptive alt text to all images'
                            })
            except:
                pass

            # Check for form inputs without labels
            try:
                inputs = self.driver.find_elements(By.CSS_SELECTOR, 'input[type="text"], input[type="email"], input[type="password"], textarea')
                for input_elem in inputs:
                    if input_elem.is_displayed():
                        input_id = input_elem.get_attribute('id')
                        aria_label = input_elem.get_attribute('aria-label')
                        aria_labelledby = input_elem.get_attribute('aria-labelledby')

                        # Check if there's an associated label
                        has_label = False
                        if input_id:
                            try:
                                labels = self.driver.find_elements(By.CSS_SELECTOR, f'label[for="{input_id}"]')
                                has_label = len(labels) > 0
                            except:
                                pass

                        if not has_label and not aria_label and not aria_labelledby:
                            field_name = input_elem.get_attribute('name') or 'unnamed'
                            self.bug_report['accessibility_issues'].append({
                                'step': step.step_number,
                                'type': 'missing_label',
                                'element': 'input',
                                'field_name': field_name,
                                'description': f"Form input '{field_name}' has no associated label",
                                'severity': 'high',
                                'wcag_criterion': '3.3.2 Labels or Instructions',
                                'recommendation': 'Add label element or aria-label attribute'
                            })
            except:
                pass

            # Check for insufficient color contrast (basic check via computed styles)
            try:
                # Check buttons and links for contrast issues
                elements = self.driver.find_elements(By.CSS_SELECTOR, 'button, a')
                for elem in elements[:10]:  # Limit to first 10 for performance
                    if elem.is_displayed():
                        try:
                            color = self.driver.execute_script(
                                "return window.getComputedStyle(arguments[0]).color;", elem)
                            bg_color = self.driver.execute_script(
                                "return window.getComputedStyle(arguments[0]).backgroundColor;", elem)

                            # Basic check - if both are similar (simplified)
                            if color and bg_color and color == bg_color:
                                self.bug_report['accessibility_issues'].append({
                                    'step': step.step_number,
                                    'type': 'contrast_issue',
                                    'element': elem.tag_name,
                                    'description': 'Potential color contrast issue detected',
                                    'severity': 'medium',
                                    'wcag_criterion': '1.4.3 Contrast (Minimum)',
                                    'recommendation': 'Ensure sufficient color contrast (4.5:1 for normal text)'
                                })
                        except:
                            pass
            except:
                pass

        except Exception as e:
            logger.debug(f"Accessibility check error: {str(e)}")

    def _check_security_vulnerabilities(self, step: TestStep):
        """Check for common security vulnerabilities"""
        try:
            from selenium.webdriver.common.by import By

            # Check for password fields without autocomplete="off" or new-password
            try:
                password_fields = self.driver.find_elements(By.CSS_SELECTOR, 'input[type="password"]')
                for pwd_field in password_fields:
                    if pwd_field.is_displayed():
                        autocomplete = pwd_field.get_attribute('autocomplete')
                        field_name = pwd_field.get_attribute('name') or 'unnamed'

                        # Check if it's a new password field without proper autocomplete
                        if 'new' in field_name.lower() or 'confirm' in field_name.lower():
                            if autocomplete != 'new-password':
                                self.bug_report['security_issues'].append({
                                    'step': step.step_number,
                                    'type': 'password_autocomplete_vulnerability',
                                    'field_name': field_name,
                                    'description': f"Password field '{field_name}' should use autocomplete='new-password'",
                                    'severity': 'medium',
                                    'recommendation': "Set autocomplete='new-password' for new password fields"
                                })
            except:
                pass

            # Check for forms submitted over HTTP instead of HTTPS
            try:
                current_url = self.driver.current_url
                if current_url.startswith('http://'):
                    forms = self.driver.find_elements(By.TAG_NAME, 'form')
                    for form in forms:
                        # Check if form has password or payment fields
                        has_sensitive = len(form.find_elements(By.CSS_SELECTOR,
                            'input[type="password"], input[name*="card"], input[name*="cvv"]')) > 0

                        if has_sensitive:
                            self.bug_report['security_issues'].append({
                                'step': step.step_number,
                                'type': 'insecure_form_submission',
                                'url': current_url,
                                'description': 'Form with sensitive data is on an insecure (HTTP) page',
                                'severity': 'critical',
                                'recommendation': 'All pages with sensitive data must use HTTPS'
                            })
                            break
            except:
                pass

            # Check for credit card fields without proper input masking
            try:
                card_fields = self.driver.find_elements(By.CSS_SELECTOR,
                    'input[name*="card"], input[id*="card"], input[placeholder*="card"]')

                for card_field in card_fields:
                    if card_field.is_displayed():
                        field_type = card_field.get_attribute('type')
                        field_name = card_field.get_attribute('name') or 'unnamed'

                        # CVV should be password type or have maxlength=4
                        if any(term in field_name.lower() for term in ['cvv', 'cvc', 'security']):
                            if field_type != 'password':
                                maxlength = card_field.get_attribute('maxlength')
                                if not maxlength or int(maxlength) > 4:
                                    self.bug_report['security_issues'].append({
                                        'step': step.step_number,
                                        'type': 'cvv_field_security',
                                        'field_name': field_name,
                                        'description': f"CVV field '{field_name}' should be type='password' or have maxlength='4'",
                                        'severity': 'high',
                                        'recommendation': 'Use type="password" for CVV fields and set maxlength="4"'
                                    })
            except:
                pass

            # Check for inline JavaScript event handlers (XSS risk)
            try:
                elements_with_inline_js = self.driver.find_elements(By.XPATH,
                    '//*[@onclick or @onload or @onerror or @onmouseover]')

                if len(elements_with_inline_js) > 0:
                    self.bug_report['security_issues'].append({
                        'step': step.step_number,
                        'type': 'inline_javascript_handlers',
                        'count': len(elements_with_inline_js),
                        'description': f'Found {len(elements_with_inline_js)} elements with inline JavaScript handlers',
                        'severity': 'low',
                        'recommendation': 'Use event listeners instead of inline JavaScript to reduce XSS risk'
                    })
            except:
                pass

        except Exception as e:
            logger.debug(f"Security check error: {str(e)}")

    async def generate_ai_bug_report(self) -> str:
        """
        Generate comprehensive AI-powered bug report

        Returns:
            Formatted bug report with AI insights
        """
        try:
            if not self.azure_client or not self.azure_client.is_configured():
                return self._generate_basic_bug_report()

            # Prepare context for AI analysis
            context = {
                'total_steps': len(self.dom_snapshots),
                'console_errors': len(self.console_errors),
                'network_errors': len([n for n in self.network_logs if n.get('is_error')]),
                'functionality_issues': self.bug_report['functionality_issues'],
                'performance_issues': self.bug_report['performance_issues'],
                'validation_issues': self.bug_report['validation_issues'],
                'accessibility_issues': self.bug_report['accessibility_issues'],
                'security_issues': self.bug_report['security_issues'],
                'screenshots': len(self.screenshots)
            }

            prompt = f"""Analyze this test automation session and provide a comprehensive bug report from a customer experience perspective.

Test Execution Summary:
- Total Steps Executed: {context['total_steps']}
- Console Errors Found: {context['console_errors']}
- Network Errors Found: {context['network_errors']}
- Failed Steps: {len(self.bug_report['functionality_issues'])}
- Performance Issues: {len(self.bug_report['performance_issues'])}
- Form Validation Issues: {len(self.bug_report['validation_issues'])}
- Accessibility Violations: {len(self.bug_report['accessibility_issues'])}
- Security Vulnerabilities: {len(self.bug_report['security_issues'])}

Detailed Issues:
{json.dumps(self.bug_report, indent=2)}

Recent Console Errors:
{json.dumps(self.console_errors[-10:], indent=2) if self.console_errors else 'None'}

Network Errors:
{json.dumps([n for n in self.network_logs if n.get('is_error')][:5], indent=2)}

Please provide:
1. Functionality Issues: Broken features, failed actions, errors
2. UI/UX Issues: Poor user experience, confusing flows, accessibility problems
3. Performance Issues: Slow loading, unresponsive pages
4. Form Validation Issues: Empty required fields (especially payment fields like card number, expiry, CVV), address dropdown selection not completed
5. Accessibility Violations: WCAG compliance issues, missing alt text, missing labels, contrast issues
6. Security Vulnerabilities: Insecure forms, weak password policies, potential XSS risks
7. Recommendations: How to fix identified issues with priority

Format as a professional QA bug report with severity levels."""

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert QA engineer analyzing test automation results to identify bugs and UX issues."
                               " Provide clear, concise, and actionable reports."
                               " Use markdown formatting."
                               " Focus on customer experience and usability."
                               " Prioritize issues by severity."
                               " Suggest improvements where applicable."
                               " Keep the report structured and easy to read."
                               " Avoid technical jargon unless necessary."
                               " Be empathetic to end-users' perspectives."
                               " Always aim to enhance overall user satisfaction."
                               " Ensure recommendations are practical and implementable."
                               " Maintain a professional tone throughout the report."
                               " Deliver actionable insights that help improve the product quality effectively."
                               " Remember to back your findings with data from the test execution."
                               " Stay objective and unbiased in your analysis."
                               " Strive for clarity and precision in your explanations."
                               " Your goal is to help the development team understand and resolve issues efficiently."
                               " Provide value through your expertise and attention to detail."
                               " Keep the end-user experience at the forefront of your analysis."
                               " Always aim to contribute positively to the product's success."
                               " Uphold the highest standards of QA reporting."
                               " Be thorough yet concise in your evaluations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self.azure_client.chat_completion_create(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )

            ai_report = response['choices'][0]['message']['content']

            # Combine with basic stats
            report = f"""# TestPilot Automation Bug Report
Date: {datetime.now().strftime('%B %d, %Y')}  
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
Prepared by: AI Quality Centre of Excellence  
Contact: siddhant.wadhwani@newfold.com

## AI Analysis

{ai_report}

## Raw Data

### Form Validation Issues
```json
{json.dumps(self.bug_report['validation_issues'], indent=2)}
```

### Accessibility Violations
```json
{json.dumps(self.bug_report['accessibility_issues'], indent=2)}
```

### Security Vulnerabilities
```json
{json.dumps(self.bug_report['security_issues'], indent=2)}
```

### Console Errors
```json
{json.dumps(self.console_errors, indent=2)}
```

### Network Errors
```json
{json.dumps([n for n in self.network_logs if n.get('is_error')], indent=2)}
```

### Performance Metrics
```json
{json.dumps(self.performance_metrics, indent=2)}
```
"""

            return report

        except Exception as e:
            logger.error(f"Error generating AI bug report: {str(e)}")
            return self._generate_basic_bug_report()

    def _generate_basic_bug_report(self) -> str:
        """Generate basic bug report without AI"""
        report = f"""# TestPilot Automation Bug Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
Prepared by: AI Quality Centre of Excellence  
Contact: siddhant.wadhwani@newfold.com

## Execution Summary
- **Total Snapshots**: {len(self.dom_snapshots)}
- **Console Errors**: {len(self.console_errors)}
- **Network Errors**: {len([n for n in self.network_logs if n.get('is_error')])}
- **Screenshots**: {len(self.screenshots)}
- **Form Validation Issues**: {len(self.bug_report['validation_issues'])}
- **Accessibility Violations**: {len(self.bug_report['accessibility_issues'])}
- **Security Vulnerabilities**: {len(self.bug_report['security_issues'])}

## Issues Found

### Functionality Issues
{json.dumps(self.bug_report['functionality_issues'], indent=2)}

### Form Validation Issues
{json.dumps(self.bug_report['validation_issues'], indent=2)}

### Accessibility Violations
{json.dumps(self.bug_report['accessibility_issues'], indent=2)}

### Security Vulnerabilities
{json.dumps(self.bug_report['security_issues'], indent=2)}

### Console Errors
{json.dumps(self.console_errors, indent=2)}

### Network Errors
{json.dumps([n for n in self.network_logs if n.get('is_error')], indent=2)}

### Performance Issues
{json.dumps(self.bug_report['performance_issues'], indent=2)}
"""
        return report

    def cleanup(self):
        """Cleanup browser resources and RobotMCP connections"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
            logger.info("✅ Browser cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Browser cleanup warning: {str(e)}")

        # Cleanup RobotMCP connection
        try:
            if self.robotmcp_helper and self.robotmcp_helper.is_connected:
                # Use synchronous shutdown to avoid event loop issues
                self.robotmcp_helper.shutdown()
                logger.info("✅ RobotMCP cleanup completed")
        except Exception as e:
            logger.debug(f"RobotMCP cleanup: {str(e)}")


class RecordingParser:
    """Parse recording JSON files and convert to test steps"""

    @staticmethod
    def parse_recording(recording_data: Dict[str, Any]) -> List[TestStep]:
        """
        Parse recording JSON and extract test steps intelligently for AI analysis

        Supports multiple recording formats:
        - Puppeteer/Playwright recordings
        - Selenium IDE exports
        - Chrome DevTools Protocol recordings
        - Custom recording formats

        Args:
            recording_data: Recording JSON data

        Returns:
            List of TestStep objects with detailed descriptions for AI analysis
        """
        steps = []

        try:
            # Support multiple recording formats
            events = recording_data.get('events',
                                       recording_data.get('actions',
                                       recording_data.get('steps', [])))

            logger.info(f"🎬 Parsing {len(events)} events from recording")

            # Track page URL for context
            current_url = recording_data.get('startUrl', recording_data.get('url', ''))

            for i, event in enumerate(events, 1):
                # Extract action type from various format fields
                action_type = (event.get('type') or
                             event.get('action') or
                             event.get('command') or
                             event.get('eventType', ''))

                # Extract selector/target from various fields
                selector = (event.get('selector') or
                          event.get('target') or
                          event.get('locator') or
                          event.get('xpath') or
                          event.get('css', ''))

                # Extract value from various fields
                value = (event.get('value') or
                        event.get('input') or
                        event.get('text') or
                        event.get('data', ''))

                # Extract element context for better description
                element_text = event.get('text', event.get('innerText', ''))
                element_type = event.get('tagName', event.get('elementType', ''))
                element_attributes = event.get('attributes', {})
                page_url = event.get('url', event.get('href', current_url))

                # Update current URL if navigation occurred
                if action_type in ['navigate', 'goto', 'navigation', 'url']:
                    current_url = value or page_url

                # Build intelligent description based on action type
                description = RecordingParser._build_action_description(
                    action_type, selector, value, element_text,
                    element_type, element_attributes, page_url
                )

                # Normalize action type to standard categories
                normalized_action = RecordingParser._normalize_action_type(action_type)

                step = TestStep(
                    step_number=i,
                    description=description,
                    action=normalized_action,
                    target=selector,
                    value=str(value) if value else ""
                )

                # Add enriched metadata as notes for AI context
                notes_parts = []
                if 'timestamp' in event:
                    notes_parts.append(f"Timestamp: {event['timestamp']}")
                if element_text:
                    notes_parts.append(f"Element Text: {element_text}")
                if element_type:
                    notes_parts.append(f"Element Type: {element_type}")
                if page_url and page_url != current_url:
                    notes_parts.append(f"Page URL: {page_url}")
                if element_attributes:
                    # Extract key attributes
                    key_attrs = {k: v for k, v in element_attributes.items()
                               if k in ['id', 'name', 'class', 'placeholder', 'aria-label', 'role']}
                    if key_attrs:
                        notes_parts.append(f"Attributes: {json.dumps(key_attrs)}")

                step.notes = " | ".join(notes_parts)

                # Skip non-actionable events (like mouse movements without clicks)
                if RecordingParser._is_actionable_event(action_type, event):
                    steps.append(step)
                    logger.debug(f"✅ Step {i}: {normalized_action} - {description}")
                else:
                    logger.debug(f"⏭️  Skipped non-actionable event: {action_type}")

            logger.info(f"✅ Parsed {len(steps)} actionable steps from recording")

        except Exception as e:
            logger.error(f"Error parsing recording: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

        return steps

    @staticmethod
    def _build_action_description(action_type: str, selector: str, value: Any,
                                  element_text: str, element_type: str,
                                  attributes: Dict, url: str) -> str:
        """
        Build an intelligent, human-readable description of the action
        that Azure OpenAI can understand and convert to test steps
        """
        action_lower = action_type.lower() if action_type else ''

        # Navigation actions
        if action_lower in ['navigate', 'goto', 'navigation', 'url', 'open']:
            return f"Navigate to URL: {value or url}"

        # Click actions
        elif action_lower in ['click', 'mousedown', 'mouseup', 'tap']:
            target_desc = element_text or attributes.get('aria-label', '') or attributes.get('title', '')
            if target_desc:
                return f"Click on '{target_desc}' {element_type or 'element'}"
            elif attributes.get('id'):
                return f"Click on {element_type or 'element'} with id '{attributes['id']}'"
            elif selector:
                return f"Click on element: {selector}"
            else:
                return f"Click on {element_type or 'element'}"

        # Input/type actions
        elif action_lower in ['type', 'input', 'keydown', 'keyup', 'fill', 'setvalue']:
            field_name = (element_text or
                        attributes.get('placeholder', '') or
                        attributes.get('name', '') or
                        attributes.get('aria-label', ''))
            if field_name:
                # Mask sensitive data in description
                display_value = value if value and len(str(value)) < 50 else '[input text]'
                if any(sensitive in field_name.lower() for sensitive in ['password', 'secret', 'token']):
                    display_value = '[sensitive data]'
                return f"Enter '{display_value}' into {field_name} field"
            elif selector:
                return f"Enter text into field: {selector}"
            else:
                return f"Enter text into {element_type or 'input'} field"

        # Select/dropdown actions
        elif action_lower in ['select', 'choose', 'dropdown']:
            field_name = element_text or attributes.get('name', '')
            if field_name:
                return f"Select '{value}' from {field_name} dropdown"
            else:
                return f"Select '{value}' from dropdown"

        # Checkbox/radio actions
        elif action_lower in ['check', 'uncheck', 'toggle']:
            label = element_text or attributes.get('aria-label', '')
            if label:
                return f"{action_type.capitalize()} '{label}' checkbox"
            else:
                return f"{action_type.capitalize()} checkbox: {selector}"

        # Wait/assertion actions
        elif action_lower in ['wait', 'waitfor', 'assert', 'verify', 'expect']:
            if value:
                return f"Wait for/verify: {value}"
            elif element_text:
                return f"Wait for/verify element with text: {element_text}"
            elif selector:
                return f"Wait for/verify element: {selector}"
            else:
                return f"Wait for element to be visible"

        # Hover actions
        elif action_lower in ['hover', 'mousemove', 'mouseover']:
            target_desc = element_text or attributes.get('aria-label', '')
            if target_desc:
                return f"Hover over '{target_desc}'"
            else:
                return f"Hover over element: {selector}"

        # Scroll actions
        elif action_lower in ['scroll', 'scrollto']:
            if value:
                return f"Scroll to: {value}"
            elif element_text:
                return f"Scroll to element with text: {element_text}"
            else:
                return f"Scroll to element: {selector}"

        # File upload actions
        elif action_lower in ['upload', 'file', 'attach']:
            return f"Upload file: {value}"

        # Screenshot/capture actions
        elif action_lower in ['screenshot', 'capture']:
            return f"Take screenshot"

        # Generic fallback
        else:
            description_parts = [action_type or 'Action']
            if element_text:
                description_parts.append(f"on '{element_text}'")
            elif selector:
                description_parts.append(f"on {selector}")
            if value:
                description_parts.append(f"with value '{value}'")
            return " ".join(description_parts)

    @staticmethod
    def _normalize_action_type(action_type: str) -> str:
        """Normalize action type to standard categories for Robot Framework"""
        if not action_type:
            return "action"

        action_lower = action_type.lower()

        # Navigation
        if action_lower in ['navigate', 'goto', 'navigation', 'url', 'open']:
            return "navigate"

        # Click
        elif action_lower in ['click', 'mousedown', 'mouseup', 'tap', 'press']:
            return "click"

        # Input
        elif action_lower in ['type', 'input', 'keydown', 'keyup', 'fill', 'setvalue', 'sendkeys']:
            return "input"

        # Select
        elif action_lower in ['select', 'choose', 'dropdown']:
            return "select"

        # Verify/Assert
        elif action_lower in ['assert', 'verify', 'expect', 'should', 'check']:
            return "verify"

        # Wait
        elif action_lower in ['wait', 'waitfor', 'sleep', 'pause']:
            return "wait"

        # Hover
        elif action_lower in ['hover', 'mousemove', 'mouseover']:
            return "hover"

        # Scroll
        elif action_lower in ['scroll', 'scrollto']:
            return "scroll"

        # Upload
        elif action_lower in ['upload', 'file', 'attach']:
            return "upload"

        # Screenshot
        elif action_lower in ['screenshot', 'capture']:
            return "screenshot"

        return action_type

    @staticmethod
    def _is_actionable_event(action_type: str, event: Dict) -> bool:
        """
        Determine if an event represents an actionable test step
        Filters out noise like mouse movements, focus events, etc.
        """
        if not action_type:
            return False

        action_lower = action_type.lower()

        # Skip pure mouse movement without clicks
        if action_lower in ['mousemove'] and 'click' not in event:
            return False

        # Skip focus events unless they have a purpose
        if action_lower in ['focus', 'blur'] and not event.get('value'):
            return False

        # Skip scroll events unless explicitly recorded as important
        if action_lower == 'scroll' and not event.get('important', True):
            return False

        # Include most other events
        return True


class TestPilotEngine:
    """Core engine for TestPilot - handles AI conversion and Robot Framework generation"""

    def __init__(self, azure_client: Optional[AzureOpenAIClient] = None):
        self.azure_client = azure_client
        self.output_dir = os.path.join(os.getcwd(), "generated_tests")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load architecture knowledge for reusability
        self.architecture_context = self._load_architecture_context()

        # Cache for scraped website data
        self.website_cache = {}

    def _extract_locators_from_url_with_selenium(self, url: str, keywords: list) -> dict:
        """
        Extract locators using Selenium for JavaScript-rendered pages

        Args:
            url: Website URL to scrape
            keywords: Keywords from test steps to help identify elements

        Returns:
            Dict of found locators with values
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException

            logger.info(f"🌐 Using Selenium to scrape dynamic content from {url}")

            # Setup Chrome in headless mode with stability options
            chrome_options = Options()
            chrome_options.add_argument('--headless=new')  # Use new headless mode
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-software-rasterizer')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-browser-side-navigation')
            chrome_options.add_argument('--disable-features=TranslateUI,BlinkGenPropertyTrees')
            chrome_options.add_argument('--remote-debugging-port=0')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

            # Exclude automation flags
            chrome_options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
            chrome_options.add_experimental_option('useAutomationExtension', False)

            # Initialize with Service for better error handling
            try:
                from selenium.webdriver.chrome.service import Service
                service = Service()
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception:
                driver = webdriver.Chrome(options=chrome_options)

            driver.set_page_load_timeout(60)  # Increased from 30 to 60
            driver.set_script_timeout(30)
            driver.implicitly_wait(10)

            try:
                driver.get(url)
                # Wait for page to be fully loaded
                WebDriverWait(driver, 10).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete'
                )

                found_locators = {}

                # Extract keyword-based elements
                for keyword in keywords:
                    keyword_parts = keyword.lower().replace('_locator', '').replace('_', ' ').split()

                    # Try to find elements matching keywords
                    for part in keyword_parts:
                        if len(part) < 3:  # Skip short words
                            continue

                        try:
                            # Search by text content
                            elements = driver.find_elements(By.XPATH, f"//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{part}')]")
                            for elem in elements[:3]:  # Check first 3 matches
                                try:
                                    if not elem.is_displayed():
                                        continue

                                    # Get best locator for this element
                                    elem_id = elem.get_attribute('id')
                                    elem_name = elem.get_attribute('name')
                                    elem_class = elem.get_attribute('class')
                                    tag_name = elem.tag_name

                                    locator_value = None
                                    if elem_id:
                                        locator_value = f"id:{elem_id}"
                                    elif elem_name and tag_name in ['input', 'select', 'textarea']:
                                        locator_value = f"name:{elem_name}"
                                    elif part in elem.text.lower():
                                        if tag_name == 'a':
                                            locator_value = f"link:{elem.text.strip()}"
                                        else:
                                            locator_value = f"xpath://{tag_name}[contains(text(), '{elem.text.strip()[:30]}')]"
                                    elif elem_class:
                                        classes = elem_class.split()
                                        if classes:
                                            locator_value = f"css:.{classes[0]}"

                                    if locator_value and keyword not in found_locators:
                                        found_locators[keyword] = locator_value
                                        logger.info(f"✅ Found via Selenium: {keyword} = {locator_value}")
                                        break
                                except Exception:
                                    continue
                        except Exception:
                            continue

                # Additional scraping for common patterns
                # Buttons
                try:
                    buttons = driver.find_elements(By.TAG_NAME, 'button')
                    for btn in buttons[:20]:
                        try:
                            if not btn.is_displayed():
                                continue
                            text = btn.text.strip().lower()
                            if text and any(word in text for word in ['explore', 'continue', 'checkout', 'submit', 'get started', 'buy', 'select']):
                                btn_id = btn.get_attribute('id')
                                locator_name = f"{text.replace(' ', '_')}_button_locator"
                                if btn_id:
                                    found_locators[locator_name] = f"id:{btn_id}"
                                else:
                                    found_locators[locator_name] = f"xpath://button[contains(text(), '{btn.text.strip()}')]"
                        except Exception:
                            continue
                except Exception:
                    pass

                # Input fields
                try:
                    inputs = driver.find_elements(By.TAG_NAME, 'input')
                    for inp in inputs[:15]:
                        try:
                            inp_type = inp.get_attribute('type')
                            inp_name = inp.get_attribute('name')
                            inp_id = inp.get_attribute('id')
                            placeholder = inp.get_attribute('placeholder')

                            if inp_name or inp_id or placeholder:
                                field_name = inp_name or inp_id or placeholder
                                locator_name = f"{field_name.lower().replace(' ', '_')}_input_locator"

                                if inp_id:
                                    found_locators[locator_name] = f"id:{inp_id}"
                                elif inp_name:
                                    found_locators[locator_name] = f"name:{inp_name}"
                        except Exception:
                            continue
                except Exception:
                    pass

                logger.info(f"🎯 Selenium found {len(found_locators)} locators")
                return found_locators

            finally:
                driver.quit()

        except ImportError:
            logger.warning("⚠️  Selenium not installed. Install with: pip install selenium")
            return {}
        except Exception as e:
            logger.error(f"❌ Selenium scraping error: {str(e)}")
            return {}

    def _extract_locators_from_url(self, url: str, keywords: list) -> dict:
        """
        Extract intelligent locators from a website URL

        Args:
            url: Website URL to scrape
            keywords: List of keywords to look for (button, menu, input, etc.)

        Returns:
            Dict of found locators with suggestions
        """
        try:
            import requests
            from bs4 import BeautifulSoup

            # Check cache first
            if url in self.website_cache:
                logger.info(f"Using cached locators for {url}")
                return self.website_cache[url]

            logger.info(f"🔍 Scraping {url} for locators...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            found_locators = {}

            # Extract common interactive elements
            # 1. Buttons and CTAs
            for btn in soup.find_all(['button', 'a'], limit=30):
                text = btn.get_text(strip=True)
                classes = btn.get('class', [])
                href = btn.get('href', '')

                # Filter relevant buttons
                if text and len(text) < 50 and (
                    any(cls for cls in classes if 'btn' in str(cls).lower() or 'button' in str(cls).lower() or 'cta' in str(cls).lower()) or
                    any(word in text.lower() for word in ['get started', 'explore', 'buy', 'continue', 'checkout', 'submit', 'sign up', 'select'])
                ):
                    locator_name = f"{text.lower().replace(' ', '_')}_button_locator"
                    if btn.get('id'):
                        found_locators[locator_name] = f"id:{btn['id']}"
                    elif text and len(text) < 30:
                        found_locators[locator_name] = f"xpath://button[contains(text(), '{text}')]" if btn.name == 'button' else f"link:{text}"
                    elif classes and len(classes) > 0:
                        found_locators[locator_name] = f"css:.{str(classes[0])}"

            # 2. Navigation menus and links
            for nav in soup.find_all(['nav', 'header', 'ul'], limit=10):
                links = nav.find_all('a', limit=20)
                for link in links:
                    text = link.get_text(strip=True)
                    href = link.get('href', '')

                    if text and len(text) < 30 and (
                        'wordpress' in text.lower() or 'hosting' in text.lower() or
                        'cloud' in text.lower() or 'plan' in text.lower() or
                        'email' in text.lower() or 'domain' in text.lower()
                    ):
                        locator_name = f"{text.lower().replace(' ', '_')}_menu_locator"
                        if link.get('id'):
                            found_locators[locator_name] = f"id:{link['id']}"
                        elif href and ('wordpress' in href.lower() or 'hosting' in href.lower() or 'cloud' in href.lower()):
                            found_locators[locator_name] = f"css:a[href*='{href.split('/')[-1]}']"
                        else:
                            found_locators[locator_name] = f"link:{text}"

            # 3. Input fields and forms
            for inp in soup.find_all(['input', 'textarea'], limit=15):
                inp_type = inp.get('type', 'text')
                name = inp.get('name', '')
                placeholder = inp.get('placeholder', '')
                inp_id = inp.get('id', '')

                if name or placeholder or inp_id:
                    field_name = name or placeholder or inp_id
                    locator_name = f"{field_name.lower().replace(' ', '_')}_input_locator"

                    if inp_id:
                        found_locators[locator_name] = f"id:{inp_id}"
                    elif name:
                        found_locators[locator_name] = f"name:{name}"
                    elif placeholder:
                        found_locators[locator_name] = f"css:input[placeholder='{placeholder}']"

            # 4. Plan/Product cards (common in hosting sites)
            for card in soup.find_all(['div', 'article'], class_=lambda x: x and ('plan' in str(x).lower() or 'product' in str(x).lower() or 'card' in str(x).lower()), limit=10):
                heading = card.find(['h2', 'h3', 'h4'])
                if heading:
                    text = heading.get_text(strip=True)
                    if text and len(text) < 30:
                        locator_name = f"{text.lower().replace(' ', '_')}_plan_locator"
                        card_class = card.get('class', [])
                        if card.get('id'):
                            found_locators[locator_name] = f"id:{card['id']}"
                        elif card_class:
                            found_locators[locator_name] = f"css:.{card_class[0]}"

            # Cache the results
            self.website_cache[url] = found_locators
            logger.info(f"✅ Found {len(found_locators)} locators from {url}")

            # Log sample of found locators for debugging
            if found_locators:
                sample = list(found_locators.items())[:5]
                logger.info(f"Sample locators: {sample}")

            return found_locators

        except requests.RequestException as e:
            logger.error(f"❌ Network error scraping {url}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"❌ Error scraping {url}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _enrich_locators_with_web_data(self, test_case: TestCase, locators: list) -> list:
        """
        Enrich generated locators with actual data from website using advanced scraping

        Args:
            test_case: TestCase with steps
            locators: List of (locator_name, description) tuples

        Returns:
            Enriched list with actual locator values where possible
        """
        # Try to find URL in test steps
        url = None
        for step in test_case.steps:
            if 'http' in step.description:
                # Extract URL from description
                import re
                urls = re.findall(r'https?://[^\s\)]+', step.description)
                if urls:
                    url = urls[0].rstrip('/').rstrip(',').rstrip('.')
                    logger.info(f"🔗 Found URL in test steps: {url}")
                    break

        if not url:
            logger.warning("⚠️  No URL found in test steps - skipping web scraping")
            logger.warning("💡 Tip: Include the website URL in your first step (e.g., 'Navigate to https://www.example.com/')")
            return [(name, desc, None) for name, desc in locators]

        # Extract locator names only for targeted scraping
        locator_names = [name for name, _ in locators]

        # Try Selenium first for JavaScript-rendered content
        scraped_data = self._extract_locators_from_url_with_selenium(url, locator_names)

        # If Selenium didn't find enough, supplement with requests-based scraping
        if len(scraped_data) < len(locators) * 0.3:  # Less than 30% found
            logger.info("📡 Supplementing with requests-based scraping...")
            requests_data = self._extract_locators_from_url(url, locator_names)
            # Merge, preferring Selenium results
            for key, value in requests_data.items():
                if key not in scraped_data:
                    scraped_data[key] = value

        if not scraped_data:
            logger.warning(f"⚠️  No locators found from {url} - using placeholders")
            return [(name, desc, None) for name, desc in locators]

        # Match and enrich
        enriched = []
        for locator_name, description in locators:
            # Try to find matching scraped locator
            actual_locator = None

            # Direct match
            if locator_name in scraped_data:
                actual_locator = scraped_data[locator_name]
                logger.info(f"✅ Direct match: {locator_name} = {actual_locator}")
            else:
                # Fuzzy match based on keywords
                name_parts = set(locator_name.replace('_locator', '').split('_'))
                best_match_score = 0
                best_match_locator = None

                for scraped_name, scraped_value in scraped_data.items():
                    scraped_parts = set(scraped_name.replace('_locator', '').replace('_button', '').replace('_input', '').split('_'))
                    # Calculate match score
                    common = name_parts & scraped_parts
                    if len(common) > best_match_score:
                        best_match_score = len(common)
                        best_match_locator = scraped_value

                # Use best match if score is good enough
                if best_match_score >= max(1, len(name_parts) // 2):
                    actual_locator = best_match_locator
                    logger.info(f"🔍 Fuzzy match: {locator_name} = {actual_locator} (score: {best_match_score})")

            enriched.append((locator_name, description, actual_locator))

        found_count = sum(1 for _, _, loc in enriched if loc)
        total_count = len(enriched)
        percentage = int(found_count/total_count*100) if total_count > 0 else 0
        logger.info(f"📊 AUTO-DETECTED {found_count}/{total_count} locators ({percentage}%)")

        return enriched


    def _load_architecture_context(self) -> str:
        """Load architecture context for keyword/locator reusability"""
        try:
            arch_file = os.path.join(ROOT_DIR, "ARCHITECTURE.md")
            if os.path.exists(arch_file):
                with open(arch_file, 'r') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading architecture: {str(e)}")

        return ""

    def _scan_existing_keywords(self) -> Dict[str, List[str]]:
        """
        Scan existing keywords from the repository

        Returns:
            Dict mapping category to list of keyword names
        """
        try:
            keywords = {
                'ui_common': [],
                'api_common': [],
                'brand_specific': []
            }

            # Scan UI common keywords
            ui_common_path = os.path.join(ROOT_DIR, "tests", "keywords", "ui", "ui_common", "common.robot")
            if os.path.exists(ui_common_path):
                with open(ui_common_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract keyword names (lines that are not indented after *** Keywords ***)
                    in_keywords_section = False
                    for line in content.split('\n'):
                        if '*** Keywords ***' in line:
                            in_keywords_section = True
                            continue
                        if in_keywords_section and line and not line.startswith((' ', '\t', '#', '[')):
                            keyword_name = line.strip()
                            if keyword_name:
                                keywords['ui_common'].append(keyword_name)

            # Scan API common keywords
            api_common_path = os.path.join(ROOT_DIR, "tests", "keywords", "api", "api_common", "common.robot")
            if os.path.exists(api_common_path):
                with open(api_common_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    in_keywords_section = False
                    for line in content.split('\n'):
                        if '*** Keywords ***' in line:
                            in_keywords_section = True
                            continue
                        if in_keywords_section and line and not line.startswith((' ', '\t', '#', '[')):
                            keyword_name = line.strip()
                            if keyword_name:
                                keywords['api_common'].append(keyword_name)

            logger.info(f"📚 Found {len(keywords['ui_common'])} UI keywords, {len(keywords['api_common'])} API keywords")
            return keywords

        except Exception as e:
            logger.error(f"Error scanning keywords: {str(e)}")
            return {'ui_common': [], 'api_common': [], 'brand_specific': []}

    def _scan_existing_locators(self, brand: str = None) -> Dict[str, str]:
        """
        Scan existing locators from the repository

        Args:
            brand: Brand name to scan (e.g., 'bhcom', 'dcom')

        Returns:
            Dict mapping locator variable names to their values
        """
        try:
            locators = {}

            # Determine locator paths to scan
            locator_dirs = []
            if brand:
                locator_dirs.append(os.path.join(ROOT_DIR, "tests", "locators", "ui", brand))
            else:
                # Scan all brands
                ui_locators_path = os.path.join(ROOT_DIR, "tests", "locators", "ui")
                if os.path.exists(ui_locators_path):
                    for brand_dir in os.listdir(ui_locators_path):
                        brand_path = os.path.join(ui_locators_path, brand_dir)
                        if os.path.isdir(brand_path):
                            locator_dirs.append(brand_path)

            # Scan Python locator files
            for locator_dir in locator_dirs:
                if not os.path.exists(locator_dir):
                    continue

                for root, dirs, files in os.walk(locator_dir):
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    # Extract locator assignments (simple pattern)
                                    for line in content.split('\n'):
                                        if '=' in line and not line.strip().startswith('#'):
                                            parts = line.split('=', 1)
                                            if len(parts) == 2:
                                                var_name = parts[0].strip()
                                                var_value = parts[1].strip().strip('"\'')
                                                if '_locator' in var_name.lower() or '_selector' in var_name.lower():
                                                    locators[var_name] = var_value
                            except Exception as e:
                                logger.debug(f"Could not read locator file {file_path}: {str(e)}")

            logger.info(f"📍 Found {len(locators)} existing locators")
            return locators

        except Exception as e:
            logger.error(f"Error scanning locators: {str(e)}")
            return {}

    async def analyze_and_generate_with_browser_automation(
        self,
        test_case: TestCase,
        base_url: str,
        headless: bool = True,
        environment: str = 'prod',
        use_robotmcp: bool = False
    ) -> Tuple[bool, str, str, str]:
        """
        Enhanced script generation with live browser automation

        This method:
        1. Initializes browser with environment-specific configuration (proxy, user agent)
        2. Executes each test step smartly
        3. Captures network logs, console errors, DOM snapshots
        4. Generates AI bug report
        5. Generates Robot Framework scripts with real locators

        Args:
            test_case: TestCase with steps
            base_url: Base URL to start automation
            headless: Run browser in headless mode
            environment: Target environment (prod, qamain, stage, jarvisqa1, jarvisqa2)
            use_robotmcp: Use RobotMCP for advanced automation

        Returns:
            Tuple of (success, script_content, file_path, bug_report)
        """
        browser_mgr = None
        try:
            logger.info(f"🚀 Starting enhanced analysis with browser automation for: {test_case.title}")

            # Initialize browser automation manager
            browser_mgr = BrowserAutomationManager(self.azure_client)

            # Reset filled forms tracker for this new test case
            browser_mgr.filled_forms.clear()
            logger.info("   ♻️  Reset form tracking for new test case")

            if not browser_mgr.initialize_browser(base_url, headless, environment):
                return False, "", "", "Failed to initialize browser"

            # Execute each step with browser automation
            for step in test_case.steps:
                success, message = await browser_mgr.execute_step_smartly(step, test_case)
                logger.info(f"Step {step.step_number}: {'✅' if success else '❌'} {message}")

            # Generate AI bug report
            bug_report = await browser_mgr.generate_ai_bug_report()

            # Save bug report
            bug_report_path = os.path.join(
                self.output_dir,
                f"bug_report_{test_case.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(bug_report_path, 'w') as f:
                f.write(bug_report)

            logger.info(f"📋 Bug report saved: {bug_report_path}")

            # Enrich test case with captured locators and variables
            logger.info(f"📍 Transferring {len(browser_mgr.captured_locators)} captured locators to test case metadata")
            logger.info(f"📊 Transferring {len(browser_mgr.captured_variables)} captured variables to test case metadata")

            test_case.metadata['captured_locators'] = browser_mgr.captured_locators
            test_case.metadata['captured_variables'] = browser_mgr.captured_variables
            test_case.metadata['dom_snapshots'] = len(browser_mgr.dom_snapshots)
            test_case.metadata['screenshots'] = [s['path'] for s in browser_mgr.screenshots]
            test_case.metadata['bug_report'] = browser_mgr.bug_report  # Add bug report for Jira ticket creation

            # Verify transfer
            logger.info(f"✅ Verified: test_case.metadata now has {len(test_case.metadata.get('captured_locators', {}))} captured locators")

            # Analyze steps with AI (using captured context)
            success, enhanced_test_case, message = await self.analyze_steps_with_ai(
                test_case,
                use_robotmcp=use_robotmcp
            )

            if not success:
                logger.warning(f"⚠️ AI analysis failed: {message}")
                enhanced_test_case = test_case

            # Generate Robot Framework script
            success, script_content, file_path = self.generate_robot_script(
                enhanced_test_case,
                include_comments=True
            )

            if success:
                logger.info(f"✅ Script generated successfully: {file_path}")
                return True, script_content, file_path, bug_report
            else:
                return False, "", file_path, bug_report

        except Exception as e:
            logger.error(f"❌ Error in browser automation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, "", "", f"Error: {str(e)}"

        finally:
            if browser_mgr:
                browser_mgr.cleanup()

    async def _use_robotmcp_for_analysis(self, test_case: TestCase) -> Tuple[bool, TestCase, str]:
        """
        Use RobotMCP for advanced test analysis and keyword discovery

        Args:
            test_case: TestCase to analyze

        Returns:
            Tuple of (success, enhanced_test_case, message)
        """
        try:
            if not ROBOTMCP_AVAILABLE or not self.robotmcp_helper:
                return False, test_case, "RobotMCP not available"

            logger.info("🤖 Using RobotMCP for advanced analysis...")

            # Connect to RobotMCP MCP server
            if not self.robotmcp_helper.is_connected:
                connected = await self.robotmcp_helper.connect()
                if not connected:
                    return False, test_case, "Failed to connect to RobotMCP server"

            # Step 1: Analyze scenario to understand intent
            logger.info("📊 Analyzing test scenario with RobotMCP...")
            scenario_description = f"{test_case.name}: {test_case.description}"
            analysis_result = await self.robotmcp_helper.analyze_scenario(
                scenario=scenario_description,
                context="web"
            )

            if "error" in analysis_result:
                logger.error(f"Scenario analysis failed: {analysis_result['error']}")
                return False, test_case, f"Analysis error: {analysis_result['error']}"

            logger.info(f"✅ Scenario analyzed - Session ID: {self.robotmcp_helper.current_session_id}")

            # Step 2: Process each test step
            enhanced_steps = []
            for step in test_case.steps:
                logger.info(f"🔍 Processing step {step.step_number}: {step.description}")

                # Discover matching keywords for this action
                keywords = await self.robotmcp_helper.discover_keywords(
                    action_description=step.description,
                    context="web"
                )

                if keywords:
                    # Use the best matching keyword
                    best_keyword = keywords[0]
                    keyword_name = best_keyword.get("name", "")
                    keyword_library = best_keyword.get("library", "")

                    logger.info(f"✅ Found keyword: {keyword_library}.{keyword_name}")

                    # Update step with Robot Framework keyword
                    step.action = f"{keyword_library}.{keyword_name}"

                    # Try to extract arguments from step description
                    # This is a simple heuristic - could be improved with AI
                    args = []
                    if "click" in step.description.lower():
                        # Extract element identifier
                        words = step.description.split()
                        for i, word in enumerate(words):
                            if word.lower() in ["button", "link", "element"]:
                                if i + 1 < len(words):
                                    args.append(words[i + 1])
                                    break
                    elif "type" in step.description.lower() or "enter" in step.description.lower():
                        # Extract input field and value
                        if " in " in step.description or " into " in step.description:
                            parts = step.description.split(" in " if " in " in step.description else " into ")
                            if len(parts) >= 2:
                                args.append(parts[1].strip())  # field
                                args.append(parts[0].strip())  # value

                    # Execute step to validate (optional - only if we want validation)
                    # Commenting out for now as it requires actual browser interaction
                    # execution_result = await self.robotmcp_helper.execute_step(
                    #     keyword=keyword_name,
                    #     arguments=args,
                    #     use_context=True
                    # )

                else:
                    logger.warning(f"⚠️ No matching keyword found for: {step.description}")
                    step.action = "Log  " + step.description  # Fallback to logging

                enhanced_steps.append(step)

            # Update test case with enhanced steps
            test_case.steps = enhanced_steps

            # Step 3: Build Robot Framework test suite
            logger.info("🏗️ Building Robot Framework test suite...")
            suite_result = await self.robotmcp_helper.build_test_suite(
                test_name=test_case.name.replace(" ", "_"),
                tags=test_case.tags if hasattr(test_case, 'tags') else [],
                documentation=test_case.description
            )

            if "error" not in suite_result:
                logger.info("✅ Test suite built successfully")
                suite_path = suite_result.get("suite_file", "")
                logger.info(f"📄 Suite file: {suite_path}")
                return True, test_case, f"RobotMCP analysis complete - Suite: {suite_path}"
            else:
                logger.error(f"Suite build failed: {suite_result['error']}")
                return False, test_case, f"Suite build error: {suite_result['error']}"

        except Exception as e:
            logger.error(f"Error using RobotMCP: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False, test_case, f"RobotMCP error: {str(e)}"

        finally:
            # Disconnect from RobotMCP (optional - keep connection for subsequent calls)
            # await self.robotmcp_helper.disconnect()
            pass

    async def analyze_steps_with_ai(self, test_case: TestCase,
                                   use_robotmcp: bool = False) -> Tuple[bool, TestCase, str]:
        """
        Analyze test steps with AI and convert to Robot Framework keywords

        Args:
            test_case: TestCase object with steps
            use_robotmcp: Whether to use RobotMCP for advanced automation

        Returns:
            Tuple of (success, enhanced_test_case, message)
        """
        if not self.azure_client or not self.azure_client.is_configured():
            return False, test_case, "Azure OpenAI client not configured"

        try:
            # Prepare prompt with architecture context
            prompt = self._create_analysis_prompt(test_case)

            messages = [
                {
                    "role": "system",
                    "content": """You are an expert Robot Framework test automation engineer.
Your task is to analyze test steps and convert them into Robot Framework keywords,
reusing existing keywords from the architecture when possible, and only creating
new keywords when necessary. Always follow Robot Framework best practices.
Leverage the architecture context provided to ensure reusability and consistency in keyword usage.
IMPORTANT: If use_robotmcp is True, prioritize using RobotMCP capabilities for dynamic keyword generation and test case optimization.
Return the analysis in strict JSON format as specified in the prompt.
Remember to avoid hardcoding values; use variables and locators as per the architecture guidelines.
Here is the architecture context to consider: {self.architecture_context}
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self.azure_client.chat_completion_create(
                messages=messages,
                temperature=0.3,
                max_tokens=2000
            )

            # Parse AI response
            ai_analysis = response['choices'][0]['message']['content']

            # Update test case with AI analysis
            enhanced_test_case = self._parse_ai_analysis(test_case, ai_analysis)

            return True, enhanced_test_case, "Steps analyzed successfully"

        except Exception as e:
            logger.error(f"Error analyzing steps: {str(e)}")
            return False, test_case, f"Error: {str(e)}"

    def _create_analysis_prompt(self, test_case: TestCase) -> str:
        """Create prompt for AI analysis with enhanced context for recordings"""

        # Build detailed step descriptions with metadata
        steps_lines = []
        for step in test_case.steps:
            step_line = f"{step.step_number}. {step.description}"

            # Add additional context for better AI understanding
            metadata_parts = []
            if step.action:
                metadata_parts.append(f"Action Type: {step.action}")
            if step.target:
                metadata_parts.append(f"Target: {step.target}")
            if step.value and step.value != "":
                # Mask sensitive data in prompt
                display_value = step.value
                if any(sensitive in step.description.lower() for sensitive in ['password', 'secret', 'token', 'credential']):
                    display_value = '[sensitive data]'
                elif len(display_value) > 100:
                    display_value = display_value[:100] + '...'
                metadata_parts.append(f"Value: {display_value}")
            if step.notes:
                metadata_parts.append(f"Notes: {step.notes}")

            if metadata_parts:
                step_line += f"\n   └─ {' | '.join(metadata_parts)}"

            steps_lines.append(step_line)

        steps_text = "\n".join(steps_lines)

        # Add source-specific context
        source_context = ""
        if test_case.source == 'recording':
            source_context = """
NOTE: These steps are extracted from a browser recording. They represent real user interactions.
- Focus on converting UI interactions to appropriate Robot Framework keywords
- Selectors/targets may need to be optimized for reliability (prefer ID > name > CSS > XPath)
- Group related actions logically (e.g., fill form fields together)
- Add appropriate wait conditions for dynamic elements
- Consider adding validation steps after key actions
"""
        elif test_case.source == 'jira' or test_case.source == 'zephyr':
            source_context = """
NOTE: These steps are from a test management system (Jira/Zephyr).
- Steps may be high-level and need to be broken down
- Focus on test intent and expected outcomes
- Add appropriate setup and teardown steps
"""

        prompt = f"""You are an expert Robot Framework test automation engineer analyzing test steps for the Jarvis Test Automation framework.

Test Case: {test_case.title}
Description: {test_case.description}
Source: {test_case.source}
{source_context}

Test Steps:
{steps_text}

IMPORTANT CONTEXT - Existing Keyword Libraries:

UI Keywords (from tests/keywords/ui/ui_common/common.robot):
- Start Browser: Opens browser to specified URL category with browser type
  Usage: Start Browser    ${{url_category}}    ${{BROWSER}}
- Common Open Browser: Opens browser with various configurations
- Go To URL: Navigate to URL with browser configuration
- Click Element: Click on element using locator
- Input Text: Enter text into input field
- Input Password: Enter password into password field
- Wait Until Element Is Visible: Wait for element to be visible
- Element Should Be Visible: Assert element is visible
- Page Should Contain: Assert page contains text
- Select From List By Label: Select from dropdown by label
- Scroll Element Into View: Scroll to element
- Close Browser: Closes the browser

Common Test Patterns:
- Test Setup uses: Start Browser    ${{url_category}}    ${{BROWSER}}
- Test Teardown uses: Common Test Teardown
- Timeouts: ${{TEST_TIMEOUT_LONG}}, ${{TEST_TIMEOUT_MEDIUM}}, ${{TEST_TIMEOUT_SHORT}}
- Variables are loaded from: tests/variables/ and tests/configs/
- Locators should reference: ${{locator_name}} not hardcoded selectors

API Keywords (from tests/keywords/api/api_common/common.robot):
- Get Request Of Api With Headers And Params: Make GET request
- Post Request Of Api With Body: Make POST request with JSON body
- Put Request Of Api With Body: Make PUT request
- Delete Request Of Api: Make DELETE request
- Validate Json Response For An API: Validate response against JSON schema
- Validate Response Of The Api: Validate HTTP status code
- Validate Response Body Value: Validate specific JSON path value
- Store Response Value To Variable: Extract value from response JSON
- Common API Test Teardown: Common teardown for API tests

Architecture Principles:
1. NEVER use hardcoded URLs - use URL categories or variables
2. NEVER hardcode locators - reference variables like ${{button_locator}}
3. ALWAYS use existing keywords from common.robot
4. Test Setup/Teardown are REQUIRED for UI tests
5. Resource imports use relative paths: ../../../keywords/ui/ui_common/common.robot
6. Variables should be defined in *** Variables *** section
7. Use proper SeleniumLibrary keywords: Click Element, Input Text, etc.
8. For API tests, use ApiLibrary keywords with proper request/response handling

Task: Analyze each test step and map to existing keywords or suggest standard SeleniumLibrary keywords.

Return JSON format:
{{
    "test_type": "ui" or "api",
    "steps": [
        {{
            "step_number": 1,
            "action": "navigate|click|input|verify|wait|get|post|validate",
            "target": "element description",
            "keyword": "Exact keyword name from above",
            "arguments": ["arg1", "arg2"],
            "notes": "Why this keyword was chosen"
        }}
    ]
}}

Rules:
- Use existing keywords when possible
- For UI: Use Click Element, Input Text, Wait Until Element Is Visible, etc.
- For API: Use Get Request Of Api, Post Request Of Api, etc.
- Arguments should use variables like ${{variable_name}} not hardcoded values
- If unsure, use SeleniumLibrary standard keywords
- Ensure JSON is well-formed and parsable
Return the analysis now.
"""
        return prompt

    def _parse_ai_analysis(self, test_case: TestCase, ai_analysis: str) -> TestCase:
        """Parse AI analysis and enhance test case"""
        try:
            # Extract JSON from AI response
            json_start = ai_analysis.find('{')
            json_end = ai_analysis.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = ai_analysis[json_start:json_end]
                analysis_data = json.loads(json_str)

                # Store test type if provided
                if 'test_type' in analysis_data:
                    test_case.metadata['test_type'] = analysis_data['test_type']

                # Update steps with AI analysis
                for i, step_data in enumerate(analysis_data.get('steps', [])):
                    if i < len(test_case.steps):
                        step = test_case.steps[i]
                        step.action = step_data.get('action', step.action)
                        step.target = step_data.get('target', step.target)
                        step.keyword = step_data.get('keyword', '')
                        step.arguments = step_data.get('arguments', [])
                        step.notes = step_data.get('notes', step.notes)

                logger.info(f"Successfully parsed AI analysis for {len(analysis_data.get('steps', []))} steps")

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing AI JSON response: {str(e)}")
            logger.error(f"AI Response: {ai_analysis[:500]}")  # Log first 500 chars
        except Exception as e:
            logger.error(f"Error parsing AI analysis: {str(e)}")

        return test_case

    def generate_robot_script(self, test_case: TestCase,
                             include_comments: bool = True) -> Tuple[bool, str, str]:
        """
        Generate production-ready Robot Framework script following repo conventions

        Args:
            test_case: TestCase object with analyzed steps
            include_comments: Whether to include explanatory comments

        Returns:
            Tuple of (success, script_content, file_path)
        """
        try:
            # Determine if this is UI or API test
            is_ui_test = self._is_ui_test(test_case)

            # Determine brand from source or default to 'generated'
            brand = test_case.metadata.get('brand', 'generated')

            # Generate test suite file
            if is_ui_test:
                suite_content = self._generate_ui_suite(test_case, include_comments)
                suite_dir = os.path.join(ROOT_DIR, "tests", "testsuite", "ui", brand)
            else:
                suite_content = self._generate_api_suite(test_case, include_comments)
                suite_dir = os.path.join(ROOT_DIR, "tests", "testsuite", "api", brand)

            os.makedirs(suite_dir, exist_ok=True)

            # Generate keyword file
            if is_ui_test:
                keyword_content = self._generate_ui_keyword_file(test_case, include_comments)
                keyword_dir = os.path.join(ROOT_DIR, "tests", "keywords", "ui", brand)
            else:
                keyword_content = self._generate_api_keyword_file(test_case, include_comments)
                keyword_dir = os.path.join(ROOT_DIR, "tests", "keywords", "api", brand)

            os.makedirs(keyword_dir, exist_ok=True)

            # Generate locator file
            locator_content = self._generate_locator_file(test_case)
            locator_dir = os.path.join(ROOT_DIR, "tests", "locators", "ui" if is_ui_test else "api", brand)
            os.makedirs(locator_dir, exist_ok=True)

            # Generate variable file if needed
            variable_content = self._generate_variable_file(test_case)
            variable_dir = os.path.join(ROOT_DIR, "tests", "variables", "ui" if is_ui_test else "api", brand)
            os.makedirs(variable_dir, exist_ok=True)

            # Create filenames
            safe_name = test_case.title.replace(' ', '_').replace('-', '_').lower()
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save files
            suite_filename = f"{safe_name}.robot"
            suite_path = os.path.join(suite_dir, suite_filename)
            with open(suite_path, 'w') as f:
                f.write(suite_content)

            keyword_filename = f"{safe_name}.robot"
            keyword_path = os.path.join(keyword_dir, keyword_filename)
            with open(keyword_path, 'w') as f:
                f.write(keyword_content)

            locator_filename = f"{safe_name}.py"
            locator_path = os.path.join(locator_dir, locator_filename)
            with open(locator_path, 'w') as f:
                f.write(locator_content)

            variable_filename = f"{safe_name}.py"
            variable_path = os.path.join(variable_dir, variable_filename)
            with open(variable_path, 'w') as f:
                f.write(variable_content)

            # Create summary content with all file paths
            summary = f"""# TestPilot Generated Test Suite

Generated {len(test_case.steps)} files for: {test_case.title}

## Files Created:

1. Test Suite: {suite_path}
2. Keywords:    {keyword_path}
3. Locators:    {locator_path}
4. Variables:   {variable_path}

## Next Steps:

1. Review and update locators in: {locator_path}
2. Review and update variables in: {variable_path}
3. Implement keyword logic in: {keyword_path}
4. Run the test: robot {suite_path}

## Usage:

The test follows the standard repo pattern:
- Test Suite calls a Test Template keyword
- Keyword file contains the actual implementation
- Locators are in separate Python file
- Variables are in separate Python file

This matches the pattern used in tests like:
- tests/testsuite/ui/bhcom/email/professional_email_new_user_purchase_flow_upp.robot
- tests/testsuite/ui/dcom/pricing_check_flows/wordpress_hosting_plans.robot
"""

            logger.info(f"Generated complete test structure: {suite_path}")
            return True, summary, suite_path

        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, "", f"Error: {str(e)}"

    def _is_ui_test(self, test_case: TestCase) -> bool:
        """Determine if test case is UI or API based on steps"""
        ui_keywords = ['navigate', 'click', 'enter', 'select', 'verify', 'wait', 'browse', 'open', 'type', 'input']
        api_keywords = ['get', 'post', 'put', 'delete', 'api', 'request', 'response', 'endpoint']

        ui_score = 0
        api_score = 0

        for step in test_case.steps:
            step_lower = step.description.lower()
            for keyword in ui_keywords:
                if keyword in step_lower:
                    ui_score += 1
            for keyword in api_keywords:
                if keyword in step_lower:
                    api_score += 1

        return ui_score > api_score

    def _generate_ui_suite(self, test_case: TestCase, include_comments: bool) -> str:
        """Generate test suite file following repo pattern"""
        lines = []
        safe_name = test_case.title.replace(' ', '_').replace('-', '_')
        keyword_name = f"Test {test_case.title}"

        # Settings
        lines.append("*** Settings ***")
        lines.append(f"Documentation    {test_case.title}")
        if test_case.description:
            lines.append(f"...              {test_case.description}")
        lines.append("Test Timeout    ${ORDER_FULFILLMENT_TIMEOUT}")
        lines.append("Test Setup      Open Browser With Proxy    ${ui_base_url}")
        lines.append("Test Teardown   Common Test Teardown")

        # Tags - 4 spaces between each tag
        brand = test_case.metadata.get('brand', 'generated')
        tags = ['ui', brand, 'testpilot'] + test_case.tags
        lines.append(f"Force Tags      {'    '.join(tags)}")
        lines.append("")

        # Resource imports - 3 levels up, and import ui_common for Test Setup/Teardown
        lines.append(f"Resource        ../../../keywords/ui/{brand}/{safe_name.lower()}.robot")
        lines.append("")

        # Test Cases - clean, no junk comments
        lines.append("*** Test Cases ***")
        lines.append(f"Test Case 1 : {test_case.title}")
        lines.append(f"    [Documentation]  {test_case.description if test_case.description else test_case.title}")
        if test_case.tags:
            # Tags under test case also need 4 spaces between each
            lines.append(f"    [Tags]    {'    '.join(test_case.tags)}")
        # Keyword call must be indented 8 spaces from the left (4 base + 4 additional)
        lines.append(f"        {keyword_name}")

        return "\n".join(lines)

    def _generate_ui_keyword_file(self, test_case: TestCase, include_comments: bool) -> str:
        """Generate keyword file with actual test implementation"""
        lines = []
        safe_name = test_case.title.replace(' ', '_').replace('-', '_')
        keyword_name = f"Test {test_case.title}"
        brand = test_case.metadata.get('brand', 'generated')

        # Settings
        lines.append("*** Settings ***")
        lines.append(f"Documentation    Keywords for {test_case.title}")
        # From keywords/ui/generated/ to keywords/ui/ui_common/ is 2 levels up (../../ui_common)
        lines.append("Resource        ../../../keywords/ui/ui_common/common.robot")
        # From keywords/ui/generated/ to locators/ui/generated/ is 3 levels up
        lines.append(f"Variables       ../../../locators/ui/{brand}/{safe_name.lower()}.py")
        lines.append(f"Variables       ../../../variables/ui/{brand}/{safe_name.lower()}.py")
        lines.append("")

        # Keywords
        lines.append("*** Keywords ***")
        lines.append(keyword_name)
        lines.append(f"    [Documentation]  Main test keyword for {test_case.title}")
        lines.append("")

        # Add initial setup - 8 spaces from left (4 base + 4 additional)
        lines.append("        # Initialize test data")
        lines.append("        Create Generic Test Data")
        lines.append("")

        # Add steps with actual implementation - all keyword calls need 8 spaces
        for step in test_case.steps:
            if include_comments:
                lines.append(f"        # Step {step.step_number}: {step.description}")

            # Generate proper keyword calls using improved pattern matching
            keyword_calls = self._generate_proper_keyword_calls(step, test_case)
            for call in keyword_calls:
                lines.append(f"        {call}")
            lines.append("")

        return "\n".join(lines)

    def _generate_proper_keyword_calls(self, step: TestStep, test_case: TestCase) -> list:
        """Generate proper keyword calls following exact repo patterns"""
        calls = []
        description = step.description.lower()
        step_num = step.step_number

        # Pattern 1: First step - navigation (browser already open)
        if step_num == 1 and any(word in description for word in ['navigate', 'open', 'go to', 'visit', 'browse']):
            calls.append("# Browser is already opened by Test Setup with ${ui_base_url}")
            calls.append("Wait Until Page Is Ready")
            return calls

        # Pattern 2: Menu navigation with submenu (common: WordPress -> WordPress Cloud)
        if '->' in description or ('select' in description and 'menu' in description):
            parts = description.split('->')
            if len(parts) >= 2:
                menu_loc = self._infer_locator_name(parts[0].strip())
                submenu_loc = self._infer_locator_name(parts[1].strip())
                calls.append(f"Wait Until Page Contains Element And Mouse Over    ${{{menu_loc}}}")
                calls.append(f"Wait Until Page Contains Element And Click    ${{{submenu_loc}}}")
            else:
                loc = self._infer_locator_name(description)
                calls.append(f"Wait Until Page Contains Element And Mouse Over    ${{{loc}}}")
            return calls

        # Pattern 3: Click button/link/CTA
        if any(word in description for word in ['click', 'press']) and any(target in description for target in ['button', 'link', 'cta', 'explore', 'continue', 'proceed']):
            calls.append("Wait Until Page Is Ready")
            loc = self._infer_locator_name(description)
            calls.append(f"Wait Until Page Contains Element And Click    ${{{loc}}}")
            return calls

        # Pattern 4: Choose/Select plan/product/option
        if any(word in description for word in ['choose', 'select', 'pick']) and any(target in description for target in ['plan', 'product', 'option', 'package', 'cloud']):
            loc = self._infer_locator_name(description)
            calls.append(f"Wait Until Page Contains Element And Click    ${{{loc}}}")
            return calls

        # Pattern 5: Enter information/form fields
        if any(word in description for word in ['enter', 'fill', 'input', 'type', 'complete form']):
            if 'random' in description or 'contact' in description or 'information' in description:
                # Multiple fields - break down
                if 'contact' in description or 'information' in description:
                    calls.append(f"Input Into Text Field    ${{contact_name_field_locator}}    ${{random_name_variable}}")
                    calls.append(f"Input Into Text Field    ${{contact_email_field_locator}}    ${{random_email_variable}}")
                    calls.append(f"Input Into Text Field    ${{contact_phone_field_locator}}    ${{random_phone_variable}}")
                else:
                    loc = self._infer_locator_name(description)
                    var = self._infer_variable_name(description)
                    calls.append(f"Input Into Text Field    ${{{loc}}}    ${{{var}}}")
            elif 'payment' in description or 'billing' in description or 'card' in description:
                # Payment information - use existing keyword
                calls.append("Enter Billing Information    ${test_card_variable}")
            elif 'domain' in description:
                # Domain search pattern
                calls.append("Enter Domain Name With Different TLD And Search    ${tld_dot_com_variable}    ${domain_search_input_locator}    ${domain_search_button_locator}")
            else:
                loc = self._infer_locator_name(description)
                var = self._infer_variable_name(description)
                calls.append(f"Input Into Text Field    ${{{loc}}}    ${{{var}}}")
            return calls

        # Pattern 6: Submit/Checkout/Payment
        if any(word in description for word in ['submit', 'checkout', 'pay']):
            if 'payment' in description:
                calls.append("Wait Until Page Contains Element And Click    ${submit_payment_button_locator}")
            else:
                loc = self._infer_locator_name(description)
                calls.append(f"Wait Until Page Contains Element And Click    ${{{loc}}}")
            return calls

        # Pattern 7: Verify/Check result
        if any(word in description for word in ['verify', 'check', 'confirm', 'validate']):
            if 'order' in description or 'success' in description or 'confirmation' in description:
                calls.append("Wait Until Page Contains Element    ${order_confirmation_locator}    ${EXPLICIT_TIMEOUT}")
                calls.append("Get Order Number From URL In Order Receipt Page")
            elif 'url' in description or 'page' in description:
                calls.append("Wait Until Page Is Ready")
                calls.append("Location Should Contain    ${expected_url_part_variable}")
            else:
                loc = self._infer_locator_name(description)
                calls.append(f"Wait Until Page Contains Element    ${{{loc}}}    ${{EXPLICIT_TIMEOUT}}")
            return calls

        # Default fallback - basic click with wait
        calls.append("Wait Until Page Is Ready")
        loc = self._infer_locator_name(description)
        calls.append(f"Wait Until Page Contains Element And Click    ${{{loc}}}")

        return calls

    def _generate_locator_file(self, test_case: TestCase) -> str:
        """Generate Python locator file with intelligent values from website scraping"""
        lines = []
        lines.append("# Locators for " + test_case.title)
        lines.append("# Generated by TestPilot with intelligent web scraping")
        lines.append("#")
        lines.append("# INSTRUCTIONS:")
        lines.append("# - Locators marked 'AUTO-DETECTED' were found from the website")
        lines.append("# - Locators marked 'NEED_TO_UPDATE' should be manually updated")
        lines.append("# - Always verify auto-detected locators work correctly")
        lines.append("#")
        lines.append("# HOW TO UPDATE:")
        lines.append("# 1. Open website in browser")
        lines.append("# 2. Right-click element → Inspect")
        lines.append("# 3. Copy selector (ID is best, then XPath, then CSS)")
        lines.append("# 4. Replace the value below")
        lines.append("#")
        lines.append("# SELECTOR FORMAT: 'strategy:value'")
        lines.append("#   id:element_id           # Best - fastest and most reliable")
        lines.append("#   xpath://div[@id='x']    # OK - XPath expressions")
        lines.append("#   css:.class-name         # Good - CSS selectors")
        lines.append("#   link:Link Text          # Good - for links with exact text")
        lines.append("")
        lines.append("# " + "="*70)
        lines.append("# LOCATORS:")
        lines.append("# " + "="*70)
        lines.append("")

        # Check if we have captured locators from browser automation
        captured_locators = test_case.metadata.get('captured_locators', {})
        has_captured = len(captured_locators) > 0

        if has_captured:
            logger.info(f"✅ Using {len(captured_locators)} CAPTURED locators from browser automation")
            logger.info(f"   📋 Captured locator keys:")
            for name in list(captured_locators.keys())[:10]:  # Show first 10
                logger.info(f"      ✓ {name} = {captured_locators[name]}")
            if len(captured_locators) > 10:
                logger.info(f"      ... and {len(captured_locators) - 10} more")
        else:
            logger.warning(f"⚠️ NO captured locators found in test_case.metadata!")
            logger.warning(f"   test_case.metadata keys: {list(test_case.metadata.keys())}")

        # Extract and enrich locators
        basic_locators = self._extract_locators_from_steps(test_case.steps)
        enriched_locators = self._enrich_locators_with_web_data(test_case, basic_locators)

        logger.info(f"   📋 Requested locator names (from step descriptions):")
        for name in [loc[0] for loc in enriched_locators][:10]:  # Show first 10
            logger.info(f"      ? {name}")
            # Check if this name exists in captured locators
            if name in captured_locators:
                logger.info(f"        ✅ MATCH FOUND in captured locators!")
            else:
                logger.info(f"        ❌ NOT FOUND in captured locators")
                # Show similar keys
                similar = [k for k in captured_locators.keys() if name.replace('_locator', '') in k or k.replace('_locator', '') in name]
                if similar:
                    logger.info(f"        💡 Similar keys: {similar[:3]}")
        if len(enriched_locators) > 10:
            logger.info(f"      ... and {len(enriched_locators) - 10} more")

        for i, enriched in enumerate(enriched_locators, 1):
            if len(enriched) == 3:
                locator_name, description, actual_value = enriched
            else:
                locator_name, description = enriched
                actual_value = None

            lines.append(f"# Step {i}: {description}")

            # Priority 1: Use captured locator from browser automation (most reliable)
            if locator_name in captured_locators:
                captured_value = captured_locators[locator_name]
                lines.append(f"{locator_name} = '{captured_value}'  # ✅ CAPTURED during browser automation - VERIFIED WORKING")
                logger.info(f"   ✅ Using CAPTURED locator for {locator_name}: {captured_value}")
            # Priority 2: Use auto-detected value from web scraping
            elif actual_value:
                lines.append(f"{locator_name} = '{actual_value}'  # AUTO-DETECTED from website")
                lines.append(f"# ✓ Found automatically - please verify this works")
            # Priority 3: Placeholder that needs manual update
            else:
                lines.append(f"{locator_name} = 'NEED_TO_UPDATE'  # TODO: Update with actual selector")
                lines.append(f"# How to find: Right-click element → Inspect → Copy selector")
                lines.append(f"# Prefer: id:element-id (fastest)")

                # CRITICAL DEBUG: Why wasn't this found?
                if i <= 5:  # Only log first 5 to avoid spam
                    logger.warning(f"   ⚠️  Locator NOT FOUND: '{locator_name}'")
                    # Check if there's a similar key
                    similar_keys = [k for k in captured_locators.keys() if locator_name.replace('_locator', '') in k or k.replace('_locator', '') in locator_name]
                    if similar_keys:
                        logger.warning(f"       Possible matches in captured dict: {similar_keys[:3]}")
                    else:
                        logger.warning(f"       No similar keys found. First 5 captured keys: {list(captured_locators.keys())[:5]}")
            lines.append("")

        if not enriched_locators:
            lines.append("# No specific locators detected - add your locators here:")
            lines.append("")
            lines.append("# Common examples:")
            lines.append("# button_locator = 'id:submit-btn'")
            lines.append("# menu_locator = 'css:nav a[href*=\"wordpress\"]'")
            lines.append("# input_locator = 'name:search'")

        lines.append("")
        lines.append("# " + "="*70)
        lines.append("# COMMON PATTERNS:")
        lines.append("# " + "="*70)
        lines.append("# For WordPress menu: link:WordPress")
        lines.append("# For Get Started button: css:.btn-get-started")
        lines.append("# For domain search: name:domain or id:domain-search")

        if has_captured:
            lines.append("")
            lines.append("# " + "="*70)
            lines.append("# BROWSER AUTOMATION STATS:")
            lines.append("# " + "="*70)
            lines.append(f"# Total captured locators: {len(captured_locators)}")
            lines.append(f"# These locators were verified to work during live browser execution")

        return "\n".join(lines)

    def _generate_variable_file(self, test_case: TestCase) -> str:
        """Generate Python variable file following repo patterns"""
        lines = []
        lines.append("# Variables for " + test_case.title)
        lines.append("# Generated by TestPilot following repo standards")
        lines.append("#")
        lines.append("# NAMING CONVENTION (from repo):")
        lines.append("# - End with _variable: test_username_variable")
        lines.append("# - Use lowercase with underscores")
        lines.append("# - Be descriptive and consistent")
        lines.append("#")
        lines.append("# COMMON PATTERNS FROM REPO:")
        lines.append("# - Plans: basic_hosting_plan_variable = 'Basic'")
        lines.append("# - Prices: wordpress_hosting_basic_1_year_price_variable = '$45.00'")
        lines.append("# - Flags: with_new_domain_variable = 'with_new_domain'")
        lines.append("# - TLD: tld_dot_com_variable = '.com'")
        lines.append("")
        lines.append("# " + "="*70)
        lines.append("# TEST-SPECIFIC VARIABLES:")
        lines.append("# " + "="*70)
        lines.append("")

        # Get captured variables from browser automation
        captured_variables = test_case.metadata.get('captured_variables', {})
        if captured_variables:
            logger.info(f"✅ Using {len(captured_variables)} CAPTURED variables from browser automation")
            for var_name, var_value in captured_variables.items():
                lines.append(f"# Captured from browser automation")
                lines.append(f"{var_name} = '{var_value}'  # ✅ CAPTURED during execution")
                lines.append("")

        # Extract variables from steps
        variables = self._extract_variables_from_steps(test_case.steps)

        if variables:
            for var_name, var_desc, default_value in variables:
                # Skip if already captured
                if var_name not in captured_variables:
                    lines.append(f"# {var_desc}")
                    lines.append(f"{var_name} = '{default_value}'")
                    lines.append("")

        # Add common repo variables that are frequently used
        lines.append("# " + "="*70)
        lines.append("# COMMON REPO VARIABLES (include as needed):")
        lines.append("# " + "="*70)
        lines.append("")
        lines.append("# Domain variables")
        lines.append("tld_dot_com_variable = '.com'")
        lines.append("# with_new_domain_variable = 'with_new_domain'")
        lines.append("# without_domain_variable = 'without_domain'")
        lines.append("")
        lines.append("# Plan variables (uncomment and modify as needed)")
        lines.append("# basic_plan_variable = 'Basic'")
        lines.append("# plus_plan_variable = 'Plus'")
        lines.append("# premium_plan_variable = 'Premium'")
        lines.append("")
        lines.append("# Billing term variables")
        lines.append("# billing_term_12_variable = '12'")
        lines.append("# billing_term_24_variable = '24'")
        lines.append("# billing_term_36_variable = '36'")
        lines.append("")
        lines.append("# URL category variables")
        lines.append("# wordpress_url_category = '/wordpress'")
        lines.append("# hosting_url_category = '/hosting'")
        lines.append("")
        lines.append("# Expected text variables")
        lines.append("# expected_success_message_variable = 'Order Successful'")
        lines.append("# expected_page_title_variable = 'WordPress Hosting'")

        return "\n".join(lines)

    def _generate_real_keyword_calls(self, step: TestStep, test_case: TestCase) -> list:
        """Generate actual keyword calls matching repo patterns exactly"""
        calls = []
        description = step.description.lower()

        # If AI provided keyword, use it
        if step.keyword and step.arguments:
            args_str = "    ".join(step.arguments)
            calls.append(f"{step.keyword}    {args_str}".rstrip())
            return calls

        # Pattern 1: Navigate/Open - common first step
        if step.step_number == 1 and any(word in description for word in ['navigate', 'open', 'go to', 'visit']):
            # Don't add Create Generic Test Data here - it's added at keyword level
            calls.append("# Browser is already opened by Test Setup with ${ui_base_url}")
            calls.append("Wait Until Page Is Ready")
            return calls

        # Pattern 2: Navigate to section via hover menu
        if any(word in description for word in ['select', 'choose', 'navigate']) and any(menu in description for menu in ['menu', 'tab', 'section', 'dropdown', '->']):
            # Extract menu items from description
            locator_menu = self._infer_locator_name(step.description.split('->')[0] if '->' in step.description else step.description)
            if '->' in step.description or 'select' in description:
                # Two-level navigation
                locator_item = self._infer_locator_name(step.description.split('->')[-1] if '->' in step.description else step.description)
                calls.append(f"Wait Until Page Contains Element And Mouse Over    ${{{locator_menu}}}")
                calls.append(f"Select Hosting Type    ${{{locator_item}}}")
            else:
                calls.append(f"Wait Until Page Contains Element And Mouse Over    ${{{locator_menu}}}")
            return calls

        # Pattern 3: Click button/link
        if any(word in description for word in ['click', 'press']) and any(target in description for target in ['button', 'link', 'cta']):
            locator = self._infer_locator_name(step.description)
            calls.append(f"Wait Until Page Contains Element And Click    ${{{locator}}}")
            return calls

        # Pattern 4: Select plan/product
        if any(word in description for word in ['choose', 'select']) and any(target in description for target in ['plan', 'product', 'option', 'package']):
            locator = self._infer_locator_name(step.description)
            calls.append(f"Select Hosting Type    ${{{locator}}}")
            return calls

        # Pattern 5: Enter text/search
        if any(word in description for word in ['enter', 'type', 'input', 'search']):
            if 'domain' in description:
                # Domain search pattern - very common
                calls.append("Enter Domain Name With Different TLD And Search    ${tld_dot_com_variable}    ${domain_search_input_locator}    ${domain_search_continue_btn_locator}")
            else:
                locator = self._infer_locator_name(step.description)
                variable = self._infer_variable_name(step.description)
                calls.append(f"Input Into Text Field    ${{{locator}}}    ${{{variable}}}")
            return calls

        # Pattern 6: Verify/Check
        if any(word in description for word in ['verify', 'check', 'validate', 'confirm']):
            if 'url' in description or 'page' in description or 'redirect' in description:
                calls.append("Wait Until Page Is Ready")
                variable = self._infer_variable_name("expected url")
                calls.append(f"Location Should Contain    ${{{variable}}}")
            elif 'price' in description or 'cost' in description:
                # Price verification pattern
                variable = self._infer_variable_name("expected price")
                calls.append(f"# TODO: Add price verification logic")
                calls.append(f"# Verify The Price    ${{{variable}}}")
            else:
                locator = self._infer_locator_name(step.description)
                calls.append(f"Wait Until Page Contains Element    ${{{locator}}}    ${{EXPLICIT_TIMEOUT}}")
            return calls

        # Pattern 7: Proceed/Continue (common pattern)
        if any(word in description for word in ['proceed', 'continue', 'next']):
            locator = self._infer_locator_name("proceed button" if 'proceed' in description else step.description)
            calls.append(f"Wait Until Page Contains Element And Click    ${{{locator}}}")
            return calls

        # Pattern 8: Wait for loading
        if any(word in description for word in ['wait', 'loading', 'spinner']):
            calls.append("Wait Until The Loading Spinner Disappears")
            calls.append("Wait Until Page Is Ready")
            return calls

        # Default fallback - basic click pattern
        calls.append("Wait Until Page Is Ready")
        locator = self._infer_locator_name(step.description)
        calls.append(f"Wait Until Page Contains Element And Click    ${{{locator}}}")

        return calls

    def _infer_locator_name(self, description: str) -> str:
        """
        Infer locator variable name from step description

        CRITICAL: This method MUST generate the same name as _capture_element_locator
        for captured locators to be found during file generation!
        """
        # Use the SAME logic as _capture_element_locator to ensure names match
        clean_desc = ''.join(c if c.isalnum() or c == ' ' else '_' for c in description.lower())
        words = [w for w in clean_desc.split() if len(w) > 2][:5]  # Filter small words (<=2 chars), take first 5

        if not words:
            return "element_locator"

        # Build locator name - EXACT same format as _capture_element_locator
        locator_name = '_'.join(words) + '_locator'
        return locator_name

    def _infer_variable_name(self, description: str) -> str:
        """Infer variable name from step description"""
        if 'username' in description.lower():
            return "test_username_variable"
        elif 'password' in description.lower():
            return "test_password_variable"
        elif 'email' in description.lower():
            return "test_email_variable"
        elif 'domain' in description.lower():
            return "domain_name_variable"
        else:
            words = [w for w in description.lower().split() if w.isalnum()]
            return '_'.join(words[:2]) + "_variable"

    def _extract_locators_from_steps(self, steps: list) -> list:
        """Extract needed locators from steps"""
        locators = []
        seen = set()

        for step in steps:
            locator_name = self._infer_locator_name(step.description)
            if locator_name not in seen:
                locators.append((locator_name, step.description))
                seen.add(locator_name)

        return locators

    def _extract_variables_from_steps(self, steps: list) -> list:
        """Extract needed variables from steps"""
        variables = []
        seen = set()

        for step in steps:
            desc_lower = step.description.lower()

            if any(word in desc_lower for word in ['enter', 'input', 'type']):
                var_name = self._infer_variable_name(step.description)
                if var_name not in seen:
                    variables.append((var_name, step.description, "UPDATE_ME"))
                    seen.add(var_name)

            if any(word in desc_lower for word in ['verify', 'check', 'expect']):
                if 'text' in desc_lower or 'message' in desc_lower:
                    var_name = "expected_text_variable"
                    if var_name not in seen:
                        variables.append((var_name, "Expected text to verify", "Expected text here"))
                        seen.add(var_name)
                elif 'url' in desc_lower:
                    var_name = "expected_url_part"
                    if var_name not in seen:
                        variables.append((var_name, "Expected URL part", "/expected/path"))
                        seen.add(var_name)

        return variables

    def _generate_api_suite(self, test_case: TestCase, include_comments: bool) -> str:
        """Generate API test suite file"""
        lines = []
        safe_name = test_case.title.replace(' ', '_').replace('-', '_')
        keyword_name = f"Test {test_case.title}"
        brand = test_case.metadata.get('brand', 'generated')

        lines.append("*** Settings ***")
        lines.append(f"Documentation    {test_case.title}")
        if test_case.description:
            lines.append(f"...              {test_case.description}")
        lines.append("Test Timeout    ${TEST_TIMEOUT_MEDIUM}")
        lines.append(f"Test Template   {keyword_name}")

        # Tags - 4 spaces between each tag
        tags = ['api', brand, 'testpilot', 'generated'] + test_case.tags
        lines.append(f"Force Tags      {'    '.join(tags)}")
        lines.append("")

        lines.append(f"Resource        ../../../../keywords/api/{brand}/{safe_name.lower()}.robot")
        lines.append("")

        lines.append("*** Test Cases ***")
        lines.append(f"Test Case 1 : {test_case.title}")
        lines.append(f"    [Documentation]  {test_case.description if test_case.description else test_case.title}")
        if test_case.tags:
            # Tags under test case also need 4 spaces between each
            lines.append(f"    [Tags]    {'    '.join(test_case.tags)}")

        return "\n".join(lines)

    def _generate_api_keyword_file(self, test_case: TestCase, include_comments: bool) -> str:
        """Generate API keyword file"""
        lines = []
        safe_name = test_case.title.replace(' ', '_').replace('-', '_')
        keyword_name = f"Test {test_case.title}"
        brand = test_case.metadata.get('brand', 'generated')

        lines.append("*** Settings ***")
        lines.append(f"Documentation    API Keywords for {test_case.title}")
        lines.append("Resource        ../../../../keywords/api/api_common/common.robot")
        lines.append(f"Variables       ../../../../variables/api/{brand}/{safe_name.lower()}.py")
        lines.append("")

        lines.append("*** Keywords ***")
        lines.append(keyword_name)
        lines.append(f"    [Documentation]  API test for {test_case.title}")
        lines.append("    [Arguments]    ${arg1}=${EMPTY}")
        lines.append("")

        # All keyword calls need 8 spaces from left (4 base + 4 additional)
        for step in test_case.steps:
            if include_comments:
                lines.append(f"        # Step {step.step_number}: {step.description}")

            keyword_call = self._generate_api_keyword_call(step)
            lines.append(f"        {keyword_call}")
            lines.append("")

        return "\n".join(lines)

    def _generate_api_keyword_call(self, step: TestStep) -> str:
        """Generate API keyword calls"""
        description = step.description.lower()

        if step.keyword and step.arguments:
            args_str = "    ".join(step.arguments)
            return f"{step.keyword}    {args_str}".rstrip()

        if 'get' in description and 'request' in description:
            return "${response}=    Get Request Of Api With Headers And Params    ${PROTOCOL}    ${API_BASE_URL}    ${api_endpoint}    ${headers}    ${params}"
        elif 'post' in description and 'request' in description:
            return "${response}=    Post Request Of Api With Body    ${PROTOCOL}    ${API_BASE_URL}    ${api_endpoint}    ${request_body}    ${headers}"
        elif 'put' in description:
            return "${response}=    Put Request Of Api With Body    ${PROTOCOL}    ${API_BASE_URL}    ${api_endpoint}    ${request_body}    ${headers}"
        elif 'delete' in description:
            return "${response}=    Delete Request Of Api    ${PROTOCOL}    ${API_BASE_URL}    ${api_endpoint}    ${headers}"
        elif any(word in description for word in ['verify', 'validate', 'check']):
            if 'status' in description or 'code' in description:
                return "Validate Response Of The Api    ${response}    ${SUCCESS_CODE}"
            else:
                return "Validate Json Response For An API    ${response}    expected_response.json    ${expected_values}"
        else:
            return f"# TODO: Implement API call - {step.description}"



def show_ui():
    """Main UI for TestPilot module"""

    st.markdown("""
    <style>
    .test-pilot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .test-pilot-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .step-item {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Helper functions for template management
    def save_template(template_name: str, test_data: dict):
        """Save current test as a template"""
        template = {
            'name': template_name,
            'title': test_data.get('title', ''),
            'description': test_data.get('description', ''),
            'priority': test_data.get('priority', 'Medium'),
            'tags': test_data.get('tags', ''),
            'steps': test_data.get('steps', []),
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat()
        }
        st.session_state.test_pilot_saved_templates[template_name] = template

        # Also save to disk for persistence
        try:
            templates_dir = os.path.join(ROOT_DIR, 'generated_tests', 'templates')
            os.makedirs(templates_dir, exist_ok=True)
            template_file = os.path.join(templates_dir, f"{template_name.replace(' ', '_')}.json")
            with open(template_file, 'w') as f:
                json.dump(template, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save template to disk: {e}")

    def load_template(template_name: str) -> dict:
        """Load a saved template"""
        if template_name in st.session_state.test_pilot_saved_templates:
            template = st.session_state.test_pilot_saved_templates[template_name]
            template['last_used'] = datetime.now().isoformat()
            return template
        return None

    def load_templates_from_disk():
        """Load saved templates from disk on startup"""
        try:
            templates_dir = os.path.join(ROOT_DIR, 'generated_tests', 'templates')
            if os.path.exists(templates_dir):
                for filename in os.listdir(templates_dir):
                    if filename.endswith('.json'):
                        template_file = os.path.join(templates_dir, filename)
                        with open(template_file, 'r') as f:
                            template = json.load(f)
                            st.session_state.test_pilot_saved_templates[template['name']] = template
        except Exception as e:
            logger.warning(f"Could not load templates from disk: {e}")

    def get_step_suggestions(current_steps: list, azure_client=None) -> list:
        """Get AI-powered suggestions for next steps based on current steps"""
        if not current_steps or not azure_client:
            # Provide common default suggestions
            return [
                "Navigate to the application login page",
                "Enter valid username and password",
                "Click on the login button",
                "Verify successful login",
                "Verify error message is displayed"
            ]

        try:
            # Use AI to suggest next steps
            steps_text = "\n".join([f"{s.get('number', i+1)}. {s.get('description', '')}"
                                   for i, s in enumerate(current_steps)])

            prompt = f"""Given these test steps:
{steps_text}

Suggest 5 logical next steps that would complete or extend this test scenario.
Return only the step descriptions, one per line, without numbering."""

            messages = [
                {"role": "system", "content": "You are a QA automation expert. Suggest logical next test steps."},
                {"role": "user", "content": prompt}
            ]

            response = azure_client.chat_completion_create(
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )

            suggestions = response['choices'][0]['message']['content'].strip().split('\n')
            return [s.strip('- ').strip() for s in suggestions if s.strip()][:5]
        except Exception as e:
            logger.debug(f"Could not get AI suggestions: {e}")
            return []

    def add_to_step_history(step_description: str):
        """Add step to history for quick reuse"""
        if step_description and step_description.strip():
            # Remove if already exists to avoid duplicates
            if step_description in st.session_state.test_pilot_step_history:
                st.session_state.test_pilot_step_history.remove(step_description)
            # Add to beginning
            st.session_state.test_pilot_step_history.insert(0, step_description)
            # Keep only last 20
            st.session_state.test_pilot_step_history = st.session_state.test_pilot_step_history[:20]

    # Header
    st.markdown("""
    <div class="test-pilot-header">
        <h1>🚀 TestPilot</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            AI-Powered Intelligent Test Automation Assistant
        </p>
        <p style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
            Convert test cases into Robot Framework scripts with AI precision
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state FIRST before using it
    if 'test_pilot_test_case' not in st.session_state:
        st.session_state.test_pilot_test_case = None
    if 'test_pilot_steps' not in st.session_state:
        st.session_state.test_pilot_steps = []
    if 'test_pilot_jira_auth' not in st.session_state:
        st.session_state.test_pilot_jira_auth = None
    if 'test_pilot_saved_templates' not in st.session_state:
        st.session_state.test_pilot_saved_templates = {}
    if 'test_pilot_step_history' not in st.session_state:
        st.session_state.test_pilot_step_history = []

    # Load templates from disk on startup (AFTER session state is initialized)
    if not st.session_state.test_pilot_saved_templates:
        load_templates_from_disk()

    # Check dependencies
    if not AZURE_AVAILABLE:
        st.warning("⚠️ Azure OpenAI client not available. AI features will be limited.")

    # Main tabs for different input methods
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Manual Entry",
        "🎫 Jira/Zephyr",
        "📤 Upload Recording",
        "⏺️ Record & Playback",
        "📊 Generated Scripts"
    ])

    # Initialize Azure client
    azure_client = None
    if AZURE_AVAILABLE:
        azure_client = AzureOpenAIClient()

    engine = TestPilotEngine(azure_client)

    # Tab 1: Manual Entry
    with tab1:
        st.markdown("### 📝 Enter Test Steps Manually")
        st.markdown("Enter your test steps in natural language, one per line.")

        # Template Management Section
        with st.expander("📚 Template Library & Quick Actions", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 💾 Save/Load Templates")

                # Save current test as template
                save_template_name = st.text_input(
                    "Template Name",
                    placeholder="e.g., Login Flow, Checkout Process",
                    key="save_template_name"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("💾 Save as Template", use_container_width=True):
                        if save_template_name:
                            test_data = {
                                'title': st.session_state.get('manual_title', ''),
                                'description': st.session_state.get('manual_description', ''),
                                'priority': st.session_state.get('manual_priority', 'Medium'),
                                'tags': st.session_state.get('manual_tags', ''),
                                'steps': st.session_state.test_pilot_steps.copy()
                            }
                            save_template(save_template_name, test_data)
                            st.success(f"✅ Template '{save_template_name}' saved!")
                            st.rerun()
                        else:
                            st.error("Please enter a template name")

                with col_b:
                    if st.button("📤 Export to JSON", use_container_width=True):
                        if st.session_state.test_pilot_steps:
                            export_data = {
                                'title': st.session_state.get('manual_title', ''),
                                'description': st.session_state.get('manual_description', ''),
                                'priority': st.session_state.get('manual_priority', 'Medium'),
                                'tags': st.session_state.get('manual_tags', ''),
                                'steps': st.session_state.test_pilot_steps,
                                'exported_at': datetime.now().isoformat()
                            }
                            st.download_button(
                                "⬇️ Download JSON",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"test_case_{int(time.time())}.json",
                                mime="application/json"
                            )

                # Load template
                if st.session_state.test_pilot_saved_templates:
                    st.markdown("#### 📂 Load Saved Template")
                    template_options = list(st.session_state.test_pilot_saved_templates.keys())
                    selected_template = st.selectbox(
                        "Select Template",
                        [""] + template_options,
                        key="selected_template"
                    )

                    col_c, col_d = st.columns(2)
                    with col_c:
                        if st.button("📥 Load Template", use_container_width=True):
                            if selected_template:
                                template = load_template(selected_template)
                                if template:
                                    # Load into session state
                                    st.session_state.manual_title = template.get('title', '')
                                    st.session_state.manual_description = template.get('description', '')
                                    st.session_state.manual_priority = template.get('priority', 'Medium')
                                    st.session_state.manual_tags = template.get('tags', '')
                                    st.session_state.test_pilot_steps = template.get('steps', []).copy()
                                    st.success(f"✅ Loaded template '{selected_template}'")
                                    st.rerun()

                    with col_d:
                        if st.button("🗑️ Delete Template", use_container_width=True):
                            if selected_template:
                                del st.session_state.test_pilot_saved_templates[selected_template]
                                # Also delete from disk
                                try:
                                    template_file = os.path.join(ROOT_DIR, 'generated_tests', 'templates',
                                                                f"{selected_template.replace(' ', '_')}.json")
                                    if os.path.exists(template_file):
                                        os.remove(template_file)
                                except:
                                    pass
                                st.success(f"🗑️ Deleted template '{selected_template}'")
                                st.rerun()
                else:
                    st.info("💡 No saved templates yet. Save your first template above!")

                # Import from JSON
                uploaded_file = st.file_uploader("📥 Import from JSON", type=['json'], key="import_json")
                if uploaded_file:
                    try:
                        import_data = json.load(uploaded_file)
                        st.session_state.manual_title = import_data.get('title', '')
                        st.session_state.manual_description = import_data.get('description', '')
                        st.session_state.manual_priority = import_data.get('priority', 'Medium')
                        st.session_state.manual_tags = import_data.get('tags', '')
                        st.session_state.test_pilot_steps = import_data.get('steps', [])
                        st.success("✅ Test case imported successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error importing: {e}")

            with col2:
                st.markdown("#### ⚡ Quick Actions")

                col_e, col_f = st.columns(2)
                with col_e:
                    if st.button("🔄 Reverse Order", use_container_width=True):
                        if st.session_state.test_pilot_steps:
                            st.session_state.test_pilot_steps.reverse()
                            # Renumber
                            for i, step in enumerate(st.session_state.test_pilot_steps):
                                step['number'] = i + 1
                            st.rerun()

                    if st.button("📋 Duplicate All", use_container_width=True):
                        if st.session_state.test_pilot_steps:
                            duplicated = st.session_state.test_pilot_steps.copy()
                            for step in duplicated:
                                new_step = step.copy()
                                new_step['number'] = len(st.session_state.test_pilot_steps) + 1
                                st.session_state.test_pilot_steps.append(new_step)
                            st.rerun()

                with col_f:
                    if st.button("🔢 Auto-number", use_container_width=True):
                        for i, step in enumerate(st.session_state.test_pilot_steps):
                            step['number'] = i + 1
                        st.success("✅ Steps renumbered")
                        st.rerun()

                    if st.button("🧹 Clear All", use_container_width=True, type="secondary"):
                        st.session_state.test_pilot_steps = []
                        st.rerun()

                # Step History (Recently Used)
                if st.session_state.test_pilot_step_history:
                    st.markdown("#### 📜 Recently Used Steps")
                    st.markdown("Click to reuse a recent step:")

                    for i, hist_step in enumerate(st.session_state.test_pilot_step_history[:5]):
                        if st.button(f"➕ {hist_step[:60]}...", key=f"hist_{i}", use_container_width=True):
                            st.session_state.test_pilot_steps.append({
                                'number': len(st.session_state.test_pilot_steps) + 1,
                                'description': hist_step
                            })
                            st.rerun()

                # AI Step Suggestions
                if AZURE_AVAILABLE and azure_client and st.session_state.test_pilot_steps:
                    st.markdown("#### 💡 AI Suggestions")
                    if st.button("🤖 Get Next Step Suggestions", use_container_width=True):
                        with st.spinner("Generating suggestions..."):
                            suggestions = get_step_suggestions(st.session_state.test_pilot_steps, azure_client)
                            if suggestions:
                                st.session_state.ai_suggestions = suggestions
                                st.rerun()

                    if hasattr(st.session_state, 'ai_suggestions') and st.session_state.ai_suggestions:
                        st.markdown("**Suggested Next Steps:**")
                        for i, suggestion in enumerate(st.session_state.ai_suggestions[:5]):
                            if st.button(f"➕ {suggestion[:60]}...", key=f"sug_{i}", use_container_width=True):
                                st.session_state.test_pilot_steps.append({
                                    'number': len(st.session_state.test_pilot_steps) + 1,
                                    'description': suggestion
                                })
                                add_to_step_history(suggestion)
                                st.rerun()

        st.markdown("---")

        # Test Case Details
        col1, col2 = st.columns([2, 1])

        with col1:
            test_title = st.text_input("Test Case Title", key="manual_title")
            test_description = st.text_area("Test Case Description",
                                           height=100, key="manual_description")

        with col2:
            test_priority = st.selectbox("Priority",
                                        ["Low", "Medium", "High", "Critical"],
                                        key="manual_priority")
            test_tags = st.text_input("Tags (comma-separated)", key="manual_tags")

        # Steps entry
        st.markdown("#### Test Steps")

        # Add step button
        col_add1, col_add2, col_add3 = st.columns([2, 2, 1])
        with col_add1:
            if st.button("➕ Add Step", key="add_step_manual", use_container_width=True):
                st.session_state.test_pilot_steps.append({
                    'number': len(st.session_state.test_pilot_steps) + 1,
                    'description': ''
                })
                st.rerun()

        with col_add2:
            if st.button("➕ Add 5 Empty Steps", key="add_5_steps", use_container_width=True):
                for _ in range(5):
                    st.session_state.test_pilot_steps.append({
                        'number': len(st.session_state.test_pilot_steps) + 1,
                        'description': ''
                    })
                st.rerun()

        with col_add3:
            step_count = len(st.session_state.test_pilot_steps)
            st.metric("Steps", step_count)

        # Display and edit steps
        steps_to_delete = []
        steps_to_duplicate = []
        steps_to_move_up = []
        steps_to_move_down = []

        for i, step in enumerate(st.session_state.test_pilot_steps):
            col1, col2 = st.columns([4, 1])
            with col1:
                step_desc = st.text_area(
                    f"Step {step['number']}",
                    value=step.get('description', ''),
                    height=80,
                    key=f"manual_step_{i}",
                    help="Enter step description in natural language"
                )
                # Update the step description in session state directly
                st.session_state.test_pilot_steps[i]['description'] = step_desc

                # Add to history when step is filled
                if step_desc and step_desc.strip() and step_desc != step.get('original_desc', ''):
                    add_to_step_history(step_desc)
                    st.session_state.test_pilot_steps[i]['original_desc'] = step_desc

            with col2:
                st.write("")  # Add spacing

                # Action buttons in a grid
                btn_col1, btn_col2 = st.columns(2)

                with btn_col1:
                    if st.button("⬆️", key=f"move_up_{i}", help="Move up", use_container_width=True):
                        if i > 0:
                            steps_to_move_up.append(i)

                    if st.button("📋", key=f"duplicate_{i}", help="Duplicate", use_container_width=True):
                        steps_to_duplicate.append(i)

                with btn_col2:
                    if st.button("⬇️", key=f"move_down_{i}", help="Move down", use_container_width=True):
                        if i < len(st.session_state.test_pilot_steps) - 1:
                            steps_to_move_down.append(i)

                    if st.button("🗑️", key=f"delete_step_{i}", help="Delete", use_container_width=True):
                        steps_to_delete.append(i)

        # Process actions
        if steps_to_move_up:
            for idx in steps_to_move_up:
                if idx > 0:
                    st.session_state.test_pilot_steps[idx], st.session_state.test_pilot_steps[idx-1] = \
                        st.session_state.test_pilot_steps[idx-1], st.session_state.test_pilot_steps[idx]
            # Renumber
            for i, step in enumerate(st.session_state.test_pilot_steps):
                step['number'] = i + 1
            st.rerun()

        if steps_to_move_down:
            for idx in steps_to_move_down:
                if idx < len(st.session_state.test_pilot_steps) - 1:
                    st.session_state.test_pilot_steps[idx], st.session_state.test_pilot_steps[idx+1] = \
                        st.session_state.test_pilot_steps[idx+1], st.session_state.test_pilot_steps[idx]
            # Renumber
            for i, step in enumerate(st.session_state.test_pilot_steps):
                step['number'] = i + 1
            st.rerun()

        if steps_to_duplicate:
            for idx in steps_to_duplicate:
                duplicated_step = st.session_state.test_pilot_steps[idx].copy()
                duplicated_step['number'] = len(st.session_state.test_pilot_steps) + 1
                st.session_state.test_pilot_steps.append(duplicated_step)
            st.rerun()

        # Remove deleted steps
        if steps_to_delete:
            for idx in reversed(steps_to_delete):
                st.session_state.test_pilot_steps.pop(idx)
            # Renumber steps
            for i, step in enumerate(st.session_state.test_pilot_steps):
                step['number'] = i + 1
            st.rerun()

        # Generate button
        st.markdown("---")
        st.markdown("### 🚀 Generation Options")

        col1, col2, col3 = st.columns(3)
        with col1:
            use_browser_automation = st.checkbox(
                "🌐 Use Browser Automation",
                value=True,
                help="Execute steps in live browser to capture real locators, network logs, and generate bug reports",
                key="use_browser_automation_manual"
            )

        with col2:
            if use_browser_automation:
                # Auto-extract URL from step 1 if available
                auto_extracted_url = ""
                if st.session_state.test_pilot_steps:
                    first_step_desc = st.session_state.test_pilot_steps[0].get('description', '')
                    # Extract URL using regex
                    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                    url_match = re.search(url_pattern, first_step_desc)
                    if url_match:
                        auto_extracted_url = url_match.group(0)

                base_url = st.text_input(
                    "Base URL",
                    value=auto_extracted_url,
                    placeholder="https://www.example.com",
                    key="base_url_manual",
                    help="Starting URL for browser automation (auto-extracted from Step 1 if available)"
                )

        with col3:
            if use_browser_automation:
                headless_mode = st.checkbox(
                    "Headless Mode",
                    value=False,
                    help="Run browser in background (no UI)",
                    key="headless_manual"
                )

        # Environment selection (new row for better visibility)
        if use_browser_automation:
            st.markdown("#### 🌍 Environment Configuration")
            col_env1, col_env2 = st.columns([2, 3])

            with col_env1:
                environment_options = EnvironmentConfig.get_available_environments()
                environment_display = [EnvironmentConfig.format_environment_display(env) for env in environment_options]

                selected_env_idx = st.selectbox(
                    "Target Environment",
                    range(len(environment_options)),
                    format_func=lambda i: environment_display[i],
                    index=0,  # Default to 'prod'
                    key="environment_selection_manual",
                    help="Select the test environment. Non-prod environments require proxy configuration."
                )

                selected_environment = environment_options[selected_env_idx]
                env_config = EnvironmentConfig.get_config(selected_environment)

            with col_env2:
                # Show environment details based on mode
                if env_config['mode'] == 'proxy':
                    st.info(f"""
                    **Selected:** {env_config['name']}  
                    **Mode:** 🔒 Proxy Mode  
                    **Proxy:** {env_config['proxy']}  
                    **User Agent:** Contains `aem_env={selected_environment}` tag
                    """)
                elif env_config['mode'] == 'user_agent':
                    st.info(f"""
                    **Selected:** {env_config['name']}  
                    **Mode:** 🏷️ User Agent Mode  
                    **Proxy:** None (direct access)  
                    **User Agent:** Contains `jarvis_env={selected_environment}` and `aem_env={selected_environment}` tags
                    
                    ℹ️ Environment routing via user agent, not proxy
                    """)
                else:
                    st.info(f"""
                    **Selected:** {env_config['name']}  
                    **Mode:** 🌐 Direct Access  
                    **Proxy:** None  
                    **User Agent:** Standard Chrome UA
                    """)
        else:
            selected_environment = 'prod'  # Default for non-browser tests

        # Generate button
        if st.button("🤖 Analyze & Generate Script", key="manual_generate", type="primary"):
            if not test_title:
                st.error("Please provide a test case title")
            elif not st.session_state.test_pilot_steps:
                st.error("Please add at least one test step")
            elif use_browser_automation and not base_url:
                st.error("Please provide a Base URL for browser automation")
            else:
                # Check if steps have content
                steps_with_content = [s for s in st.session_state.test_pilot_steps if s.get('description', '').strip()]
                if not steps_with_content:
                    st.error("Please add descriptions to your test steps")
                else:
                    spinner_text = "🌐 Executing browser automation and analyzing..." if use_browser_automation else "Analyzing test steps with AI..."
                    with st.spinner(spinner_text):
                        try:
                            # Create test case
                            test_case = TestCase(
                                id=f"MANUAL-{int(time.time())}",
                                title=test_title,
                                description=test_description,
                                priority=test_priority,
                                tags=[tag.strip() for tag in test_tags.split(',') if tag.strip()],
                                source='manual'
                            )

                            # Add steps
                            for step_data in st.session_state.test_pilot_steps:
                                if step_data.get('description', '').strip():
                                    test_case.steps.append(TestStep(
                                        step_number=step_data['number'],
                                        description=step_data['description']
                                    ))

                            # Choose generation path: browser automation or standard
                            if use_browser_automation:
                                # Use enhanced browser automation
                                st.info(f"🌐 Starting browser automation on {env_config['name']} environment...")

                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)

                                success, script_content, file_path, bug_report = loop.run_until_complete(
                                    engine.analyze_and_generate_with_browser_automation(
                                        test_case,
                                        base_url,
                                        headless=headless_mode,
                                        environment=selected_environment,
                                        use_robotmcp=False
                                    )
                                )
                                loop.close()

                                if success:
                                    st.markdown('<div class="success-message">', unsafe_allow_html=True)
                                    st.success(f"✅ Script generated with browser automation!")
                                    st.markdown('</div>', unsafe_allow_html=True)

                                    # Show bug report
                                    with st.expander("🐛 Bug Analysis Report", expanded=True):
                                        st.markdown(bug_report)

                                        # Add Jira ticket creation section
                                        st.markdown("---")
                                        st.markdown("### 🎫 Create Jira Tickets")

                                        # Check if Jira is authenticated
                                        if st.session_state.get('test_pilot_jira_auth'):
                                            col1, col2 = st.columns([3, 1])

                                            with col1:
                                                jira_project = st.text_input(
                                                    "Jira Project Key",
                                                    value="QA",
                                                    placeholder="e.g., QA, TEST, PROJ",
                                                    help="Enter the Jira project key where bugs should be created",
                                                    key="bug_jira_project"
                                                )

                                            # Get all bugs from the report
                                            all_bugs = []
                                            # Get bug report data from test_case metadata which is populated during browser automation
                                            bug_report_data = test_case.metadata.get('bug_report', {})

                                            # Collect validation issues
                                            for bug in bug_report_data.get('validation_issues', []):
                                                all_bugs.append({
                                                    'summary': f"Validation Issue: {bug.get('field_name', 'Unknown field')} - {bug.get('type', 'Unknown')}",
                                                    'type': bug.get('type', 'validation_issue'),
                                                    'severity': bug.get('severity', 'medium'),
                                                    'description': bug.get('description', ''),
                                                    'field_name': bug.get('field_name', ''),
                                                    'field_type': bug.get('field_type', ''),
                                                    'step': bug.get('step', ''),
                                                    'recommendation': bug.get('recommendation', ''),
                                                    'is_payment_field': bug.get('is_payment_field', False)
                                                })

                                            # Collect accessibility issues
                                            for bug in bug_report_data.get('accessibility_issues', []):
                                                all_bugs.append({
                                                    'summary': f"Accessibility Issue: {bug.get('type', 'Unknown')} - {bug.get('wcag_criterion', '')}",
                                                    'type': bug.get('type', 'accessibility_issue'),
                                                    'severity': bug.get('severity', 'medium'),
                                                    'description': bug.get('description', ''),
                                                    'element': bug.get('element', ''),
                                                    'step': bug.get('step', ''),
                                                    'wcag_criterion': bug.get('wcag_criterion', ''),
                                                    'recommendation': bug.get('recommendation', '')
                                                })

                                            # Collect security issues
                                            for bug in bug_report_data.get('security_issues', []):
                                                all_bugs.append({
                                                    'summary': f"Security Issue: {bug.get('type', 'Unknown')}",
                                                    'type': bug.get('type', 'security_issue'),
                                                    'severity': bug.get('severity', 'high'),
                                                    'description': bug.get('description', ''),
                                                    'step': bug.get('step', ''),
                                                    'url': bug.get('url', ''),
                                                    'recommendation': bug.get('recommendation', '')
                                                })

                                            # Collect functionality issues
                                            for bug in bug_report_data.get('functionality_issues', []):
                                                all_bugs.append({
                                                    'summary': f"Functionality Issue: {bug.get('description', 'Unknown')[:80]}",
                                                    'type': 'functionality_issue',
                                                    'severity': bug.get('severity', 'high'),
                                                    'description': bug.get('description', ''),
                                                    'error': bug.get('error', ''),
                                                    'step': bug.get('step', ''),
                                                    'recommendation': 'Please investigate and fix this functionality issue'
                                                })

                                            if all_bugs:
                                                st.info(f"Found {len(all_bugs)} bugs that can be created as Jira tickets")

                                                # Display bugs with create ticket buttons
                                                for idx, bug in enumerate(all_bugs):
                                                    with st.container():
                                                        col_bug, col_btn = st.columns([4, 1])

                                                        with col_bug:
                                                            severity_emoji = {
                                                                'critical': '🔴',
                                                                'high': '🟠',
                                                                'medium': '🟡',
                                                                'low': '🟢'
                                                            }.get(bug['severity'], '⚪')

                                                            st.markdown(f"{severity_emoji} **{bug['summary']}**")
                                                            st.caption(f"Severity: {bug['severity'].upper()} | Type: {bug['type']}")

                                                        with col_btn:
                                                            if st.button("Create Ticket", key=f"create_jira_{idx}", type="secondary"):
                                                                if jira_project:
                                                                    jira_integration = st.session_state.test_pilot_jira_auth
                                                                    success, ticket_key, msg = jira_integration.create_bug_ticket(
                                                                        bug, jira_project
                                                                    )

                                                                    if success:
                                                                        st.success(f"✅ Created ticket: {ticket_key}")
                                                                        st.markdown(f"[View Ticket]({jira_integration.base_url}/browse/{ticket_key})")
                                                                    else:
                                                                        st.error(f"❌ Failed: {msg}")
                                                                else:
                                                                    st.warning("Please enter a Jira project key")

                                                # Bulk create option
                                                st.markdown("---")
                                                if st.button("🎫 Create All Tickets", key="create_all_jira", type="primary"):
                                                    if jira_project:
                                                        jira_integration = st.session_state.test_pilot_jira_auth
                                                        created_tickets = []
                                                        failed_tickets = []

                                                        progress_bar = st.progress(0)
                                                        status_text = st.empty()

                                                        for idx, bug in enumerate(all_bugs):
                                                            status_text.text(f"Creating ticket {idx + 1} of {len(all_bugs)}...")
                                                            success, ticket_key, msg = jira_integration.create_bug_ticket(
                                                                bug, jira_project
                                                            )

                                                            if success:
                                                                created_tickets.append(ticket_key)
                                                            else:
                                                                failed_tickets.append(bug['summary'])

                                                            progress_bar.progress((idx + 1) / len(all_bugs))

                                                        status_text.empty()
                                                        progress_bar.empty()

                                                        if created_tickets:
                                                            st.success(f"✅ Created {len(created_tickets)} Jira tickets")
                                                            st.markdown("**Created Tickets:**")
                                                            for ticket in created_tickets:
                                                                st.markdown(f"- [{ticket}]({jira_integration.base_url}/browse/{ticket})")

                                                        if failed_tickets:
                                                            st.warning(f"⚠️ Failed to create {len(failed_tickets)} tickets")
                                                    else:
                                                        st.warning("Please enter a Jira project key")
                                            else:
                                                st.info("No bugs found in this report")
                                        else:
                                            st.info("🔐 Please authenticate with Jira in the 'From Jira/Zephyr' tab to create tickets")

                                    # Show captured data summary
                                    if test_case.metadata.get('captured_locators'):
                                        with st.expander("📍 Captured Locators", expanded=False):
                                            st.json(test_case.metadata['captured_locators'])

                                    if test_case.metadata.get('screenshots'):
                                        with st.expander("📸 Screenshots", expanded=False):
                                            for screenshot in test_case.metadata['screenshots'][:5]:
                                                if os.path.exists(screenshot):
                                                    st.image(screenshot, caption=os.path.basename(screenshot), use_container_width=True)

                                    # Show preview
                                    with st.expander("📜 Preview Generated Script", expanded=True):
                                        st.code(script_content, language='robotframework')

                                    # Download button
                                    st.download_button(
                                        label="⬇️ Download Script",
                                        data=script_content,
                                        file_name=os.path.basename(file_path),
                                        mime="text/plain",
                                        key="download_manual_script_browser"
                                    )

                                    st.info(f"📁 Script saved to: {file_path}")

                                    # Notification
                                    if NOTIFICATIONS_AVAILABLE:
                                        notifications.add_notification(
                                            module_name="test_pilot",
                                            status="success",
                                            message=f"Generated script with browser automation: {test_title}",
                                            details=f"Script: {file_path}\nBug report available"
                                        )
                                else:
                                    st.error(f"❌ Browser automation failed: {file_path}")

                            else:
                                # Standard generation path (without browser automation)
                                # Analyze with AI if available
                                if AZURE_AVAILABLE and azure_client and azure_client.is_configured():
                                    try:
                                        # Use asyncio properly for Streamlit
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        success, enhanced_test_case, message = loop.run_until_complete(
                                            engine.analyze_steps_with_ai(test_case)
                                        )
                                        loop.close()

                                        if success:
                                            test_case = enhanced_test_case
                                            st.info(f"✅ AI Analysis: {message}")
                                        else:
                                            st.warning(f"AI Analysis failed: {message}. Generating basic script...")
                                    except Exception as e:
                                        st.warning(f"AI Analysis error: {str(e)}. Generating basic script...")
                                else:
                                    st.info("ℹ️ Generating script without AI analysis (Azure OpenAI not configured)")

                                # Generate script
                                st.session_state.test_pilot_test_case = test_case
                                success, script_content, file_path = engine.generate_robot_script(test_case)

                                # Show results for standard generation path only
                                if success:
                                    st.markdown('<div class="success-message">', unsafe_allow_html=True)
                                    st.success(f"✅ Script generated successfully!")
                                    st.markdown('</div>', unsafe_allow_html=True)

                                    # Show preview
                                    with st.expander("📜 Preview Generated Script", expanded=True):
                                        st.code(script_content, language='robotframework')

                                    # Download button
                                    st.download_button(
                                        label="⬇️ Download Script",
                                        data=script_content,
                                        file_name=os.path.basename(file_path),
                                        mime="text/plain",
                                        key="download_manual_script_standard"
                                    )

                                    st.info(f"📁 Script saved to: {file_path}")

                                    # Notification
                                    if NOTIFICATIONS_AVAILABLE:
                                        notifications.add_notification(
                                            module_name="test_pilot",
                                            status="success",
                                            message=f"Generated script for: {test_title}",
                                            details=f"Script saved to: {file_path}"
                                        )
                                else:
                                    st.error(f"❌ Failed to generate script: {file_path}")

                        except Exception as e:
                            logger.error(f"Error in script generation: {str(e)}")
                            st.error(f"❌ Error generating script: {str(e)}")
                            st.exception(e)

    # Tab 2: Jira/Zephyr Integration
    with tab2:
        st.markdown("### Fetch Test Cases from Jira/Zephyr")

        # Authentication section
        with st.expander("🔐 Jira Authentication", expanded=not st.session_state.test_pilot_jira_auth):
            col1, col2 = st.columns(2)

            with col1:
                jira_host = st.text_input("Jira Host",
                                         value="https://jira.newfold.com",
                                         placeholder="https://jira.newfold.com",
                                         key="jira_host")
                jira_username = st.text_input("Username/Email", key="jira_username")

            with col2:
                auth_type = st.selectbox("Authentication Type",
                                        ["API Token", "Password"],
                                        key="jira_auth_type")
                jira_token = st.text_input("API Token/Password",
                                          type="password", key="jira_token")

            if st.button("🔑 Authenticate", key="jira_auth_button"):
                if not jira_host or not jira_username or not jira_token:
                    st.error("Please fill in all authentication fields")
                else:
                    with st.spinner("Authenticating..."):
                        integration = JiraZephyrIntegration()
                        credential_type = "token" if auth_type == "API Token" else "password"

                        success, message = integration.authenticate(
                            jira_host, jira_username, jira_token, credential_type
                        )

                        if success:
                            st.session_state.test_pilot_jira_auth = integration
                            st.success(message)
                        else:
                            st.error(message)

        # Fetch test case
        if st.session_state.test_pilot_jira_auth:
            st.markdown("#### Fetch Test Case")

            col1, col2 = st.columns([3, 1])

            with col1:
                issue_key = st.text_input("Issue Key",
                                         placeholder="PROJ-123",
                                         key="jira_issue_key")

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                fetch_button = st.button("🔍 Fetch", key="jira_fetch_button", type="primary")

            if fetch_button:
                if not issue_key:
                    st.error("Please enter an issue key")
                else:
                    with st.spinner(f"Fetching {issue_key}..."):
                        integration = st.session_state.test_pilot_jira_auth
                        success, test_case, message = integration.fetch_zephyr_test_case(issue_key)

                        if success:
                            st.session_state.test_pilot_test_case = test_case
                            st.session_state.test_pilot_test_source = 'jira'
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

        # Display fetched test case (outside the button callback)
        if st.session_state.get('test_pilot_test_case') and st.session_state.get('test_pilot_test_source') == 'jira':
            test_case = st.session_state.test_pilot_test_case

            st.markdown('<div class="test-pilot-card">', unsafe_allow_html=True)
            st.markdown(f"### 📋 {test_case.title}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**ID:** {test_case.id}")
            with col2:
                st.markdown(f"**Priority:** {test_case.priority}")
            with col3:
                st.markdown(f"**Source:** {test_case.source.upper()}")

            if test_case.tags:
                st.markdown(f"**Tags:** {', '.join(test_case.tags)}")

            if test_case.description:
                with st.expander("📝 Description", expanded=False):
                    st.write(test_case.description)

            if test_case.preconditions:
                with st.expander("⚙️ Preconditions", expanded=False):
                    st.write(test_case.preconditions)

            st.markdown("### 📋 Test Steps:")
            for step in test_case.steps:
                with st.container():
                    st.markdown(f'<div class="step-item">', unsafe_allow_html=True)
                    st.markdown(f"**Step {step.step_number}:** {step.description}")

                    if step.value:
                        st.markdown(f"*📊 Test Data:* `{step.value}`")
                    if step.notes:
                        st.markdown(f"*✅ Expected Result:* {step.notes}")
                    if step.action:
                        st.markdown(f"*📝 Actual Result:* {step.action}")

                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                if st.button("🤖 Analyze & Generate Script", key="jira_generate_button", type="primary", use_container_width=True):
                    st.session_state.test_pilot_generate_triggered = True
                    st.rerun()

            with col2:
                if st.button("✏️ Edit Steps", key="jira_edit_button", use_container_width=True):
                    st.session_state.test_pilot_editing = True
                    st.rerun()

            with col3:
                if st.button("🗑️ Clear", key="jira_clear_button", use_container_width=True):
                    st.session_state.test_pilot_test_case = None
                    st.session_state.test_pilot_test_source = None
                    st.rerun()

        # Handle generate script action
        if st.session_state.get('test_pilot_generate_triggered') and st.session_state.get('test_pilot_test_case'):
            test_case = st.session_state.test_pilot_test_case

            with st.spinner("🔄 Analyzing test steps with AI and generating Robot Framework script..."):
                try:
                    # Analyze with AI if available
                    if AZURE_AVAILABLE and azure_client and azure_client.is_configured():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            success, enhanced_test_case, msg = loop.run_until_complete(
                                engine.analyze_steps_with_ai(test_case)
                            )
                            loop.close()

                            if success:
                                test_case = enhanced_test_case
                                st.session_state.test_pilot_test_case = enhanced_test_case
                                st.info(f"✅ AI Analysis: {msg}")
                            else:
                                st.warning(f"⚠️ AI Analysis: {msg}. Generating script with default patterns...")
                        except Exception as e:
                            logger.error(f"AI Analysis error: {str(e)}")
                            st.warning(f"⚠️ AI Analysis error: {str(e)}. Generating script with default patterns...")
                    else:
                        st.info("ℹ️ Generating script without AI analysis (Azure OpenAI not configured)")

                    # Generate script
                    success, script_content, file_path = engine.generate_robot_script(test_case, include_comments=True)

                    if success:
                        st.success("✅ Script generated successfully!")

                        # Store in session state
                        st.session_state.test_pilot_generated_script = script_content
                        st.session_state.test_pilot_script_path = file_path

                        st.markdown("### 📜 Generated Script Preview")

                        # Show file info
                        st.info(f"""
**📁 Files Generated:**
- Test Suite: `{file_path}`
- Keywords: Check keywords directory
- Locators: Check locators directory
- Variables: Check variables directory

**Next Steps:**
1. Review the generated scripts below
2. Update locators with actual element selectors
3. Update variables with test data
4. Run: `robot {file_path}`
                        """)

                        with st.expander("📄 Test Suite File", expanded=True):
                            st.code(script_content, language='robotframework')

                        # Download button
                        st.download_button(
                            label="⬇️ Download Test Suite",
                            data=script_content,
                            file_name=os.path.basename(file_path),
                            mime="text/plain",
                            key="download_jira_script"
                        )

                        if NOTIFICATIONS_AVAILABLE:
                            notifications.add_notification(
                                module_name="test_pilot",
                                message=f"Generated script from {test_case.id}: {test_case.title}",
                                level="success"
                            )
                    else:
                        st.error(f"❌ Failed to generate script: {file_path}")

                except Exception as e:
                    logger.error(f"Error generating script: {str(e)}")
                    import traceback
                    st.error(f"❌ Error: {str(e)}")
                    st.code(traceback.format_exc())

                # Clear the trigger
                st.session_state.test_pilot_generate_triggered = False

    # Tab 3: Upload Recording
    with tab3:
        st.markdown("### Upload Recording JSON")
        st.markdown("Upload a recording file from browser automation tools")

        uploaded_file = st.file_uploader("Choose recording file",
                                        type=['json'],
                                        key="recording_upload")

        if uploaded_file:
            try:
                recording_data = json.load(uploaded_file)

                st.success("✅ Recording file loaded successfully")

                # Show recording metadata if available
                recording_info = []
                if 'title' in recording_data:
                    recording_info.append(f"**Title:** {recording_data['title']}")
                if 'description' in recording_data:
                    recording_info.append(f"**Description:** {recording_data['description']}")
                if 'startUrl' in recording_data or 'url' in recording_data:
                    url = recording_data.get('startUrl', recording_data.get('url'))
                    recording_info.append(f"**URL:** {url}")

                # Detect recording format
                format_detected = "Unknown"
                if 'events' in recording_data:
                    format_detected = "Events-based (Puppeteer/Playwright)"
                elif 'actions' in recording_data:
                    format_detected = "Actions-based (Chrome Recorder)"
                elif 'steps' in recording_data:
                    format_detected = "Steps-based (Custom)"

                recording_info.append(f"**Format Detected:** {format_detected}")

                if recording_info:
                    st.markdown("**Recording Information:**")
                    for info in recording_info:
                        st.markdown(f"- {info}")

                st.markdown("---")

                # Parse recording with enhanced parser
                with st.spinner("🔍 Analyzing recording and extracting actionable steps..."):
                    steps = RecordingParser.parse_recording(recording_data)

                if not steps:
                    st.warning("⚠️ No actionable steps found in recording. Please check the recording format.")
                    st.info("""
                    **Expected recording format:**
                    ```json
                    {
                        "title": "Test name",
                        "description": "Test description",
                        "startUrl": "https://example.com",
                        "events": [
                            {
                                "type": "click|input|navigate",
                                "selector": "CSS or XPath selector",
                                "value": "input value (if applicable)",
                                "text": "element text",
                                "timestamp": "ISO timestamp"
                            }
                        ]
                    }
                    ```
                    """)
                    st.stop()

                # Create test case
                test_case = TestCase(
                    id=f"RECORDING-{int(time.time())}",
                    title=recording_data.get('title', 'Recorded Test'),
                    description=recording_data.get('description', 'Test generated from recording'),
                    source='recording',
                    steps=steps
                )

                st.session_state.test_pilot_test_case = test_case

                # Display steps with enhanced information
                st.markdown("#### Recorded Steps")
                st.info(f"📊 Parsed {len(steps)} actionable steps from recording")

                for step in steps:
                    st.markdown(f'<div class="step-item">', unsafe_allow_html=True)

                    # Main description with action badge
                    action_emoji = {
                        'navigate': '🌐', 'click': '👆', 'input': '⌨️', 'select': '📋',
                        'verify': '✓', 'wait': '⏱️', 'hover': '👋', 'scroll': '📜',
                        'upload': '📤', 'screenshot': '📸'
                    }.get(step.action, '▶️')

                    st.markdown(f"**Step {step.step_number}:** {action_emoji} {step.description}")

                    # Show action type
                    if step.action:
                        st.markdown(f"*Action Type:* `{step.action}`")

                    # Show target/selector
                    if step.target:
                        st.markdown(f"*Target Selector:* `{step.target}`")

                    # Show value (mask sensitive data)
                    if step.value:
                        display_value = step.value
                        if any(sensitive in step.description.lower() for sensitive in ['password', 'secret', 'token']):
                            display_value = '[sensitive data masked]'
                        st.markdown(f"*Value:* `{display_value}`")

                    # Show metadata notes
                    if step.notes:
                        with st.expander(f"📝 Additional Context for Step {step.step_number}"):
                            st.text(step.notes)

                    st.markdown('</div>', unsafe_allow_html=True)

                # Generate button
                if st.button("🤖 Generate Script", key="recording_generate", type="primary"):
                    with st.spinner("Generating script..."):
                        try:
                            # Analyze with AI if available
                            if AZURE_AVAILABLE and azure_client and azure_client.is_configured():
                                try:
                                    st.info("🤖 Analyzing steps with Azure OpenAI...")

                                    # Show what will be analyzed
                                    with st.expander("📋 Steps being analyzed", expanded=False):
                                        for step in test_case.steps:
                                            st.text(f"Step {step.step_number}: {step.action} - {step.description}")

                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    success, enhanced_test_case, msg = loop.run_until_complete(
                                        engine.analyze_steps_with_ai(test_case)
                                    )
                                    loop.close()

                                    if success:
                                        test_case = enhanced_test_case
                                        st.success(f"✅ AI Analysis Complete: {msg}")

                                        # Show AI-enhanced keywords
                                        keywords_found = [step.keyword for step in test_case.steps if step.keyword]
                                        if keywords_found:
                                            st.info(f"🔑 AI identified {len(keywords_found)} Robot Framework keywords")
                                            with st.expander("View Mapped Keywords"):
                                                for i, step in enumerate(test_case.steps):
                                                    if step.keyword:
                                                        st.markdown(f"**Step {step.step_number}:** `{step.keyword}`")
                                                        if step.arguments:
                                                            st.markdown(f"  └─ Arguments: {', '.join(step.arguments)}")
                                    else:
                                        st.warning(f"⚠️ AI Analysis had issues: {msg}. Using original steps...")
                                except Exception as e:
                                    logger.error(f"AI Analysis error: {str(e)}")
                                    st.warning(f"⚠️ AI Analysis error: {str(e)}. Using original steps...")
                            else:
                                st.info("ℹ️ Generating script without AI analysis (Azure OpenAI not configured)")

                            # Generate script
                            st.info("📝 Generating Robot Framework script...")
                            success, script_content, file_path = engine.generate_robot_script(test_case)

                            if success:
                                st.success(f"✅ Script generated successfully!")

                                with st.expander("📜 Preview", expanded=True):
                                    st.code(script_content, language='robotframework')

                                st.download_button(
                                    label="⬇️ Download Script",
                                    data=script_content,
                                    file_name=os.path.basename(file_path),
                                    mime="text/plain",
                                    key="download_recording_script"
                                )

                                st.info(f"📁 Script saved to: {file_path}")

                                if NOTIFICATIONS_AVAILABLE:
                                    notifications.add_notification(
                                        module_name="test_pilot",
                                        status="success",
                                        message=f"Generated script from recording",
                                        details=f"Script saved to: {file_path}"
                                    )
                            else:
                                st.error(f"❌ Failed to generate script: {file_path}")
                        except Exception as e:
                            logger.error(f"Error generating recording script: {str(e)}")
                            st.error(f"❌ Error: {str(e)}")
                            st.exception(e)

            except Exception as e:
                st.error(f"Error processing recording file: {str(e)}")

    # Tab 4: Record & Playback
    with tab4:
        st.markdown("### ⏺️ Record & Playback")
        st.markdown("Record your browser actions in real-time and automatically generate test scripts with AI analysis")

        # Initialize session state for recording
        if 'test_pilot_recording' not in st.session_state:
            st.session_state.test_pilot_recording = False
        if 'test_pilot_recorded_actions' not in st.session_state:
            st.session_state.test_pilot_recorded_actions = []
        if 'test_pilot_recording_thread' not in st.session_state:
            st.session_state.test_pilot_recording_thread = None
        if 'test_pilot_start_url' not in st.session_state:
            st.session_state.test_pilot_start_url = ""
        if 'test_pilot_recording_metadata' not in st.session_state:
            st.session_state.test_pilot_recording_metadata = {}

        # Recording configuration
        st.markdown("#### 🎬 Recording Configuration")

        col1, col2 = st.columns([3, 1])

        with col1:
            initial_url = st.text_input(
                "Initial URL (Optional - will auto-detect from first page)",
                placeholder="Leave empty to auto-detect, or enter: https://www.example.com",
                help="Starting URL will be automatically captured when browser opens. You can also specify one here.",
                key="record_start_url",
                value=st.session_state.test_pilot_start_url
            )

        with col2:
            browser_choice = st.selectbox(
                "Browser",
                ["Chrome", "Firefox"],
                key="record_browser",
                help="Browser to use for recording"
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            headless = st.checkbox(
                "Headless Mode",
                value=False,
                help="Run browser in background (no UI)",
                key="record_headless"
            )

        with col2:
            capture_screenshots = st.checkbox(
                "Capture Screenshots",
                value=True,
                help="Take screenshots at key steps",
                key="record_screenshots"
            )

        with col3:
            smart_wait = st.checkbox(
                "Smart Wait Detection",
                value=True,
                help="Automatically detect wait conditions",
                key="record_smart_wait"
            )

        st.markdown("---")

        # Recording controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if not st.session_state.test_pilot_recording:
                if st.button("🔴 Start Recording", type="primary", use_container_width=True, key="start_record_btn"):
                    try:
                        from selenium import webdriver
                        from selenium.webdriver.common.by import By
                        from selenium.webdriver.support.events import EventFiringWebDriver, AbstractEventListener
                        from selenium.webdriver.chrome.options import Options as ChromeOptions
                        from selenium.webdriver.firefox.options import Options as FirefoxOptions
                        from selenium.webdriver.support.ui import WebDriverWait
                        from selenium.webdriver.support import expected_conditions as EC

                        # Create smart event listener for capturing actions
                        class SmartRecordingListener(AbstractEventListener):
                            """Intelligent event listener that captures user actions with rich context"""

                            def __init__(self):
                                self.actions = []
                                self.start_url = None
                                self.last_url = None
                                self.page_load_times = []

                            def before_navigate_to(self, url, driver):
                                """Capture navigation before it happens"""
                                try:
                                    # Skip about:blank as it's not a real navigation
                                    if url == 'about:blank':
                                        return

                                    # Capture first real URL as start URL
                                    if self.start_url is None:
                                        self.start_url = url
                                        st.session_state.test_pilot_start_url = url
                                        logger.info(f"🌐 Starting URL captured: {url}")

                                    self.actions.append({
                                        'type': 'navigate',
                                        'action': 'navigation',
                                        'value': url,
                                        'url': url,
                                        'timestamp': datetime.now().isoformat(),
                                        'description': f'Navigate to {url}'
                                    })
                                except Exception as e:
                                    # Silently handle - browser might be closing
                                    logger.debug(f"Error in before_navigate_to: {e}")

                            def after_navigate_to(self, url, driver):
                                """Capture page state after navigation and re-inject JS recorder"""
                                try:
                                    # Skip about:blank
                                    if url == 'about:blank':
                                        return

                                    self.last_url = url
                                    title = driver.title

                                    # Update last action with page title
                                    if self.actions and self.actions[-1]['type'] == 'navigate':
                                        self.actions[-1]['title'] = title
                                        self.actions[-1]['description'] = f'Navigate to {title} ({url})'

                                    # Re-inject JavaScript recorder after navigation
                                    if hasattr(st.session_state, 'test_pilot_js_recorder'):
                                        try:
                                            time.sleep(0.5)  # Brief wait for page to stabilize
                                            driver.execute_script(st.session_state.test_pilot_js_recorder)
                                            logger.info(f"✅ JS recorder re-injected after navigation to {url}")
                                        except Exception as e:
                                            logger.debug(f"Could not re-inject JS recorder: {e}")
                                except Exception as e:
                                    # Silently handle - browser might be closing
                                    logger.debug(f"Error in after_navigate_to: {e}")

                            def before_click(self, element, driver):
                                """Capture click action with element context"""
                                try:
                                    # Get element details
                                    tag_name = element.tag_name
                                    element_text = element.text.strip() if element.text else ""
                                    element_id = element.get_attribute('id')
                                    element_name = element.get_attribute('name')
                                    element_class = element.get_attribute('class')
                                    element_type = element.get_attribute('type')
                                    aria_label = element.get_attribute('aria-label')
                                    placeholder = element.get_attribute('placeholder')

                                    # Build selector preference: id > name > class > xpath
                                    selector = None
                                    if element_id:
                                        selector = f"id:{element_id}"
                                    elif element_name and tag_name in ['input', 'select', 'textarea', 'button']:
                                        selector = f"name:{element_name}"
                                    elif element_class:
                                        classes = element_class.split()
                                        if classes:
                                            selector = f"css:.{classes[0]}"

                                    # Build description
                                    description = "Click on "
                                    if element_text:
                                        description += f"'{element_text}' {tag_name}"
                                    elif aria_label:
                                        description += f"'{aria_label}' {tag_name}"
                                    elif placeholder:
                                        description += f"{placeholder} field"
                                    elif element_id:
                                        description += f"{tag_name} with id '{element_id}'"
                                    else:
                                        description += f"{tag_name} element"

                                    self.actions.append({
                                        'type': 'click',
                                        'action': 'click',
                                        'selector': selector,
                                        'target': selector,
                                        'text': element_text,
                                        'innerText': element_text,
                                        'tagName': tag_name,
                                        'elementType': tag_name,
                                        'attributes': {
                                            'id': element_id,
                                            'name': element_name,
                                            'class': element_class,
                                            'type': element_type,
                                            'aria-label': aria_label,
                                            'placeholder': placeholder
                                        },
                                        'url': self.last_url or driver.current_url,
                                        'timestamp': datetime.now().isoformat(),
                                        'description': description
                                    })
                                except Exception as e:
                                    logger.error(f"Error capturing click: {e}")

                            def after_change_value_of(self, element, driver, value=None):
                                """Capture input/change actions"""
                                try:
                                    tag_name = element.tag_name
                                    element_text = element.text.strip() if element.text else ""
                                    element_id = element.get_attribute('id')
                                    element_name = element.get_attribute('name')
                                    element_class = element.get_attribute('class')
                                    element_type = element.get_attribute('type')
                                    aria_label = element.get_attribute('aria-label')
                                    placeholder = element.get_attribute('placeholder')
                                    current_value = element.get_attribute('value')

                                    # Build selector
                                    selector = None
                                    if element_id:
                                        selector = f"id:{element_id}"
                                    elif element_name:
                                        selector = f"name:{element_name}"
                                    elif element_class:
                                        classes = element_class.split()
                                        if classes:
                                            selector = f"css:.{classes[0]}"

                                    # Build description
                                    field_name = aria_label or placeholder or element_name or element_id or f"{tag_name} field"

                                    # Determine action type
                                    action_type = 'input' if tag_name in ['input', 'textarea'] else 'select'

                                    # Mask sensitive data
                                    display_value = current_value
                                    is_sensitive = element_type in ['password'] or any(
                                        sensitive in field_name.lower()
                                        for sensitive in ['password', 'secret', 'token', 'ssn', 'credit', 'cvv']
                                    )

                                    if is_sensitive:
                                        description = f"Enter [sensitive data] into {field_name} field"
                                    else:
                                        description = f"Enter '{current_value}' into {field_name} field"

                                    self.actions.append({
                                        'type': action_type,
                                        'action': action_type,
                                        'selector': selector,
                                        'target': selector,
                                        'value': current_value if not is_sensitive else '[MASKED]',
                                        'text': field_name,
                                        'tagName': tag_name,
                                        'elementType': tag_name,
                                        'attributes': {
                                            'id': element_id,
                                            'name': element_name,
                                            'class': element_class,
                                            'type': element_type,
                                            'aria-label': aria_label,
                                            'placeholder': placeholder
                                        },
                                        'url': self.last_url or driver.current_url,
                                        'timestamp': datetime.now().isoformat(),
                                        'description': description,
                                        'is_sensitive': is_sensitive
                                    })
                                except Exception as e:
                                    logger.error(f"Error capturing input: {e}")

                            def on_exception(self, exception, driver):
                                """Capture exceptions during recording"""
                                # Check if this is a browser closure exception
                                exception_str = str(exception).lower()
                                if any(msg in exception_str for msg in [
                                    'target window already closed',
                                    'web view not found',
                                    'no such window',
                                    'session deleted',
                                    'chrome not reachable',
                                    'browser has been closed'
                                ]):
                                    # Browser was closed - this is expected, don't log as error
                                    logger.debug(f"Browser closed: {exception}")
                                else:
                                    # Unexpected exception - log it
                                    logger.warning(f"Recording exception: {exception}")

                        # Setup driver based on browser choice
                        logger.info(f"🎬 Starting recording with {browser_choice} browser (visible mode)...")

                        if browser_choice == "Chrome":
                            options = ChromeOptions()
                            # Always visible mode for recording (user needs to interact)
                            # Open in incognito and maximized
                            options.add_argument('--incognito')
                            options.add_argument('--start-maximized')
                            options.add_argument('--no-sandbox')
                            options.add_argument('--disable-dev-shm-usage')
                            options.add_argument('--disable-gpu')
                            options.add_argument('--disable-software-rasterizer')
                            options.add_argument('--disable-blink-features=AutomationControlled')
                            options.add_argument('--disable-browser-side-navigation')
                            options.add_argument('--disable-features=TranslateUI,BlinkGenPropertyTrees')
                            options.add_argument('--remote-debugging-port=0')
                            options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
                            options.add_experimental_option('useAutomationExtension', False)

                            # Initialize with Service for better error handling
                            try:
                                from selenium.webdriver.chrome.service import Service
                                service = Service()
                                base_driver = webdriver.Chrome(service=service, options=options)
                                logger.info("   ✅ Chrome started with Service")
                            except Exception as service_error:
                                logger.warning(f"   ⚠️ Service init failed: {service_error}, using fallback")
                                base_driver = webdriver.Chrome(options=options)

                            # Set timeouts
                            base_driver.set_page_load_timeout(60)
                            base_driver.set_script_timeout(30)
                            base_driver.implicitly_wait(10)

                        elif browser_choice == "Firefox":
                            options = FirefoxOptions()
                            # Always visible mode for recording
                            # Open in private browsing
                            options.add_argument('-private')
                            base_driver = webdriver.Firefox(options=options)
                            # Set timeouts
                            base_driver.set_page_load_timeout(60)
                            base_driver.set_script_timeout(30)
                            base_driver.implicitly_wait(10)
                        else:
                            st.error(f"Browser {browser_choice} not supported")
                            base_driver = None

                        if base_driver:
                            # Ensure browser window is maximized for better visibility
                            try:
                                base_driver.maximize_window()
                            except Exception:
                                try:
                                    base_driver.set_window_size(1920, 1080)
                                except Exception:
                                    pass

                            # Create event listener
                            listener = SmartRecordingListener()

                            # Wrap driver with event listener
                            driver = EventFiringWebDriver(base_driver, listener)

                            # Inject JavaScript to capture user actions
                            # This is more reliable than EventFiringWebDriver for user interactions
                            js_recorder = """
                            // Initialize action storage
                            window._testPilotActions = window._testPilotActions || [];
                            window._testPilotInitialized = true;

                            console.log('[TestPilot] Action recording initialized');

                            // Helper to get best selector for element
                            function getElementSelector(element) {
                                if (element.id) return 'id:' + element.id;
                                if (element.name) return 'name:' + element.name;
                                if (element.className) {
                                    var classes = element.className.split(' ').filter(c => c.trim());
                                    if (classes.length > 0) return 'css:.' + classes[0];
                                }
                                return null;
                            }

                            // Capture clicks with capture phase to catch everything
                            document.addEventListener('click', function(e) {
                                var element = e.target;
                                console.log('[TestPilot] Click captured on:', element.tagName);

                                var elementInfo = {
                                    type: 'click',
                                    tagName: element.tagName.toLowerCase(),
                                    id: element.id || null,
                                    name: element.name || null,
                                    className: element.className || null,
                                    text: element.textContent ? element.textContent.trim().substring(0, 100) : null,
                                    placeholder: element.placeholder || null,
                                    ariaLabel: element.getAttribute('aria-label') || null,
                                    type: element.type || null,
                                    href: element.href || null,
                                    timestamp: new Date().toISOString(),
                                    url: window.location.href
                                };
                                window._testPilotActions.push(elementInfo);
                                console.log('[TestPilot] Total actions:', window._testPilotActions.length);
                            }, true);

                            // Capture input changes
                            document.addEventListener('input', function(e) {
                                var element = e.target;
                                if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
                                    console.log('[TestPilot] Input captured on:', element.name || element.id);

                                    var isPassword = element.type === 'password';
                                    var elementInfo = {
                                        type: 'input',
                                        tagName: element.tagName.toLowerCase(),
                                        id: element.id || null,
                                        name: element.name || null,
                                        className: element.className || null,
                                        placeholder: element.placeholder || null,
                                        ariaLabel: element.getAttribute('aria-label') || null,
                                        inputType: element.type || null,
                                        value: isPassword ? '[MASKED]' : element.value,
                                        timestamp: new Date().toISOString(),
                                        url: window.location.href
                                    };
                                    window._testPilotActions.push(elementInfo);
                                    console.log('[TestPilot] Total actions:', window._testPilotActions.length);
                                }
                            }, true);

                            // Capture select changes
                            document.addEventListener('change', function(e) {
                                var element = e.target;
                                if (element.tagName === 'SELECT') {
                                    console.log('[TestPilot] Select captured on:', element.name || element.id);

                                    var elementInfo = {
                                        type: 'select',
                                        tagName: element.tagName.toLowerCase(),
                                        id: element.id || null,
                                        name: element.name || null,
                                        className: element.className || null,
                                        value: element.value,
                                        selectedText: element.options[element.selectedIndex] ? element.options[element.selectedIndex].text : null,
                                        timestamp: new Date().toISOString(),
                                        url: window.location.href
                                    };
                                    window._testPilotActions.push(elementInfo);
                                    console.log('[TestPilot] Total actions:', window._testPilotActions.length);
                                }
                            }, true);

                            // Log initialization success
                            console.log('[TestPilot] Recording active. Actions:', window._testPilotActions.length);
                            """

                            # Store driver and recording info
                            st.session_state.test_pilot_recording = True
                            st.session_state.test_pilot_recorder_driver = driver
                            st.session_state.test_pilot_recorder_listener = listener
                            st.session_state.test_pilot_recorded_actions = []
                            st.session_state.test_pilot_recording_start_time = datetime.now()
                            st.session_state.test_pilot_js_recorder = js_recorder  # Store JS for re-injection
                            st.session_state.test_pilot_live_action_count = 0  # Initialize action counter
                            st.session_state.test_pilot_recording_metadata = {
                                'browser': browser_choice,
                                'headless': False,  # Always False for Record & Playback (user needs visible browser)
                                'capture_screenshots': capture_screenshots,
                                'smart_wait': smart_wait,
                                'start_time': datetime.now().isoformat()
                            }

                            # Navigate to initial URL if provided
                            if initial_url:
                                driver.get(initial_url)
                                st.session_state.test_pilot_start_url = initial_url
                                # Inject JS recorder after page load
                                try:
                                    time.sleep(1)  # Wait for page to stabilize
                                    driver.execute_script(js_recorder)
                                    logger.info("✅ JavaScript action recorder injected")
                                except Exception as e:
                                    logger.warning(f"Could not inject JS recorder: {e}")
                            else:
                                # Open blank page - user will navigate to desired site
                                driver.get("about:blank")

                                # Start URL polling thread to detect when user navigates away
                                import threading
                                def poll_for_url_change():
                                    """Poll until user navigates away from about:blank"""
                                    max_attempts = 300  # 5 minutes max
                                    attempt = 0
                                    while attempt < max_attempts:
                                        try:
                                            if not st.session_state.get('test_pilot_recording', False):
                                                break  # Recording stopped

                                            current = driver.current_url
                                            if current != 'about:blank':
                                                # User navigated! Capture the URL
                                                logger.info(f"🌐 User navigated to: {current}")
                                                listener.start_url = current
                                                listener.last_url = current
                                                st.session_state.test_pilot_start_url = current

                                                # Add navigation action
                                                listener.actions.append({
                                                    'type': 'navigate',
                                                    'action': 'navigation',
                                                    'value': current,
                                                    'url': current,
                                                    'timestamp': datetime.now().isoformat(),
                                                    'description': f'Navigate to {current}'
                                                })

                                                # Inject JS recorder
                                                try:
                                                    time.sleep(1)
                                                    driver.execute_script(js_recorder)
                                                    logger.info("✅ JS recorder injected after first navigation")
                                                except Exception as e:
                                                    logger.warning(f"Could not inject JS: {e}")

                                                break  # Stop polling

                                            time.sleep(1)
                                            attempt += 1
                                        except Exception as e:
                                            logger.debug(f"URL poll error: {e}")
                                            break

                                # Start polling in background
                                poll_thread = threading.Thread(target=poll_for_url_change, daemon=True)
                                poll_thread.start()

                            # Start browser window monitoring thread to detect closure
                            import threading
                            def monitor_browser_closure():
                                """Monitor browser and auto-stop recording if window is closed"""
                                while st.session_state.get('test_pilot_recording', False):
                                    try:
                                        # Try to get current URL - will fail if browser closed
                                        _ = driver.current_url
                                        time.sleep(2)  # Check every 2 seconds
                                    except Exception as e:
                                        # Check if this is a browser closure exception
                                        exception_str = str(e).lower()
                                        is_browser_closed = any(msg in exception_str for msg in [
                                            'target window already closed',
                                            'web view not found',
                                            'no such window',
                                            'session deleted',
                                            'chrome not reachable',
                                            'invalid session id',
                                            'browser has been closed'
                                        ])

                                        if is_browser_closed:
                                            # Browser was closed by user - expected behavior
                                            logger.info(f"🔴 Browser window closed by user - stopping recording automatically")
                                        else:
                                            # Unexpected error
                                            logger.warning(f"Browser monitoring error: {e}")

                                        # Collect final actions
                                        try:
                                            listener = st.session_state.get('test_pilot_recorder_listener')
                                            nav_actions = listener.actions if listener else []

                                            # Note: Cannot collect JS actions since browser is closed
                                            # Use only navigation actions captured by listener
                                            st.session_state.test_pilot_recorded_actions = nav_actions
                                            st.session_state.test_pilot_recording_metadata['end_time'] = datetime.now().isoformat()
                                            st.session_state.test_pilot_recording_metadata['stopped_by'] = 'user_closed_browser'
                                            st.session_state.test_pilot_recording_metadata['total_actions'] = len(nav_actions)

                                            logger.info(f"✅ Auto-stopped recording. Captured {len(nav_actions)} navigation actions")
                                        except Exception as capture_error:
                                            logger.error(f"Error capturing final actions: {capture_error}")

                                        # Mark recording as stopped
                                        st.session_state.test_pilot_recording = False
                                        st.session_state.test_pilot_recording_stopped = True
                                        break

                            # Start monitoring in background
                            monitor_thread = threading.Thread(target=monitor_browser_closure, daemon=True)
                            monitor_thread.start()

                            st.success(f"✅ Recording started with {browser_choice}!")
                            if initial_url:
                                st.info(f"🌐 Browser opened at: {initial_url}")
                            else:
                                st.info("🌐 Browser opened. Navigate to your desired website to begin recording.")
                            st.warning("⚠️ **Perform your actions in the browser window**, then click '⏹️ Stop Recording' when done")
                            st.info("💡 **Tip:** Recording will automatically stop if you close the browser window")
                            st.rerun()

                    except ImportError as ie:
                        st.error(f"❌ Selenium not installed. Please run: `pip install selenium`")
                        logger.error(f"Import error: {ie}")
                    except Exception as e:
                        st.error(f"❌ Error starting recording: {str(e)}")
                        logger.error(f"Recording start error: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())

        with col2:
            if st.session_state.test_pilot_recording:
                if st.button("📸 Capture Screenshot", use_container_width=True, key="capture_state_btn"):
                    try:
                        driver = st.session_state.test_pilot_recorder_driver
                        listener = st.session_state.test_pilot_recorder_listener
                        js_recorder = st.session_state.test_pilot_js_recorder

                        current_url = driver.current_url
                        page_title = driver.title

                        # Re-inject JS recorder in case page changed
                        try:
                            driver.execute_script(js_recorder)
                            logger.info("🔄 Re-injected JavaScript recorder")
                        except Exception as e:
                            logger.warning(f"Could not re-inject JS recorder: {e}")

                        # Capture page screenshot
                        screenshot_dir = os.path.join(ROOT_DIR, "screenshots", "recordings")
                        os.makedirs(screenshot_dir, exist_ok=True)

                        screenshot_path = os.path.join(
                            screenshot_dir,
                            f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        )
                        driver.save_screenshot(screenshot_path)

                        # Add to actions
                        listener.actions.append({
                            'type': 'screenshot',
                            'action': 'screenshot',
                            'url': current_url,
                            'title': page_title,
                            'screenshot': screenshot_path,
                            'timestamp': datetime.now().isoformat(),
                            'description': f'Screenshot captured: {page_title}'
                        })

                        st.success(f"📸 Screenshot captured: {page_title}")

                    except Exception as e:
                        st.error(f"Error capturing screenshot: {str(e)}")
                        logger.error(f"Screenshot error: {e}")

        with col3:
            if st.session_state.test_pilot_recording:
                if st.button("⏹️ Stop Recording", type="secondary", use_container_width=True, key="stop_record_btn"):
                    try:
                        driver = st.session_state.test_pilot_recorder_driver
                        listener = st.session_state.test_pilot_recorder_listener

                        # Capture final state
                        final_url = driver.current_url
                        final_title = driver.title

                        logger.info(f"🛑 Stopping recording. Current URL: {final_url}")

                        # Collect JavaScript-captured actions
                        js_actions = []
                        try:
                            # First check if JS recorder was initialized
                            js_initialized = driver.execute_script("return window._testPilotInitialized === true;")
                            if not js_initialized:
                                st.warning("⚠️ JavaScript recorder was not active. Actions may not be captured.")
                                logger.warning("JS recorder was not initialized")

                            js_actions_raw = driver.execute_script("return window._testPilotActions || [];")
                            logger.info(f"📊 Retrieved {len(js_actions_raw)} actions from JavaScript recorder")

                            # Log action breakdown for debugging
                            if js_actions_raw:
                                action_types = {}
                                for action in js_actions_raw:
                                    action_type = action.get('type', 'unknown')
                                    action_types[action_type] = action_types.get(action_type, 0) + 1
                                logger.info(f"📊 JS Action breakdown: {action_types}")
                                st.info(f"📊 JavaScript captured: {', '.join([f'{count} {type}' for type, count in action_types.items()])}")
                            else:
                                st.warning("⚠️ No JavaScript actions captured. Check if you interacted with the page.")

                            # Convert JS actions to our format
                            for js_action in js_actions_raw:
                                # Build selector
                                selector = None
                                if js_action.get('id'):
                                    selector = f"id:{js_action['id']}"
                                elif js_action.get('name'):
                                    selector = f"name:{js_action['name']}"
                                elif js_action.get('className'):
                                    classes = js_action['className'].split()
                                    if classes:
                                        selector = f"css:.{classes[0]}"

                                # Build description
                                action_type = js_action.get('type', 'action')
                                element_text = js_action.get('text', '')
                                tag_name = js_action.get('tagName', 'element')

                                if action_type == 'click':
                                    if element_text:
                                        description = f"Click on '{element_text[:50]}' {tag_name}"
                                    elif js_action.get('ariaLabel'):
                                        description = f"Click on '{js_action['ariaLabel']}' {tag_name}"
                                    elif js_action.get('id'):
                                        description = f"Click on {tag_name} with id '{js_action['id']}'"
                                    else:
                                        description = f"Click on {tag_name} element"
                                elif action_type == 'input':
                                    field_name = js_action.get('ariaLabel') or js_action.get('placeholder') or js_action.get('name') or 'field'
                                    value = js_action.get('value', '')
                                    is_sensitive = js_action.get('inputType') == 'password' or 'password' in field_name.lower()
                                    if is_sensitive:
                                        description = f"Enter [sensitive data] into {field_name}"
                                    else:
                                        description = f"Enter '{value[:50]}' into {field_name}"
                                elif action_type == 'select':
                                    field_name = js_action.get('name') or js_action.get('id') or 'dropdown'
                                    selected_text = js_action.get('selectedText', js_action.get('value', ''))
                                    description = f"Select '{selected_text}' from {field_name}"
                                else:
                                    description = f"{action_type} on {tag_name}"

                                # Create action object
                                action = {
                                    'type': action_type,
                                    'action': action_type,
                                    'selector': selector,
                                    'target': selector,
                                    'value': js_action.get('value', ''),
                                    'text': element_text,
                                    'innerText': element_text,
                                    'tagName': tag_name,
                                    'elementType': tag_name,
                                    'attributes': {
                                        'id': js_action.get('id'),
                                        'name': js_action.get('name'),
                                        'class': js_action.get('className'),
                                        'type': js_action.get('inputType') or js_action.get('type'),
                                        'aria-label': js_action.get('ariaLabel'),
                                        'placeholder': js_action.get('placeholder')
                                    },
                                    'url': js_action.get('url', final_url),
                                    'timestamp': js_action.get('timestamp', datetime.now().isoformat()),
                                    'description': description,
                                    'is_sensitive': js_action.get('inputType') == 'password'
                                }
                                js_actions.append(action)
                        except Exception as e:
                            logger.error(f"⚠️ Error retrieving JavaScript actions: {e}")
                            st.error(f"⚠️ Could not retrieve JavaScript actions: {e}")
                            import traceback
                            logger.error(traceback.format_exc())

                        # Get navigation actions from listener
                        navigation_actions = listener.actions
                        logger.info(f"📊 Retrieved {len(navigation_actions)} navigation actions from listener")

                        # Merge actions (navigation + JS actions), sorted by timestamp
                        all_actions = navigation_actions + js_actions

                        # Sort by timestamp
                        try:
                            all_actions.sort(key=lambda x: x.get('timestamp', ''))
                        except Exception:
                            pass  # If sorting fails, use unsorted

                        logger.info(f"✅ Total merged actions: {len(all_actions)}")

                        # Store actions
                        st.session_state.test_pilot_recorded_actions = all_actions
                        st.session_state.test_pilot_recording_metadata['end_time'] = datetime.now().isoformat()
                        st.session_state.test_pilot_recording_metadata['final_url'] = final_url
                        st.session_state.test_pilot_recording_metadata['final_title'] = final_title
                        st.session_state.test_pilot_recording_metadata['total_actions'] = len(all_actions)
                        st.session_state.test_pilot_recording_metadata['js_actions'] = len(js_actions)
                        st.session_state.test_pilot_recording_metadata['navigation_actions'] = len(navigation_actions)

                        # Auto-detect start URL if not set or still about:blank
                        if not st.session_state.test_pilot_start_url or st.session_state.test_pilot_start_url == 'about:blank':
                            if listener.start_url and listener.start_url != 'about:blank':
                                st.session_state.test_pilot_start_url = listener.start_url
                            elif final_url and final_url != 'about:blank':
                                # Use final URL as start URL if no navigation was captured
                                st.session_state.test_pilot_start_url = final_url
                                logger.info(f"🌐 Using final URL as start URL: {final_url}")

                        # Close driver
                        driver.quit()

                        st.session_state.test_pilot_recording = False
                        st.session_state.test_pilot_recording_stopped = True

                        logger.info(f"✅ Recording stopped. Captured {len(all_actions)} total actions ({len(navigation_actions)} navigation, {len(js_actions)} interactions)")
                        st.success(f"✅ Recording stopped! Captured {len(all_actions)} actions ({len(js_actions)} interactions)")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error stopping recording: {str(e)}")
                        logger.error(f"Stop recording error: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        st.session_state.test_pilot_recording = False
                        st.rerun()

        # Display recording status (after button columns)
        if st.session_state.test_pilot_recording:
            st.markdown("---")
            st.info("🔴 **Recording in progress...** Perform actions in the browser window")

            # Continuous polling to capture actions and URL changes (non-blocking)
            browser_closed = False
            try:
                driver = st.session_state.test_pilot_recorder_driver
                listener = st.session_state.test_pilot_recorder_listener
                js_recorder = st.session_state.test_pilot_js_recorder

                # Check current URL (for manual navigation detection)
                current_url = driver.current_url

                # Detect URL change (user typed new URL or clicked link)
                if current_url != 'about:blank' and current_url != listener.last_url:
                    logger.info(f"🔄 URL change detected: {current_url}")

                    # Capture as navigation if different
                    if listener.last_url and listener.last_url != 'about:blank':
                        listener.actions.append({
                            'type': 'navigate',
                            'action': 'navigation',
                            'value': current_url,
                            'url': current_url,
                            'timestamp': datetime.now().isoformat(),
                            'description': f'Navigate to {current_url}'
                        })

                    # Update starting URL if not set or was about:blank
                    if not listener.start_url or listener.start_url == 'about:blank':
                        listener.start_url = current_url
                        st.session_state.test_pilot_start_url = current_url
                        logger.info(f"🌐 Starting URL captured: {current_url}")

                    listener.last_url = current_url

                    # Re-inject JS recorder on new page
                    try:
                        driver.execute_script(js_recorder)
                        logger.info("✅ JS recorder re-injected after URL change")
                    except Exception:
                        pass

                # Collect intermediate JS actions
                try:
                    js_actions_count = driver.execute_script("return (window._testPilotActions || []).length;")

                    # Update action counter in session
                    if 'test_pilot_live_action_count' not in st.session_state:
                        st.session_state.test_pilot_live_action_count = 0

                    total_count = len(listener.actions) + js_actions_count
                    st.session_state.test_pilot_live_action_count = total_count

                except Exception:
                    pass

            except Exception as e:
                # Browser might be closed by user
                exception_str = str(e).lower()
                is_browser_closed = any(msg in exception_str for msg in [
                    'target window already closed',
                    'web view not found',
                    'no such window',
                    'session deleted',
                    'chrome not reachable',
                    'invalid session id',
                    'browser has been closed'
                ])

                if is_browser_closed:
                    # Browser closed - mark for handling after buttons are shown
                    browser_closed = True
                    logger.debug(f"Browser closed during poll: {e}")
                else:
                    # Unexpected error
                    logger.warning(f"Error in recording poll: {e}")

            # Show recording info with live updates
            col1, col2 = st.columns(2)

            with col1:
                if st.session_state.test_pilot_start_url and st.session_state.test_pilot_start_url != 'about:blank':
                    st.markdown(f"**Starting URL:** {st.session_state.test_pilot_start_url}")
                else:
                    st.markdown("**Starting URL:** ⏳ Waiting for navigation...")

            with col2:
                recording_duration = datetime.now() - st.session_state.test_pilot_recording_start_time
                st.markdown(f"**Duration:** {str(recording_duration).split('.')[0]}")

            # Show live action count
            if hasattr(st.session_state, 'test_pilot_live_action_count'):
                st.metric("Actions Captured (Live)", st.session_state.test_pilot_live_action_count)
                if st.session_state.test_pilot_live_action_count > 0:
                    st.success(f"✅ Recording {st.session_state.test_pilot_live_action_count} actions...")
                else:
                    st.warning("⏳ Perform actions in the browser to start capturing...")

            # Check if browser was closed - show message AFTER displaying info
            if browser_closed:
                st.warning("🔴 **Browser window was closed** - Recording stopped automatically")
                st.info("📋 Please wait while we save your recorded actions...")
                time.sleep(2)  # Give monitoring thread time to complete
                st.rerun()

        # Auto-refresh during recording (placed at the end after ALL UI elements)
        if st.session_state.test_pilot_recording:
            time.sleep(1)  # Refresh every second to update live counts
            st.rerun()

        # Display recorded actions with enhanced visualization
        if st.session_state.test_pilot_recorded_actions and not st.session_state.test_pilot_recording:
            st.markdown("---")
            st.markdown("### 📋 Recorded Actions")

            actions = st.session_state.test_pilot_recorded_actions
            metadata = st.session_state.test_pilot_recording_metadata

            # Show recording summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Actions", len(actions))
            with col2:
                navigate_count = sum(1 for a in actions if a.get('type') == 'navigate')
                st.metric("Navigations", navigate_count)
            with col3:
                click_count = sum(1 for a in actions if a.get('type') == 'click')
                st.metric("Clicks", click_count)
            with col4:
                input_count = sum(1 for a in actions if a.get('type') in ['input', 'select'])
                st.metric("Inputs", input_count)

            # Show start URL
            if st.session_state.test_pilot_start_url:
                st.info(f"🌐 **Starting URL (auto-detected):** {st.session_state.test_pilot_start_url}")

            # Show how recording was stopped
            if metadata.get('stopped_by') == 'user_closed_browser':
                st.warning("⚠️ **Recording was automatically stopped when browser window was closed by user**")
                st.info("💡 Note: Only navigation actions were captured. Click/input actions require the Stop Recording button for full capture.")

            st.markdown("---")

            # Display actions with rich context
            st.markdown("#### 🎬 Captured Actions")

            for i, action in enumerate(actions, 1):
                # Get action type safely with fallback
                action_type = action.get('type', 'unknown')

                action_emoji = {
                    'navigate': '🌐', 'click': '👆', 'input': '⌨️', 'select': '📋',
                    'screenshot': '📸', 'wait': '⏱️'
                }.get(action_type, '▶️')

                # Get description safely with fallback
                action_description = action.get('description', '')
                if not action_description and action_type:
                    action_description = action_type.title() if action_type != 'unknown' else 'Action'

                with st.expander(f"**Step {i}:** {action_emoji} {action_description}", expanded=False):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Action Type:** `{action_type}`")
                        if action.get('selector'):
                            st.markdown(f"**Selector:** `{action['selector']}`")
                        if action.get('value') and not action.get('is_sensitive'):
                            st.markdown(f"**Value:** `{action['value']}`")
                        elif action.get('is_sensitive'):
                            st.markdown(f"**Value:** `[MASKED - Sensitive Data]`")
                        if action.get('url'):
                            st.markdown(f"**URL:** {action['url']}")
                        if action.get('timestamp'):
                            st.markdown(f"**Timestamp:** {action['timestamp']}")
                        else:
                            st.markdown(f"**Timestamp:** N/A")

                    with col2:
                        if action.get('attributes'):
                            st.markdown("**📝 Element Attributes:**")
                            attrs = {k: v for k, v in action['attributes'].items() if v}
                            st.json(attrs)

                        if action.get('screenshot'):
                            if os.path.exists(action['screenshot']):
                                st.image(action['screenshot'], width=200)

            # Export and generate options
            st.markdown("---")
            st.markdown("### 💾 Export & Generate")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Export as JSON for reuse
                recording_json = {
                    'title': f"Recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'description': 'Browser recording captured with TestPilot',
                    'startUrl': st.session_state.test_pilot_start_url,
                    'url': st.session_state.test_pilot_start_url,
                    'recorded_at': metadata.get('start_time'),
                    'metadata': metadata,
                    'events': actions  # Use 'events' format for compatibility with upload
                }
                json_str = json.dumps(recording_json, indent=2)

                st.download_button(
                    label="📄 Export JSON",
                    data=json_str,
                    file_name=f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_recording_json",
                    use_container_width=True,
                    help="Export as JSON for upload and reuse"
                )

            with col2:
                # Export as human-readable steps
                steps_text = f"# Test Recording - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                steps_text += f"Starting URL: {st.session_state.test_pilot_start_url}\n\n"
                steps_text += "## Test Steps:\n\n"
                for i, action in enumerate(actions, 1):
                    steps_text += f"{i}. {action.get('description', action['type'])}\n"
                    if action.get('selector'):
                        steps_text += f"   Selector: {action['selector']}\n"
                    if action.get('value') and not action.get('is_sensitive'):
                        steps_text += f"   Value: {action['value']}\n"
                    steps_text += "\n"

                st.download_button(
                    label="📝 Export Steps",
                    data=steps_text,
                    file_name=f"test_steps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_recording_steps",
                    use_container_width=True,
                    help="Export as readable test steps"
                )

            with col3:
                # Generate with AI Analysis
                if st.button("🤖 Generate with AI", type="primary", use_container_width=True, key="generate_with_ai_btn",
                           help="Use Azure OpenAI to analyze and generate optimized Robot Framework script"):
                    st.session_state.test_pilot_generate_from_recording_ai = True
                    st.rerun()

            with col4:
                if st.button("🗑️ Clear Recording", use_container_width=True, key="clear_recording_btn"):
                    st.session_state.test_pilot_recorded_actions = []
                    st.session_state.test_pilot_recording_stopped = False
                    st.session_state.test_pilot_start_url = ""
                    st.session_state.test_pilot_recording_metadata = {}
                    st.rerun()

        # Generate Robot script from recording with AI analysis
        if st.session_state.get('test_pilot_generate_from_recording_ai') and st.session_state.test_pilot_recorded_actions:
            st.markdown("---")
            st.markdown("### 🤖 Generating Robot Framework Script with AI Analysis")

            with st.spinner("🔄 Processing recording and analyzing with Azure OpenAI..."):
                try:
                    # Create recording data in proper format
                    recording_data = {
                        'title': f"Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        'description': 'Test generated from browser recording',
                        'startUrl': st.session_state.test_pilot_start_url,
                        'url': st.session_state.test_pilot_start_url,
                        'events': st.session_state.test_pilot_recorded_actions
                    }

                    # Use enhanced RecordingParser to parse actions
                    st.info("📋 Parsing recorded actions...")
                    steps = RecordingParser.parse_recording(recording_data)

                    if not steps:
                        st.error("❌ No actionable steps found in recording")
                        st.session_state.test_pilot_generate_from_recording_ai = False
                        st.stop()

                    st.success(f"✅ Parsed {len(steps)} actionable steps from recording")

                    # Display parsed steps
                    with st.expander("📋 Parsed Steps", expanded=False):
                        for step in steps:
                            st.markdown(f"**Step {step.step_number}:** {step.description}")
                            if step.action:
                                st.markdown(f"  └─ Action: `{step.action}`")
                            if step.target:
                                st.markdown(f"  └─ Target: `{step.target}`")

                    # Create test case
                    test_case = TestCase(
                        id=f"REC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        title=f"Recorded Test {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        description=f"Test case generated from browser recording. Starting URL: {st.session_state.test_pilot_start_url}",
                        source='recording',
                        steps=steps,
                        metadata={
                            'start_url': st.session_state.test_pilot_start_url,
                            'recording_metadata': st.session_state.test_pilot_recording_metadata
                        }
                    )

                    # Analyze with AI if available
                    if AZURE_AVAILABLE and azure_client and azure_client.is_configured():
                        try:
                            st.info("🤖 Analyzing steps with Azure OpenAI...")

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            success, enhanced_test_case, msg = loop.run_until_complete(
                                engine.analyze_steps_with_ai(test_case)
                            )
                            loop.close()

                            if success:
                                test_case = enhanced_test_case
                                st.success(f"✅ AI Analysis Complete: {msg}")

                                # Show AI-enhanced keywords
                                keywords_found = [step.keyword for step in test_case.steps if step.keyword]
                                if keywords_found:
                                    st.info(f"🔑 AI identified {len(keywords_found)} Robot Framework keywords")
                                    with st.expander("View Mapped Keywords", expanded=True):
                                        for step in test_case.steps:
                                            if step.keyword:
                                                st.markdown(f"**Step {step.step_number}:** `{step.keyword}`")
                                                if step.arguments:
                                                    st.markdown(f"  └─ Arguments: {', '.join(step.arguments)}")
                            else:
                                st.warning(f"⚠️ AI Analysis had issues: {msg}. Using default patterns...")
                        except Exception as e:
                            logger.error(f"AI Analysis error: {str(e)}")
                            st.warning(f"⚠️ AI Analysis error: {str(e)}. Using default patterns...")
                    else:
                        st.info("ℹ️ Generating script without AI analysis (Azure OpenAI not configured)")

                    # Generate Robot Framework script
                    st.info("📝 Generating Robot Framework script...")
                    success, script_content, file_path = engine.generate_robot_script(test_case, include_comments=True)

                    if success:
                        st.success("✅ Robot Framework script generated successfully!")

                        st.markdown("### 📜 Generated Script Preview")

                        # Show file info
                        st.info(f"""
**📁 Files Generated:**
- Test Suite: `{file_path}`
- Keywords: Check keywords directory
- Locators: Check locators directory
- Variables: Check variables directory

**Next Steps:**
1. Review the generated scripts below
2. Update locators with actual element selectors
3. Update variables with test data
4. Run: `robot {file_path}`
                        """)

                        with st.expander("📄 Test Suite File", expanded=True):
                            st.code(script_content, language='robotframework')

                        # Download button
                        st.download_button(
                            label="⬇️ Download Robot Framework Script",
                            data=script_content,
                            file_name=os.path.basename(file_path),
                            mime="text/plain",
                            key="download_recording_robot_ai",
                            use_container_width=True
                        )

                        if NOTIFICATIONS_AVAILABLE:
                            notifications.add_notification(
                                module_name="test_pilot",
                                status="success",
                                message=f"Generated script from recording with AI analysis",
                                details=f"Script saved to: {file_path}"
                            )
                    else:
                        st.error(f"❌ Failed to generate script: {file_path}")

                except Exception as e:
                    st.error(f"❌ Error generating script: {str(e)}")
                    logger.error(f"Recording script generation error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

            # Clear the trigger
            st.session_state.test_pilot_generate_from_recording_ai = False

    # Tab 5: Generated Scripts
    with tab5:
        st.markdown("### Generated Scripts")

        # List generated scripts
        if os.path.exists(engine.output_dir):
            script_files = [f for f in os.listdir(engine.output_dir)
                           if f.startswith('test_pilot_') and f.endswith('.robot')]

            if script_files:
                st.markdown(f"Found {len(script_files)} generated scripts")

                # Sort by modification time (newest first)
                script_files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(engine.output_dir, x)),
                    reverse=True
                )

                for script_file in script_files:
                    file_path = os.path.join(engine.output_dir, script_file)
                    file_size = os.path.getsize(file_path)
                    mod_time = datetime.fromtimestamp(
                        os.path.getmtime(file_path)
                    ).strftime('%Y-%m-%d %H:%M:%S')

                    with st.expander(f"📄 {script_file}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Size:** {file_size} bytes")
                        with col2:
                            st.markdown(f"**Modified:** {mod_time}")
                        with col3:
                            with open(file_path, 'r') as f:
                                content = f.read()
                                st.download_button(
                                    label="⬇️ Download",
                                    data=content,
                                    file_name=script_file,
                                    mime="text/plain",
                                    key=f"download_{script_file}"
                                )

                        # Show preview
                        with open(file_path, 'r') as f:
                            st.code(f.read(), language='robotframework')
            else:
                st.info("No scripts generated yet. Start by creating a test case in one of the other tabs.")
        else:
            st.info("Output directory not found. Generate your first script to get started.")


# Main entry point
if __name__ == "__main__":
    show_ui()
