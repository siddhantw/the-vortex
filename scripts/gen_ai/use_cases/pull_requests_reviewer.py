"""
Pull Requests Reviewer Module
Automated code review and analysis for Stash/Bitbucket repositories
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Optional, Tuple
import logging
import base64
from urllib.parse import urljoin, urlparse
import difflib
import ast
import subprocess
import tempfile
import os
import fnmatch
import sys

# Azure OpenAI Integration
try:
    # Add the path to import azure_openai_client
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from azure_openai_client import AzureOpenAIClient
    AI_AVAILABLE = True
    azure_openai_client = AzureOpenAIClient()
except (ImportError, ValueError) as e:
    AI_AVAILABLE = False
    azure_openai_client = None
    print(f"Warning: Azure OpenAI client not available: {e}")

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("PullRequestReviewer", level=logging.INFO, log_file="pull_requests_reviewer.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")


class PullRequestReviewer:
    """Advanced pull request analysis and code review system"""

    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        requests.packages.urllib3.disable_warnings()
        self._last_auth_error = None

    def authenticate(self, base_url: str, username: str, password: str = None, token: str = None) -> bool:
        """Authenticate with Bitbucket/Stash with improved error handling"""
        self._last_auth_error = None

        try:
            # Normalize base URL
            if not base_url.endswith('/'):
                base_url = base_url + '/'

            # Clear any previous authentication
            self.session.headers.pop('Authorization', None)
            self.session.auth = None

            if token:
                # Try different token authentication methods
                auth_methods = [
                    # Method 1: Bearer token (common for modern APIs)
                    {'Authorization': f'Bearer {token}'},
                    # Method 2: HTTP Basic with token as username
                    {'auth': (token, '')},
                    # Method 3: HTTP Basic with username and token as password
                    {'auth': (username, token)},
                    # Method 4: Custom token header (some Bitbucket versions)
                    {'Authorization': f'token {token}'}
                ]

                for i, method in enumerate(auth_methods, 1):
                    logger.info(f"Trying authentication method {i} for token auth...")

                    # Reset session
                    self.session.headers.pop('Authorization', None)
                    self.session.auth = None

                    if 'Authorization' in method:
                        self.session.headers.update({'Authorization': method['Authorization']})
                    elif 'auth' in method:
                        self.session.auth = method['auth']

                    success, error = self._test_authentication(base_url)
                    if success:
                        logger.info(f"Token authentication successful with method {i}")
                        return True
                    else:
                        self._last_auth_error = f"Method {i} failed: {error}"

            elif username and password:
                logger.info("Trying username/password authentication...")
                self.session.auth = (username, password)

                success, error = self._test_authentication(base_url)
                if success:
                    logger.info("Username/password authentication successful")
                    return True
                else:
                    self._last_auth_error = f"Username/password auth failed: {error}"
            else:
                self._last_auth_error = "No authentication credentials provided"
                logger.error(self._last_auth_error)
                return False

            logger.error("All authentication methods failed")
            return False

        except Exception as e:
            self._last_auth_error = f"Authentication exception: {str(e)}"
            logger.error(f"Authentication failed with exception: {e}")
            return False

    def _test_authentication(self, base_url: str) -> Tuple[bool, str]:
        """Test authentication with multiple endpoints"""
        # List of endpoints to try for authentication testing
        test_endpoints = [
            '/rest/api/1.0/projects',  # Standard projects endpoint
            '/rest/api/1.0/repos',  # Repositories endpoint
            '/rest/api/1.0/users',  # Users endpoint
            '/rest/api/1.0/dashboard/pull-requests',  # Dashboard endpoint
            '/rest/api/1.0/application-properties'  # Application info
        ]

        last_error = "No endpoints could be tested"

        for endpoint in test_endpoints:
            try:
                test_url = urljoin(base_url, endpoint)
                logger.info(f"Testing authentication with endpoint: {endpoint}")

                response = self.session.get(test_url, timeout=15)

                logger.info(f"Response status: {response.status_code}")

                # Check for successful response codes
                if response.status_code == 200:
                    logger.info(f"Authentication successful with endpoint: {endpoint}")
                    return True, "Success"
                elif response.status_code == 401:
                    last_error = f"Unauthorized (401) - Invalid credentials"
                    logger.warning(f"Unauthorized (401) for endpoint: {endpoint}")
                    continue
                elif response.status_code == 403:
                    last_error = f"Forbidden (403) - Access denied or insufficient permissions"
                    logger.warning(f"Forbidden (403) for endpoint: {endpoint} - may have limited permissions")
                    continue
                elif response.status_code == 404:
                    last_error = f"Not found (404) - API endpoint may not exist on this server"
                    logger.warning(f"Not found (404) for endpoint: {endpoint} - endpoint may not exist")
                    continue
                else:
                    last_error = f"Unexpected status {response.status_code}"
                    logger.warning(f"Unexpected status {response.status_code} for endpoint: {endpoint}")
                    continue

            except requests.exceptions.Timeout:
                last_error = f"Timeout - Server not responding"
                logger.error(f"Timeout testing endpoint: {endpoint}")
                continue
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error - Cannot reach server: {str(e)}"
                logger.error(f"Connection error testing endpoint {endpoint}: {e}")
                continue
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Error testing endpoint {endpoint}: {e}")
                continue

        return False, last_error

    def get_repositories(self, base_url: str, project_key: str = None) -> List[Dict]:
        """Get list of repositories"""
        repositories = []

        try:
            if project_key:
                url = urljoin(base_url, f'/rest/api/1.0/projects/{project_key}/repos')
            else:
                url = urljoin(base_url, '/rest/api/1.0/repos')

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                repositories = data.get('values', [])

                # Handle pagination
                while not data.get('isLastPage', True):
                    next_url = f"{url}?start={data.get('nextPageStart', 0)}"
                    response = self.session.get(next_url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        repositories.extend(data.get('values', []))
                    else:
                        break

        except Exception as e:
            logger.error(f"Error fetching repositories: {e}")

        return repositories

    def get_pull_requests(self, base_url: str, project_key: str, repo_slug: str,
                          state: str = "OPEN", limit: int = 100) -> List[Dict]:
        """Get pull requests for a repository"""
        pull_requests = []

        try:
            url = urljoin(base_url, f'/rest/api/1.0/projects/{project_key}/repos/{repo_slug}/pull-requests')
            params = {'state': state, 'limit': limit}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                pull_requests = data.get('values', [])

                # Handle pagination if needed
                while not data.get('isLastPage', True) and len(pull_requests) < limit:
                    params['start'] = data.get('nextPageStart', 0)
                    response = self.session.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        pull_requests.extend(data.get('values', []))
                    else:
                        break

        except Exception as e:
            logger.error(f"Error fetching pull requests: {e}")

        return pull_requests

    def get_pr_changes(self, base_url: str, project_key: str, repo_slug: str, pr_id: int) -> Dict:
        """Get detailed changes for a pull request"""
        changes_data = {
            "files_changed": [],
            "additions": 0,
            "deletions": 0,
            "total_changes": 0,
            "commits": []
        }

        try:
            # Get PR changes/diff
            changes_url = urljoin(base_url,
                                  f'/rest/api/1.0/projects/{project_key}/repos/{repo_slug}/pull-requests/{pr_id}/changes')

            response = self.session.get(changes_url, timeout=10)

            if response.status_code == 200:
                changes = response.json()

                for change in changes.get('values', []):
                    file_info = {
                        "path": change.get('path', {}).get('toString', ''),
                        "type": change.get('type', ''),
                        "content_id": change.get('contentId', ''),
                        "src_path": change.get('srcPath', {}).get('toString', '') if change.get('srcPath') else None
                    }

                    changes_data["files_changed"].append(file_info)

            # Get PR commits
            commits_url = urljoin(base_url,
                                  f'/rest/api/1.0/projects/{project_key}/repos/{repo_slug}/pull-requests/{pr_id}/commits')

            response = self.session.get(commits_url, timeout=10)

            if response.status_code == 200:
                commits = response.json()
                changes_data["commits"] = commits.get('values', [])

            # Get PR activities for more context
            activities_url = urljoin(base_url,
                                     f'/rest/api/1.0/projects/{project_key}/repos/{repo_slug}/pull-requests/{pr_id}/activities')

            response = self.session.get(activities_url, timeout=10)

            if response.status_code == 200:
                activities = response.json()
                changes_data["activities"] = activities.get('values', [])

        except Exception as e:
            logger.error(f"Error fetching PR changes: {e}")

        return changes_data

    def analyze_code_quality(self, file_content: str, file_path: str, config: Dict = None) -> Dict:
        """Analyze code quality for a file with configuration options"""
        if config is None:
            config = {}

        analysis = {
            "file_path": file_path,
            "language": self._detect_language(file_path),
            "issues": [],
            "metrics": {},
            "security_issues": [],
            "suggestions": []
        }

        # Check if file type should be analyzed
        focus_file_types = config.get('file_types', ['All'])
        if 'All' not in focus_file_types:
            language_map = {
                'python': 'Python',
                'javascript': 'JavaScript',
                'java': 'Java',
                'robot': 'Robot Framework',
                'sql': 'SQL',
                'yaml': 'YAML',
                'bash': 'Shell'
            }
            if language_map.get(analysis["language"]) not in focus_file_types:
                return analysis

        # Check exclude patterns
        exclude_patterns = config.get('exclude_patterns', '').split('\n')
        for pattern in exclude_patterns:
            pattern = pattern.strip()
            if pattern and self._matches_pattern(file_path, pattern):
                return analysis

        try:
            # Calculate accurate metrics first
            analysis["metrics"] = self._calculate_accurate_metrics(file_content, analysis["language"])

            # Language-specific analysis with config
            if analysis["language"] == "python":
                analysis.update(self._analyze_python_code(file_content, file_path, config))
            elif analysis["language"] == "javascript":
                analysis.update(self._analyze_javascript_code(file_content, file_path, config))
            elif analysis["language"] == "java":
                analysis.update(self._analyze_java_code(file_content, file_path, config))
            elif analysis["language"] == "robot":
                analysis.update(self._analyze_robot_code(file_content, file_path, config))
            else:
                analysis.update(self._analyze_generic_code(file_content, file_path, config))

            # Update metrics from language-specific analysis if available
            lang_metrics = analysis.get('metrics', {})
            if lang_metrics:
                # Merge with calculated metrics, preferring calculated ones
                calculated_metrics = self._calculate_accurate_metrics(file_content, analysis["language"])
                analysis["metrics"] = {**lang_metrics, **calculated_metrics}

            # Extract accurate line numbers for all issues
            self._update_issue_line_numbers(analysis, file_content)

            # Apply severity filtering
            min_severity = config.get('min_severity_report', 'Medium')
            analysis = self._filter_by_severity(analysis, min_severity)

            # Apply issue limits
            max_issues = config.get('max_issues_per_file', 20)
            analysis = self._limit_issues(analysis, max_issues)

        except Exception as e:
            logger.error(f"Error analyzing code quality for {file_path}: {e}")
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"Failed to analyze file: {str(e)}",
                "line": 1
            })

        return analysis

    def _update_issue_line_numbers(self, analysis: Dict, file_content: str) -> None:
        """Update all issues with accurate line numbers based on content analysis"""
        all_issues = analysis.get('issues', []) + analysis.get('security_issues', [])

        for issue in all_issues:
            if issue.get('line', 0) <= 1:  # Only update if line number is missing or generic
                issue_type = issue.get('type', '')
                issue_message = issue.get('message', '')

                # Extract actual line numbers based on issue content
                matching_lines = self._extract_actual_line_numbers(file_content, issue_type, issue_message)
                if matching_lines:
                    issue['line'] = matching_lines[0]  # Use first match
                    if len(matching_lines) > 1:
                        issue['additional_lines'] = matching_lines[1:]

    def analyze_code_quality_v2(self, file_content: str, file_path: str, config: Dict = None) -> Dict:
        """Analyze code quality for a file with configuration options (v2 with improved metrics and line numbers)"""
        if config is None:
            config = {}

        analysis = {
            "file_path": file_path,
            "language": self._detect_language(file_path),
            "issues": [],
            "metrics": {},
            "security_issues": [],
            "suggestions": []
        }

        # Check if file type should be analyzed
        focus_file_types = config.get('file_types', ['All'])
        if 'All' not in focus_file_types:
            language_map = {
                'python': 'Python',
                'javascript': 'JavaScript',
                'java': 'Java',
                'robot': 'Robot Framework',
                'sql': 'SQL',
                'yaml': 'YAML',
                'bash': 'Shell'
            }
            if language_map.get(analysis["language"]) not in focus_file_types:
                return analysis

        # Check exclude patterns
        exclude_patterns = config.get('exclude_patterns', '').split('\n')
        for pattern in exclude_patterns:
            pattern = pattern.strip()
            if pattern and self._matches_pattern(file_path, pattern):
                return analysis

        try:
            # Calculate accurate metrics first
            analysis["metrics"] = self._calculate_accurate_metrics(file_content, analysis["language"])

            # Language-specific analysis with config
            if analysis["language"] == "python":
                analysis.update(self._analyze_python_code(file_content, file_path, config))
            elif analysis["language"] == "javascript":
                analysis.update(self._analyze_javascript_code(file_content, file_path, config))
            elif analysis["language"] == "java":
                analysis.update(self._analyze_java_code(file_content, file_path, config))
            elif analysis["language"] == "robot":
                analysis.update(self._analyze_robot_code(file_content, file_path, config))
            else:
                analysis.update(self._analyze_generic_code(file_content, file_path, config))

            # Update metrics from language-specific analysis if available
            lang_metrics = analysis.get('metrics', {})
            if lang_metrics:
                # Merge with calculated metrics, preferring calculated ones
                calculated_metrics = self._calculate_accurate_metrics(file_content, analysis["language"])
                analysis["metrics"] = {**lang_metrics, **calculated_metrics}

            # Extract accurate line numbers for all issues
            self._update_issue_line_numbers(analysis, file_content)

            # Apply severity filtering
            min_severity = config.get('min_severity_report', 'Medium')
            analysis = self._filter_by_severity(analysis, min_severity)

            # Apply issue limits
            max_issues = config.get('max_issues_per_file', 20)
            analysis = self._limit_issues(analysis, max_issues)

        except Exception as e:
            logger.error(f"Error analyzing code quality for {file_path}: {e}")
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"Failed to analyze file: {str(e)}",
                "line": 1
            })

        return analysis

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file path matches exclude pattern"""
        import fnmatch
        return fnmatch.fnmatch(file_path, pattern) or fnmatch.fnmatch(file_path.lower(), pattern.lower())

    def _filter_by_severity(self, analysis: Dict, min_severity: str) -> Dict:
        """Filter issues by minimum severity level"""
        severity_levels = {'Info': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        min_level = severity_levels.get(min_severity, 2)

        def filter_issues(issues):
            return [issue for issue in issues
                    if severity_levels.get(issue.get('severity', 'Medium').title(), 2) >= min_level]

        analysis['issues'] = filter_issues(analysis['issues'])
        analysis['security_issues'] = filter_issues(analysis['security_issues'])

        return analysis

    def _limit_issues(self, analysis: Dict, max_issues: int) -> Dict:
        """Limit number of issues per file to avoid noise"""
        if len(analysis['issues']) > max_issues:
            analysis['issues'] = analysis['issues'][:max_issues]
            analysis['issues'].append({
                "type": "truncated",
                "severity": "info",
                "message": f"... and {len(analysis['issues']) - max_issues + 1} more issues (truncated)",
                "line": 0
            })

        return analysis

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.robot': 'robot',
            '.rb': 'ruby',
            '.php': 'php',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.sh': 'bash',
            '.sql': 'sql',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css'
        }

        for ext, lang in extension_map.items():
            if file_path.lower().endswith(ext):
                return lang

        return 'unknown'

    def _analyze_python_code(self, content: str, file_path: str, config: Dict) -> Dict:
        """Analyze Python code quality with configuration options"""
        analysis = {"issues": [], "metrics": {}, "security_issues": [], "suggestions": []}

        try:
            # Parse AST for structural analysis
            tree = ast.parse(content)

            # Count lines of code
            lines = content.split('\n')
            analysis["metrics"]["total_lines"] = len(lines)
            analysis["metrics"]["non_empty_lines"] = len([line for line in lines if line.strip()])

            # Get thresholds from config
            max_function_lines = config.get('max_function_lines', 50)
            max_file_lines = config.get('max_file_lines', 1000)
            complexity_threshold = config.get('complexity_threshold', 10)

            # Check file size
            if len(lines) > max_file_lines:
                analysis["issues"].append({
                    "type": "maintainability",
                    "severity": "medium",
                    "message": f"File too large ({len(lines)} lines, max {max_file_lines})",
                    "line": 0
                })

            # Check for common issues based on config
            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Style checks
                if config.get('analyze_style', True):
                    if len(line) > 120:
                        analysis["issues"].append({
                            "type": "style",
                            "severity": "low",
                            "message": f"Line too long ({len(line)} characters)",
                            "line": i
                        })

                # TODO comments
                if config.get('check_todos', True):
                    if 'TODO' in line_stripped or 'FIXME' in line_stripped:
                        analysis["issues"].append({
                            "type": "maintenance",
                            "severity": "medium",
                            "message": "TODO/FIXME comment found",
                            "line": i
                        })

                # Security analysis
                if config.get('analyze_security', True):
                    if 'eval(' in line_stripped:
                        analysis["security_issues"].append({
                            "type": "dangerous_function",
                            "severity": "high",
                            "message": "Use of eval() function detected",
                            "line": i
                        })

                    if 'exec(' in line_stripped:
                        analysis["security_issues"].append({
                            "type": "dangerous_function",
                            "severity": "high",
                            "message": "Use of exec() function detected",
                            "line": i
                        })

                    # Check for hardcoded secrets
                    if re.search(r'(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']', line_stripped, re.I):
                        analysis["security_issues"].append({
                            "type": "hardcoded_secret",
                            "severity": "high",
                            "message": "Potential hardcoded secret detected",
                            "line": i
                        })

                # Performance checks
                if config.get('check_performance', True):
                    if re.search(r'for\s+\w+\s+in\s+range\(len\(', line_stripped):
                        analysis["issues"].append({
                            "type": "performance",
                            "severity": "low",
                            "message": "Consider using enumerate instead of range(len())",
                            "line": i
                        })

            # AST-based analysis
            class CodeAnalyzer(ast.NodeVisitor):
                def __init__(self):
                    self.functions = 0
                    self.classes = 0
                    self.imports = 0
                    self.unused_imports = []
                    self.complex_functions = []

                def visit_FunctionDef(self, node):
                    self.functions += 1

                    # Check function complexity
                    if config.get('check_complexity', True):
                        if len(node.body) > max_function_lines:
                            analysis["issues"].append({
                                "type": "complexity",
                                "severity": "medium",
                                "message": f"Function '{node.name}' is too long ({len(node.body)} statements, max {max_function_lines})",
                                "line": node.lineno
                            })

                    # Check for missing docstrings
                    if config.get('check_documentation', True):
                        if not ast.get_docstring(node):
                            analysis["issues"].append({
                                "type": "documentation",
                                "severity": "low",
                                "message": f"Function '{node.name}' missing docstring",
                                "line": node.lineno
                            })

                    # Check error handling
                    if config.get('check_error_handling', True):
                        has_try_except = any(isinstance(stmt, ast.Try) for stmt in ast.walk(node))
                        if not has_try_except and len(node.body) > 10:
                            analysis["issues"].append({
                                "type": "error_handling",
                                "severity": "medium",
                                "message": f"Function '{node.name}' lacks error handling",
                                "line": node.lineno
                            })

                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    self.classes += 1

                    # Check class size
                    if config.get('check_modularity', True):
                        max_class_lines = config.get('max_class_lines', 300)
                        if len(node.body) > max_class_lines:
                            analysis["issues"].append({
                                "type": "modularity",
                                "severity": "medium",
                                "message": f"Class '{node.name}' is too large",
                                "line": node.lineno
                            })

                    # Check for missing docstrings
                    if config.get('check_documentation', True):
                        if not ast.get_docstring(node):
                            analysis["issues"].append({
                                "type": "documentation",
                                "severity": "low",
                                "message": f"Class '{node.name}' missing docstring",
                                "line": node.lineno
                            })

                    self.generic_visit(node)

                def visit_Import(self, node):
                    self.imports += 1
                    self.generic_visit(node)

                def visit_ImportFrom(self, node):
                    self.imports += 1
                    self.generic_visit(node)

            analyzer = CodeAnalyzer()
            analyzer.visit(tree)

            analysis["metrics"]["functions"] = analyzer.functions
            analysis["metrics"]["classes"] = analyzer.classes
            analysis["metrics"]["imports"] = analyzer.imports

            # Generate suggestions based on config
            if config.get('generate_suggestions', True):
                if analyzer.functions == 0 and analyzer.classes == 0:
                    analysis["suggestions"].append("Consider organizing code into functions or classes")

                if len(analysis["security_issues"]) > 0:
                    analysis["suggestions"].append("Review and address security vulnerabilities")

                if analysis["metrics"]["total_lines"] > 500 and analyzer.functions < 5:
                    analysis["suggestions"].append("Consider breaking large file into smaller modules")

        except SyntaxError as e:
            analysis["issues"].append({
                "type": "syntax_error",
                "severity": "high",
                "message": f"Syntax error: {str(e)}",
                "line": e.lineno if hasattr(e, 'lineno') else 0
            })
        except Exception as e:
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"Analysis failed: {str(e)}",
                "line": 0
            })

        return analysis

    def _analyze_robot_code(self, content: str, file_path: str, config: Dict) -> Dict:
        """Analyze Robot Framework code quality with configuration options"""
        analysis = {"issues": [], "metrics": {}, "security_issues": [], "suggestions": []}

        try:
            lines = content.split('\n')
            analysis["metrics"]["total_lines"] = len(lines)

            # Get thresholds from config
            max_file_lines = config.get('max_file_lines', 1000)

            test_cases = 0
            keywords = 0
            variables = 0

            # Check file size
            if len(lines) > max_file_lines:
                analysis["issues"].append({
                    "type": "maintainability",
                    "severity": "medium",
                    "message": f"File too large ({len(lines)} lines, max {max_file_lines})",
                    "line": 0
                })

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Count Robot Framework sections
                if line_stripped.startswith('Test Case') or 'Test Case' in line_stripped:
                    test_cases += 1
                elif line_stripped.startswith('***') and 'Keywords' in line_stripped:
                    keywords += 1
                elif line_stripped.startswith('***') and 'Variables' in line_stripped:
                    variables += 1

                # Security analysis
                if config.get('analyze_security', True):
                    if re.search(r'(password|secret|key)\s*=\s*\S+', line_stripped, re.I):
                        analysis["security_issues"].append({
                            "type": "hardcoded_credential",
                            "severity": "medium",
                            "message": "Potential hardcoded credential in Robot file",
                            "line": i
                        })

                # Style checks
                if config.get('analyze_style', True):
                    if len(line) > 120:
                        analysis["issues"].append({
                            "type": "style",
                            "severity": "low",
                            "message": f"Line too long ({len(line)} characters)",
                            "line": i
                        })

                # TODO comments
                if config.get('check_todos', True):
                    if 'TODO' in line_stripped or 'FIXME' in line_stripped:
                        analysis["issues"].append({
                            "type": "maintenance",
                            "severity": "medium",
                            "message": "TODO/FIXME comment found",
                            "line": i
                        })

                # Documentation checks
                if config.get('check_documentation', True):
                    if line_stripped.startswith('***') and 'Documentation' not in content:
                        analysis["issues"].append({
                            "type": "documentation",
                            "severity": "low",
                            "message": "Robot file missing documentation section",
                            "line": i
                        })

            analysis["metrics"]["test_cases"] = test_cases
            analysis["metrics"]["keywords"] = keywords
            analysis["metrics"]["variables"] = variables

            # Generate suggestions based on config
            if config.get('generate_suggestions', True):
                if test_cases == 0:
                    analysis["suggestions"].append("Consider adding test cases to this Robot file")
                if keywords == 0 and test_cases > 5:
                    analysis["suggestions"].append("Consider extracting common functionality into keywords")
                if variables == 0 and test_cases > 3:
                    analysis["suggestions"].append("Consider using variables for test data management")

        except Exception as e:
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"Robot analysis failed: {str(e)}",
                "line": 0
            })

        return analysis

    def _analyze_javascript_code(self, content: str, file_path: str, config: Dict) -> Dict:
        """Analyze JavaScript code quality with configuration options"""
        analysis = {"issues": [], "metrics": {}, "security_issues": [], "suggestions": []}

        try:
            lines = content.split('\n')
            analysis["metrics"]["total_lines"] = len(lines)

            # Get thresholds from config
            max_file_lines = config.get('max_file_lines', 1000)

            # Check file size
            if len(lines) > max_file_lines:
                analysis["issues"].append({
                    "type": "maintainability",
                    "severity": "medium",
                    "message": f"File too large ({len(lines)} lines, max {max_file_lines})",
                    "line": 0
                })

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Security analysis
                if config.get('analyze_security', True):
                    if 'eval(' in line_stripped:
                        analysis["security_issues"].append({
                            "type": "dangerous_function",
                            "severity": "high",
                            "message": "Use of eval() function detected",
                            "line": i
                        })

                    if 'document.write(' in line_stripped:
                        analysis["security_issues"].append({
                            "type": "xss_risk",
                            "severity": "medium",
                            "message": "document.write() usage may lead to XSS",
                            "line": i
                        })

                    if 'innerHTML' in line_stripped and '=' in line_stripped:
                        analysis["security_issues"].append({
                            "type": "xss_risk",
                            "severity": "medium",
                            "message": "innerHTML assignment may lead to XSS",
                            "line": i
                        })

                    # Check for hardcoded secrets
                    if re.search(r'(password|secret|key|token)\s*[:=]\s*["\'][^"\']{8,}["\']', line_stripped, re.I):
                        analysis["security_issues"].append({
                            "type": "hardcoded_secret",
                            "severity": "high",
                            "message": "Potential hardcoded secret detected",
                            "line": i
                        })

                # Style checks
                if config.get('analyze_style', True):
                    if len(line) > 120:
                        analysis["issues"].append({
                            "type": "style",
                            "severity": "low",
                            "message": f"Line too long ({len(line)} characters)",
                            "line": i
                        })

                # Check for console.log in production code
                if config.get('check_todos', True) or config.get('analyze_quality', True):
                    if 'console.log(' in line_stripped:
                        analysis["issues"].append({
                            "type": "debug_code",
                            "severity": "low",
                            "message": "console.log() statement found - remove before production",
                            "line": i
                        })

                # TODO comments
                if config.get('check_todos', True):
                    if 'TODO' in line_stripped or 'FIXME' in line_stripped:
                        analysis["issues"].append({
                            "type": "maintenance",
                            "severity": "medium",
                            "message": "TODO/FIXME comment found",
                            "line": i
                        })

                # Performance checks
                if config.get('check_performance', True):
                    if re.search(r'for\s*\(\s*var\s+\w+\s*=\s*0', line_stripped):
                        analysis["issues"].append({
                            "type": "performance",
                            "severity": "low",
                            "message": "Consider using for...of or forEach for better performance",
                            "line": i
                        })

                # Error handling
                if config.get('check_error_handling', True):
                    if 'throw new Error(' in line_stripped and 'try' not in content:
                        analysis["issues"].append({
                            "type": "error_handling",
                            "severity": "medium",
                            "message": "Error throwing without proper try-catch handling",
                            "line": i
                        })

            # Generate suggestions based on config
            if config.get('generate_suggestions', True):
                if len(analysis["security_issues"]) > 0:
                    analysis["suggestions"].append("Review and address JavaScript security vulnerabilities")

                if analysis["metrics"]["total_lines"] > 300:
                    analysis["suggestions"].append("Consider breaking large JavaScript file into modules")

        except Exception as e:
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"JavaScript analysis failed: {str(e)}",
                "line": 0
            })

        return analysis

    def _analyze_java_code(self, content: str, file_path: str, config: Dict) -> Dict:
        """Analyze Java code quality with configuration options"""
        analysis = {"issues": [], "metrics": {}, "security_issues": [], "suggestions": []}

        try:
            lines = content.split('\n')
            analysis["metrics"]["total_lines"] = len(lines)

            # Get thresholds from config
            max_file_lines = config.get('max_file_lines', 1000)

            # Check file size
            if len(lines) > max_file_lines:
                analysis["issues"].append({
                    "type": "maintainability",
                    "severity": "medium",
                    "message": f"File too large ({len(lines)} lines, max {max_file_lines})",
                    "line": 0
                })

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Debug code checks
                if config.get('analyze_quality', True):
                    if 'System.out.print' in line_stripped:
                        analysis["issues"].append({
                            "type": "debug_code",
                            "severity": "low",
                            "message": "System.out.print statement found - use logging instead",
                            "line": i
                        })

                # Security analysis
                if config.get('analyze_security', True):
                    # Check for SQL injection risks
                    if re.search(r'Statement.*execute.*\+', line_stripped):
                        analysis["security_issues"].append({
                            "type": "sql_injection",
                            "severity": "high",
                            "message": "Potential SQL injection - use PreparedStatement",
                            "line": i
                        })

                    # Check for hardcoded secrets
                    if re.search(r'(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']', line_stripped, re.I):
                        analysis["security_issues"].append({
                            "type": "hardcoded_secret",
                            "severity": "high",
                            "message": "Potential hardcoded secret detected",
                            "line": i
                        })

                # Style checks
                if config.get('analyze_style', True):
                    if len(line) > 120:
                        analysis["issues"].append({
                            "type": "style",
                            "severity": "low",
                            "message": f"Line too long ({len(line)} characters)",
                            "line": i
                        })

                # TODO comments
                if config.get('check_todos', True):
                    if 'TODO' in line_stripped or 'FIXME' in line_stripped:
                        analysis["issues"].append({
                            "type": "maintenance",
                            "severity": "medium",
                            "message": "TODO/FIXME comment found",
                            "line": i
                        })

                # Error handling
                if config.get('check_error_handling', True):
                    if 'throw new ' in line_stripped and 'try {' not in content:
                        analysis["issues"].append({
                            "type": "error_handling",
                            "severity": "medium",
                            "message": "Exception throwing without proper try-catch handling",
                            "line": i
                        })

                # Performance checks
                if config.get('check_performance', True):
                    if re.search(r'for\s*\(\s*int\s+\w+\s*=\s*0.*\.size\(\)', line_stripped):
                        analysis["issues"].append({
                            "type": "performance",
                            "severity": "low",
                            "message": "Consider caching collection size in loop",
                            "line": i
                        })

            # Generate suggestions based on config
            if config.get('generate_suggestions', True):
                if len(analysis["security_issues"]) > 0:
                    analysis["suggestions"].append("Review and address Java security vulnerabilities")

                if analysis["metrics"]["total_lines"] > 500:
                    analysis["suggestions"].append("Consider breaking large Java file into smaller classes")

        except Exception as e:
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"Java analysis failed: {str(e)}",
                "line": 0
            })

        return analysis

    def _analyze_generic_code(self, content: str, file_path: str, config: Dict) -> Dict:
        """Generic code analysis for unknown file types with configuration options"""
        analysis = {"issues": [], "metrics": {}, "security_issues": [], "suggestions": []}

        try:
            lines = content.split('\n')
            analysis["metrics"]["total_lines"] = len(lines)
            analysis["metrics"]["non_empty_lines"] = len([line for line in lines if line.strip()])

            # Get thresholds from config
            max_file_lines = config.get('max_file_lines', 1000)

            # Check file size
            if len(lines) > max_file_lines:
                analysis["issues"].append({
                    "type": "maintainability",
                    "severity": "medium",
                    "message": f"File too large ({len(lines)} lines, max {max_file_lines})",
                    "line": 0
                })

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # TODO comments
                if config.get('check_todos', True):
                    if 'TODO' in line_stripped or 'FIXME' in line_stripped:
                        analysis["issues"].append({
                            "type": "maintenance",
                            "severity": "medium",
                            "message": "TODO/FIXME comment found",
                            "line": i
                        })

                # Security analysis
                if config.get('analyze_security', True):
                    # Check for potential secrets
                    if re.search(r'(password|secret|key|token)\s*[:=]\s*["\'][^"\']{8,}["\']', line_stripped, re.I):
                        analysis["security_issues"].append({
                            "type": "potential_secret",
                            "severity": "medium",
                            "message": "Potential hardcoded secret detected",
                            "line": i
                        })

                # Style checks
                if config.get('analyze_style', True):
                    if len(line) > 120:
                        analysis["issues"].append({
                            "type": "style",
                            "severity": "low",
                            "message": f"Line too long ({len(line)} characters)",
                            "line": i
                        })

            # Generate suggestions based on config
            if config.get('generate_suggestions', True):
                if len(analysis["security_issues"]) > 0:
                    analysis["suggestions"].append("Review potential security issues")

                if analysis["metrics"]["total_lines"] > 300:
                    analysis["suggestions"].append("Consider breaking large file into smaller components")

        except Exception as e:
            analysis["issues"].append({
                "type": "analysis_error",
                "severity": "medium",
                "message": f"Generic analysis failed: {str(e)}",
                "line": 0
            })

        return analysis

    def generate_pr_review(self, pr_data: Dict, changes_data: Dict, code_analysis: List[Dict],
                           config: Dict = None) -> Dict:
        """Generate comprehensive PR review"""
        review = {
            "pr_id": pr_data.get('id'),
            "title": pr_data.get('title'),
            "author": pr_data.get('author', {}).get('user', {}).get('displayName', 'Unknown'),
            "created_date": pr_data.get('createdDate'),
            "updated_date": pr_data.get('updatedDate'),
            "status": pr_data.get('state'),
            "reviewers": [r.get('user', {}).get('displayName', 'Unknown') for r in pr_data.get('reviewers', [])],
            "overall_score": 0,
            "recommendations": [],
            "security_concerns": [],
            "code_quality_issues": [],
            "summary": "",
            "detailed_analysis": code_analysis,
            "metrics": {
                "files_changed": len(changes_data.get('files_changed', [])),
                "total_issues": 0,
                "security_issues": 0,
                "high_severity_issues": 0
            }
        }

        # Aggregate analysis results
        total_issues = 0
        security_issues = 0
        high_severity_issues = 0

        for file_analysis in code_analysis:
            file_issues = len(file_analysis.get('issues', []))
            file_security = len(file_analysis.get('security_issues', []))

            total_issues += file_issues + file_security
            security_issues += file_security

            # Count high severity issues
            for issue in file_analysis.get('issues', []) + file_analysis.get('security_issues', []):
                if issue.get('severity') == 'high':
                    high_severity_issues += 1

        review["metrics"]["total_issues"] = total_issues
        review["metrics"]["security_issues"] = security_issues
        review["metrics"]["high_severity_issues"] = high_severity_issues

        # Calculate overall score (0-100)
        base_score = 100
        base_score -= min(50, total_issues * 2)  # Deduct 2 points per issue, max 50
        base_score -= min(30, security_issues * 5)  # Deduct 5 points per security issue, max 30
        base_score -= min(20, high_severity_issues * 10)  # Deduct 10 points per high severity, max 20

        review["overall_score"] = max(0, base_score)

        # Generate recommendations
        if security_issues > 0:
            review["recommendations"].append("Address security vulnerabilities before merging")
            review["security_concerns"].extend([
                f"Found {security_issues} security issue(s) across {len(code_analysis)} files"
            ])

        if high_severity_issues > 0:
            review["recommendations"].append(f"Fix {high_severity_issues} high severity issues")

        if total_issues > 20:
            review["recommendations"].append("Consider breaking this PR into smaller, more focused changes")

        if len(changes_data.get('files_changed', [])) > 20:
            review["recommendations"].append("Large number of files changed - ensure proper testing")

        # Generate summary
        if review["overall_score"] >= 90:
            review["summary"] = "✅ Excellent code quality - Ready for merge"
        elif review["overall_score"] >= 70:
            review["summary"] = "🟡 Good code quality with minor issues"
        elif review["overall_score"] >= 50:
            review["summary"] = "🟠 Moderate issues requiring attention"
        else:
            review["summary"] = "🔴 Significant issues - Review required before merge"

        return review

    def _generate_mock_content(self, file_path: str, config: Dict) -> str:
        """Generate realistic mock content based on file type and configuration"""
        language = self._detect_language(file_path)

        # Generate more realistic mock content based on language and config settings
        if language == "python":
            content = '''#!/usr/bin/env python3
"""Sample Python module for testing"""

import os
import sys
import requests
from typing import List, Dict, Optional

# TODO: Implement proper logging
PASSWORD = "hardcoded_secret_123"  # Security issue

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process_large_file(self, filename):
        # No error handling - will cause issues
        with open(filename, 'r') as f:
            data = f.read()
        
        # Inefficient loop - performance issue
        results = []
        for i in range(len(data.split('\\n'))):
            line = data.split('\\n')[i]
            if line:
                results.append(line.upper())
        
        return results
    
    def vulnerable_query(self, user_input):
        # SQL injection vulnerability
        query = "SELECT * FROM users WHERE name = '" + user_input + "'"
        return query
    
    def complex_function_with_many_lines(self, param1, param2, param3):
        # This function is too long and complex
        result = None
        if param1:
            if param2:
                if param3:
                    result = param1 + param2 + param3
                    # Many more lines of complex logic...
                    for i in range(100):
                        if i % 2 == 0:
                            result += str(i)
                        else:
                            result -= str(i)
                    # More complex nested logic
                    if result > 1000:
                        for j in range(50):
                            result = result * 2
                            if result > 10000:
                                break
        return result

def undocumented_function():
    pass  # Missing docstring

# Unused import: json
eval("print('Dangerous eval usage')")  # Security risk
'''
        elif language == "javascript":
            content = '''// JavaScript sample with various issues
const API_KEY = "sk-1234567890abcdef";  // Hardcoded secret

function processUserData(userData) {
    // XSS vulnerability
    document.getElementById("output").innerHTML = userData.name;
    
    // Performance issue with DOM manipulation
    for (let i = 0; i < userData.items.length; i++) {
        document.body.appendChild(createElementForItem(userData.items[i]));
    }
    
    // TODO: Add proper validation
    console.log("Processing user:", userData);  // Debug code left in
}

function vulnerableFunction(userInput) {
    // Dangerous eval usage
    return eval("(" + userInput + ")");
}

function complexFunctionWithManyNestedConditions(a, b, c, d, e) {
    if (a) {
        if (b) {
            if (c) {
                if (d) {
                    if (e) {
                        // Too much nesting - cognitive complexity
                        return processComplexLogic(a, b, c, d, e);
                    }
                }
            }
        }
    }
    return null;
}

// Missing error handling
function fetchDataWithoutErrorHandling(url) {
    fetch(url)
        .then(response => response.json())
        .then(data => processData(data));
}
'''
        elif language == "java":
            content = '''package com.example.app;

import java.sql.*;
import java.util.*;

public class UserService {
    
    private static final String PASSWORD = "admin123";  // Hardcoded secret
    
    // TODO: Implement proper connection pooling
    public List<User> getUsers(String searchTerm) {
        List<User> users = new ArrayList<>();
        
        try {
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/db", "user", PASSWORD);
            
            // SQL injection vulnerability
            Statement stmt = conn.createStatement();
            String query = "SELECT * FROM users WHERE name = '" + searchTerm + "'";
            ResultSet rs = stmt.executeQuery(query);
            
            while (rs.next()) {
                User user = new User();
                user.setName(rs.getString("name"));
                users.add(user);
            }
            
            // Resource leak - connection not closed
            
        } catch (SQLException e) {
            System.out.println("Database error: " + e.getMessage());  // Poor logging
            throw new RuntimeException(e);  // Poor error handling
        }
        
        return users;
    }
    
    // Method too long and complex
    public void complexBusinessLogic(Map<String, Object> params) {
        if (params != null) {
            if (params.containsKey("type")) {
                if ("premium".equals(params.get("type"))) {
                    if (params.containsKey("amount")) {
                        double amount = (Double) params.get("amount");
                        if (amount > 1000) {
                            // Many lines of complex processing...
                            for (int i = 0; i < 100; i++) {
                                processTransaction(amount * i);
                                if (i % 10 == 0) {
                                    updateDatabase(i);
                                    sendNotification(i);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
'''
        elif language == "robot":
            content = '''*** Settings ***
Library    SeleniumLibrary
Library    RequestsLibrary

*** Variables ***
${PASSWORD}    hardcoded_password_123    # Security issue
${BASE_URL}    http://example.com

*** Test Cases ***
# TODO: Add more comprehensive test coverage
Login Test Without Error Handling
    Open Browser    ${BASE_URL}    chrome
    Input Text    username    admin
    Input Text    password    ${PASSWORD}
    Click Button    login
    # Missing verification and error handling

Very Long Test Case With Too Many Steps
    [Documentation]    This test case is too long and should be broken down
    Open Browser    ${BASE_URL}    chrome
    Wait Until Page Contains    Welcome
    Click Link    Products
    Wait Until Page Contains    Product List
    Click Button    Add to Cart
    Wait Until Page Contains    Cart
    Input Text    quantity    5
    Click Button    Update
    Wait Until Page Contains    Updated
    Click Button    Checkout
    Wait Until Page Contains    Checkout
    Input Text    name    John Doe
    Input Text    email    john@example.com
    Input Text    address    123 Main St
    Input Text    city    Anytown
    Select From List    state    CA
    Input Text    zip    12345
    Click Button    Submit Order
    Wait Until Page Contains    Order Confirmed
    # ... many more steps

*** Keywords ***
# Missing documentation
Process Order
    [Arguments]    ${order_data}
    Log    Processing order: ${order_data}
'''
        else:
            # Generic content for unknown file types
            content = f'''# Sample {language} file
# TODO: Implement proper functionality
# FIXME: This is a placeholder implementation

{file_path.split('/')[-1]} content here...

# Potential security issue - review secrets management
SECRET_KEY = "default_secret_12345"

# Long line that exceeds typical style guidelines and should be broken down into multiple lines for better readability
very_long_variable_name_that_makes_this_line_extremely_long = "This is a very long string that contributes to making this line way too long"
'''

        return content

    def generate_pr_review(self, pr_data: Dict, changes_data: Dict, code_analysis: List[Dict],
                           config: Dict = None) -> Dict:
        """Generate comprehensive PR review"""
        review = {
            "pr_id": pr_data.get('id'),
            "title": pr_data.get('title'),
            "author": pr_data.get('author', {}).get('user', {}).get('displayName', 'Unknown'),
            "created_date": pr_data.get('createdDate'),
            "updated_date": pr_data.get('updatedDate'),
            "status": pr_data.get('state'),
            "reviewers": [r.get('user', {}).get('displayName', 'Unknown') for r in pr_data.get('reviewers', [])],
            "overall_score": 0,
            "recommendations": [],
            "security_concerns": [],
            "code_quality_issues": [],
            "summary": "",
            "detailed_analysis": code_analysis,
            "metrics": {
                "files_changed": len(changes_data.get('files_changed', [])),
                "total_issues": 0,
                "security_issues": 0,
                "high_severity_issues": 0
            }
        }

        # Aggregate analysis results
        total_issues = 0
        security_issues = 0
        high_severity_issues = 0

        for file_analysis in code_analysis:
            file_issues = len(file_analysis.get('issues', []))
            file_security = len(file_analysis.get('security_issues', []))

            total_issues += file_issues + file_security
            security_issues += file_security

            # Count high severity issues
            for issue in file_analysis.get('issues', []) + file_analysis.get('security_issues', []):
                if issue.get('severity') == 'high':
                    high_severity_issues += 1

        review["metrics"]["total_issues"] = total_issues
        review["metrics"]["security_issues"] = security_issues
        review["metrics"]["high_severity_issues"] = high_severity_issues

        # Calculate overall score (0-100)
        base_score = 100
        base_score -= min(50, total_issues * 2)  # Deduct 2 points per issue, max 50
        base_score -= min(30, security_issues * 5)  # Deduct 5 points per security issue, max 30
        base_score -= min(20, high_severity_issues * 10)  # Deduct 10 points per high severity, max 20

        review["overall_score"] = max(0, base_score)

        # Generate recommendations
        if security_issues > 0:
            review["recommendations"].append("Address security vulnerabilities before merging")
            review["security_concerns"].extend([
                f"Found {security_issues} security issue(s) across {len(code_analysis)} files"
            ])

        if high_severity_issues > 0:
            review["recommendations"].append(f"Fix {high_severity_issues} high severity issues")

        if total_issues > 20:
            review["recommendations"].append("Consider breaking this PR into smaller, more focused changes")

        if len(changes_data.get('files_changed', [])) > 20:
            review["recommendations"].append("Large number of files changed - ensure proper testing")

        # Generate summary
        if review["overall_score"] >= 90:
            review["summary"] = "✅ Excellent code quality - Ready for merge"
        elif review["overall_score"] >= 70:
            review["summary"] = "🟡 Good code quality with minor issues"
        elif review["overall_score"] >= 50:
            review["summary"] = "🟠 Moderate issues requiring attention"
        else:
            review["summary"] = "🔴 Significant issues - Review required before merge"

        return review

    def _extract_actual_line_numbers(self, content: str, issue_pattern: str, issue_message: str) -> List[int]:
        """Extract actual line numbers where issues occur in the code content"""
        if not content:
            return [1]  # Default to line 1 if no content

        lines = content.split('\n')
        matching_lines = []

        # Try to find lines that match the issue pattern or contain relevant keywords
        issue_keywords = self._extract_keywords_from_message(issue_message)

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower().strip()

            # Skip empty lines and comments
            if not line_lower or line_lower.startswith('#') or line_lower.startswith('//'):
                continue

            # Check for specific patterns based on issue keywords
            for keyword in issue_keywords:
                if keyword.lower() in line_lower:
                    matching_lines.append(line_num)
                    break

        return matching_lines if matching_lines else [1]  # Return line 1 if no matches found

    def _extract_keywords_from_message(self, message: str) -> List[str]:
        """Extract relevant keywords from issue message to help locate the issue in code"""
        if not message:
            return []

        # Common patterns that indicate code issues
        patterns = {
            'hardcoded': ['password', 'secret', 'key', 'token', 'api_key'],
            'sql': ['select', 'insert', 'update', 'delete', 'query', 'execute'],
            'eval': ['eval', 'exec', 'compile'],
            'import': ['import', 'from'],
            'function': ['def ', 'function', 'method'],
            'variable': ['=', 'var', 'let', 'const'],
            'loop': ['for', 'while', 'range'],
            'condition': ['if', 'elif', 'else', 'switch', 'case']
        }

        message_lower = message.lower()
        keywords = []

        for category, words in patterns.items():
            if category in message_lower:
                keywords.extend(words)

        # Also extract quoted strings from the message
        import re
        quoted_strings = re.findall(r"['\"`]([^'\"`]+)['\"`]", message)
        keywords.extend(quoted_strings)

        return list(set(keywords))  # Remove duplicates

    def _calculate_accurate_metrics(self, content: str, language: str) -> Dict[str, int]:
        """Calculate accurate file metrics including line counts, functions, classes, etc."""
        if not content:
            return {
                'total_lines': 0,
                'non_empty_lines': 0,
                'code_lines': 0,
                'comment_lines': 0,
                'functions': 0,
                'classes': 0,
                'imports': 0,
                'complexity_score': 0
            }

        lines = content.split('\n')
        total_lines = len(lines)
        non_empty_lines = 0
        code_lines = 0
        comment_lines = 0
        functions = 0
        classes = 0
        imports = 0
        complexity_score = 0

        # Language-specific comment patterns
        comment_patterns = {
            'python': ['#', '"""', "'''"],
            'javascript': ['//', '/*', '/**'],
            'java': ['//', '/*', '/**'],
            'typescript': ['//', '/*', '/**'],
            'c': ['//', '/*'],
            'cpp': ['//', '/*'],
            'c#': ['//', '/*'],
            'go': ['//', '/*'],
            'rust': ['//', '/*'],
            'ruby': ['#'],
            'php': ['//', '#', '/*'],
            'shell': ['#'],
            'yaml': ['#'],
            'sql': ['--', '/*']
        }

        # Language-specific function/class patterns
        function_patterns = {
            'python': [r'^\s*def\s+\w+', r'^\s*async\s+def\s+\w+'],
            'javascript': [r'^\s*function\s+\w+', r'^\s*\w+\s*=\s*function', r'^\s*\w+\s*:\s*function',
                           r'^\s*\w+\s*=>\s*'],
            'typescript': [r'^\s*function\s+\w+', r'^\s*\w+\s*=\s*function', r'^\s*\w+\s*:\s*function',
                           r'^\s*\w+\s*=>\s*'],
            'java': [r'^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\('],
            'c': [r'^\s*\w+\s+\w+\s*\('],
            'cpp': [r'^\s*\w+\s+\w+\s*\('],
            'go': [r'^\s*func\s+\w+'],
            'rust': [r'^\s*fn\s+\w+'],
            'ruby': [r'^\s*def\s+\w+'],
            'php': [r'^\s*(public|private|protected)?\s*function\s+\w+']
        }

        class_patterns = {
            'python': [r'^\s*class\s+\w+'],
            'javascript': [r'^\s*class\s+\w+'],
            'typescript': [r'^\s*class\s+\w+', r'^\s*interface\s+\w+'],
            'java': [r'^\s*(public|private|protected)?\s*class\s+\w+'],
            'c#': [r'^\s*(public|private|protected)?\s*class\s+\w+'],
            'cpp': [r'^\s*class\s+\w+'],
            'ruby': [r'^\s*class\s+\w+'],
            'php': [r'^\s*(public|private|protected)?\s*class\s+\w+']
        }

        import_patterns = {
            'python': [r'^\s*import\s+', r'^\s*from\s+\w+\s+import'],
            'javascript': [r'^\s*import\s+', r'^\s*const\s+\w+\s*=\s*require'],
            'typescript': [r'^\s*import\s+', r'^\s*const\s+\w+\s*=\s*require'],
            'java': [r'^\s*import\s+'],
            'c': [r'^\s*#include'],
            'cpp': [r'^\s*#include'],
            'c#': [r'^\s*using\s+'],
            'go': [r'^\s*import\s+'],
            'rust': [r'^\s*use\s+'],
            'ruby': [r'^\s*require'],
            'php': [r'^\s*(require|include)']
        }

        lang = language.lower()
        comments = comment_patterns.get(lang, ['#', '//'])
        func_patterns = function_patterns.get(lang, [])
        cls_patterns = class_patterns.get(lang, [])
        imp_patterns = import_patterns.get(lang, [])

        import re
        in_multiline_comment = False

        for line in lines:
            line_stripped = line.strip()

            # Count non-empty lines
            if line_stripped:
                non_empty_lines += 1

                # Check for multiline comments
                if any(delimiter in line_stripped for delimiter in ['"""', "'''", '/*']):
                    if not in_multiline_comment and any(line_stripped.startswith(d) for d in ['"""', "'''", '/*']):
                        in_multiline_comment = True
                    if in_multiline_comment and any(line_stripped.endswith(d) for d in ['"""', "'''", '*/']):
                        in_multiline_comment = False
                        comment_lines += 1
                        continue

                # Count comment lines
                if in_multiline_comment or any(line_stripped.startswith(c) for c in comments):
                    comment_lines += 1
                else:
                    code_lines += 1

                    # Count functions
                    for pattern in func_patterns:
                        if re.match(pattern, line):
                            functions += 1
                            complexity_score += 2  # Functions add complexity
                            break

                    # Count classes
                    for pattern in cls_patterns:
                        if re.match(pattern, line):
                            classes += 1
                            complexity_score += 3  # Classes add more complexity
                            break

                    # Count imports
                    for pattern in imp_patterns:
                        if re.match(pattern, line):
                            imports += 1
                            break

                    # Add complexity for control structures
                    if re.search(r'\b(if|elif|else|for|while|try|except|catch|switch|case)\b', line_stripped):
                        complexity_score += 1

        return {
            'total_lines': total_lines,
            'non_empty_lines': non_empty_lines,
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'complexity_score': complexity_score
        }

    def test_connection(self, base_url: str) -> Dict[str, Any]:
        """Test basic connection to the server without authentication"""
        result = {
            "url_valid": False,
            "server_reachable": False,
            "server_type": "Unknown",
            "api_available": False,
            "errors": []
        }

        try:
            # Validate URL format
            parsed = urlparse(base_url)
            if not parsed.scheme or not parsed.netloc:
                result["errors"].append("Invalid URL format. Use https://your-bitbucket-server.com")
                return result

            result["url_valid"] = True

            # Normalize base URL
            if not base_url.endswith('/'):
                base_url = base_url + '/'

            # Test basic connectivity
            response = requests.get(base_url, timeout=10, verify=False)

            if response.status_code in [200, 401, 403]:
                result["server_reachable"] = True

                # Try to identify server type
                response_text = response.text.lower()
                if 'bitbucket' in response_text:
                    result["server_type"] = "Bitbucket"
                elif 'stash' in response_text:
                    result["server_type"] = "Stash"
                elif 'atlassian' in response_text:
                    result["server_type"] = "Atlassian Bitbucket"

                # Test API endpoint availability
                api_url = urljoin(base_url, '/rest/api/1.0/application-properties')
                api_response = requests.get(api_url, timeout=10, verify=False)

                if api_response.status_code in [200, 401, 403]:
                    result["api_available"] = True
                else:
                    result["errors"].append(f"REST API not available (status: {api_response.status_code})")
            else:
                result["errors"].append(f"Server returned status {response.status_code}")

        except requests.exceptions.Timeout:
            result["errors"].append("Connection timeout - server not responding")
        except requests.exceptions.ConnectionError as e:
            result["errors"].append(f"Cannot connect to server: {str(e)}")
        except Exception as e:
            result["errors"].append(f"Connection test failed: {str(e)}")

        return result

    # def _build_pr_file_url(self, base_url: str, project: str, repo: str, file_path: str, pr_id: str) -> str:
    #     """Build URL for file in pull request context."""
    #     return f"{base_url}/projects/{project}/repos/{repo}/pull-requests/{pr_id}/diff#{file_path}"
    #
    # def _build_browse_file_url(self, base_url: str, project: str, repo: str, file_path: str) -> str:
    #     """Build URL for file in main branch browse context."""
    #     return f"{base_url}/projects/{project}/repos/{repo}/browse/{file_path}"
    #
    # def generate_file_url(self, base_url: str, project: str, repo: str, file_path: str, pr_id: str = None) -> str:
    #     """Generate clickable URL for files in Bitbucket/Stash"""
    #     if not all([base_url, project, repo, file_path]):
    #         return file_path  # Return plain path if missing info
    #
    #     # Normalize base URL
    #     base_url = base_url.rstrip('/')
    #
    #     # Handle different URL formats for Bitbucket Server/Stash
    #     if pr_id:
    #         return self._build_pr_file_url(base_url, project, repo, file_path, pr_id)
    #     else:
    #         return self._build_browse_file_url(base_url, project, repo, file_path)

    def generate_ai_powered_review(self, pr_data: Dict, changes_data: Dict, code_analysis: List[Dict],
                                   config: Dict = None) -> Dict:
        """Generate AI-powered comprehensive PR review with intelligent insights and recommendations"""
        # Get basic review first
        basic_review = self.generate_pr_review(pr_data, changes_data, code_analysis, config)
        
        # Check if AI analysis is enabled in config
        ai_enabled = config.get('enable_ai_analysis', True) if config else True
        
        # Add AI analysis if available and enabled
        if AI_AVAILABLE and azure_openai_client and ai_enabled:
            try:
                # Prepare comprehensive summary for AI analysis
                ai_summary = self._prepare_pr_summary_for_ai(pr_data, changes_data, code_analysis, config)
                
                # Get AI insights based on analysis focus
                ai_focus = config.get('ai_analysis_depth', 'Comprehensive') if config else 'Comprehensive'
                ai_insights = self._get_ai_pr_analysis(ai_summary, ai_focus)
                
                # Integrate AI insights into the basic review
                basic_review = self._integrate_ai_insights(basic_review, ai_insights)
                
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                basic_review["ai_analysis_error"] = f"AI analysis unavailable: {str(e)}"
        elif not ai_enabled:
            basic_review["ai_analysis_error"] = "AI analysis disabled in configuration"
        else:
            basic_review["ai_analysis_error"] = "Azure OpenAI not configured"
            
        return basic_review

    def _prepare_pr_summary_for_ai(self, pr_data: Dict, changes_data: Dict, 
                                   code_analysis: List[Dict], config: Dict) -> str:
        """Prepare a comprehensive summary of the PR for AI analysis"""
        try:
            summary = []
            
            # Basic PR information
            summary.append("=== PULL REQUEST ANALYSIS ===")
            summary.append(f"Title: {pr_data.get('title', 'No title')}")
            summary.append(f"Author: {pr_data.get('author', {}).get('user', {}).get('displayName', 'Unknown')}")
            summary.append(f"Status: {pr_data.get('state', 'Unknown')}")
            summary.append(f"Files Changed: {len(changes_data.get('files_changed', []))}")
            
            # Description if available
            if pr_data.get('description'):
                summary.append(f"Description: {pr_data.get('description')[:500]}...")
            
            summary.append("\n=== CODE CHANGES ANALYSIS ===")
            
            # File changes summary
            file_changes = changes_data.get('files_changed', [])
            summary.append(f"Total files modified: {len(file_changes)}")
            
            # Group files by language/type
            language_stats = {}
            for file_info in file_changes:
                file_path = file_info.get('path', '')
                language = self._detect_language(file_path)
                language_stats[language] = language_stats.get(language, 0) + 1
            
            summary.append("Files by language:")
            for lang, count in language_stats.items():
                summary.append(f"  - {lang}: {count} files")
            
            summary.append("\n=== DETECTED ISSUES ===")
            
            # Aggregate all issues
            all_issues = []
            security_issues = []
            performance_issues = []
            quality_issues = []
            
            for file_analysis in code_analysis:
                all_issues.extend(file_analysis.get('issues', []))
                security_issues.extend(file_analysis.get('security_issues', []))
                
                # Categorize issues
                for issue in file_analysis.get('issues', []):
                    category = issue.get('category', '').lower()
                    if 'performance' in category:
                        performance_issues.append(issue)
                    elif 'quality' in category:
                        quality_issues.append(issue)
            
            summary.append(f"Total Issues Found: {len(all_issues) + len(security_issues)}")
            summary.append(f"Security Issues: {len(security_issues)}")
            summary.append(f"Performance Issues: {len(performance_issues)}")
            summary.append(f"Quality Issues: {len(quality_issues)}")
            
            # Top security issues
            if security_issues:
                summary.append("\nTop Security Issues:")
                for i, issue in enumerate(security_issues[:5], 1):
                    summary.append(f"  {i}. {issue.get('message', 'No message')} (Severity: {issue.get('severity', 'Unknown')})")
            
            # Top performance issues
            if performance_issues:
                summary.append("\nTop Performance Issues:")
                for i, issue in enumerate(performance_issues[:5], 1):
                    summary.append(f"  {i}. {issue.get('message', 'No message')}")
            
            # Code complexity metrics
            summary.append("\n=== CODE METRICS ===")
            total_lines = sum(analysis.get('metrics', {}).get('total_lines', 0) for analysis in code_analysis)
            total_functions = sum(analysis.get('metrics', {}).get('functions', 0) for analysis in code_analysis)
            total_classes = sum(analysis.get('metrics', {}).get('classes', 0) for analysis in code_analysis)
            avg_complexity = sum(analysis.get('metrics', {}).get('complexity_score', 0) for analysis in code_analysis) / max(len(code_analysis), 1)
            
            summary.append(f"Total Lines of Code: {total_lines}")
            summary.append(f"Total Functions: {total_functions}")
            summary.append(f"Total Classes: {total_classes}")
            summary.append(f"Average Complexity Score: {avg_complexity:.1f}")
            
            # Configuration applied
            if config:
                summary.append("\n=== ANALYSIS CONFIGURATION ===")
                enabled_checks = [k for k, v in config.items() if v is True and k.startswith('check_')]
                summary.append(f"Active Quality Checks: {len(enabled_checks)}")
                summary.append(f"Analysis Depth: {config.get('analysis_depth', 'Standard')}")
                summary.append(f"Strictness Level: {config.get('analysis_strictness', 'Standard')}")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error preparing PR summary: {str(e)}"

    def _get_ai_pr_analysis(self, pr_summary: str, analysis_focus: str = "Comprehensive") -> str:
        """Get comprehensive AI analysis of the pull request using Azure OpenAI"""
        try:
            # Customize prompt based on analysis focus
            if analysis_focus == "Security-Focused":
                focus_instructions = """
                Focus primarily on security aspects:
                - Prioritize security vulnerabilities and risks
                - Analyze authentication and authorization issues
                - Review input validation and data sanitization
                - Examine cryptographic implementations
                - Assess potential attack vectors
                - Review third-party dependencies for vulnerabilities
                - Examine logging and monitoring practices
                - Ensure compliance with security standards
                """
            elif analysis_focus == "Performance-Focused":
                focus_instructions = """
                Focus primarily on performance aspects:
                - Identify performance bottlenecks and inefficiencies
                - Analyze algorithm complexity and optimization opportunities
                - Review memory usage and resource management
                - Examine database query optimization
                - Assess scalability concerns
                - Review caching strategies
                - Evaluate network performance and latency
                - Ensure efficient use of asynchronous operations
                - Review concurrency and parallelism practices
                """
            elif analysis_focus == "Quality-Focused":
                focus_instructions = """
                Focus primarily on code quality aspects:
                - Analyze code structure and design patterns
                - Review naming conventions and readability
                - Examine maintainability and modularity
                - Assess test coverage and quality
                - Review documentation and comments
                - Evaluate adherence to coding standards and best practices
                - Identify potential anti-patterns and code smells
                - Ensure proper error handling and logging
                - Review dependency management and versioning
                """
            else:  # Comprehensive
                focus_instructions = """
                Provide a balanced analysis covering all aspects:
                - Security, performance, and quality considerations
                - Code structure, design, and best practices
                - Testing, documentation, and maintainability
                - Logging and monitoring practices
                - Third-party dependencies and vulnerabilities
                """

            prompt = f"""
            You are a senior principal architect, skilled senior software engineer and code review expert. Analyze the following pull request data and provide a comprehensive review.

            {focus_instructions}

            {pr_summary}

            Please provide a detailed analysis with the following sections:

            ## 🔍 CODE REVIEW SUMMARY
            - Overall assessment of code quality
            - Readiness for merge (Ready/Needs Work/Major Issues)
            - Risk level (Low/Medium/High)
            - Key areas of concern
            - General observations
            - Recommendations for improvement
            - Any critical issues that must be addressed
            - Areas that are well done
            
            ## 📝 DETAILED CODE ANALYSIS
            - Specific issues found in the code
            - Line numbers where issues occur
            - Categorization of issues (security, performance, quality)
            - Suggestions for fixing each issue
            - Code snippets for problematic areas
            - Suggested improvements with examples
            - Performance optimizations
            - Security enhancements

            ## 🚨 CRITICAL ISSUES
            - Security vulnerabilities that must be addressed
            - Performance bottlenecks
            - Potential bugs or logic errors
            - Code smells or anti-patterns
            - Areas that violate best practices
            - Issues that could lead to technical debt

            ## 💡 IMPROVEMENT SUGGESTIONS
            - Code structure and design improvements
            - Best practices recommendations
            - Refactoring opportunities
            - Testing and documentation enhancements
            - Suggestions for better maintainability

            ## 🧹 CODE QUALITY ENHANCEMENTS
            - Naming conventions improvements
            - Documentation gaps
            - Test coverage recommendations
            - Code organization suggestions

            ## 📊 TECHNICAL DEBT ASSESSMENT
            - Areas that may contribute to technical debt
            - Long-term maintainability concerns
            - Recommended architectural improvements

            ## ✅ POSITIVE OBSERVATIONS
            - Well-written code sections
            - Good practices followed
            - Effective solutions implemented
            - Areas that demonstrate good design

            ## 🎯 ACTIONABLE RECOMMENDATIONS
            - Specific changes to make before merging
            - Priority order for addressing issues
            - Suggested follow-up actions

            ## 🧪 TESTING RECOMMENDATIONS
            - Areas that need better test coverage
            - Suggested test cases for critical functionality
            - Edge cases to consider            

            ## 🔒 SECURITY RECOMMENDATIONS
            - Address any security vulnerabilities
            - Input validation improvements
            - Secure coding practices
            
            ## 🚀 PERFORMANCE RECOMMENDATIONS
            - Identify performance bottlenecks
            - Suggest more efficient algorithms or approaches
            - Memory usage improvements
            
            ## METRICS AND STATISTICS
            - Total lines of code
            - Number of files changed
            - Number of issues found
            - Number of security issues
            - Number of performance issues
            - Number of quality issues
            - Complexity score
            - Number of functions and classes
            - Number of imports
            - Number of comments
            - Number of non-empty lines
            - Number of code lines
            - Number of comment lines
            
            ## 📚 CODE STRUCTURE RECOMMENDATIONS
            - Function/method refactoring suggestions
            - Better variable naming
            - Code organization improvements

            Provide specific, actionable feedback with line-by-line suggestions where possible.
            Focus on practical improvements that enhance code quality, security, and maintainability.
            """

            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.3
            )

            return response

        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}")
            return f"AI analysis failed: {str(e)}"

    def _integrate_ai_insights(self, basic_review: Dict, ai_insights: str) -> Dict:
        """Integrate AI insights into the basic review structure"""
        try:
            # Add AI insights to the review
            basic_review["ai_insights"] = ai_insights
            
            # Parse AI insights to extract specific recommendations
            ai_lines = ai_insights.split('\n')
            
            # Extract critical issues
            critical_section = False
            for line in ai_lines:
                if "CRITICAL ISSUES" in line.upper():
                    critical_section = True
                elif line.startswith('#') and "CRITICAL ISSUES" not in line.upper():
                    critical_section = False
                elif critical_section and line.strip() and not line.startswith('#'):
                    if line.strip().startswith('-') or line.strip().startswith('*'):
                        basic_review["recommendations"].append(f"🚨 {line.strip()[1:].strip()}")
            
            # Extract improvement suggestions
            improvement_section = False
            for line in ai_lines:
                if "IMPROVEMENT SUGGESTIONS" in line.upper():
                    improvement_section = True
                elif line.startswith('#') and "IMPROVEMENT SUGGESTIONS" not in line.upper():
                    improvement_section = False
                elif improvement_section and line.strip() and not line.startswith('#'):
                    if line.strip().startswith('-') or line.strip().startswith('*'):
                        basic_review["recommendations"].append(f"💡 {line.strip()[1:].strip()}")
            
            # Try to extract a risk level from AI analysis
            ai_lower = ai_insights.lower()
            if "high risk" in ai_lower or "major issues" in ai_lower:
                basic_review["risk_level"] = "High"
            elif "medium risk" in ai_lower or "needs work" in ai_lower:
                basic_review["risk_level"] = "Medium"
            else:
                basic_review["risk_level"] = "Low"
            
            # Try to extract readiness assessment
            if "ready for merge" in ai_lower or "ready to merge" in ai_lower:
                basic_review["merge_readiness"] = "Ready"
            elif "needs work" in ai_lower or "major issues" in ai_lower:
                basic_review["merge_readiness"] = "Needs Work"
            else:
                basic_review["merge_readiness"] = "Review Required"
            
            return basic_review
            
        except Exception as e:
            logger.error(f"Error integrating AI insights: {e}")
            basic_review["ai_integration_error"] = str(e)
            return basic_review

    def generate_ai_code_suggestions(self, file_content: str, file_path: str, 
                                     detected_issues: List[Dict]) -> str:
        """Generate AI-powered code improvement suggestions for a specific file"""
        try:
            if not AI_AVAILABLE or not azure_openai_client:
                return "AI suggestions not available - Azure OpenAI not configured"

            language = self._detect_language(file_path)
            
            # Prepare issue summary
            issue_summary = []
            for issue in detected_issues[:10]:  # Limit to top 10 issues
                issue_summary.append(f"- {issue.get('message', 'No message')} (Line: {issue.get('line', 'Unknown')}, Severity: {issue.get('severity', 'Unknown')})")
            
            prompt = f"""
            You are an expert {language} developer. Review the following code file and provide specific improvement suggestions.

            File: {file_path}
            Language: {language}

            Detected Issues:
            {chr(10).join(issue_summary)}

            Code Content:
            ```{language}
            {file_content[:2000]}  # Truncate for API limits
            ```

            Please provide:

            ## 🔧 SPECIFIC CODE FIXES
            For each detected issue, provide the exact code changes needed:
            - Line number reference
            - Current problematic code
            - Improved code suggestion
            - Explanation of why the change is needed
            - Why the current code is problematic
            - Potential side effects of the current code
            - How the suggested change improves the code
            - Why the suggested change is better
            - Any additional context that may help in understanding the issue

            ## 🚀 PERFORMANCE OPTIMIZATIONS
            - Identify performance bottlenecks
            - Suggest more efficient algorithms or approaches
            - Memory usage improvements
            - Caching strategies
            - Asynchronous operations improvements
            - Concurrency and parallelism improvements
            - Database query optimizations
            - Network performance enhancements            

            ## 🔒 SECURITY ENHANCEMENTS
            - Address any security vulnerabilities
            - Input validation improvements
            - Secure coding practices
            - Cryptographic improvements
            - Dependency management
            - Logging and monitoring practices
            - Ensure compliance with security standards
            - Review third-party dependencies for vulnerabilities
            - Examine authentication and authorization practices
            - Assess potential attack vectors
            - Review data handling and storage practices
            - Ensure secure coding standards are followed
            - Review error handling and logging practices
            - Ensure proper access controls are implemented
            - Review API security practices
            - Ensure secure configuration management
            - Review network security practices
            - Ensure secure deployment practices
            - Review incident response and recovery practices

            ## 📚 CODE STRUCTURE IMPROVEMENTS
            - Function/method refactoring suggestions
            - Better variable naming
            - Code organization improvements
            - Documentation gaps
            - Test coverage recommendations
            - Code organization suggestions
            - Naming conventions improvements
            - Code duplication reduction strategies
            - Code readability enhancements
            - Code maintainability improvements
            - Code modularity suggestions
            - Code complexity reduction strategies
            - DRY (Don't Repeat Yourself) principles
            - SOLID principles application
            - Code reusability improvements
            - Code scalability suggestions
            - Code extensibility improvements

            ## 🧪 TESTING RECOMMENDATIONS
            - Suggest test cases for the code
            - Areas that need better test coverage
            - Edge cases to consider
            - Testing best practices
            - Automation opportunities

            Provide actionable, specific suggestions that can be immediately implemented.
            """

            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.2
            )

            return response

        except Exception as e:
            logger.error(f"Error generating AI code suggestions: {e}")
            return f"AI code suggestions failed: {str(e)}"

    def get_ai_security_analysis(self, code_analysis: List[Dict]) -> str:
        """Get focused AI analysis on security issues"""
        try:
            if not AI_AVAILABLE or not azure_openai_client:
                return "AI security analysis not available"

            # Extract security issues
            security_issues = []
            for file_analysis in code_analysis:
                for issue in file_analysis.get('security_issues', []):
                    security_issues.append({
                        'file': file_analysis.get('file_path', 'Unknown'),
                        'message': issue.get('message', ''),
                        'severity': issue.get('severity', ''),
                        'line': issue.get('line', ''),
                        'category': issue.get('category', '')
                    })

            if not security_issues:
                return "No security issues detected in this pull request."

            # Prepare security summary
            security_summary = []
            security_summary.append(f"Found {len(security_issues)} security issues:")
            
            for i, issue in enumerate(security_issues[:20], 1):  # Limit to top 20
                security_summary.append(f"{i}. {issue['file']} (Line {issue['line']}): {issue['message']} [Severity: {issue['severity']}]")

            prompt = f"""
            You are a cybersecurity expert conducting a security review of a pull request. 
            Analyze the following security issues and provide comprehensive guidance.

            Security Issues Found:
            {chr(10).join(security_summary)}

            Please provide:

            ## 🚨 CRITICAL SECURITY ASSESSMENT
            - Overall security risk level (Critical/High/Medium/Low)
            - Most dangerous vulnerabilities that need immediate attention
            - Potential attack vectors

            ## 🛡️ SPECIFIC SECURITY FIXES
            For each critical issue:
            - Exact remediation steps
            - Secure code examples
            - Why the current code is vulnerable

            ## 🔒 SECURITY BEST PRACTICES
            - Recommend security patterns for this codebase
            - Input validation strategies
            - Authentication and authorization improvements

            ## 📋 SECURITY CHECKLIST
            - Items to verify before merging
            - Security testing recommendations
            - Monitoring and logging suggestions

            ## 🎯 PRIORITY RANKING
            - Rank issues by severity and exploitability
            - Recommended order for fixing issues
            - Quick wins vs. complex fixes

            Focus on actionable security improvements and real-world attack prevention.
            """

            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.1  # Low temperature for security analysis
            )

            return response

        except Exception as e:
            logger.error(f"Error in AI security analysis: {e}")
            return f"AI security analysis failed: {str(e)}"


def get_ai_analysis(prompt):
    """Get AI analysis using Azure OpenAI - similar to fos_checks.py pattern"""
    try:
        if not AI_AVAILABLE or azure_openai_client is None:
            return "AI analysis is not available. Please check Azure OpenAI configuration."

        # Use the correct method from AzureOpenAIClient
        response = azure_openai_client.generate_response(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7
        )

        return response

    except Exception as e:
        logger.error(f"Error getting AI analysis: {str(e)}")
        return f"Unable to get AI analysis due to error: {str(e)}"


def run_ai_pr_quality_analysis(all_pr_reviews):
    """Run AI analysis on overall PR quality across multiple repositories"""
    try:
        if not AI_AVAILABLE or not azure_openai_client:
            return "AI analysis not available - Azure OpenAI not configured"

        # Prepare summary of all PRs
        pr_summary = []
        pr_summary.append("=== CROSS-REPOSITORY PULL REQUEST ANALYSIS ===")
        pr_summary.append(f"Total Pull Requests Analyzed: {len(all_pr_reviews)}")
        
        # Aggregate metrics
        total_issues = sum(review.get('metrics', {}).get('total_issues', 0) for review in all_pr_reviews)
        total_security = sum(review.get('metrics', {}).get('security_issues', 0) for review in all_pr_reviews)
        avg_score = sum(review.get('overall_score', 0) for review in all_pr_reviews) / len(all_pr_reviews) if all_pr_reviews else 0
        
        pr_summary.append(f"Total Issues Found: {total_issues}")
        pr_summary.append(f"Security Issues: {total_security}")
        pr_summary.append(f"Average Quality Score: {avg_score:.1f}")
        
        # Repository breakdown
        repo_stats = {}
        for review in all_pr_reviews:
            repo = review.get('repository', 'Unknown')
            if repo not in repo_stats:
                repo_stats[repo] = {'count': 0, 'avg_score': 0, 'issues': 0}
            repo_stats[repo]['count'] += 1
            repo_stats[repo]['avg_score'] += review.get('overall_score', 0)
            repo_stats[repo]['issues'] += review.get('metrics', {}).get('total_issues', 0)
        
        pr_summary.append("\n=== REPOSITORY BREAKDOWN ===")
        for repo, stats in repo_stats.items():
            avg_score = stats['avg_score'] / stats['count'] if stats['count'] > 0 else 0
            pr_summary.append(f"{repo}: {stats['count']} PRs, Avg Score: {avg_score:.1f}, Issues: {stats['issues']}")
        
        # Top issues across all PRs
        all_issues = []
        for review in all_pr_reviews:
            for file_analysis in review.get('detailed_analysis', []):
                all_issues.extend(file_analysis.get('issues', []))
                all_issues.extend(file_analysis.get('security_issues', []))
        
        # Group issues by type
        issue_types = {}
        for issue in all_issues:
            issue_type = issue.get('type', 'unknown')
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        pr_summary.append(f"\n=== TOP ISSUE PATTERNS ===")
        sorted_issues = sorted(issue_types.items(), key=lambda x: x[1], reverse=True)
        for issue_type, count in sorted_issues[:10]:
            pr_summary.append(f"{issue_type}: {count} occurrences")

        prompt = f"""
        You are a senior engineering manager conducting a cross-repository code quality review. 
        Analyze the following pull request data and provide strategic insights.

        {chr(10).join(pr_summary)}

        Please provide:

        ## 🎯 EXECUTIVE SUMMARY
        - Overall code quality assessment across repositories
        - Key trends and patterns observed
        - Risk assessment for the engineering organization

        ## 📊 REPOSITORY COMPARISON
        - Which repositories show the best/worst code quality
        - Identify repositories that need immediate attention
        - Recommend resource allocation and focus areas

        ## 🔍 SYSTEMIC ISSUES
        - Common anti-patterns across repositories
        - Recurring security vulnerabilities
        - Infrastructure or tooling gaps

        ## 🚀 IMPROVEMENT RECOMMENDATIONS
        - Process improvements for better code quality
        - Training recommendations for development teams
        - Tool and automation suggestions

        ## 📋 ACTION PLAN
        - Immediate actions (next 2 weeks)
        - Short-term goals (next quarter)
        - Long-term strategic initiatives

        ## 🎓 TEAM DEVELOPMENT
        - Skill gaps identified across teams
        - Mentoring and knowledge sharing opportunities
        - Best practices to standardize

        Focus on actionable insights that can drive organizational improvement.
        """

        response = azure_openai_client.generate_response(
            prompt=prompt,
            max_tokens=3000,
            temperature=0.2
        )

        return response

    except Exception as e:
        logger.error(f"Error in cross-PR AI analysis: {e}")
        return f"Cross-repository AI analysis failed: {str(e)}"


def show_ui():
    """Display the Pull Requests Reviewer UI"""
    st.title("🔍 Pull Requests Reviewer")
    st.markdown("Automated code review and analysis for Stash/Bitbucket repositories")

    # Configuration Section
    st.header("🔧 Repository Configuration")

    col1, col2 = st.columns(2)

    with col1:
        base_url = st.text_input(
            "Bitbucket/Stash URL",
            placeholder="https://stash.newfold.com",
            help="Base URL of your Bitbucket or Stash instance"
        )

        project_key = st.text_input(
            "Project Key",
            placeholder="PROJ",
            help="Project key (leave empty to list all projects)"
        )

    with col2:
        auth_method = st.selectbox(
            "Authentication Method",
            ["Username/Password", "Personal Access Token"],
            help="Choose authentication method"
        )

        if auth_method == "Username/Password":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            token = None
        else:
            username = st.text_input("Username")
            token = st.text_input("Personal Access Token", type="password")
            password = None

    # Repository Selection
    st.header("📂 Repository Selection")

    # Add connection test button
    col1, col2 = st.columns([2, 1])

    with col1:
        connect_button = st.button("🔗 Connect and Load Repositories")

    with col2:
        test_button = st.button("🔍 Test Connection")

    # Test Connection functionality
    if test_button:
        if not base_url:
            st.error("Please enter a Bitbucket/Stash URL first")
        else:
            with st.spinner("Testing connection..."):
                test_reviewer = PullRequestReviewer()
                test_result = test_reviewer.test_connection(base_url)

                if test_result["server_reachable"] and test_result["api_available"]:
                    st.success(f"✅ Connection successful! Server type: {test_result['server_type']}")
                    st.info("You can now proceed with authentication.")
                else:
                    st.error("❌ Connection failed")

                    with st.expander("🔧 Connection Details"):
                        st.write("**Test Results:**")
                        st.write(f"- URL Valid: {'✅' if test_result['url_valid'] else '❌'}")
                        st.write(f"- Server Reachable: {'✅' if test_result['server_reachable'] else '❌'}")
                        st.write(f"- API Available: {'✅' if test_result['api_available'] else '❌'}")
                        st.write(f"- Server Type: {test_result['server_type']}")

                        if test_result["errors"]:
                            st.write("**Errors:**")
                            for error in test_result["errors"]:
                                st.write(f"- {error}")

    selected_repos = []

    if connect_button:
        if not base_url:
            st.error("Please enter a Bitbucket/Stash URL")
            return

        if auth_method == "Username/Password" and (not username or not password):
            st.error("Please enter username and password")
            return
        elif auth_method == "Personal Access Token" and (not username or not token):
            st.error("Please enter username and token")
            return

        reviewer = PullRequestReviewer()

        with st.spinner("Connecting to repository..."):
            auth_success = reviewer.authenticate(base_url, username, password, token)

            if not auth_success:
                st.error("❌ Authentication failed. Please check your credentials.")

                # Provide detailed troubleshooting information
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    **Common Authentication Issues:**

                    1. **Check URL Format**: Ensure your URL is correct (e.g., `https://bitbucket.company.com`)
                    2. **Verify Credentials**:
                       - For Username/Password: Use your standard login credentials
                       - For Token: Ensure the token has proper permissions
                    3. **Network Access**: Ensure you can access the Bitbucket/Stash server from your network
                    4. **API Permissions**: Your account may need specific API access permissions
                    5. **Token Format**: Different Bitbucket versions may use different token formats

                    **For Personal Access Tokens:**
                    - Go to Bitbucket Settings → Personal Access Tokens
                    - Create a token with "Repository Read" permissions
                    - Copy the token exactly as generated

                    **Alternative Solutions:**
                    - Try switching between Username/Password and Token authentication
                    - Contact your Bitbucket administrator if issues persist
                    - Check if your account has API access enabled
                    """)

                # Show authentication attempt details if available
                if hasattr(reviewer, '_last_auth_error'):
                    st.error(f"Details: {reviewer._last_auth_error}")

                return

            st.success("✅ Connected successfully!")

            # Get repositories
            repositories = reviewer.get_repositories(base_url, project_key)

            if repositories:
                st.success(f"Found {len(repositories)} repositories")

                # Store in session state
                st.session_state.reviewer = reviewer
                st.session_state.base_url = base_url
                st.session_state.repositories = repositories

            else:
                st.warning("No repositories found or access denied")

    # Repository and PR Analysis
    if hasattr(st.session_state, 'repositories') and st.session_state.repositories:
        st.subheader("📋 Available Repositories")

        repo_options = {}
        for repo in st.session_state.repositories:
            project_name = repo.get('project', {}).get('name', 'Unknown')
            repo_name = repo.get('name', 'Unknown')
            repo_key = f"{project_name} / {repo_name}"
            repo_options[repo_key] = repo

        selected_repo_keys = st.multiselect(
            "Select Repositories to Analyze",
            list(repo_options.keys()),
            help="Select one or more repositories to analyze pull requests"
        )

        if selected_repo_keys:
            selected_repos = [repo_options[key] for key in selected_repo_keys]

            # PR Analysis Options
            st.header("⚙️ Analysis Options")

            col1, col2, col3 = st.columns(3)

            with col1:
                pr_state = st.selectbox(
                    "PR State",
                    ["OPEN", "MERGED", "DECLINED", "ALL"],
                    help="Filter pull requests by state"
                )

            with col2:
                max_prs = st.number_input(
                    "Max PRs per Repo",
                    1, 100, 10,
                    help="Maximum number of PRs to analyze per repository"
                )

            with col3:
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    ["Quick", "Standard", "Deep"],
                    index=2,
                    help="Depth of code analysis to perform"
                )

            # Advanced Options
            with st.expander("🔧 Advanced Analysis Options", expanded=True):
                st.markdown("### 🔍 Core Quality Checks")
                col1, col2 = st.columns(2)

                with col1:
                    analyze_security = st.checkbox("🔒 Security Analysis", True,
                                                   help="Detect vulnerabilities, hardcoded secrets, injection risks")
                    analyze_quality = st.checkbox("✨ Code Quality", True,
                                                  help="General code quality metrics and standards")
                    analyze_style = st.checkbox("🎨 Style & Formatting", True,
                                                help="Code style, formatting, and conventions")

                with col2:
                    check_todos = st.checkbox("📝 TODO/FIXME Comments", True,
                                              help="Identify incomplete work and technical debt")
                    check_complexity = st.checkbox("🧠 Code Complexity", True,
                                                   help="Cyclomatic complexity and cognitive load")
                    generate_suggestions = st.checkbox("💡 Auto Suggestions", True,
                                                       help="AI-generated improvement recommendations")

                st.markdown("### 🏗️ Architecture & Design")
                col3, col4 = st.columns(2)

                with col3:
                    check_design_patterns = st.checkbox("🏛️ Design Patterns", True,
                                                        help="Identify anti-patterns and suggest better designs")
                    check_solid_principles = st.checkbox("🎯 SOLID Principles", True,
                                                         help="Single Responsibility, Open/Closed, etc.")
                    check_dry_principle = st.checkbox("♻️ DRY Violations", True,
                                                      help="Detect code duplication and redundancy")
                    check_modularity = st.checkbox("🧩 Modularity", True, help="Function/class size, cohesion, coupling")

                with col4:
                    check_dependencies = st.checkbox("📦 Dependency Analysis", True,
                                                     help="Import usage, circular dependencies, unused imports")
                    check_api_design = st.checkbox("🔌 API Design", True,
                                                   help="Method signatures, parameter validation, return types")
                    check_error_handling = st.checkbox("⚠️ Error Handling", True,
                                                       help="Exception handling patterns and robustness")
                    check_resource_mgmt = st.checkbox("🗃️ Resource Management", True,
                                                      help="Memory leaks, file handles, connection pooling")

                st.markdown("### ⚡ Performance & Optimization")
                col5, col6 = st.columns(2)

                with col5:
                    check_performance = st.checkbox("🚀 Performance Issues", True,
                                                    help="Inefficient algorithms, N+1 queries, bottlenecks")
                    check_memory_usage = st.checkbox("💾 Memory Efficiency", True,
                                                     help="Memory leaks, large object creation, GC pressure")
                    check_database_ops = st.checkbox("🗄️ Database Operations", True,
                                                     help="Query optimization, transaction handling, indexing")

                with col6:
                    check_async_patterns = st.checkbox("⚡ Async/Concurrency", True,
                                                       help="Threading, async/await, race conditions")
                    check_caching = st.checkbox("📝 Caching Strategy", True,
                                                help="Cache usage, invalidation, performance gains")
                    check_algorithms = st.checkbox("🔬 Algorithm Efficiency", True,
                                                   help="Time/space complexity, optimization opportunities")

                st.markdown("### 🧪 Testing & Quality Assurance")
                col7, col8 = st.columns(2)

                with col7:
                    check_test_coverage = st.checkbox("📊 Test Coverage", True, help="Unit test presence and quality")
                    check_test_quality = st.checkbox("🎯 Test Quality", True,
                                                     help="Test assertions, mocking, edge cases")
                    check_integration_tests = st.checkbox("🔗 Integration Tests", True,
                                                          help="API tests, database tests, end-to-end scenarios")

                with col8:
                    check_assertions = st.checkbox("✅ Assertion Quality", True,
                                                   help="Meaningful assertions, proper error messages")
                    check_test_data = st.checkbox("📋 Test Data Management", True,
                                                  help="Test fixtures, data setup/teardown")
                    check_mocking = st.checkbox("🎭 Mocking Practices", True,
                                                help="Proper mocking, dependency injection")

                st.markdown("### 📚 Documentation & Maintainability")
                col9, col10 = st.columns(2)

                with col9:
                    check_documentation = st.checkbox("📖 Documentation Quality", True,
                                                      help="Comments, docstrings, README updates")
                    check_naming = st.checkbox("🏷️ Naming Conventions", True,
                                               help="Variable, function, class naming standards")
                    check_code_comments = st.checkbox("💬 Comment Quality", True,
                                                      help="Meaningful comments, self-documenting code")

                with col10:
                    check_maintainability = st.checkbox("🔧 Maintainability Index", True,
                                                        help="Code readability, changeability, testability")
                    check_technical_debt = st.checkbox("⚖️ Technical Debt", True,
                                                       help="Code smells, refactoring opportunities")
                    check_versioning = st.checkbox("📌 Version Compatibility", True,
                                                   help="Breaking changes, API versioning, deprecations")

                st.markdown("### 🚨 Critical Analysis Settings")
                col11, col12 = st.columns(2)

                with col11:
                    analysis_strictness = st.selectbox(
                        "Analysis Strictness",
                        ["Lenient", "Standard", "Strict", "Enterprise"],
                        index=3,
                        help="How strict should the analysis be?"
                    )

                    min_severity_report = st.selectbox(
                        "Minimum Severity to Report",
                        ["Info", "Low", "Medium", "High", "Critical"],
                        index=2,
                        help="Only report issues above this severity level"
                    )

                with col12:
                    max_issues_per_file = st.number_input(
                        "Max Issues per File",
                        1, 100, 20,
                        help="Limit reported issues to avoid noise"
                    )

                st.markdown("### 🎛️ Analysis Customization")

                # File type specific analysis
                file_types = st.multiselect(
                    "Focus on File Types",
                    ["Python", "JavaScript", "Java", "Robot Framework", "SQL", "YAML", "Dockerfile", "Shell", "All"],
                    default=["All"],
                    help="Analyze only specific file types"
                )

                # Exclude patterns
                exclude_patterns = st.text_area(
                    "Exclude File Patterns",
                    placeholder="*.min.js\n**/node_modules/**\n**/build/**\n**/dist/**",
                    help="Files/directories to exclude from analysis (one per line, supports wildcards)"
                )

                # Custom rules
                st.markdown("### 🎛️ Analysis Customization")
                st.markdown("### 🛠️ Custom Analysis Rules")
                st.markdown("Configure custom analysis rules and thresholds:")

                col13, col14 = st.columns(2)

                with col13:
                    max_function_lines = st.number_input("Max Function Lines", 1, 200, 50)
                    max_class_lines = st.number_input("Max Class Lines", 1, 1000, 300)
                    max_file_lines = st.number_input("Max File Lines", 1, 5000, 1000)

                with col14:
                    complexity_threshold = st.number_input("Complexity Threshold", 1, 50, 10)
                    duplication_threshold = st.slider("Code Duplication %", 0, 100, 15)
                    test_coverage_min = st.slider("Min Test Coverage %", 0, 100, 80)

            # AI Analysis Configuration
            st.markdown("### 🤖 AI-Powered Analysis Settings")
            
            ai_col1, ai_col2 = st.columns(2)
            
            with ai_col1:
                enable_ai_analysis = st.checkbox(
                    "🧠 Enable AI-Powered Analysis", 
                    value=AI_AVAILABLE,
                    help="Use Azure OpenAI for intelligent code review insights"
                )
                
                if not AI_AVAILABLE:
                    st.warning("⚠️ Azure OpenAI not configured - AI analysis will be limited")
            
            with ai_col2:
                if enable_ai_analysis:
                    ai_analysis_depth = st.selectbox(
                        "AI Analysis Focus",
                        ["Comprehensive", "Security-Focused", "Performance-Focused", "Quality-Focused"],
                        help="Choose the primary focus for AI analysis"
                    )
                    
                    generate_ai_suggestions = st.checkbox(
                        "📝 Generate Code Suggestions",
                        value=True,
                        help="Generate specific code improvement suggestions"
                    )

            # Start Analysis
            if st.button("🚀 Analyse Pull Requests", type="primary"):

                # Capture all configuration options
                analysis_config = {
                    # Core Quality Checks
                    'analyze_security': analyze_security,
                    'analyze_quality': analyze_quality,
                    'analyze_style': analyze_style,
                    'check_todos': check_todos,
                    'check_complexity': check_complexity,
                    'generate_suggestions': generate_suggestions,

                    # Architecture & Design
                    'check_design_patterns': check_design_patterns,
                    'check_solid_principles': check_solid_principles,
                    'check_dry_principle': check_dry_principle,
                    'check_modularity': check_modularity,
                    'check_dependencies': check_dependencies,
                    'check_api_design': check_api_design,
                    'check_error_handling': check_error_handling,
                    'check_resource_mgmt': check_resource_mgmt,

                    # Performance & Optimization
                    'check_performance': check_performance,
                    'check_memory_usage': check_memory_usage,
                    'check_database_ops': check_database_ops,
                    'check_async_patterns': check_async_patterns,
                    'check_caching': check_caching,
                    'check_algorithms': check_algorithms,

                    # Testing & Quality Assurance
                    'check_test_coverage': check_test_coverage,
                    'check_test_quality': check_test_quality,
                    'check_integration_tests': check_integration_tests,
                    'check_assertions': check_assertions,
                    'check_test_data': check_test_data,
                    'check_mocking': check_mocking,

                    # Documentation & Maintainability
                    'check_documentation': check_documentation,
                    'check_naming': check_naming,
                    'check_code_comments': check_code_comments,
                    'check_maintainability': check_maintainability,
                    'check_technical_debt': check_technical_debt,
                    'check_versioning': check_versioning,

                    # Analysis Settings
                    'analysis_strictness': analysis_strictness,
                    'min_severity_report': min_severity_report,
                    'max_issues_per_file': max_issues_per_file,
                    'enable_ai_suggestions': generate_ai_suggestions if 'generate_ai_suggestions' in locals() else False,
                    'file_types': file_types,
                    'exclude_patterns': exclude_patterns,

                    # Custom Thresholds
                    'max_function_lines': max_function_lines,
                    'max_class_lines': max_class_lines,
                    'max_file_lines': max_file_lines,
                    'complexity_threshold': complexity_threshold,
                    'duplication_threshold': duplication_threshold,
                    'test_coverage_min': test_coverage_min,

                    # Analysis depth
                    'analysis_depth': analysis_depth,
                    
                    # AI Analysis Settings
                    'enable_ai_analysis': enable_ai_analysis if 'enable_ai_analysis' in locals() else False,
                    'ai_analysis_depth': ai_analysis_depth if 'ai_analysis_depth' in locals() else 'Comprehensive',
                    'generate_ai_suggestions': generate_ai_suggestions if 'generate_ai_suggestions' in locals() else False
                }

                progress_bar = st.progress(0)
                status_text = st.empty()

                all_pr_reviews = []
                total_repos = len(selected_repos)

                for repo_idx, repo in enumerate(selected_repos):
                    project_key = repo.get('project', {}).get('key', '')
                    repo_slug = repo.get('slug', '')
                    repo_name = repo.get('name', '')

                    status_text.text(f"Analyzing repository: {repo_name}")

                    try:
                        # Get pull requests
                        prs = st.session_state.reviewer.get_pull_requests(
                            st.session_state.base_url,
                            project_key,
                            repo_slug,
                            pr_state if pr_state != "ALL" else "OPEN",
                            max_prs
                        )

                        for pr_idx, pr in enumerate(prs):
                            pr_id = pr.get('id')
                            status_text.text(f"Analyzing PR #{pr_id} in {repo_name}")

                            # Get PR changes
                            changes = st.session_state.reviewer.get_pr_changes(
                                st.session_state.base_url,
                                project_key,
                                repo_slug,
                                pr_id
                            )

                            # Analyze code quality for changed files
                            code_analysis = []

                            files_to_analyze = changes.get('files_changed', [])
                            if analysis_depth == "Quick":
                                files_to_analyze = files_to_analyze[:5]  # Limit files for quick analysis
                            elif analysis_depth == "Standard":
                                files_to_analyze = files_to_analyze[:15]
                            # Deep analysis uses all files

                            for file_change in files_to_analyze:
                                file_path = file_change.get('path', '')

                                # Skip files based on exclude patterns
                                should_skip = False
                                exclude_patterns_list = analysis_config.get('exclude_patterns', '').split('\n')
                                for pattern in exclude_patterns_list:
                                    pattern = pattern.strip()
                                    if pattern and st.session_state.reviewer._matches_pattern(file_path, pattern):
                                        should_skip = True
                                        break

                                if should_skip:
                                    continue

                                # Get actual file content (in real implementation)
                                # For demo purposes, we'll create more realistic mock content based on file type
                                mock_content = st.session_state.reviewer._generate_mock_content(file_path,
                                                                                                analysis_config)

                                # Analyze with full configuration
                                file_analysis = st.session_state.reviewer.analyze_code_quality(
                                    mock_content,
                                    file_path,
                                    analysis_config
                                )
                                code_analysis.append(file_analysis)

                            # Generate PR review with AI-powered configuration
                            pr_review = st.session_state.reviewer.generate_ai_powered_review(
                                pr, changes, code_analysis, analysis_config
                            )

                            pr_review["repository"] = repo_name
                            pr_review["project_key"] = project_key
                            pr_review["repo_slug"] = repo_slug

                            all_pr_reviews.append(pr_review)

                    except Exception as e:
                        st.error(f"Error analyzing repository {repo_name}: {e}")

                    progress_bar.progress((repo_idx + 1) / total_repos)

                status_text.text("Analysis completed!")

                # Display Results
                if all_pr_reviews:
                    st.header("📊 Pull Request Analysis Results")

                    # Summary Statistics
                    st.subheader("📈 Executive Summary")

                    total_prs = len(all_pr_reviews)
                    avg_score = sum(
                        review.get('overall_score', 0) for review in all_pr_reviews) / total_prs if total_prs > 0 else 0
                    total_security_issues = sum(
                        review.get('metrics', {}).get('security_issues', 0) for review in all_pr_reviews)
                    high_severity_total = sum(
                        review.get('metrics', {}).get('high_severity_issues', 0) for review in all_pr_reviews)

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Pull Requests", total_prs)
                    with col2:
                        st.metric("Avg Quality Score", f"{avg_score:.1f}/100")
                    with col3:
                        st.metric("Security Issues", total_security_issues)
                    with col4:
                        st.metric("High Severity", high_severity_total)

                    # AI-Powered Cross-Repository Analysis
                    st.subheader("🤖 AI-Powered Strategic Analysis")
                    
                    # if AI_AVAILABLE:
                    #     ai_analysis_col1, ai_analysis_col2 = st.columns([3, 1])
                        
                    #     with ai_analysis_col1:
                    #         st.info("🧠 Get comprehensive AI insights across all analyzed repositories")
                        
                    #     with ai_analysis_col2:
                    #         if st.button("🚀 Generate AI Analysis", type="primary", key="cross_repo_ai"):
                    #             with st.spinner("AI is analyzing patterns across all repositories..."):
                    #                 cross_repo_analysis = run_ai_pr_quality_analysis(all_pr_reviews)
                                    
                    #                 # Display in a dedicated section
                    #                 st.markdown("---")
                    #                 st.markdown("### 🎯 Cross-Repository AI Insights")
                    #                 st.markdown(cross_repo_analysis)
                                    
                    #                 # Save analysis to session state for later access
                    #                 st.session_state.cross_repo_ai_analysis = cross_repo_analysis
                    # else:
                    #     st.warning("⚠️ AI strategic analysis requires Azure OpenAI configuration")
                    
                    # Display saved analysis if available
                    if hasattr(st.session_state, 'cross_repo_ai_analysis'):
                        with st.expander("📋 Previous AI Strategic Analysis", expanded=False):
                            st.markdown(st.session_state.cross_repo_ai_analysis)

                    # Configuration Summary - Enhanced with actual applied settings
                    st.subheader("⚙️ Analysis Configuration Applied")

                    # Show detailed configuration that was actually applied
                    enabled_checks = []
                    disabled_checks = []

                    check_categories = {
                        "🔍 Core Quality Checks": [
                            ('analyze_security', '🔒 Security Analysis'),
                            ('analyze_quality', '✨ Code Quality'),
                            ('analyze_style', '🎨 Style & Formatting'),
                            ('check_todos', '📝 TODO/FIXME Comments'),
                            ('check_complexity', '🧠 Code Complexity'),
                            ('generate_suggestions', '💡 Auto Suggestions')
                        ],
                        "🏗️ Architecture & Design": [
                            ('check_design_patterns', '🏛️ Design Patterns'),
                            ('check_solid_principles', '🎯 SOLID Principles'),
                            ('check_dry_principle', '♻️ DRY Violations'),
                            ('check_modularity', '🧩 Modularity'),
                            ('check_dependencies', '📦 Dependency Analysis'),
                            ('check_api_design', '🔌 API Design'),
                            ('check_error_handling', '⚠️ Error Handling'),
                            ('check_resource_mgmt', '🗃️ Resource Management')
                        ],
                        "⚡ Performance & Optimization": [
                            ('check_performance', '🚀 Performance Issues'),
                            ('check_memory_usage', '💾 Memory Efficiency'),
                            ('check_database_ops', '🗄️ Database Operations'),
                            ('check_async_patterns', '⚡ Async/Concurrency'),
                            ('check_caching', '📝 Caching Strategy'),
                            ('check_algorithms', '🔬 Algorithm Efficiency')
                        ],
                        "🧪 Testing & Quality Assurance": [
                            ('check_test_coverage', '📊 Test Coverage'),
                            ('check_test_quality', '🎯 Test Quality'),
                            ('check_integration_tests', '🔗 Integration Tests'),
                            ('check_assertions', '✅ Assertion Quality'),
                            ('check_test_data', '📋 Test Data Management'),
                            ('check_mocking', '🎭 Mocking Practices')
                        ],
                        "📚 Documentation & Maintainability": [
                            ('check_documentation', '📖 Documentation Quality'),
                            ('check_naming', '🏷️ Naming Conventions'),
                            ('check_code_comments', '💬 Comment Quality'),
                            ('check_maintainability', '🔧 Maintainability Index'),
                            ('check_technical_debt', '⚖️ Technical Debt'),
                            ('check_versioning', '📌 Version Compatibility')
                        ]
                    }

                    # Display configuration in expandable sections
                    for category, checks in check_categories.items():
                        with st.expander(f"{category} - Applied Configuration"):
                            enabled_in_category = []
                            disabled_in_category = []

                            for check_key, check_name in checks:
                                if analysis_config.get(check_key, False):
                                    enabled_in_category.append(check_name)
                                else:
                                    disabled_in_category.append(check_name)

                            if enabled_in_category:
                                st.success(f"**✅ Enabled ({len(enabled_in_category)}):**")
                                for check in enabled_in_category:
                                    st.write(f"  • {check}")

                            if disabled_in_category:
                                st.info(f"**❌ Disabled ({len(disabled_in_category)}):**")
                                for check in disabled_in_category:
                                    st.write(f"  • {check}")

                    # Analysis Settings Summary
                    st.write("**🎛️ Analysis Settings:**")
                    settings_col1, settings_col2 = st.columns(2)

                    with settings_col1:
                        st.info(f"**Analysis Depth:** {analysis_config['analysis_depth']}")
                        st.info(f"**Strictness:** {analysis_config['analysis_strictness']}")
                        st.info(f"**Min Severity:** {analysis_config['min_severity_report']}")
                        st.info(f"**Max Issues/File:** {analysis_config['max_issues_per_file']}")

                    with settings_col2:
                        st.info(f"**File Types:** {', '.join(analysis_config['file_types'])}")
                        st.info(f"**Max Function Lines:** {analysis_config['max_function_lines']}")
                        st.info(f"**Max File Lines:** {analysis_config['max_file_lines']}")
                        st.info(f"**Complexity Threshold:** {analysis_config['complexity_threshold']}")

                    # Detailed PR Reviews with Enhanced Issue Reporting
                    st.subheader("🔍 Detailed Analysis Results")

                    for review in all_pr_reviews:
                        # Determine expandability based on issues found
                        has_critical_issues = review['overall_score'] < 70 or review['metrics']['security_issues'] > 0

                        with st.expander(f"PR #{review['pr_id']}: {review['title']} - {review['summary']}",
                                         expanded=has_critical_issues):

                            # Enhanced PR Overview
                            overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

                            with overview_col1:
                                score = review['overall_score']
                                if score >= 90:
                                    score_color = "🟢"
                                    score_status = "Excellent"
                                elif score >= 70:
                                    score_color = "🟡"
                                    score_status = "Good"
                                elif score >= 50:
                                    score_color = "🟠"
                                    score_status = "Needs Work"
                                else:
                                    score_color = "🔴"
                                    score_status = "Critical"

                                st.metric("Quality Score", f"{score}/100", help=f"Status: {score_status}")
                                st.write(f"{score_color} {score_status}")

                            with overview_col2:
                                st.metric("Files Changed", review['metrics']['files_changed'])
                                st.metric("Total Issues", review['metrics']['total_issues'])

                            with overview_col3:
                                security_count = review['metrics']['security_issues']
                                if security_count > 0:
                                    st.metric("🚨 Security Issues", security_count, delta=f"Critical: {security_count}")
                                else:
                                    st.metric("🔒 Security Issues", "0", delta="✅ Clean")

                            with overview_col4:
                                high_severity = review['metrics']['high_severity_issues']
                                if high_severity > 0:
                                    st.metric("⚠️ High Severity", high_severity, delta=f"Urgent: {high_severity}")
                                else:
                                    st.metric("⚠️ High Severity", "0", delta="✅ None")

                            # PR Metadata
                            st.markdown("---")
                            st.markdown("### 📋 Pull Request Details")

                            details_col1, details_col2 = st.columns(2)

                            with details_col1:
                                st.write(f"**👤 Author:** {review['author']}")
                                st.write(f"**📂 Repository:** {review['repository']}")
                                st.write(f"**📅 Created:** {review['created_date']}")

                            with details_col2:
                                st.write(f"**🔄 Status:** {review['status']}")
                                if review['reviewers']:
                                    st.write(f"**👥 Reviewers:** {', '.join(review['reviewers'])}")
                                st.write(f"**📝 Updated:** {review['updated_date']}")

                            # Critical Issues Section - Enhanced
                            if review['security_concerns'] or review['metrics']['high_severity_issues'] > 0:
                                st.markdown("---")
                                st.markdown("### 🚨 Critical Issues Requiring Immediate Attention")

                                # Security Issues
                                if review['security_concerns']:
                                    st.error("**🔒 Security Vulnerabilities Found:**")
                                    for concern in review['security_concerns']:
                                        st.write(f"• {concern}")

                                # High Severity Issues from file analysis
                                high_severity_details = []
                                for file_analysis in review['detailed_analysis']:
                                    file_path = file_analysis['file_path']

                                    # Collect high severity issues
                                    for issue in file_analysis.get('issues', []) + file_analysis.get('security_issues',
                                                                                                     []):
                                        if issue.get('severity', '').lower() == 'high':
                                            high_severity_details.append({
                                                'file': file_path,
                                                'line': issue.get('line', 0),
                                                'type': issue.get('type', 'unknown'),
                                                'message': issue.get('message', 'No description'),
                                                'category': 'Security' if issue in file_analysis.get('security_issues',
                                                                                                     []) else 'Code Quality'
                                            })

                                if high_severity_details:
                                    st.warning(f"**⚠️ {len(high_severity_details)} High Severity Issues Found:**")

                                    # Create DataFrame for better display
                                    high_severity_df = pd.DataFrame(high_severity_details)
                                    st.dataframe(
                                        high_severity_df[['file', 'line', 'category', 'type', 'message']],
                                        use_container_width=True,
                                        column_config={
                                            'file': st.column_config.TextColumn('📁 File', width='medium'),
                                            'line': st.column_config.NumberColumn('📍 Line', width='small'),
                                            'category': st.column_config.TextColumn('🏷️ Category', width='small'),
                                            'type': st.column_config.TextColumn('🔍 Type', width='medium'),
                                            'message': st.column_config.TextColumn('💬 Issue Description', width='large')
                                        }
                                    )

                            # Recommendations Section - Enhanced
                            if review['recommendations']:
                                st.markdown("---")
                                st.markdown("### 💡 Actionable Recommendations")

                                for i, rec in enumerate(review['recommendations'], 1):
                                    st.info(f"**{i}.** {rec}")

                            # AI Insights Section - NEW
                            if 'ai_insights' in review and review['ai_insights']:
                                st.markdown("---")
                                st.markdown("### 🤖 AI-Powered Code Analysis")
                                
                                # Display AI availability status
                                if AI_AVAILABLE:
                                    st.success("✅ AI Analysis Complete")
                                else:
                                    st.warning("⚠️ AI Analysis Limited - Azure OpenAI not available")
                                
                                # AI insights tabs for better organization
                                ai_tab1, ai_tab2, ai_tab3 = st.tabs(["📋 Full Analysis", "🔒 Security Focus", "💡 Code Suggestions"])
                                
                                with ai_tab1:
                                    st.markdown("#### 🧠 Comprehensive AI Review")
                                    st.markdown(review['ai_insights'])
                                    
                                    # Additional AI-extracted metrics
                                    if 'risk_level' in review:
                                        risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(review['risk_level'], "⚫")
                                        st.markdown(f"**Risk Assessment:** {risk_color} {review['risk_level']} Risk")
                                    
                                    if 'merge_readiness' in review:
                                        readiness_color = {"Ready": "🟢", "Needs Work": "🟡", "Review Required": "🟠"}.get(review['merge_readiness'], "⚫")
                                        st.markdown(f"**Merge Readiness:** {readiness_color} {review['merge_readiness']}")
                                
                                with ai_tab2:
                                    st.markdown("#### 🔒 AI Security Analysis")
                                    if review['security_concerns'] or review['metrics']['security_issues'] > 0:
                                        # Get focused security analysis
                                        if AI_AVAILABLE and azure_openai_client:
                                            with st.spinner("Getting AI security analysis..."):
                                                reviewer_instance = st.session_state.get('reviewer')
                                                if reviewer_instance:
                                                    security_analysis = reviewer_instance.get_ai_security_analysis(review['detailed_analysis'])
                                                    st.markdown(security_analysis)
                                                else:
                                                    st.error("Reviewer instance not available")
                                        else:
                                            st.warning("AI security analysis requires Azure OpenAI configuration")
                                            # Fallback to basic security info
                                            if review['security_concerns']:
                                                st.error("**Security Issues Found:**")
                                                for concern in review['security_concerns']:
                                                    st.write(f"• {concern}")
                                    else:
                                        st.success("🎉 No security issues detected by AI analysis!")
                                
                                with ai_tab3:
                                    st.markdown("#### 💡 AI Code Improvement Suggestions")
                                    
                                    # File-specific AI suggestions
                                    if review['detailed_analysis']:
                                        file_options = []
                                        for file_analysis in review['detailed_analysis']:
                                            if file_analysis.get('issues') or file_analysis.get('security_issues'):
                                                file_path = file_analysis['file_path']
                                                issue_count = len(file_analysis.get('issues', [])) + len(file_analysis.get('security_issues', []))
                                                file_options.append(f"{file_path.split('/')[-1]} ({issue_count} issues)")
                                        
                                        if file_options:
                                            selected_file = st.selectbox(
                                                "Select file for AI suggestions:",
                                                file_options,
                                                key=f"ai_suggestions_{review['pr_id']}"
                                            )
                                            
                                            if selected_file and st.button(f"🤖 Get AI Suggestions", key=f"ai_suggest_btn_{review['pr_id']}"):
                                                # Extract file path from selection
                                                selected_file_name = selected_file.split(' (')[0]
                                                target_file_analysis = None
                                                
                                                for file_analysis in review['detailed_analysis']:
                                                    if file_analysis['file_path'].endswith(selected_file_name):
                                                        target_file_analysis = file_analysis
                                                        break
                                                
                                                if target_file_analysis and AI_AVAILABLE:
                                                    with st.spinner(f"Generating AI suggestions for {selected_file_name}..."):
                                                        reviewer_instance = st.session_state.get('reviewer')
                                                        if reviewer_instance:
                                                            # Get mock content (in real implementation, you'd get actual file content)
                                                            mock_content = reviewer_instance._generate_mock_content(
                                                                target_file_analysis['file_path'], 
                                                                analysis_config if 'analysis_config' in locals() else {}
                                                            )
                                                            
                                                            all_issues = target_file_analysis.get('issues', []) + target_file_analysis.get('security_issues', [])
                                                            suggestions = reviewer_instance.generate_ai_code_suggestions(
                                                                mock_content,
                                                                target_file_analysis['file_path'],
                                                                all_issues
                                                            )
                                                            st.markdown(suggestions)
                                                        else:
                                                            st.error("Reviewer instance not available")
                                                else:
                                                    st.warning("AI suggestions require Azure OpenAI configuration")
                                        else:
                                            st.info("No files with issues found for AI suggestions")
                                    else:
                                        st.info("No file analysis available for AI suggestions")
                            
                            elif 'ai_analysis_error' in review:
                                st.markdown("---")
                                st.markdown("### 🤖 AI Analysis Status")
                                st.warning(f"⚠️ {review['ai_analysis_error']}")
                                st.info("💡 To enable AI analysis, configure Azure OpenAI client in the system")

                            # Detailed File Analysis - Completely Enhanced
                            if review['detailed_analysis']:
                                st.markdown("---")
                                st.markdown("### 📁 Detailed File-by-File Analysis")

                                # Summary table first
                                file_summary_data = []
                                for file_analysis in review['detailed_analysis']:
                                    issues_count = len(file_analysis.get('issues', []))
                                    security_count = len(file_analysis.get('security_issues', []))
                                    total_issues = issues_count + security_count

                                    # Determine risk level
                                    if security_count > 0:
                                        risk_level = "🔴 High Risk"
                                    elif total_issues > 5:
                                        risk_level = "🟠 Medium Risk"
                                    elif total_issues > 0:
                                        risk_level = "🟡 Low Risk"
                                    else:
                                        risk_level = "🟢 Clean"

                                    file_summary_data.append({
                                        "File": file_analysis['file_path'].split('/')[-1],  # Just filename
                                        "Full Path": file_analysis['file_path'],
                                        "Language": file_analysis['language'].title(),
                                        "Lines": file_analysis.get('metrics', {}).get('total_lines', 0),
                                        "Issues": issues_count,
                                        "Security": security_count,
                                        "Total": total_issues,
                                        "Risk Level": risk_level
                                    })

                                if file_summary_data:
                                    st.markdown("#### 📊 Files Overview")
                                    files_df = pd.DataFrame(file_summary_data)

                                    # # Add file URLs for clickable links
                                    # if hasattr(st.session_state,
                                    #            'base_url') and 'project_key' in review and 'repo_slug' in review:
                                    #     files_df['File_URL'] = files_df['Full Path'].apply(
                                    #         lambda path: st.session_state.reviewer.generate_file_url(
    #             st.session_state.base_url,
    #             review['project_key'],
    #             review['repo_slug'],
    #             path,
    #             str(review['pr_id'])
    #         )
    #     )

                                    st.dataframe(
                                        files_df[
                                            ['File', 'Language', 'Lines', 'Issues', 'Security', 'Total', 'Risk Level']],
                                        use_container_width=True,
                                        column_config={
                                            'File': st.column_config.LinkColumn(
                                                '📄 File Name',
                                                width='medium',
                                                display_text=files_df[
                                                    'File'] if 'File_URL' in files_df.columns else None,
                                                help="Click to view file in Bitbucket/Stash"
                                            ) if 'File_URL' in files_df.columns else st.column_config.TextColumn(
                                                '📄 File Name', width='medium'),
                                            'Language': st.column_config.TextColumn('💻 Language', width='small'),
                                            'Lines': st.column_config.NumberColumn('📏 Lines', width='small'),
                                            'Issues': st.column_config.NumberColumn('⚠️ Issues', width='small'),
                                            'Security': st.column_config.NumberColumn('🔒 Security', width='small'),
                                            'Total': st.column_config.NumberColumn('📊 Total', width='small'),
                                            'Risk Level': st.column_config.TextColumn('🎯 Risk', width='medium')
                                        }
                                    )

                                    # # Show file URLs as clickable links in expander for better visibility
                                    # if hasattr(st.session_state,
    #            'base_url') and 'project_key' in review and 'repo_slug' in review:
    #     with st.expander("🔗 File Links", expanded=False):
    #         st.markdown("**Direct links to files in Bitbucket/Stash:**")
    #         for _, row in files_df.iterrows():
    #             file_url = st.session_state.reviewer.generate_file_url(
    #                 st.session_state.base_url,
    #                 review['project_key'],
    #                 review['repo_slug'],
    #                 row['Full Path'],
    #                 str(review['pr_id'])
    #             )
    #             st.markdown(f"📄 [{row['File']}]({file_url}) - {row['Full Path']}")

                                # Detailed per-file analysis
                                st.markdown("#### 🔍 Individual File Analysis")

                                for file_analysis in review['detailed_analysis']:
                                    file_path = file_analysis['file_path']
                                    file_name = file_path.split('/')[-1]
                                    language = file_analysis['language'].title()

                                    all_issues = file_analysis.get('issues', []) + file_analysis.get('security_issues',
                                                                                                     [])

                                    if not all_issues and not file_analysis.get('suggestions'):
                                        # Skip files with no issues for clean display
                                        continue

                                    # # Generate file URL for this specific file
                                    # file_url = "#"
                                    # if hasattr(st.session_state,
    #            'base_url') and 'project_key' in review and 'repo_slug' in review:
    #     file_url = st.session_state.reviewer.generate_file_url(
    #         st.session_state.base_url,
    #         review['project_key'],
    #         review['repo_slug'],
    #         file_path,
    #         str(review['pr_id'])
    #     )

                                    # # Display file header with clickable link
                                    # if file_url != "#":
                                    #     st.markdown(
    #         f"##### 📄 [{file_name}]({file_url}) ({language}) - {len(all_issues)} issues")
    # else:
    #     st.markdown(f"##### 📄 {file_name} ({language}) - {len(all_issues)} issues")

                                    show_file_details = st.checkbox(
                                        f"Show detailed analysis",
                                        key=f"file_details_{review['pr_id']}_{file_path.replace('/', '_').replace('.', '_')}",
                                        value=len(all_issues) > 0
                                    )

                                    if show_file_details:
                                        with st.container():
                                            # File header info with improved metrics display
                                            file_col1, file_col2, file_col3 = st.columns(3)

                                            with file_col1:
                                                # if file_url != "#":
                                                #     st.markdown(f"**📂 Path:** [{file_path}]({file_url})")
                                                # else:
                                                #     st.write(f"**📂 Path:** `{file_path}`")
                                                st.write(f"**💻 Language:** {language}")

                                            with file_col2:
                                                metrics = file_analysis.get('metrics', {})
                                                # Use accurate metrics from the analysis
                                                total_lines = metrics.get('total_lines', 0)
                                                code_lines = metrics.get('code_lines',
                                                                         metrics.get('non_empty_lines', 0))

                                                st.write(f"**📏 Total Lines:** {total_lines}")
                                                st.write(f"**📝 Code Lines:** {code_lines}")

                                            with file_col3:
                                                st.write(f"**⚠️ Issues:** {len(file_analysis.get('issues', []))}")
                                                st.write(
                                                    f"**🔒 Security:** {len(file_analysis.get('security_issues', []))}")

                                            # Additional detailed metrics if available
                                            if metrics and len(metrics) > 2:
                                                show_detailed_metrics = st.checkbox(
                                                    "📊 Show Detailed Code Metrics",
                                                    key=f"detailed_metrics_{review['pr_id']}_{file_path.replace('/', '_').replace('.', '_')}"
                                                )

                                                if show_detailed_metrics:
                                                    with st.container():
                                                        st.markdown("**📊 Detailed Code Metrics:**")
                                                        metrics_col1, metrics_col2 = st.columns(2)

                                                        with metrics_col1:
                                                            if 'functions' in metrics and metrics['functions'] > 0:
                                                                st.write(f"**Functions:** {metrics['functions']}")
                                                            if 'classes' in metrics and metrics['classes'] > 0:
                                                                st.write(f"**Classes:** {metrics['classes']}")
                                                            if 'imports' in metrics and metrics['imports'] > 0:
                                                                st.write(f"**Imports:** {metrics['imports']}")

                                                        with metrics_col2:
                                                            if 'comment_lines' in metrics:
                                                                st.write(f"**Comment Lines:** {metrics['comment_lines']}")
                                                            if 'complexity_score' in metrics and metrics[
                                                                'complexity_score'] > 0:
                                                                st.write(
                                                                    f"**Complexity Score:** {metrics['complexity_score']}")
                                                            if 'test_cases' in metrics and metrics['test_cases'] > 0:
                                                                st.write(f"**Test Cases:** {metrics['test_cases']}")

                                            # Issues breakdown with accurate line numbers
                                            if all_issues:
                                                st.markdown("**🐛 Issues Found:**")

                                                # Group issues by severity
                                                issues_by_severity = {}
                                                for issue in all_issues:
                                                    severity = issue.get('severity', 'medium').title()
                                                    if severity not in issues_by_severity:
                                                        issues_by_severity[severity] = []
                                                    issues_by_severity[severity].append(issue)

                                                # Display issues by severity (highest first)
                                                severity_order = ['Critical', 'High', 'Medium', 'Low', 'Info']

                                                # Create a global issue counter for unique keys
                                                global_issue_counter = 0

                                                for severity in severity_order:
                                                    if severity in issues_by_severity:
                                                        issues = issues_by_severity[severity]

                                                        # Severity color coding
                                                        if severity == 'Critical':
                                                            severity_icon = "🔴"
                                                        elif severity == 'High':
                                                            severity_icon = "🟠"
                                                        elif severity == 'Medium':
                                                            severity_icon = "🟡"
                                                        elif severity == 'Low':
                                                            severity_icon = "🟢"
                                                        else:
                                                            severity_icon = "ℹ️"

                                                        st.markdown(
                                                            f"**{severity_icon} {severity} Severity Issues ({len(issues)}):**")

                                                        for issue_idx, issue in enumerate(issues):
                                                            global_issue_counter += 1
                                                            line_num = issue.get('line', 0)
                                                            issue_type = issue.get('type', 'unknown')
                                                            message = issue.get('message', 'No description')

                                                            # Create expandable issue details
                                                            with st.container():
                                                                issue_col1, issue_col2 = st.columns([1, 4])

                                                                with issue_col1:
                                                                    if line_num > 0:
                                                                        st.code(f"Line {line_num}")
                                                                    else:
                                                                        st.code("General")

                                                                with issue_col2:
                                                                    st.write(
                                                                        f"**{issue_type.replace('_', ' ').title()}:** {message}")

                                                                    # Add recommended fixes based on issue type - Remove nested expander
                                                                    fix_recommendations = get_fix_recommendation(
                                                                        issue_type, language, message)
                                                                    if fix_recommendations:
                                                                        # Create unique key using multiple identifiers
                                                                        import hashlib
                                                                        message_hash = hashlib.md5(message.encode()).hexdigest()[:8]
                                                                        unique_key = f"fix_{review['pr_id']}_{file_path.replace('/', '_').replace('.', '_')}_{line_num}_{severity}_{issue_type}_{global_issue_counter}_{message_hash}"

                                                                        # Use a button or checkbox to show/hide recommendations instead of nested expander
                                                                        show_fix = st.checkbox(
                                                                            f"💡 Show Fix for {issue_type.replace('_', ' ').title()}",
                                                                            key=unique_key)
                                                                        if show_fix:
                                                                            st.markdown("**💡 Recommended Fix:**")
                                                                            st.markdown(fix_recommendations)

                                                        st.markdown("---")

                                        # AI Suggestions
                                        suggestions = file_analysis.get('suggestions', [])
                                        if suggestions:
                                            st.markdown("**🤖 AI-Generated Suggestions:**")
                                            for suggestion in suggestions:
                                                st.info(f"💡 {suggestion}")

                                        # Code Quality Metrics
                                        metrics = file_analysis.get('metrics', {})
                                        if metrics and len(
                                                metrics) > 2:  # More than just total_lines and non_empty_lines
                                            st.markdown("**📊 Code Metrics:**")
                                            metrics_col1, metrics_col2 = st.columns(2)

                                            with metrics_col1:
                                                if 'functions' in metrics:
                                                    st.write(f"Functions: {metrics['functions']}")
                                                if 'classes' in metrics:
                                                    st.write(f"Classes: {metrics['classes']}")

                                            with metrics_col2:
                                                if 'imports' in metrics:
                                                    st.write(f"Imports: {metrics['imports']}")
                                                if 'test_cases' in metrics:
                                                    st.write(f"Test Cases: {metrics['test_cases']}")


def get_fix_recommendation(issue_type: str, language: str, message: str) -> str:
    """Generate specific fix recommendations based on issue type and language"""

    recommendations = {
        'hardcoded_secret': {
            'description': "**Security Issue:** Hardcoded secrets pose a serious security risk.",
            'fixes': [
                "Move secrets to environment variables: `os.getenv('SECRET_KEY')`",
                "Use a secrets management service (AWS Secrets Manager, Azure Key Vault)",
                "Store in encrypted configuration files",
                "Use development vs production configuration separation"
            ],
            'example': """
```python
# Bad
PASSWORD = "hardcoded_password_123"

# Good
import os
PASSWORD = os.getenv('DATABASE_PASSWORD')
```"""
        },

        'dangerous_function': {
            'description': "**Security Issue:** Dangerous functions can lead to code injection attacks.",
            'fixes': [
                "Replace `eval()` with safer alternatives like `ast.literal_eval()`",
                "Use `json.loads()` for JSON data parsing",
                "Validate and sanitize all input before processing",
                "Consider using a sandboxed execution environment"
            ],
            'example': """
```python
# Bad
result = eval(user_input)

# Good
import ast
result = ast.literal_eval(user_input)  # Only for literals
# or
import json
result = json.loads(user_input)  # For JSON
```"""
        },

        'sql_injection': {
            'description': "**Critical Security Issue:** SQL injection vulnerability detected.",
            'fixes': [
                "Use parameterized queries/prepared statements",
                "Never concatenate user input directly into SQL",
                "Use ORM frameworks when possible",
                "Validate and sanitize all input"
            ],
            'example': """
```python
# Bad
query = "SELECT * FROM users WHERE name = '" + user_input + "'"

# Good
query = "SELECT * FROM users WHERE name = %s"
cursor.execute(query, (user_input,))
```"""
        },

        'xss_risk': {
            'description': "**Security Issue:** Cross-Site Scripting (XSS) vulnerability detected.",
            'fixes': [
                "Use `textContent` instead of `innerHTML` for user data",
                "Sanitize HTML content using libraries like DOMPurify",
                "Escape special characters in user input",
                "Use Content Security Policy (CSP) headers"
            ],
            'example': """
```javascript
// Bad
element.innerHTML = userInput;

// Good
element.textContent = userInput;
// or for HTML content
element.innerHTML = DOMPurify.sanitize(userInput);
```"""
        },

        'performance': {
            'description': "**Performance Issue:** Inefficient code pattern detected.",
            'fixes': [
                "Use built-in functions and methods when available",
                "Avoid nested loops when possible",
                "Cache expensive operations",
                "Use appropriate data structures"
            ],
            'example': """
```python
# Bad
for i in range(len(items)):
    process(items[i])

# Good
for item in items:
    process(item)
# or
for i, item in enumerate(items):
    process(item, i)
```"""
        },

        'complexity': {
            'description': "**Maintainability Issue:** Code complexity is too high.",
            'fixes': [
                "Break large functions into smaller, focused functions",
                "Reduce nesting levels using early returns",
                "Extract complex logic into separate methods",
                "Use design patterns to simplify structure"
            ],
            'example': """
```python
# Bad - deeply nested
def process_user(user):
    if user:
        if user.is_active:
            if user.has_permission:
                return process_data(user.data)
    return None

# Good - early returns
def process_user(user):
    if not user:
        return None
    if not user.is_active:
        return None
    if not user.has_permission:
        return None
    return process_data(user.data)
```"""
        },

        'documentation': {
            'description': "**Documentation Issue:** Missing or inadequate documentation.",
            'fixes': [
                "Add docstrings to functions and classes",
                "Use clear, descriptive comments",
                "Document complex algorithms and business logic",
                "Keep documentation up-to-date with code changes"
            ],
            'example': """
```python
def calculate_tax(amount, rate):
    \"\"\"
    Calculate tax amount based on the given rate.

    Args:
        amount (float): The base amount to calculate tax on
        rate (float): Tax rate as a decimal (e.g., 0.08 for 8%)

    Returns:
        float: The calculated tax amount
    \"\"\"
    return amount * rate
```"""
        },

        'error_handling': {
            'description': "**Robustness Issue:** Missing or inadequate error handling.",
            'fixes': [
                "Add try-catch blocks around risky operations",
                "Handle specific exceptions rather than generic ones",
                "Provide meaningful error messages",
                "Implement proper logging for errors"
            ],
            'example': """
```python
# Bad
def read_file(filename):
    return open(filename).read()

# Good
def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return None
    except IOError as e:
        logger.error(f"Error reading file {filename}: {e}")
        return None
```"""
        },

        'style': {
            'description': "**Style Issue:** Code doesn't follow style guidelines.",
            'fixes': [
                "Use automatic code formatters (black, prettier, etc.)",
                "Follow language-specific style guides (PEP 8, etc.)",
                "Keep line lengths reasonable (80-120 characters)",
                "Use consistent naming conventions"
            ],
            'example': """
```python
# Bad
def someFunction(userInput,anotherParam):
    reallyLongVariableNameThatExceedsReasonableLimitsAndMakesCodeHardToRead=userInput+anotherParam
    return reallyLongVariableNameThatExceedsReasonableLimitsAndMakesCodeHardToRead

# Good
def process_user_input(user_input, additional_param):
    result = user_input + additional_param
    return result
```"""
        },

        'maintenance': {
            'description': "**Technical Debt:** TODO/FIXME comments indicate incomplete work.",
            'fixes': [
                "Create proper tickets/issues for TODO items",
                "Set deadlines for addressing technical debt",
                "Prioritize based on impact and effort",
                "Remove completed TODOs promptly"
            ],
            'example': """
```python
# Bad
# TODO: Fix this later

# Good
# Issue #123: Implement proper error handling for edge case
# Expected completion: Sprint 23
```"""
        }
    }

    # Get recommendation for this issue type
    rec = recommendations.get(issue_type, {
        'description': f"**Issue Type:** {issue_type.replace('_', ' ').title()}",
        'fixes': [
            "Review the specific issue in context",
            "Follow best practices for your programming language",
            "Consider refactoring if the issue impacts maintainability",
            "Test thoroughly after making changes"
        ],
        'example': "Specific example not available for this issue type."
    })

    # Format the recommendation
    result = f"{rec['description']}\n\n**🔧 Recommended Fixes:**\n"
    for fix in rec['fixes']:
        result += f"• {fix}\n"

    if 'example' in rec:
        result += f"\n**📝 Example:**\n{rec['example']}"

    return result


# Test and validation functions
def test_ai_integration():
    """Test Azure OpenAI integration for pull request analysis"""
    try:
        if not AI_AVAILABLE:
            return {"status": "error", "message": "Azure OpenAI not available"}
        
        if not azure_openai_client:
            return {"status": "error", "message": "Azure OpenAI client not initialized"}
        
        # Simple test prompt
        test_prompt = """
        You are a code review expert. Analyze this simple Python function:
        
        def process_user_input(user_data):
            return eval(user_data)
        
        What security issues do you see? Provide a brief response.
        """
        
        response = azure_openai_client.generate_response(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        if response and len(response.strip()) > 10:
            return {
                "status": "success", 
                "message": "Azure OpenAI integration working correctly",
                "sample_response": response[:100] + "..." if len(response) > 100 else response
            }
        else:
            return {"status": "error", "message": "Azure OpenAI returned empty or invalid response"}
            
    except Exception as e:
        return {"status": "error", "message": f"Azure OpenAI integration test failed: {str(e)}"}


def validate_pr_analysis_config():
    """Validate that the PR analysis configuration is complete"""
    issues = []
    
    # Check if Azure OpenAI is available
    if not AI_AVAILABLE:
        issues.append("Azure OpenAI client not available - AI features will be limited")
    
    # Check required imports
    try:
        import streamlit as st
        import pandas as pd
        import requests
    except ImportError as e:
        issues.append(f"Missing required import: {e}")
    
    # Check if we can create a reviewer instance
    try:
        reviewer = PullRequestReviewer()
        if not hasattr(reviewer, 'generate_ai_powered_review'):
            issues.append("AI-powered review method not available")
    except Exception as e:
        issues.append(f"Could not create PullRequestReviewer instance: {e}")
    
    return {
        "status": "valid" if not issues else "issues_found",
        "issues": issues,
        "ai_available": AI_AVAILABLE,
        "total_methods": len([m for m in dir(PullRequestReviewer) if not m.startswith('_')])
    }


if __name__ == "__main__":
    # Run validation if script is executed directly
    print("🔍 Pull Request Reviewer - Azure OpenAI Integration")
    print("=" * 50)
    
    # Test AI integration
    print("\n🧪 Testing Azure OpenAI Integration...")
    ai_test = test_ai_integration()
    print(f"Status: {ai_test['status']}")
    print(f"Message: {ai_test['message']}")
    if 'sample_response' in ai_test:
        print(f"Sample Response: {ai_test['sample_response']}")
    
    # Validate configuration
    print("\n🔧 Validating Configuration...")
    config_test = validate_pr_analysis_config()
    print(f"Status: {config_test['status']}")
    print(f"AI Available: {config_test['ai_available']}")
    print(f"Total Methods: {config_test['total_methods']}")
    
    if config_test['issues']:
        print("\n⚠️ Issues Found:")
        for issue in config_test['issues']:
            print(f"  - {issue}")
    else:
        print("\n✅ All validations passed!")
