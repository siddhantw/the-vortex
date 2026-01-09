"""
Robot Framework Dashboard Analytics Module
Integrates with robotframework-dashboard to provide AI-powered insights
on test results across multiple Jenkins runs using Azure OpenAI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from requests.auth import HTTPBasicAuth
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from collections import defaultdict
from urllib.parse import quote
from io import BytesIO
import xml.etree.ElementTree as ET
import re

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("RFDashboardAnalytics", level=logging.INFO, log_file="rf_dashboard_analytics.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# Ensure parent directory is in path to import shared modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import Azure OpenAI Client
try:
    from azure_openai_client import AzureOpenAIClient
    AZURE_AVAILABLE = True
    logger.info("Azure OpenAI client imported successfully")
except ImportError as e:
    AZURE_AVAILABLE = False
    logger.warning(f"Azure OpenAI client not available: {e}")

# Import notifications module
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    logger.warning("Notifications module not available")

# Import self-healing analyzer
try:
    listeners_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'listeners')
    sys.path.insert(0, listeners_path)
    from healing_analyzer import HealingAnalyzer
    HEALING_ANALYZER_AVAILABLE = True
    logger.info("Healing analyzer loaded successfully")
except ImportError as e:
    HEALING_ANALYZER_AVAILABLE = False
    HealingAnalyzer = None  # Define for type checking
    logger.warning(f"Healing analyzer not available: {e}")

# Import PDF generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    PDF_AVAILABLE = True
    logger.info("PDF generation libraries loaded successfully")
except ImportError as e:
    PDF_AVAILABLE = False
    logger.warning(f"PDF generation not available: {e}. Install with: pip install reportlab")


class RobotFrameworkDashboardClient:
    """Client to interact with Robot Framework Dashboard and Jenkins"""

    def __init__(self, jenkins_url: str, username: str, api_token: str):
        """
        Initialize the RF Dashboard client

        Args:
            jenkins_url: Jenkins server URL
            username: Jenkins username
            api_token: Jenkins API token
        """
        self.jenkins_url = jenkins_url.rstrip('/')
        self.username = username
        self.api_token = api_token
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, api_token)
        self.session.verify = False  # For self-signed certificates
        # Disable SSL warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def get_jobs(self, folder_path: str = "") -> List[Dict[str, Any]]:
        """Get all Jenkins jobs, including jobs inside folders

        Args:
            folder_path: Path to folder (e.g., 'job/FolderName') for recursive fetching
        """
        # Build URL - either root or specific folder
        if folder_path:
            url = f"{self.jenkins_url}/{folder_path}/api/json"
        else:
            url = f"{self.jenkins_url}/api/json"

        try:
            params = {'tree': 'jobs[name,url,color,lastBuild[number],_class]'}

            logger.info(f"Fetching jobs from: {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            items = data.get('jobs', [])

            all_jobs = []
            folders_found = []

            for item in items:
                item_class = item.get('_class', '')
                item_name = item.get('name', 'Unknown')

                # Check if this is a folder
                if 'folder' in item_class.lower() or 'Folder' in item_class:
                    folders_found.append(item_name)
                    # Recursively get jobs from this folder
                    folder_url_path = f"job/{quote(item_name, safe='')}" if not folder_path else f"{folder_path}/job/{quote(item_name, safe='')}"
                    logger.info(f"Found folder: {item_name}, fetching jobs from it...")
                    folder_jobs = self.get_jobs(folder_url_path)
                    # Add folder prefix to job names for clarity
                    for job in folder_jobs:
                        job['folder'] = f"{folder_path}/job/{item_name}" if folder_path else f"job/{item_name}"
                        job['display_name'] = f"{item_name}/{job.get('display_name', job['name'])}"
                    all_jobs.extend(folder_jobs)
                else:
                    # This is a regular job
                    item['folder'] = folder_path
                    item['display_name'] = item_name
                    all_jobs.append(item)

            if not folder_path:
                logger.info(f"Successfully fetched {len(all_jobs)} jobs from Jenkins (including {len(folders_found)} folders)")
                if folders_found:
                    logger.info(f"Folders found: {folders_found}")
                # Log first few job names for debugging
                if all_jobs:
                    sample_jobs = [j.get('display_name', j.get('name', 'Unknown')) for j in all_jobs[:5]]
                    logger.info(f"Sample jobs: {sample_jobs}")

            return all_jobs
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching jobs from {url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching jobs from {url}: {type(e).__name__}: {e}")
            return []

    def get_job_builds(self, job_info: Dict[str, Any], max_builds: int = 50) -> List[Dict[str, Any]]:
        """Get builds for a specific job

        Args:
            job_info: Job dictionary containing 'name', 'folder', and 'display_name'
            max_builds: Maximum number of builds to fetch
        """
        job_name = job_info.get('name')
        folder_path = job_info.get('folder', '')
        display_name = job_info.get('display_name', job_name)

        # Build the full job path
        if folder_path:
            # folder_path already includes 'job/' prefixes
            url = f"{self.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/api/json"
        else:
            url = f"{self.jenkins_url}/job/{quote(job_name, safe='')}/api/json"

        try:
            params = {
                'tree': f'builds[number,result,timestamp,duration,url]{{0,{max_builds}}}'
            }

            logger.info(f"Fetching builds for job '{display_name}' from: {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            builds = data.get('builds', [])
            logger.info(f"Successfully fetched {len(builds)} builds for job: {display_name}")

            if builds:
                logger.info(f"Build numbers: {[b.get('number') for b in builds[:5]]}")

            return builds
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Job not found: {display_name}")
                logger.error(f"Attempted URL: {url}")
            else:
                logger.error(f"HTTP {e.response.status_code} error fetching builds for {display_name}: {e}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching builds for {display_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching builds for {display_name}: {type(e).__name__}: {e}")
            return []

    def get_build_test_results(self, job_info: Dict[str, Any], build_number: int) -> Optional[Dict[str, Any]]:
        """Get Robot Framework test results for a specific build

        Args:
            job_info: Job dictionary containing 'name', 'folder', and 'display_name'
            build_number: Build number
        """
        job_name = job_info.get('name')
        folder_path = job_info.get('folder', '')
        display_name = job_info.get('display_name', job_name)

        # Build the full job path
        if folder_path:
            url = f"{self.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/{build_number}/robot/api/json"
        else:
            url = f"{self.jenkins_url}/job/{quote(job_name, safe='')}/{build_number}/robot/api/json"

        try:
            logger.info(f"Fetching RF results from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Log the structure we received for debugging
            logger.info(f"RF API Response keys for build #{build_number}: {list(data.keys())}")
            logger.info(f"RF API Response sample: totalCount={data.get('totalCount', 'N/A')}, passCount={data.get('passCount', 'N/A')}, total={data.get('total', 'N/A')}")

            return data
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"No Robot Framework results found for {display_name} #{build_number} (404)")
            else:
                logger.error(f"HTTP {e.response.status_code} error fetching test results for {display_name} #{build_number}")
                if e.response.text:
                    logger.error(f"Response text: {e.response.text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error fetching test results for {display_name} #{build_number}: {type(e).__name__}: {e}")
            return None

    def get_build_output_xml(self, job_name: str, build_number: int, folder_path: str = "") -> Optional[str]:
        """Get Robot Framework output.xml for detailed analysis"""
        try:
            if folder_path:
                url = f"{self.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/{build_number}/robot/report/output.xml"
            else:
                url = f"{self.jenkins_url}/job/{quote(job_name, safe='')}/{build_number}/robot/report/output.xml"

            logger.info(f"Fetching output.xml from: {url}")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Error fetching output.xml for {job_name} #{build_number}: {e}")
            return None

    def get_build_console_log(self, job_name: str, build_number: int, folder_path: str = "") -> Optional[str]:
        """Get Jenkins console log for a build"""
        try:
            if folder_path:
                url = f"{self.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/{build_number}/consoleText"
            else:
                url = f"{self.jenkins_url}/job/{quote(job_name, safe='')}/{build_number}/consoleText"

            logger.info(f"Fetching console log from: {url}")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            # Get last 50KB to avoid memory issues with very large logs
            log_text = response.text
            if len(log_text) > 50000:
                log_text = log_text[-50000:]
            return log_text
        except Exception as e:
            logger.warning(f"Error fetching console log for {job_name} #{build_number}: {e}")
            return None

    def get_build_log_html(self, job_name: str, build_number: int, folder_path: str = "") -> Optional[str]:
        """Get Robot Framework log.html for a build"""
        try:
            if folder_path:
                url = f"{self.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/{build_number}/robot/report/log.html"
            else:
                url = f"{self.jenkins_url}/job/{quote(job_name, safe='')}/{build_number}/robot/report/log.html"

            logger.info(f"Fetching log.html from: {url}")
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Error fetching log.html for {job_name} #{build_number}: {e}")
            return None


    def parse_output_xml(self, xml_content: str) -> Dict[str, Any]:
        """Parse Robot Framework output.xml for detailed test information"""
        try:
            root = ET.fromstring(xml_content)

            parsed_data = {
                'suites': [],
                'failed_tests_details': [],
                'statistics': {},
                'errors': []
            }

            # Parse test suites recursively
            def parse_suite(suite_elem, parent_name=""):
                suite_name = suite_elem.get('name', 'Unknown')
                full_name = f"{parent_name}.{suite_name}" if parent_name else suite_name

                suite_info = {
                    'name': full_name,
                    'tests': []
                }

                # Parse tests in this suite
                for test_elem in suite_elem.findall('.//test'):
                    test_name = test_elem.get('name', 'Unknown')
                    status_elem = test_elem.find('status')

                    if status_elem is not None:
                        status = status_elem.get('status', 'UNKNOWN')
                        start_time = status_elem.get('starttime', '')
                        end_time = status_elem.get('endtime', '')

                        test_info = {
                            'name': test_name,
                            'full_name': f"{full_name}.{test_name}",
                            'status': status,
                            'start_time': start_time,
                            'end_time': end_time
                        }

                        # Get failure message if test failed
                        if status == 'FAIL':
                            message = status_elem.text if status_elem.text else ''

                            # Get detailed keyword failure information
                            failed_keywords = []
                            for kw_elem in test_elem.findall('.//kw'):
                                kw_status = kw_elem.find('status')
                                if kw_status is not None and kw_status.get('status') == 'FAIL':
                                    kw_name = kw_elem.get('name', 'Unknown')
                                    kw_message = kw_status.text if kw_status.text else ''
                                    failed_keywords.append({
                                        'keyword': kw_name,
                                        'message': kw_message
                                    })

                            test_info['message'] = message
                            test_info['failed_keywords'] = failed_keywords
                            parsed_data['failed_tests_details'].append(test_info)

                        suite_info['tests'].append(test_info)

                # Recursively parse child suites
                for child_suite in suite_elem.findall('suite'):
                    child_info = parse_suite(child_suite, full_name)
                    parsed_data['suites'].append(child_info)

                return suite_info

            # Parse all top-level suites
            for suite in root.findall('.//suite'):
                suite_info = parse_suite(suite)
                parsed_data['suites'].append(suite_info)

            # Parse statistics
            stats = root.find('.//statistics')
            if stats is not None:
                total_elem = stats.find('.//total/stat')
                if total_elem is not None:
                    parsed_data['statistics'] = {
                        'total': total_elem.get('pass', '0'),
                        'pass': total_elem.get('pass', '0'),
                        'fail': total_elem.get('fail', '0')
                    }

            # Parse errors
            for error_elem in root.findall('.//errors/msg'):
                parsed_data['errors'].append(error_elem.text or '')

            return parsed_data

        except ET.ParseError as e:
            logger.error(f"Error parsing output.xml: {e}")
            return {'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error parsing output.xml: {e}")
            return {'error': str(e)}


class RFTestMetrics:
    """Container for Robot Framework test metrics"""

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.execution_time = 0
        self.pass_rate = 0.0
        self.build_number = 0
        self.timestamp = None
        self.test_details = []
        self.suite_details = []
        self.failed_test_details = []
        self.keyword_statistics = {}
        self.tag_statistics = {}
        self.console_log = None
        self.detailed_failures = []


class RFDashboardAnalyzer:
    """Analyzer for Robot Framework Dashboard data with AI insights"""

    def __init__(self, azure_client: Optional[AzureOpenAIClient] = None):
        """Initialize analyzer with optional Azure OpenAI client"""
        self.azure_client = azure_client
        self.metrics_history = []

    def parse_robot_results(self, robot_data: Dict[str, Any], build_info: Dict[str, Any]) -> RFTestMetrics:
        """Parse Robot Framework test results from Jenkins API"""
        metrics = RFTestMetrics()

        try:
            build_num = build_info.get('number', 'unknown')

            # Log the complete structure we received for debugging
            logger.info(f"=== Parsing Build #{build_num} ===")
            logger.info(f"Robot data keys: {list(robot_data.keys())}")
            logger.info(f"Robot data sample: {str(robot_data)[:500]}")

            # Basic statistics - try different field names as Jenkins Robot Plugin versions vary
            # Try all possible field name combinations
            metrics.total_tests = (robot_data.get('totalCount') or
                                 robot_data.get('total') or
                                 robot_data.get('overallTotal') or 0)
            metrics.passed_tests = (robot_data.get('passCount') or
                                  robot_data.get('passed') or
                                  robot_data.get('overallPassed') or 0)
            metrics.failed_tests = (robot_data.get('failCount') or
                                  robot_data.get('failed') or
                                  robot_data.get('overallFailed') or 0)
            metrics.skipped_tests = (robot_data.get('skipCount') or
                                   robot_data.get('skipped') or 0)

            # Try multiple field names for execution time (in milliseconds)
            # First try robot_data (test execution time)
            metrics.execution_time = (robot_data.get('duration') or
                                     robot_data.get('totalDuration') or
                                     robot_data.get('overallDuration') or
                                     robot_data.get('time') or 0)

            # If not found in robot_data, try build_info (build-level duration)
            if metrics.execution_time == 0 and build_info:
                metrics.execution_time = (build_info.get('duration') or
                                         build_info.get('estimatedDuration') or 0)
                if metrics.execution_time > 0:
                    logger.info(f"Using build duration from build_info: {metrics.execution_time}ms")

            logger.info(f"Extracted: total={metrics.total_tests}, passed={metrics.passed_tests}, failed={metrics.failed_tests}, execution_time={metrics.execution_time}ms")

            # Calculate pass rate
            if metrics.total_tests > 0:
                metrics.pass_rate = (metrics.passed_tests / metrics.total_tests) * 100
            else:
                # If totalCount is 0 but we have passed/failed, recalculate
                actual_total = metrics.passed_tests + metrics.failed_tests + metrics.skipped_tests
                if actual_total > 0:
                    logger.info(f"Recalculating total from components: {actual_total}")
                    metrics.total_tests = actual_total
                    metrics.pass_rate = (metrics.passed_tests / metrics.total_tests) * 100
                else:
                    # Still 0? Try to count from suites
                    logger.warning(f"All test counts are 0 for build #{build_num}, will try to extract from suites")

            metrics.build_number = build_info.get('number', 0)

            # Handle timestamp - ensure it's valid
            timestamp_ms = build_info.get('timestamp', 0)
            if timestamp_ms > 0:
                metrics.timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
            else:
                metrics.timestamp = datetime.now()

            logger.info(f"Build #{metrics.build_number}: {metrics.passed_tests}/{metrics.total_tests} passed ({metrics.pass_rate:.1f}%)")

            # Extract test details from suites
            if 'suites' in robot_data:
                logger.info(f"Found 'suites' key, type: {type(robot_data['suites'])}")
                if isinstance(robot_data['suites'], list):
                    logger.info(f"Processing {len(robot_data['suites'])} suites")
                    self._extract_suite_details(robot_data['suites'], metrics)
                else:
                    # Sometimes it's a single suite object
                    self._extract_suite_details([robot_data['suites']], metrics)
            elif 'suite' in robot_data:
                # Alternative structure
                logger.info(f"Found 'suite' key")
                self._extract_suite_details([robot_data['suite']], metrics)
            else:
                logger.warning(f"No 'suites' or 'suite' key found in robot_data")

            # If we still have 0 tests after extraction, log it
            if metrics.total_tests == 0 and len(metrics.test_details) > 0:
                logger.info(f"Recalculating total from extracted test details: {len(metrics.test_details)}")
                metrics.total_tests = len(metrics.test_details)
                metrics.passed_tests = len([t for t in metrics.test_details if t.get('status') in ['PASS', 'PASSED']])
                metrics.failed_tests = len([t for t in metrics.test_details if t.get('status') in ['FAIL', 'FAILED', 'FAILURE']])
                if metrics.total_tests > 0:
                    metrics.pass_rate = (metrics.passed_tests / metrics.total_tests) * 100

            # If execution_time is still 0 or missing, try to calculate from suite_details or test_details
            if metrics.execution_time == 0:
                if metrics.suite_details:
                    # Sum up suite durations
                    suite_duration = sum(s.get('duration', 0) for s in metrics.suite_details)
                    if suite_duration > 0:
                        metrics.execution_time = suite_duration
                        logger.info(f"Calculated execution_time from suite_details: {metrics.execution_time}ms")
                elif metrics.test_details:
                    # Sum up test durations as last resort
                    test_duration = sum(t.get('duration', 0) for t in metrics.test_details)
                    if test_duration > 0:
                        metrics.execution_time = test_duration
                        logger.info(f"Calculated execution_time from test_details: {metrics.execution_time}ms")

                if metrics.execution_time == 0:
                    logger.warning(f"Build #{build_num}: execution_time is still 0 after all fallback attempts")

            logger.info(f"Final metrics for build #{build_num}: total={metrics.total_tests}, passed={metrics.passed_tests}, failed={metrics.failed_tests}, execution_time={metrics.execution_time}ms, test_details={len(metrics.test_details)}")

        except Exception as e:
            logger.error(f"Error parsing robot results for build #{build_info.get('number', 'unknown')}: {type(e).__name__}: {e}")
            logger.error(f"Robot data structure: {str(robot_data)[:1000]}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return metrics

    def _extract_suite_details(self, suites: List[Dict], metrics: RFTestMetrics):
        """Extract detailed information from test suites"""
        if not suites:
            logger.warning("No suites provided to extract")
            return

        logger.info(f"Extracting from {len(suites)} suite(s)")

        for idx, suite in enumerate(suites):
            if not isinstance(suite, dict):
                logger.warning(f"Skipping invalid suite at index {idx}: {type(suite)}")
                continue

            suite_name = suite.get('name', 'Unknown')
            logger.info(f"Processing suite: {suite_name}, keys: {list(suite.keys())}")

            suite_info = {
                'name': suite_name,
                'passed': suite.get('passCount', suite.get('passed', 0)),
                'failed': suite.get('failCount', suite.get('failed', 0)),
                'duration': suite.get('duration', 0)
            }
            metrics.suite_details.append(suite_info)

            # Extract test cases - handle different field names
            test_cases = suite.get('testCases', suite.get('cases', suite.get('tests', [])))
            logger.info(f"Suite '{suite_name}' has {len(test_cases) if test_cases else 0} test cases")

            if test_cases:
                for test_idx, test in enumerate(test_cases):
                    if not isinstance(test, dict):
                        logger.warning(f"Skipping invalid test at index {test_idx} in suite '{suite_name}'")
                        continue

                    test_name = test.get('name', 'Unknown')
                    test_status = test.get('status', test.get('result', 'UNKNOWN'))

                    test_info = {
                        'name': test_name,
                        'status': test_status,
                        'duration': test.get('duration', 0),
                        'suite': suite_name
                    }
                    metrics.test_details.append(test_info)

                    logger.debug(f"  Test: {test_name} - {test_status}")

                    # Track failed tests
                    status = test_info['status'].upper()
                    if status in ['FAIL', 'FAILED', 'FAILURE']:
                        failed_info = test_info.copy()
                        failed_info['message'] = test.get('errorMsg', test.get('error', test.get('message', 'No error message')))
                        metrics.failed_test_details.append(failed_info)

            # Recursively process nested suites - handle different field names
            child_suites = suite.get('childSuites', suite.get('children', suite.get('suites', [])))
            if child_suites:
                logger.info(f"Suite '{suite_name}' has {len(child_suites)} child suite(s)")
                self._extract_suite_details(child_suites, metrics)
            else:
                logger.debug(f"Suite '{suite_name}' has no child suites")

    def analyze_trends(self, metrics_list: List[RFTestMetrics]) -> Dict[str, Any]:
        """Analyze trends across multiple test runs"""
        if not metrics_list:
            logger.warning("No metrics provided for trend analysis")
            return {}

        # Filter out metrics with no tests
        valid_metrics = [m for m in metrics_list if m.total_tests > 0]
        if not valid_metrics:
            logger.warning("No valid metrics (all have 0 tests)")
            return {
                'total_runs': len(metrics_list),
                'average_pass_rate': 0.0,
                'error': 'No valid test data found'
            }

        try:
            # Use valid_metrics for calculations
            pass_rates = [m.pass_rate for m in valid_metrics]
            exec_times = [m.execution_time for m in valid_metrics]

            analysis = {
                'total_runs': len(valid_metrics),
                'average_pass_rate': float(np.mean(pass_rates)) if pass_rates else 0.0,
                'pass_rate_trend': self._calculate_trend(pass_rates),
                'average_execution_time': float(np.mean(exec_times)) if exec_times else 0.0,
                'execution_time_trend': self._calculate_trend(exec_times),
                'stability_score': self._calculate_stability(valid_metrics),
                'most_failed_tests': self._get_most_failed_tests(valid_metrics),
                'slowest_tests': self._get_slowest_tests(valid_metrics),
                'flaky_tests': self._detect_flaky_tests(valid_metrics)
            }

            logger.info(f"Analysis complete: {len(valid_metrics)} builds, avg pass rate: {analysis['average_pass_rate']:.1f}%")

            return analysis
        except Exception as e:
            logger.error(f"Error in analyze_trends: {e}")
            return {
                'total_runs': len(metrics_list),
                'error': str(e)
            }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if values are trending up, down, or stable"""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "degrading"

    def _calculate_stability(self, metrics_list: List[RFTestMetrics]) -> float:
        """Calculate test stability score (0-100)"""
        if not metrics_list:
            return 0.0

        pass_rates = [m.pass_rate for m in metrics_list]

        # Stability is inverse of variance
        variance = np.var(pass_rates)
        stability = max(0, 100 - variance)

        return round(stability, 2)

    def enrich_metrics_with_detailed_analysis(self, metrics_list: List[RFTestMetrics],
                                               jenkins_client: 'RobotFrameworkDashboardClient',
                                               job_info: Dict[str, Any],
                                               max_builds_to_analyze: int = 5) -> List[RFTestMetrics]:
        """Enrich metrics with detailed analysis from output.xml and console logs"""

        logger.info(f"Starting detailed analysis enrichment for up to {max_builds_to_analyze} builds...")

        builds_to_analyze = metrics_list[:max_builds_to_analyze]
        job_name = job_info.get('name')
        folder_path = job_info.get('folder', '')

        for i, metrics in enumerate(builds_to_analyze):
            build_num = metrics.build_number
            logger.info(f"Enriching build #{build_num} ({i+1}/{len(builds_to_analyze)})...")

            try:
                # Fetch output.xml for detailed error information
                xml_content = jenkins_client.get_build_output_xml(job_name, build_num, folder_path)
                if xml_content:
                    parsed_xml = jenkins_client.parse_output_xml(xml_content)
                    if 'failed_tests_details' in parsed_xml:
                        metrics.detailed_failures = parsed_xml['failed_tests_details']
                        logger.info(f"  - Extracted {len(metrics.detailed_failures)} detailed failure records from XML")

                # Fetch console log for build-level errors
                console_log = jenkins_client.get_build_console_log(job_name, build_num, folder_path)
                if console_log:
                    metrics.console_log = console_log
                    # Extract common error patterns from console
                    error_patterns = self._extract_error_patterns_from_log(console_log)
                    if error_patterns:
                        logger.info(f"  - Identified {len(error_patterns)} error patterns in console log")
                        # Store error patterns in metrics for later analysis
                        if not hasattr(metrics, 'error_patterns'):
                            metrics.error_patterns = []
                        metrics.error_patterns.extend(error_patterns)

            except Exception as e:
                logger.warning(f"Error enriching build #{build_num}: {e}")
                continue

        logger.info("Detailed analysis enrichment complete")
        return metrics_list

    def _extract_error_patterns_from_log(self, console_log: str) -> List[Dict[str, str]]:
        """Extract common error patterns from console log"""
        patterns = []

        # Common error patterns
        error_indicators = [
            (r'(?:ERROR|FATAL|CRITICAL):\s*(.{0,200})', 'Error'),
            (r'Exception:\s*(.{0,200})', 'Exception'),
            (r'(?:Timeout|TIMEOUT):\s*(.{0,200})', 'Timeout'),
            (r'(?:Connection refused|ConnectionError):\s*(.{0,200})', 'Connection Error'),
            (r'(?:ElementNotFound|NoSuchElement):\s*(.{0,200})', 'Element Not Found'),
            (r'(?:AssertionError|Assertion Failed):\s*(.{0,200})', 'Assertion Failed'),
            (r'(?:PermissionError|Access Denied):\s*(.{0,200})', 'Permission Error'),
        ]

        for pattern, error_type in error_indicators:
            matches = re.finditer(pattern, console_log, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                patterns.append({
                    'type': error_type,
                    'message': match.group(1).strip() if match.group(1) else match.group(0).strip()
                })
                if len(patterns) >= 20:  # Limit to 20 patterns
                    break
            if len(patterns) >= 20:
                break

        return patterns

    def _get_most_failed_tests(self, metrics_list: List[RFTestMetrics]) -> List[Dict[str, Any]]:
        """Get tests that fail most frequently with detailed error information"""
        failure_count = defaultdict(int)
        test_run_count = defaultdict(int)  # Track how many times each test actually ran
        failure_details = defaultdict(lambda: {
            'messages': [],
            'suites': set(),
            'builds': [],
            'error_types': defaultdict(int),
            'stack_traces': []
        })

        for metrics in metrics_list:
            build_num = metrics.build_number
            
            # Track tests that failed and all tests that ran in this build
            tests_failed_in_build = set()
            tests_in_build = set()

            # Use detailed failures if available
            if hasattr(metrics, 'detailed_failures') and metrics.detailed_failures:
                for failed_test in metrics.detailed_failures:
                    test_name = failed_test.get('name', 'Unknown')
                    # Only count one failure per test per build
                    if test_name not in tests_failed_in_build:
                        failure_count[test_name] += 1
                        tests_failed_in_build.add(test_name)
                        failure_details[test_name]['builds'].append(build_num)
                    tests_in_build.add(test_name)

                    message = failed_test.get('message', '')
                    failure_details[test_name]['messages'].append(message)

                    # Extract suite name from full_name (e.g., "Suite.Subsuite.TestName" -> "Suite.Subsuite")
                    full_name = failed_test.get('full_name', '')
                    if full_name and '.' in full_name:
                        # Get everything except the last part (test name)
                        suite_name = '.'.join(full_name.split('.')[:-1])
                        failure_details[test_name]['suites'].add(suite_name if suite_name else 'Unknown')
                    else:
                        # Fallback: try to get suite from other fields
                        suite_name = failed_test.get('suite', 'Unknown')
                        failure_details[test_name]['suites'].add(suite_name)

                    # Categorize error type
                    error_type = self._categorize_error(message)
                    failure_details[test_name]['error_types'][error_type] += 1

                    # Store failed keywords as stack trace
                    if 'failed_keywords' in failed_test:
                        failure_details[test_name]['stack_traces'].append({
                            'build': build_num,
                            'keywords': failed_test['failed_keywords']
                        })
            else:
                # Fallback to basic failed test details
                for failed_test in metrics.failed_test_details:
                    test_name = failed_test['name']
                    # Only count one failure per test per build
                    if test_name not in tests_failed_in_build:
                        failure_count[test_name] += 1
                        tests_failed_in_build.add(test_name)
                        failure_details[test_name]['builds'].append(build_num)
                    tests_in_build.add(test_name)
                    failure_details[test_name]['messages'].append(failed_test.get('message', ''))
                    failure_details[test_name]['suites'].add(failed_test.get('suite', 'Unknown'))

            # Track all tests from test_details (includes both passed and failed)
            if hasattr(metrics, 'test_details') and metrics.test_details:
                for test in metrics.test_details:
                    test_name = test.get('name', 'Unknown')
                    tests_in_build.add(test_name)

                    # Capture suite info for all tests (even passing ones)
                    # This ensures we have suite data even if a test passes sometimes
                    suite_name = test.get('suite', 'Unknown')
                    if suite_name and suite_name != 'Unknown':
                        failure_details[test_name]['suites'].add(suite_name)

            # Increment run count for all tests that ran in this build
            for test_name in tests_in_build:
                test_run_count[test_name] += 1

        # Sort by failure count
        sorted_failures = sorted(failure_count.items(), key=lambda x: x[1], reverse=True)

        result = []
        for test_name, count in sorted_failures[:10]:
            details = failure_details[test_name]
            
            # Calculate actual failure rate based on how many times the test ran
            actual_runs = test_run_count.get(test_name, count)  # Fallback to failure count if no run data
            actual_failure_rate = (count / actual_runs) * 100 if actual_runs > 0 else 0

            # Get most common error type
            common_error = max(details['error_types'].items(), key=lambda x: x[1])[0] if details['error_types'] else 'Unknown'

            result.append({
                'test': test_name,
                'failure_count': count,
                'total_runs': actual_runs,
                'failure_rate': round(actual_failure_rate, 1),
                'suites': list(details['suites']) if details['suites'] else ['Unknown'],
                'sample_messages': details['messages'][:3],
                'all_messages': details['messages'],
                'failed_in_builds': details['builds'],
                'common_error_type': common_error,
                'error_breakdown': dict(details['error_types']),
                'stack_traces': details['stack_traces'][:3]  # Include up to 3 stack traces
            })

        return result

    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message into common error types"""
        error_message_lower = error_message.lower()

        if 'timeout' in error_message_lower or 'timed out' in error_message_lower:
            return 'Timeout'
        elif 'element' in error_message_lower and ('not found' in error_message_lower or 'not visible' in error_message_lower):
            return 'Element Not Found'
        elif 'assertion' in error_message_lower or 'expected' in error_message_lower:
            return 'Assertion Failed'
        elif 'connection' in error_message_lower or 'network' in error_message_lower:
            return 'Connection Error'
        elif 'permission' in error_message_lower or 'access denied' in error_message_lower:
            return 'Permission Error'
        elif 'exception' in error_message_lower or 'error' in error_message_lower:
            return 'Exception'
        else:
            return 'Other'

    def _get_slowest_tests(self, metrics_list: List[RFTestMetrics]) -> List[Dict[str, Any]]:
        """Get slowest running tests with detailed timing analysis"""
        test_durations = defaultdict(lambda: {'durations': [], 'builds': [], 'suites': set()})

        for metrics in metrics_list:
            build_num = metrics.build_number
            for test in metrics.test_details:
                test_name = test['name']
                test_durations[test_name]['durations'].append(test['duration'])
                test_durations[test_name]['builds'].append(build_num)
                test_durations[test_name]['suites'].add(test.get('suite', 'Unknown'))

        # Calculate statistics for each test
        test_stats = []
        for test_name, data in test_durations.items():
            durations = data['durations']
            if not durations:
                continue

            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            min_duration = np.min(durations)
            std_dev = np.std(durations) if len(durations) > 1 else 0

            # Calculate trend (is it getting slower?)
            if len(durations) >= 3:
                recent_avg = np.mean(durations[-3:])
                older_avg = np.mean(durations[:3] if len(durations) > 3 else durations)
                trend = "Slowing Down" if recent_avg > older_avg * 1.1 else ("Speeding Up" if recent_avg < older_avg * 0.9 else "Stable")
            else:
                trend = "Insufficient Data"

            test_stats.append({
                'test': test_name,
                'avg_duration': round(avg_duration, 2),
                'max_duration': round(max_duration, 2),
                'min_duration': round(min_duration, 2),
                'std_dev': round(std_dev, 2),
                'trend': trend,
                'run_count': len(durations),
                'suites': list(data['suites']),
                'measured_in_builds': data['builds'][:5],  # Show first 5 builds
                'all_durations': [round(d, 2) for d in durations]
            })

        # Sort by average duration
        test_stats.sort(key=lambda x: x['avg_duration'], reverse=True)

        return test_stats[:10]

    def _detect_flaky_tests(self, metrics_list: List[RFTestMetrics]) -> List[Dict[str, Any]]:
        """Detect tests that pass/fail inconsistently with detailed failure pattern analysis"""
        test_results = defaultdict(lambda: {
            'statuses': [],
            'builds': [],
            'failure_messages': [],
            'durations': [],
            'suites': set()
        })

        for metrics in metrics_list:
            build_num = metrics.build_number

            # Collect test execution data
            for test in metrics.test_details:
                test_name = test['name']
                test_results[test_name]['statuses'].append(test['status'])
                test_results[test_name]['builds'].append(build_num)
                test_results[test_name]['durations'].append(test.get('duration', 0))
                test_results[test_name]['suites'].add(test.get('suite', 'Unknown'))

            # Collect failure messages from detailed failures if available
            if hasattr(metrics, 'detailed_failures') and metrics.detailed_failures:
                for failure in metrics.detailed_failures:
                    test_name = failure.get('name', 'Unknown')
                    if test_name in test_results:
                        test_results[test_name]['failure_messages'].append({
                            'build': build_num,
                            'message': failure.get('message', 'No message')
                        })

        flaky_tests = []
        for test_name, data in test_results.items():
            results = data['statuses']
            if len(results) < 3:
                continue

            # Count passes and fails
            passes = results.count('PASS')
            fails = results.count('FAIL')
            total = len(results)

            # Flaky if both passes and fails exist and neither dominates completely
            if passes > 0 and fails > 0:
                flakiness = min(passes, fails) / total
                if flakiness > 0.2:  # At least 20% inconsistency

                    # Analyze failure pattern
                    failure_pattern = self._analyze_flaky_pattern(results, data['builds'])

                    # Check if timing-related (fails are significantly slower)
                    avg_pass_duration = 0
                    avg_fail_duration = 0
                    timing_related = False

                    if data['durations']:
                        avg_pass_duration = np.mean([d for i, d in enumerate(data['durations']) if results[i] == 'PASS']) if passes > 0 else 0
                        avg_fail_duration = np.mean([d for i, d in enumerate(data['durations']) if results[i] == 'FAIL']) if fails > 0 else 0
                        timing_related = abs(avg_fail_duration - avg_pass_duration) > avg_pass_duration * 0.3 if avg_pass_duration > 0 else False

                    # Determine likely root cause
                    root_cause_hints = []
                    if timing_related:
                        root_cause_hints.append("Timing/Race condition")
                    if failure_pattern == "Intermittent":
                        root_cause_hints.append("Non-deterministic behavior")
                    if failure_pattern == "Consecutive Failures":
                        root_cause_hints.append("Environment-dependent issue")

                    # Analyze failure messages for patterns
                    unique_errors = set()
                    for fm in data['failure_messages']:
                        error_type = self._categorize_error(fm['message'])
                        unique_errors.add(error_type)

                    if len(unique_errors) > 1:
                        root_cause_hints.append("Multiple error types")

                    flaky_tests.append({
                        'test': test_name,
                        'total_runs': total,
                        'passes': passes,
                        'fails': fails,
                        'flakiness_score': round(flakiness * 100, 2),
                        'failure_pattern': failure_pattern,
                        'timing_related': timing_related,
                        'avg_pass_duration': round(avg_pass_duration, 2) if passes > 0 and data['durations'] else 0,
                        'avg_fail_duration': round(avg_fail_duration, 2) if fails > 0 and data['durations'] else 0,
                        'root_cause_hints': root_cause_hints,
                        'unique_error_types': list(unique_errors),
                        'suites': list(data['suites']),
                        'failure_builds': [data['builds'][i] for i, s in enumerate(results) if s == 'FAIL'],
                        'sample_failure_messages': [fm['message'][:100] for fm in data['failure_messages'][:3]]
                    })

        # Sort by flakiness score
        flaky_tests.sort(key=lambda x: x['flakiness_score'], reverse=True)

        return flaky_tests[:10]

    def _analyze_flaky_pattern(self, statuses: List[str], builds: List[int]) -> str:
        """Analyze the pattern of flaky test failures"""
        # Check for consecutive failures
        consecutive_fails = 0
        max_consecutive = 0

        for status in statuses:
            if status == 'FAIL':
                consecutive_fails += 1
                max_consecutive = max(max_consecutive, consecutive_fails)
            else:
                consecutive_fails = 0

        if max_consecutive >= 3:
            return "Consecutive Failures"

        # Check for alternating pattern
        alternations = 0
        for i in range(len(statuses) - 1):
            if statuses[i] != statuses[i+1]:
                alternations += 1

        if alternations > len(statuses) * 0.6:
            return "Alternating"

        return "Intermittent"

    def generate_ai_insights(self, analysis: Dict[str, Any], metrics_list: List[RFTestMetrics] = None) -> Dict[str, Any]:
        """Generate comprehensive AI-powered insights with root cause analysis"""
        if not self.azure_client or not AZURE_AVAILABLE:
            return self._generate_enhanced_basic_insights(analysis, metrics_list or [])

        try:
            # Perform deep analysis
            root_causes = self._analyze_root_causes(analysis, metrics_list or [])
            patterns = self._detect_test_patterns(metrics_list or [])
            quality_score = self._calculate_quality_score(analysis, metrics_list or [])

            # Prepare comprehensive prompt for AI analysis
            prompt = self._create_comprehensive_analysis_prompt(
                analysis, root_causes, patterns, quality_score
            )

            # Get AI insights with more tokens for detailed analysis
            response = self.azure_client.generate_response(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.7
            )

            # Parse AI response
            insights = self._parse_ai_response(response)

            # If parsing failed, use enhanced basic insights
            if insights is None:
                logger.warning("AI response parsing failed, using enhanced basic insights")
                return self._generate_enhanced_basic_insights(analysis, metrics_list or [])

            # Enhance with calculated metrics
            insights['root_causes'] = root_causes
            insights['patterns'] = patterns
            insights['quality_score'] = quality_score

            # Ensure we have the actionable structure
            if 'immediate_actions' not in insights:
                actionable = self._generate_actionable_items(analysis, root_causes)
                insights['immediate_actions'] = actionable.get('immediate', [])
                insights['short_term_recommendations'] = actionable.get('short_term', [])
                insights['long_term_strategy'] = actionable.get('long_term', [])
            else:
                # Normalize actionable items from AI to ensure complete data
                insights['immediate_actions'] = self._normalize_actionable_items(insights.get('immediate_actions', []), analysis, root_causes, 'immediate')
                insights['short_term_recommendations'] = self._normalize_actionable_items(insights.get('short_term_recommendations', []), analysis, root_causes, 'short_term')
                insights['long_term_strategy'] = self._normalize_actionable_items(insights.get('long_term_strategy', []), analysis, root_causes, 'long_term')

            # Normalize quick_wins - AI might return strings instead of dicts
            if 'quick_wins' in insights:
                insights['quick_wins'] = self._normalize_quick_wins(insights['quick_wins'], analysis)
            else:
                insights['quick_wins'] = self._identify_quick_wins(analysis)

            # Normalize critical_issues - AI might return improper structure
            if 'critical_issues' in insights:
                insights['critical_issues'] = self._normalize_critical_issues(insights['critical_issues'], analysis, root_causes)
            else:
                insights['critical_issues'] = self._identify_critical_issues(analysis, root_causes)

            return insights

        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return self._generate_enhanced_basic_insights(analysis, metrics_list or [])

    def _analyze_root_causes(self, analysis: Dict, metrics_list: List[RFTestMetrics]) -> Dict[str, List[str]]:
        """Analyze root causes of failures using pattern matching and error analysis with specific details"""
        root_causes = {
            'environmental': [],
            'test_design': [],
            'application': [],
            'infrastructure': [],
            'timing': [],
            'data': []
        }

        try:
            # Get key metrics
            failed_tests = analysis.get('most_failed_tests', [])
            flaky_tests = analysis.get('flaky_tests', [])
            slow_tests = analysis.get('slowest_tests', [])
            stability_score = analysis.get('stability_score', 0)
            pass_rate = analysis.get('average_pass_rate', 0)

            # Environmental issues - Flaky tests indicate environment instability
            if flaky_tests:
                flakiness_severity = 'CRITICAL' if len(flaky_tests) > 10 else 'HIGH' if len(flaky_tests) > 5 else 'MEDIUM' if len(flaky_tests) > 2 else 'LOW'
                avg_flakiness = sum(t.get('flakiness_score', 0) for t in flaky_tests) / len(flaky_tests)

                example_tests = ', '.join([t['test'][:40] + '...' if len(t['test']) > 40 else t['test'] for t in flaky_tests[:2]])

                root_causes['environmental'].append(
                    f"[{flakiness_severity}] {len(flaky_tests)} flaky tests detected (avg flakiness: {avg_flakiness:.1f}%) - "
                    f"indicates environment instability, race conditions, or timing issues. "
                    f"Examples: {example_tests}"
                )

                # Add specific recommendation based on flaky patterns
                highly_flaky = [t for t in flaky_tests if t.get('flakiness_score', 0) > 40]
                if highly_flaky:
                    root_causes['environmental'].append(
                        f"[HIGH] {len(highly_flaky)} tests with >40% flakiness - severe instability requiring "
                        f"immediate investigation of test environment, data setup, or wait strategies"
                    )

            # Stability analysis
            if stability_score < 70:
                variance = 100 - stability_score
                root_causes['environmental'].append(
                    f"[HIGH] Low stability score ({stability_score:.1f}/100, variance: {variance:.1f}) - "
                    f"Pass rates vary significantly between builds, indicating inconsistent test environment, "
                    f"data dependencies, or external service issues"
                )
            elif stability_score < 85:
                root_causes['environmental'].append(
                    f"[MEDIUM] Moderate stability ({stability_score:.1f}/100) - "
                    f"Some variation in test results between builds. Check for shared resources or data refresh issues"
                )

            # Timing issues - Slow tests may cause timeouts
            if slow_tests:
                very_slow = [t for t in slow_tests if t.get('avg_duration', 0) > 60000]  # > 1 minute
                if very_slow:
                    total_slow_time = sum(t.get('avg_duration', 0) for t in very_slow) / 1000
                    example_slow = ', '.join([t['test'][:30] + '...' for t in very_slow[:2]])

                    root_causes['timing'].append(
                        f"[HIGH] {len(very_slow)} tests taking >60s (total: {total_slow_time:.1f}s) - "
                        f"potential timeout issues, excessive waits, or inefficient operations. "
                        f"Examples: {example_slow}"
                    )

                    # Identify extremely slow tests
                    extremely_slow = [t for t in very_slow if t.get('avg_duration', 0) > 120000]  # > 2 minutes
                    if extremely_slow:
                        root_causes['timing'].append(
                            f"[CRITICAL] {len(extremely_slow)} tests taking >2 minutes - "
                            f"review for unnecessary sleeps, unoptimized operations, or database queries"
                        )

            # Test design issues - High failure rate indicates poor test design
            if failed_tests:
                chronic_failures = [t for t in failed_tests if t.get('failure_rate', 0) > 70]
                if chronic_failures:
                    avg_failure_rate = sum(t.get('failure_rate', 0) for t in chronic_failures) / len(chronic_failures)
                    example_chronic = ', '.join([t['test'][:35] + '...' for t in chronic_failures[:2]])

                    root_causes['test_design'].append(
                        f"[CRITICAL] {len(chronic_failures)} tests failing >70% (avg: {avg_failure_rate:.1f}%) - "
                        f"likely broken tests, invalid assertions, or outdated test logic. "
                        f"Examples: {example_chronic}"
                    )

                # Tests failing 50-70% - maintenance needed
                moderate_failures = [t for t in failed_tests if 50 < t.get('failure_rate', 0) <= 70]
                if moderate_failures:
                    root_causes['test_design'].append(
                        f"[HIGH] {len(moderate_failures)} tests failing 50-70% - "
                        f"requires test maintenance, assertion review, or test data validation"
                    )

                # Check if many tests are failing from same suite
                suite_failures = {}
                for test in failed_tests:
                    for suite in test.get('suites', []):
                        suite_failures[suite] = suite_failures.get(suite, 0) + 1

                problematic_suites = [s for s, count in suite_failures.items() if count >= 3]
                if problematic_suites:
                    suite_names = ', '.join(problematic_suites[:2])
                    root_causes['test_design'].append(
                        f"[MEDIUM] Multiple failures in suite(s): {suite_names} - "
                        f"may indicate suite-level setup/teardown issues or related test dependencies"
                    )

            # Application issues - Degrading trend indicates app regression
            trend = analysis.get('pass_rate_trend', 'stable')
            if trend == 'degrading':
                root_causes['application'].append(
                    f"[HIGH] Pass rate declining over time (currently {pass_rate:.1f}%) - "
                    f"indicates application regression, new bugs introduced, or deteriorating code quality. "
                    f"Review recent commits and feature changes"
                )

            # Application issues - Low pass rate
            if pass_rate < 75:
                root_causes['application'].append(
                    f"[CRITICAL] Very low pass rate ({pass_rate:.1f}%) - "
                    f"significant application issues or major test suite problems. "
                    f"Immediate investigation required to identify if failures are in app or tests"
                )
            elif pass_rate < 85:
                root_causes['application'].append(
                    f"[HIGH] Below-target pass rate ({pass_rate:.1f}%, target: 95%) - "
                    f"review recent failures to determine if caused by application bugs or test issues"
                )

            # Analyze error message patterns from actual failures
            error_patterns = self._extract_error_patterns(metrics_list)
            for category, patterns in error_patterns.items():
                if category in root_causes and patterns:
                    root_causes[category].extend(patterns)

            # Add context and examples when no specific issues found
            very_slow = [t for t in slow_tests if t.get('avg_duration', 0) > 60000] if slow_tests else []

            for category in root_causes:
                if not root_causes[category]:
                    if category == 'environmental':
                        if len(flaky_tests) == 0 and stability_score >= 90:
                            root_causes[category].append("[GOOD] Environment is stable - no flaky tests detected")
                    elif category == 'test_design':
                        if not failed_tests or all(t.get('failure_rate', 0) < 50 for t in failed_tests):
                            root_causes[category].append("[GOOD] Test design appears sound - no chronic failures")
                    elif category == 'application':
                        if pass_rate >= 95 and trend != 'degrading':
                            root_causes[category].append("[GOOD] Application quality is healthy")
                    elif category == 'timing':
                        if not very_slow:
                            root_causes[category].append("[GOOD] Test execution times are acceptable")

        except Exception as e:
            logger.error(f"Error analyzing root causes: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return root_causes

    def _extract_error_patterns(self, metrics_list: List[RFTestMetrics]) -> Dict[str, List[str]]:
        """Extract and categorize common error patterns from failure messages"""
        from collections import Counter

        patterns = defaultdict(list)
        error_keywords = {
            'environmental': ['connection refused', 'timeout', 'network', 'unreachable', 'dns', 'socket'],
            'application': ['assertion failed', 'unexpected value', 'incorrect', 'null pointer', 'index out'],
            'infrastructure': ['database', 'server error', 'service unavailable', '503', '502', '500'],
            'data': ['missing data', 'invalid format', 'not found', 'empty', 'null'],
            'timing': ['timeout', 'wait', 'stale element', 'no such element']
        }

        all_errors = []
        for metrics in metrics_list:
            for failed_test in metrics.failed_test_details:
                message = failed_test.get('message', '').lower()
                if message and len(message) > 10:
                    all_errors.append(message)

        # Count occurrences of error patterns
        error_counts = Counter()
        for error in all_errors:
            for category, keywords in error_keywords.items():
                for keyword in keywords:
                    if keyword in error:
                        error_counts[(category, keyword)] += 1

        # Report significant patterns (appearing in 3+ failures)
        for (category, keyword), count in error_counts.most_common(15):
            if count >= 3:
                severity = 'CRITICAL' if count > 10 else 'HIGH' if count > 5 else 'MEDIUM'
                patterns[category].append(
                    f"[{severity}] '{keyword}' appears in {count} failure messages - investigate {keyword}-related issues"
                )

        return dict(patterns)

    def _detect_test_patterns(self, metrics_list: List[RFTestMetrics]) -> Dict[str, Any]:
        """Detect temporal and recurring patterns in test execution"""
        patterns = {
            'temporal': [],
            'recurring': [],
            'correlation': [],
            'anomalies': []
        }

        if len(metrics_list) < 5:
            return patterns

        try:
            # Temporal patterns - Time-based analysis
            timestamps = [m.timestamp for m in metrics_list]
            pass_rates = [m.pass_rate for m in metrics_list]

            # Weekend vs Weekday analysis
            weekend_data = [(ts, pr) for ts, pr in zip(timestamps, pass_rates) if ts.weekday() >= 5]
            weekday_data = [(ts, pr) for ts, pr in zip(timestamps, pass_rates) if ts.weekday() < 5]

            if weekend_data and weekday_data:
                weekend_avg = np.mean([pr for _, pr in weekend_data])
                weekday_avg = np.mean([pr for _, pr in weekday_data])
                difference = abs(weekend_avg - weekday_avg)

                if difference > 5:
                    trend = "higher" if weekend_avg > weekday_avg else "lower"
                    patterns['temporal'].append(
                        f"Weekend pass rate ({weekend_avg:.1f}%) is {difference:.1f}% {trend} than weekday ({weekday_avg:.1f}%) - may indicate environment or data refresh issues"
                    )

                # Time-of-day analysis
                hours = [ts.hour for ts in timestamps]
                if len(set(hours)) > 3:
                    business_hours_data = [(ts, pr) for ts, pr in zip(timestamps, pass_rates)
                                          if 9 <= ts.hour <= 17]
                    off_hours_data = [(ts, pr) for ts, pr in zip(timestamps, pass_rates)
                                     if ts.hour < 9 or ts.hour > 17]

                    if business_hours_data and off_hours_data:
                        bh_avg = np.mean([pr for _, pr in business_hours_data])
                        oh_avg = np.mean([pr for _, pr in off_hours_data])
                        difference = abs(bh_avg - oh_avg)

                        if difference > 5:
                            patterns['temporal'].append(
                                f"Business hours pass rate differs by {difference:.1f}% from off-hours - possible load or concurrency issues"
                            )

                # Recurring failure patterns
                test_failure_builds = defaultdict(list)
                for metrics in metrics_list:
                    for failed_test in metrics.failed_test_details:
                        test_failure_builds[failed_test['name']].append(metrics.build_number)

                # Find consecutively failing tests
                for test_name, builds in test_failure_builds.items():
                    if len(builds) >= 3:
                        builds_sorted = sorted(builds)
                        consecutive_count = 1
                        max_consecutive = 1

                        for i in range(1, len(builds_sorted)):
                            if builds_sorted[i] - builds_sorted[i-1] == 1:
                                consecutive_count += 1
                                max_consecutive = max(max_consecutive, consecutive_count)
                            else:
                                consecutive_count = 1

                        if max_consecutive >= 3:
                            patterns['recurring'].append(
                                f"Test '{test_name}' failed {max_consecutive} consecutive times (builds {builds_sorted[0]}-{builds_sorted[-1]}) - systematic issue"
                            )

                # Detect anomalies using statistical methods
                if len(pass_rates) >= 10:
                    mean_pass_rate = np.mean(pass_rates)
                    std_pass_rate = np.std(pass_rates)

                    for i, (ts, pr) in enumerate(zip(timestamps, pass_rates)):
                        # Z-score for anomaly detection
                        if std_pass_rate > 0:
                            z_score = abs((pr - mean_pass_rate) / std_pass_rate)
                            if z_score > 2:  # More than 2 standard deviations
                                build_num = metrics_list[i].build_number if i < len(metrics_list) else 'unknown'
                                patterns['anomalies'].append(
                                    f"Build #{build_num} on {ts.strftime('%Y-%m-%d')} shows unusual pass rate ({pr:.1f}%) - investigate for special conditions"
                                )

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        return patterns

    def _calculate_quality_score(self, analysis: Dict, metrics_list: List[RFTestMetrics] = None) -> Dict[str, Any]:
        """Calculate comprehensive test suite quality score"""
        if metrics_list is None:
            metrics_list = []

        quality_metrics = {
            'overall_score': 0,
            'reliability': 0,
            'performance': 0,
            'coverage': 0,
            'maintainability': 0,
            'grade': 'F',
            'breakdown': {}
        }

        try:
            # Reliability score (40% weight) - Based on pass rate and stability
            pass_rate = analysis.get('average_pass_rate', 0)
            stability = analysis.get('stability_score', 0)
            flaky_count = len(analysis.get('flaky_tests', []))

            reliability_base = (pass_rate * 0.65) + (stability * 0.35)

            # Penalty for flaky tests
            if flaky_count > 10:
                reliability = reliability_base * 0.7
            elif flaky_count > 5:
                reliability = reliability_base * 0.8
            elif flaky_count > 0:
                reliability = reliability_base * 0.9
            else:
                reliability = reliability_base

            quality_metrics['reliability'] = round(reliability, 1)
            quality_metrics['breakdown']['pass_rate'] = round(pass_rate, 1)
            quality_metrics['breakdown']['stability'] = round(stability, 1)
            quality_metrics['breakdown']['flaky_penalty'] = round((reliability_base - reliability), 1)

            # Performance score (25% weight) - Based on execution time and test count
            avg_exec_time = analysis.get('average_execution_time', 0) / 1000  # Convert to seconds
            exec_trend = analysis.get('execution_time_trend', '')

            # Get average test count to normalize performance
            total_tests_avg = 0
            if metrics_list:
                total_tests_avg = np.mean([m.total_tests for m in metrics_list if m.total_tests > 0])

            if avg_exec_time > 0:
                # Calculate time per test for normalized performance assessment
                time_per_test = avg_exec_time / max(total_tests_avg, 1)

                # Score based on time per test (more fair assessment)
                if time_per_test < 0.5:  # < 0.5 seconds per test - Excellent
                    performance = 95
                elif time_per_test < 2:  # < 2 seconds per test - Good
                    performance = 85
                elif time_per_test < 5:  # < 5 seconds per test - Acceptable
                    performance = 70
                elif time_per_test < 10:  # < 10 seconds per test - Poor
                    performance = 55
                else:  # > 10 seconds per test - Very Poor
                    performance = 40

                # Additional consideration for absolute time (very long suites)
                if avg_exec_time > 1800:  # > 30 minutes absolute time
                    performance *= 0.9
                elif avg_exec_time > 3600:  # > 1 hour absolute time
                    performance *= 0.8

                # Adjust based on trend
                if exec_trend == 'degrading':
                    performance *= 0.9
                elif exec_trend == 'improving':
                    performance = min(100, performance * 1.05)

                quality_metrics['performance'] = round(performance, 1)
                quality_metrics['breakdown']['avg_exec_time'] = round(avg_exec_time, 1)
                quality_metrics['breakdown']['time_per_test'] = round(time_per_test, 2)
                quality_metrics['breakdown']['avg_test_count'] = round(total_tests_avg, 1)
            else:
                quality_metrics['performance'] = 50

            # Maintainability score (20% weight) - Based on failure patterns
            failed_tests = analysis.get('most_failed_tests', [])
            chronic_failures = [t for t in failed_tests if t.get('failure_rate', 0) > 50]

            if len(chronic_failures) > 10:
                maintainability = 40
            elif len(chronic_failures) > 5:
                maintainability = 60
            elif len(chronic_failures) > 0:
                maintainability = 75
            else:
                maintainability = 90

            # Additional penalty for high total failure count
            if len(failed_tests) > 20:
                maintainability *= 0.85

            quality_metrics['maintainability'] = round(maintainability, 1)
            quality_metrics['breakdown']['chronic_failures'] = len(chronic_failures)
            quality_metrics['breakdown']['total_failures'] = len(failed_tests)

            # Coverage score (15% weight) - Based on test execution breadth and consistency
            total_runs = analysis.get('total_runs', 0)

            # Data confidence (sample size factor)
            if total_runs >= 30:
                data_confidence = 1.0  # Full confidence
            elif total_runs >= 20:
                data_confidence = 0.9
            elif total_runs >= 10:
                data_confidence = 0.8
            elif total_runs >= 5:
                data_confidence = 0.65
            else:
                data_confidence = 0.5  # Limited confidence

            # Actual coverage based on test execution consistency
            # Calculate what percentage of tests consistently run
            test_execution_consistency = 100  # Start with perfect score

            # If we have flaky tests, it indicates inconsistent execution
            flaky_count = len(analysis.get('flaky_tests', []))
            if flaky_count > 0:
                # Reduce score based on flaky tests (they indicate coverage gaps)
                flaky_penalty = min(20, flaky_count * 2)
                test_execution_consistency -= flaky_penalty

            # Check if we have a good distribution of test runs
            if metrics_list and len(metrics_list) > 1:
                test_counts = [m.total_tests for m in metrics_list if m.total_tests > 0]
                if test_counts:
                    # Calculate coefficient of variation for test count stability
                    mean_tests = np.mean(test_counts)
                    std_tests = np.std(test_counts)
                    if mean_tests > 0:
                        cv = std_tests / mean_tests
                        # High variation means inconsistent coverage (tests being skipped)
                        if cv > 0.3:  # >30% variation
                            test_execution_consistency *= 0.7
                        elif cv > 0.2:  # >20% variation
                            test_execution_consistency *= 0.85

            # Combine data confidence with execution consistency
            coverage = test_execution_consistency * data_confidence

            quality_metrics['coverage'] = round(coverage, 1)
            quality_metrics['breakdown']['total_runs'] = total_runs
            quality_metrics['breakdown']['data_confidence'] = round(data_confidence * 100, 1)
            quality_metrics['breakdown']['execution_consistency'] = round(test_execution_consistency, 1)

            # Overall score (weighted average)
            overall = (
                quality_metrics['reliability'] * 0.40 +
                quality_metrics['performance'] * 0.25 +
                quality_metrics['maintainability'] * 0.20 +
                quality_metrics['coverage'] * 0.15
            )
            quality_metrics['overall_score'] = round(overall, 1)

            # Assign letter grade
            if overall >= 90:
                quality_metrics['grade'] = 'A'
                quality_metrics['grade_description'] = 'Excellent'
            elif overall >= 80:
                quality_metrics['grade'] = 'B'
                quality_metrics['grade_description'] = 'Good'
            elif overall >= 70:
                quality_metrics['grade'] = 'C'
                quality_metrics['grade_description'] = 'Acceptable'
            elif overall >= 60:
                quality_metrics['grade'] = 'D'
                quality_metrics['grade_description'] = 'Poor'
            else:
                quality_metrics['grade'] = 'F'
                quality_metrics['grade_description'] = 'Failing'

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")

        return quality_metrics

    def _generate_actionable_items(self, analysis: Dict, root_causes: Dict) -> Dict[str, List[Dict]]:
            """Generate prioritized, actionable items based on analysis with specific details"""
            actionable = {
                'immediate': [],  # Within 24 hours
                'short_term': [],  # Within 1 week
                'long_term': []  # Within 1 month
            }

            try:
                # Get key metrics
                pass_rate = analysis.get('average_pass_rate', 0)
                stability = analysis.get('stability_score', 0)
                flaky_tests = analysis.get('flaky_tests', [])
                failed_tests = analysis.get('most_failed_tests', [])
                slow_tests = analysis.get('slowest_tests', [])
                trend = analysis.get('pass_rate_trend', 'stable')

                # IMMEDIATE ACTIONS (24 hours) - Critical issues

                # 1. Critical pass rate
                if pass_rate < 75:
                    failed_test_names = [t['test'] for t in failed_tests[:5]] if failed_tests else ['Review all failing tests in detailed analysis']
                    actionable['immediate'].append({
                        'priority': 'CRITICAL',
                        'action': 'Investigate and fix top failing tests immediately',
                        'reason': f'Pass rate at {pass_rate:.1f}% is critically low (target: 95%+)',
                        'tests': failed_test_names,
                        'estimated_impact': f'Could improve pass rate by 10-20% (from {pass_rate:.1f}% to ~{min(95, pass_rate + 15):.1f}%)',
                        'effort': 'High',
                        'details': [
                            f'Review error logs for top {len(failed_test_names)} failing tests',
                            'Identify common failure patterns',
                            'Fix or disable broken tests',
                            'Re-run analysis to verify improvements'
                        ]
                    })
                elif pass_rate < 85:
                    failed_test_names = [t['test'] for t in failed_tests[:3]] if failed_tests else ['Review failing tests']
                    actionable['immediate'].append({
                        'priority': 'HIGH',
                        'action': 'Address failing tests to reach 95% target',
                        'reason': f'Pass rate at {pass_rate:.1f}% is below acceptable threshold',
                        'tests': failed_test_names,
                        'estimated_impact': f'Achieve target pass rate (need +{95 - pass_rate:.1f}%)',
                        'effort': 'Medium to High',
                        'details': [
                            'Focus on tests failing >70% of the time',
                            'Review recent code changes',
                            'Update tests for feature changes'
                        ]
                    })

                # 2. High flaky test count
                if len(flaky_tests) > 10:
                    flaky_test_names = [t['test'] for t in flaky_tests[:5]]
                    actionable['immediate'].append({
                        'priority': 'CRITICAL',
                        'action': 'Stabilize highly flaky tests',
                        'reason': f'{len(flaky_tests)} flaky tests causing severe instability',
                        'tests': flaky_test_names,
                        'estimated_impact': f'Could improve stability by {len(flaky_tests) * 2:.1f} points (from {stability:.1f} to ~{min(100, stability + len(flaky_tests) * 2):.1f})',
                        'effort': 'High',
                        'details': [
                            'Add explicit waits for dynamic content',
                            'Remove hard-coded sleep statements',
                            'Check for race conditions',
                            'Verify test data setup/cleanup'
                        ]
                    })
                elif len(flaky_tests) > 5:
                    flaky_test_names = [t['test'] for t in flaky_tests[:5]]
                    actionable['immediate'].append({
                        'priority': 'HIGH',
                        'action': 'Fix flaky tests undermining reliability',
                        'reason': f'{len(flaky_tests)} flaky tests detected - reducing confidence in results',
                        'tests': flaky_test_names,
                        'estimated_impact': f'Restore test reliability, improve stability score',
                        'effort': 'Medium to High',
                        'details': [
                            'Investigate environment inconsistencies',
                            'Review wait strategies',
                            'Check for shared state issues'
                        ]
                    })

                # 3. Degrading trend
                if trend == 'degrading':
                    actionable['immediate'].append({
                        'priority': 'HIGH',
                        'action': 'Investigate declining pass rate trend',
                        'reason': 'Quality is degrading over time - indicates systematic issues',
                        'tests': [t['test'] for t in failed_tests[:3]] if failed_tests else ['Review recent builds'],
                        'estimated_impact': 'Stop quality decline, restore stability',
                        'effort': 'Medium',
                        'details': [
                            'Review recent application changes',
                            'Check for new failures vs existing issues',
                            'Identify if specific builds introduced problems',
                            'Analyze commit history correlation'
                        ]
                    })

                # SHORT-TERM ACTIONS (1 week) - Important improvements

                # 1. Moderate flaky tests
                if 0 < len(flaky_tests) <= 5:
                    flaky_test_names = [t['test'] for t in flaky_tests]
                    actionable['short_term'].append({
                        'priority': 'MEDIUM',
                        'action': 'Eliminate remaining flaky tests',
                        'reason': f'{len(flaky_tests)} flaky tests detected - achievable to fix all',
                        'tests': flaky_test_names,
                        'estimated_impact': f'Achieve 100% stability (currently {stability:.1f}/100)',
                        'effort': 'Medium',
                        'details': [
                            'Add retry logic for transient failures',
                            'Improve synchronization',
                            'Document fixes for future reference'
                        ]
                    })

                # 2. Slow test optimization
                very_slow = [t for t in slow_tests if t.get('avg_duration', 0) > 60000]
                if very_slow:
                    slow_test_names = [t['test'] for t in very_slow[:5]]
                    total_time = sum(t.get('avg_duration', 0) for t in very_slow[:10]) / 1000
                    potential_saving = total_time * 0.3  # 30% improvement
                    actionable['short_term'].append({
                        'priority': 'MEDIUM',
                        'action': 'Optimize slow-running tests',
                        'reason': f'{len(very_slow)} tests taking >60s each - significant performance impact',
                        'tests': slow_test_names,
                        'estimated_impact': f'Save ~{potential_saving:.1f}s per run (30% improvement)',
                        'effort': 'Low to Medium',
                        'details': [
                            'Replace sleep statements with smart waits',
                            'Optimize data setup (use APIs vs UI)',
                            'Parallelize independent operations',
                            'Review and reduce timeout values'
                        ]
                    })

                # 3. Environmental stability
                if root_causes.get('environmental'):
                    actionable['short_term'].append({
                        'priority': 'HIGH',
                        'action': 'Stabilize test environment',
                        'reason': f'{len(root_causes["environmental"])} environment-related issues detected',
                        'details': root_causes['environmental'][:5],
                        'estimated_impact': 'Reduce flakiness by 50%, improve reliability',
                        'effort': 'Medium',
                        'tests': ['Environment configuration review required']
                    })

                # 4. Failed test maintenance
                chronic_failures = [t for t in failed_tests if t.get('failure_rate', 0) > 70]
                if chronic_failures:
                    chronic_test_names = [t['test'] for t in chronic_failures[:3]]
                    actionable['short_term'].append({
                        'priority': 'MEDIUM',
                        'action': 'Fix or remove chronically failing tests',
                        'reason': f'{len(chronic_failures)} tests failing >70% - likely broken or obsolete',
                        'tests': chronic_test_names,
                        'estimated_impact': f'Improve pass rate by ~{sum(t.get("failure_rate", 0) for t in chronic_failures[:3]) / len(chronic_failures[:3]) / 10:.1f}%',
                        'effort': 'Medium',
                        'details': [
                            'Determine if tests are still valid',
                            'Update for application changes',
                            'Disable if no longer relevant',
                            'Document decisions'
                        ]
                    })

                # LONG-TERM ACTIONS (1 month) - Strategic improvements

                # 1. Test design improvements
                if root_causes.get('test_design'):
                    actionable['long_term'].append({
                        'priority': 'MEDIUM',
                        'action': 'Refactor poorly designed tests',
                        'reason': f'{len(root_causes["test_design"])} test design issues identified',
                        'details': root_causes['test_design'][:5],
                        'estimated_impact': 'Improved maintainability, reduced false failures',
                        'effort': 'High',
                        'tests': ['Test suite architecture review needed']
                    })

                # 2. Continuous monitoring
                if stability < 90 or pass_rate < 95:
                    actionable['long_term'].append({
                        'priority': 'MEDIUM',
                        'action': 'Implement continuous quality monitoring',
                        'reason': f'Current metrics (Pass: {pass_rate:.1f}%, Stability: {stability:.1f}) need ongoing tracking',
                        'details': [
                            'Set up automated alerts for pass rate drops >5%',
                            'Weekly quality review meetings',
                            'Dashboard with trend visualizations',
                            'Integration with CI/CD pipeline',
                            'Flaky test detection automation'
                        ],
                        'estimated_impact': 'Early issue detection, prevent quality degradation',
                        'effort': 'Low to Medium',
                        'tests': ['System-wide monitoring implementation']
                    })

                # 3. Test suite optimization
                if len(slow_tests) > 15 or analysis.get('average_execution_time', 0) > 300000:
                    avg_time = analysis.get('average_execution_time', 0) / 1000
                    actionable['long_term'].append({
                        'priority': 'LOW',
                        'action': 'Comprehensive test suite optimization',
                        'reason': f'Average execution time ({avg_time:.1f}s) can be significantly improved',
                        'details': [
                            'Audit all tests for efficiency',
                            'Implement parallel execution',
                            'Optimize test data management',
                            'Review test coverage vs execution time',
                            'Consider test prioritization strategies'
                        ],
                        'estimated_impact': f'Reduce total execution time by 40-50% ({avg_time * 0.5:.1f}s savings)',
                        'effort': 'High',
                        'tests': ['Full suite performance audit']
                    })

                # 4. Infrastructure improvements
                if root_causes.get('infrastructure') or root_causes.get('timing'):
                    all_infra_issues = root_causes.get('infrastructure', []) + root_causes.get('timing', [])
                    actionable['long_term'].append({
                        'priority': 'LOW',
                        'action': 'Upgrade test infrastructure',
                        'reason': f'{len(all_infra_issues)} infrastructure/timing issues detected',
                        'details': all_infra_issues[:5] + [
                            'Evaluate test environment resources',
                            'Consider containerization',
                            'Review network configuration',
                            'Assess database performance'
                        ],
                        'estimated_impact': 'Improved stability, faster execution, better reliability',
                        'effort': 'High',
                        'tests': ['Infrastructure assessment required']
                    })

                # Ensure we always have at least some recommendations
                if not actionable['immediate'] and not actionable['short_term'] and not actionable['long_term']:
                    if pass_rate >= 95 and stability >= 90 and len(flaky_tests) == 0:
                        # Excellent state - maintenance mode
                        actionable['long_term'].append({
                            'priority': 'LOW',
                            'action': 'Maintain excellent test suite quality',
                            'reason': 'Test suite is in excellent condition',
                            'details': [
                                'Continue regular monitoring',
                                'Keep tests updated with application changes',
                                'Document best practices',
                                'Share success with team'
                            ],
                            'estimated_impact': 'Maintain high quality standards',
                            'effort': 'Low',
                            'tests': ['Ongoing maintenance']
                        })
                    else:
                        # Good state - minor improvements
                        actionable['short_term'].append({
                            'priority': 'LOW',
                            'action': 'Fine-tune test suite',
                            'reason': 'Test suite is in good condition with minor improvement opportunities',
                            'details': [
                                'Review and optimize slower tests',
                                'Update documentation',
                                'Implement preventive measures',
                                'Regular quality reviews'
                            ],
                            'estimated_impact': 'Incremental improvements to quality',
                            'effort': 'Low',
                            'tests': ['Minor optimizations']
                        })

            except Exception as e:
                logger.error(f"Error generating actionable items: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")

            return actionable

    def _create_comprehensive_analysis_prompt(self, analysis: Dict, root_causes: Dict,
                                             patterns: Dict, quality_score: Dict) -> str:
        """Create comprehensive prompt with all analysis data"""
        prompt = f"""
    You are an expert test automation analyst with deep knowledge of Robot Framework and software quality engineering.

    Analyze this comprehensive test suite data and provide actionable, specific insights:

    ## QUALITY ASSESSMENT:
    Overall Grade: {quality_score.get('grade', 'N/A')} ({quality_score.get('overall_score', 0):.1f}/100)
    - Reliability: {quality_score.get('reliability', 0):.1f}/100
    - Performance: {quality_score.get('performance', 0):.1f}/100
    - Maintainability: {quality_score.get('maintainability', 0):.1f}/100
    - Coverage: {quality_score.get('coverage', 0):.1f}/100

    ## CURRENT METRICS:
    - Total Runs Analyzed: {analysis.get('total_runs', 0)}
    - Average Pass Rate: {analysis.get('average_pass_rate', 0):.1f}%
    - Pass Rate Trend: {analysis.get('pass_rate_trend', 'Unknown')}
    - Stability Score: {analysis.get('stability_score', 0):.1f}/100
    - Average Execution Time: {analysis.get('average_execution_time', 0)/1000:.1f} seconds

    ## IDENTIFIED ROOT CAUSES:
    **Environmental Issues:**
    {chr(10).join('  - ' + cause for cause in root_causes.get('environmental', ['None detected']))}

    **Application Issues:**
    {chr(10).join('  - ' + cause for cause in root_causes.get('application', ['None detected']))}

    **Test Design Issues:**
    {chr(10).join('  - ' + cause for cause in root_causes.get('test_design', ['None detected']))}

    **Infrastructure Issues:**
    {chr(10).join('  - ' + cause for cause in root_causes.get('infrastructure', ['None detected']))}

    **Timing Issues:**
    {chr(10).join('  - ' + cause for cause in root_causes.get('timing', ['None detected']))}

    ## DETECTED PATTERNS:
    **Temporal Patterns:**
    {chr(10).join('  - ' + p for p in patterns.get('temporal', ['None detected']))}

    **Recurring Issues:**
    {chr(10).join('  - ' + p for p in patterns.get('recurring', ['None detected']))}

    **Anomalies:**
    {chr(10).join('  - ' + p for p in patterns.get('anomalies', ['None detected']))}

    ## TOP FAILING TESTS:
    {chr(10).join('  - ' + t.get('test', 'Unknown') + f" (Failed {t.get('failure_count', 0)} times, {t.get('failure_rate', 0):.1f}% failure rate)" for t in analysis.get('most_failed_tests', [])[:10])}

    ## FLAKY TESTS:
    {chr(10).join('  - ' + t.get('test', 'Unknown') + f" ({t.get('flakiness_score', 0):.1f}% flakiness, {t.get('passes', 0)}P/{t.get('fails', 0)}F)" for t in analysis.get('flaky_tests', [])[:10])}

    Based on this comprehensive analysis, provide:

    1. **Executive Summary** (2-3 sentences):
       - Overall health assessment
       - Most critical concern
       - Urgency level

    2. **Critical Issues** (Top 5 ranked by severity):
       Format each as:
       - **[SEVERITY LEVEL]** Issue Name
       - Root Cause: Why it's happening
       - Impact: Effect on system/team
       - Evidence: Data supporting this
       
    3. **Immediate Actions** (Within 24 hours):
       For each provide:
       - Specific action to take
       - Expected outcome
       - Resources needed
       - Success criteria

    4. **Short-term Recommendations** (Within 1 week):
       Prioritized list with:
       - Action item
       - Why it matters
       - Estimated effort
       - Expected ROI

    5. **Long-term Strategy** (1-3 months):
       Strategic improvements for:
       - Test suite architecture
       - Process improvements
       - Tool/framework enhancements

    6. **Predicted Impact Timeline**:
       What happens if critical issues aren't fixed:
       - Week 1: ...
       - Week 2: ...
       - Month 1: ...

    7. **Success Metrics & Targets**:
       Specific, measurable goals:
       - Pass Rate: Current X% → Target Y% by [date]
       - Stability: Current X → Target Y
       - Execution Time: Current X → Target Y
       - Flaky Tests: Current X → Target 0

    8. **Quick Wins** (Highest ROI, Lowest Effort):
       3-5 actionable items that can be completed quickly with significant impact

    Format response as JSON with keys: executive_summary, critical_issues (array of objects), immediate_actions (array), short_term_recommendations (array), long_term_strategy (array), predicted_impact (object with timeline keys), success_metrics (object), quick_wins (array)
    """
        return prompt

    def _generate_enhanced_basic_insights(self, analysis: Dict, metrics_list: List[RFTestMetrics]) -> Dict[str, Any]:
        """Generate enhanced basic insights without AI"""
        root_causes = self._analyze_root_causes(analysis, metrics_list)
        patterns = self._detect_test_patterns(metrics_list)
        quality_score = self._calculate_quality_score(analysis, metrics_list)
        actionable = self._generate_actionable_items(analysis, root_causes)

        insights = {
            'executive_summary': self._create_executive_summary(analysis, quality_score),
            'critical_issues': self._identify_critical_issues(analysis, root_causes),
            'root_causes': root_causes,
            'patterns': patterns,
            'quality_score': quality_score,
            'immediate_actions': actionable.get('immediate', []),
            'short_term_recommendations': actionable.get('short_term', []),
            'long_term_strategy': actionable.get('long_term', []),
            'quick_wins': self._identify_quick_wins(analysis),
            'success_metrics': self._define_success_metrics(analysis, quality_score)
        }

        return insights

    def _create_executive_summary(self, analysis: Dict, quality_score: Dict) -> str:
        """Create executive summary"""
        grade = quality_score.get('grade', 'N/A')
        score = quality_score.get('overall_score', 0)
        grade_desc = quality_score.get('grade_description', '')
        pass_rate = analysis.get('average_pass_rate', 0)
        trend = analysis.get('pass_rate_trend', 'stable')

        if grade in ['A', 'B'] and pass_rate >= 85:
            urgency = "LOW"
            status = f"Test suite is in {grade_desc.lower()} condition"
        elif grade in ['C'] or (grade in ['B'] and pass_rate < 85):
            urgency = "MEDIUM"
            status = f"Test suite requires attention"
        else:
            urgency = "HIGH"
            status = f"Test suite needs immediate action"

        summary = f"{status} (Grade {grade}, {score:.1f}/100). "
        summary += f"Pass rate is {pass_rate:.1f}% and {trend}. "

        # Add most critical concern
        flaky_count = len(analysis.get('flaky_tests', []))
        failed_count = len(analysis.get('most_failed_tests', []))

        if pass_rate < 75:
            summary += f"CRITICAL: Low pass rate requires immediate investigation. "
        elif flaky_count > 10:
            summary += f"CONCERN: {flaky_count} flaky tests undermining reliability. "
        elif failed_count > 15:
            summary += f"CONCERN: {failed_count} tests failing consistently. "
        else:
            summary += "Continue monitoring for stability. "

        summary += f"Urgency: {urgency}."

        return summary

    def _identify_critical_issues(self, analysis: Dict, root_causes: Dict) -> List[Dict]:
        """Identify and prioritize critical issues"""
        issues = []

        pass_rate = analysis.get('average_pass_rate', 0)
        stability = analysis.get('stability_score', 0)
        flaky_tests = analysis.get('flaky_tests', [])
        failed_tests = analysis.get('most_failed_tests', [])

        # Critical pass rate
        if pass_rate < 70:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'Very Low Pass Rate',
                'root_cause': 'Multiple test failures indicating systemic issues',
                'impact': 'High risk of production incidents, loss of confidence in releases',
                'evidence': f'Pass rate at {pass_rate:.1f}%, {len(failed_tests)} failing tests'
            })
        elif pass_rate < 85:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Below Target Pass Rate',
                'root_cause': 'Test failures need investigation and fixes',
                'impact': 'Reduced quality assurance effectiveness',
                'evidence': f'Pass rate at {pass_rate:.1f}%, target is 95%+'
            })

        # Flaky tests
        if len(flaky_tests) > 10:
            issues.append({
                'severity': 'CRITICAL',
                'issue': 'Extensive Test Flakiness',
                'root_cause': root_causes.get('environmental', ['Environment instability'])[0] if root_causes.get('environmental') else 'Environment instability',
                'impact': 'Cannot trust test results, wasting developer time on false failures',
                'evidence': f'{len(flaky_tests)} flaky tests, stability score: {stability:.1f}/100'
            })
        elif len(flaky_tests) > 5:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Significant Test Flakiness',
                'root_cause': 'Race conditions or timing issues in tests',
                'impact': 'Unreliable test results, false positives/negatives',
                'evidence': f'{len(flaky_tests)} flaky tests detected'
            })

        # Degrading trend
        if analysis.get('pass_rate_trend') == 'degrading':
            issues.append({
                'severity': 'HIGH',
                'issue': 'Declining Pass Rate Trend',
                'root_cause': root_causes.get('application', ['Application regressions'])[0] if root_causes.get('application') else 'Application quality degrading',
                'impact': 'Quality deteriorating over time, technical debt accumulating',
                'evidence': f'Pass rate trending downward over {analysis.get("total_runs", 0)} runs'
            })

        # Chronic failures
        chronic = [t for t in failed_tests if t.get('failure_rate', 0) > 70]
        if len(chronic) > 5:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Multiple Chronically Failing Tests',
                'root_cause': root_causes.get('test_design', ['Test design issues'])[0] if root_causes.get('test_design') else 'Invalid test assertions or outdated tests',
                'impact': 'Tests not providing value, consuming CI/CD resources',
                'evidence': f'{len(chronic)} tests failing >70% of the time'
            })

        # Low stability
        if stability < 60:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Extremely Unstable Test Suite',
                'root_cause': 'Inconsistent test environment or data dependencies',
                'impact': 'Cannot rely on test results for release decisions',
                'evidence': f'Stability score: {stability:.1f}/100 (target: 90+)'
            })

        return issues[:5]  # Return top 5

    def _normalize_critical_issues(self, critical_issues: Any, analysis: Dict, root_causes: Dict) -> List[Dict]:
        """Normalize critical issues from AI response - ensure proper structure"""
        if not isinstance(critical_issues, list):
            logger.warning(f"critical_issues is not a list (type: {type(critical_issues)}), generating from analysis")
            return self._identify_critical_issues(analysis, root_causes)

        normalized = []
        for item in critical_issues:
            if isinstance(item, dict):
                # Validate and normalize dict structure
                issue_text = item.get('issue', item.get('title', item.get('description', 'Unknown Issue')))

                # If issue is still empty or just whitespace, create a meaningful description
                if not issue_text or issue_text.strip() == '':
                    # Try to create description from root_cause or impact
                    if item.get('root_cause'):
                        issue_text = f"Issue: {item.get('root_cause')[:50]}"
                    elif item.get('impact'):
                        issue_text = f"Impact: {item.get('impact')[:50]}"
                    else:
                        issue_text = 'Unspecified Issue'

                normalized.append({
                    'severity': item.get('severity', item.get('level', 'MEDIUM')).upper(),
                    'issue': issue_text,
                    'root_cause': item.get('root_cause', item.get('cause', 'Not identified')),
                    'impact': item.get('impact', item.get('consequence', 'Unknown impact')),
                    'evidence': item.get('evidence', item.get('details', 'No evidence provided'))
                })
            elif isinstance(item, str):
                # AI returned a string, convert to structured format
                normalized.append({
                    'severity': 'MEDIUM',
                    'issue': item,
                    'root_cause': 'AI-generated issue (details not specified)',
                    'impact': 'To be investigated',
                    'evidence': 'See AI insights for context'
                })
            else:
                logger.warning(f"Unexpected critical issue item type: {type(item)}, skipping")
                continue

        # If normalization resulted in empty list, generate from analysis
        if not normalized:
            logger.info("No valid critical issues from AI, generating from analysis")
            return self._identify_critical_issues(analysis, root_causes)

        return normalized

    def _identify_quick_wins(self, analysis: Dict) -> List[Dict]:
        """Identify quick win opportunities with meaningful insights"""
        wins = []

        slow_tests = analysis.get('slowest_tests', [])
        if slow_tests:
            total_slow_time = sum(t.get('avg_duration', 0) for t in slow_tests[:5]) / 1000
            potential_saving = total_slow_time * 0.25  # 25% improvement estimate
            wins.append({
                'action': 'Optimize slowest tests',
                'tests': [t['test'] for t in slow_tests[:3]],
                'reason': f'Top {len(slow_tests)} tests taking {total_slow_time:.1f}s combined - optimization opportunities exist',
                'potential_saving': f'Save ~{potential_saving:.1f}s per test run (25% improvement)',
                'effort': 'Low - 2-4 hours to optimize waits and test data setup',
                'roi': 'High - Faster feedback loops, reduced CI/CD time'
            })

        failed_tests = analysis.get('most_failed_tests', [])
        if failed_tests:
            easy_fixes = [t for t in failed_tests if t.get('failure_rate', 0) > 90]
            if easy_fixes:
                avg_failure_rate = sum(t.get('failure_rate', 0) for t in easy_fixes[:3]) / len(easy_fixes[:3])
                pass_rate = analysis.get('average_pass_rate', 0)
                improvement = min(10, (100 - pass_rate) / 2)
                wins.append({
                    'action': 'Fix or disable consistently failing tests',
                    'tests': [t['test'] for t in easy_fixes[:3]],
                    'reason': f'{len(easy_fixes)} tests failing >{avg_failure_rate:.0f}% are likely broken or outdated, not catching real bugs',
                    'potential_saving': f'Improve pass rate by ~{improvement:.1f}% (from {pass_rate:.1f}% to {pass_rate + improvement:.1f}%)',
                    'effort': 'Low - 1-2 hours to review and fix/disable each test',
                    'roi': 'Very High - Quick pass rate improvement with minimal effort'
                })

        flaky_tests = analysis.get('flaky_tests', [])
        if 1 <= len(flaky_tests) <= 3:
            avg_flakiness = sum(t.get('flakiness_score', 0) for t in flaky_tests) / len(flaky_tests)
            stability = analysis.get('stability_score', 0)
            wins.append({
                'action': 'Stabilize small number of flaky tests',
                'tests': [t['test'] for t in flaky_tests],
                'reason': f'Only {len(flaky_tests)} flaky tests (avg {avg_flakiness:.1f}% flakiness) - achievable to fix all',
                'potential_saving': f'Achieve 100% stability score (currently {stability:.1f}/100), eliminate false failures',
                'effort': 'Medium - 4-6 hours to add proper waits and stabilize',
                'roi': 'Very High - Eliminate all flakiness, restore confidence in test results'
            })
        elif 4 <= len(flaky_tests) <= 6:
            avg_flakiness = sum(t.get('flakiness_score', 0) for t in flaky_tests[:3]) / 3
            wins.append({
                'action': 'Fix most critical flaky tests first',
                'tests': [t['test'] for t in flaky_tests[:3]],
                'reason': f'Tackle top 3 most flaky tests (avg {avg_flakiness:.1f}% flakiness) to demonstrate quick progress',
                'potential_saving': 'Improve stability by 30-40 points, reduce false failure noise',
                'effort': 'Medium - 6-8 hours for top 3 tests',
                'roi': 'High - Significant stability improvement with focused effort'
            })

        # Add test data cleanup opportunity if execution time is high
        avg_exec_time = analysis.get('average_execution_time', 0) / 1000
        if avg_exec_time > 300 and not any('data' in w['action'].lower() for w in wins):
            wins.append({
                'action': 'Optimize test data setup using API calls',
                'tests': ['All tests with UI-based data setup'],
                'reason': f'Average execution time of {avg_exec_time:.1f}s suggests UI-heavy data preparation',
                'potential_saving': f'Reduce execution time by 30-40% (~{avg_exec_time * 0.35:.1f}s savings)',
                'effort': 'Medium - 8-12 hours to convert UI setup to API calls',
                'roi': 'Very High - Faster test execution, more stable tests, reduced maintenance'
            })

        # If no quick wins found, provide maintenance recommendations
        if not wins:
            pass_rate = analysis.get('average_pass_rate', 0)
            stability = analysis.get('stability_score', 0)
            if pass_rate >= 95 and stability >= 90:
                wins.append({
                    'action': 'Document and share test suite best practices',
                    'tests': ['Test suite architecture documentation'],
                    'reason': 'Test suite is in excellent condition - capture and share the success factors',
                    'potential_saving': 'Prevent regression, enable knowledge transfer, 2-3 hours saved on future troubleshooting',
                    'effort': 'Low - 2-3 hours to document patterns and practices',
                    'roi': 'Medium - Long-term maintenance efficiency and team knowledge sharing'
                })
            else:
                wins.append({
                    'action': 'Implement automated quality monitoring',
                    'tests': ['CI/CD pipeline integration'],
                    'reason': 'Proactive monitoring will catch issues before they become critical',
                    'potential_saving': 'Catch quality degradation early, save 5-10 hours per incident',
                    'effort': 'Medium - 4-6 hours to set up monitoring and alerts',
                    'roi': 'High - Early detection prevents larger issues'
                })

        return wins

    def _normalize_quick_wins(self, quick_wins: Any, analysis: Dict) -> List[Dict]:
        """Normalize quick wins from AI response - convert strings to dicts if needed"""
        if not isinstance(quick_wins, list):
            logger.warning(f"quick_wins is not a list (type: {type(quick_wins)}), generating from analysis")
            return self._identify_quick_wins(analysis)

        normalized = []
        has_valid_items = False

        for item in quick_wins:
            if isinstance(item, dict):
                # Check if dict has meaningful values (not placeholders)
                action = item.get('action', 'Unknown action')
                reason = item.get('reason', 'Not specified')
                potential_saving = item.get('potential_saving', 'Unknown')
                effort = item.get('effort', 'Unknown')
                roi = item.get('roi', 'Unknown')

                # If dict has mostly unknown/placeholder values, skip it
                if (action != 'Unknown action' and
                    reason != 'Not specified' and
                    potential_saving not in ['Unknown', 'To be determined'] and
                    effort not in ['Unknown', 'To be assessed']):
                    normalized.append({
                        'action': action,
                        'tests': item.get('tests', []),
                        'reason': reason,
                        'potential_saving': potential_saving,
                        'effort': effort,
                        'roi': roi
                    })
                    has_valid_items = True
                else:
                    logger.debug(f"Skipping quick win with placeholder values: {action}")
            elif isinstance(item, str):
                # AI returned a string - don't use placeholder, regenerate from analysis instead
                logger.info(f"Quick win item is string, will regenerate from analysis: {item[:50]}")
            else:
                logger.warning(f"Unexpected quick win item type: {type(item)}, skipping")
                continue

        # If normalization resulted in empty list or only placeholders, generate from analysis
        if not normalized or not has_valid_items:
            logger.info("No valid quick wins from AI (empty or placeholders), generating from analysis")
            return self._identify_quick_wins(analysis)

        return normalized

    def _define_success_metrics(self, analysis: Dict, quality_score: Dict) -> Dict[str, Any]:
        """Define success metrics with current vs target, specific timelines based on severity"""
        current_pass = analysis.get('average_pass_rate', 0)
        current_stability = analysis.get('stability_score', 0)
        current_exec = analysis.get('average_execution_time', 0) / 1000
        current_flaky = len(analysis.get('flaky_tests', []))
        current_grade = quality_score.get('grade', 'F')
        total_runs = analysis.get('total_runs', 0)

        # Calculate realistic timelines based on severity
        pass_gap = 95 - current_pass
        stability_gap = 90 - current_stability

        # Pass rate timeline - more urgent if critically low
        if current_pass < 70:
            pass_timeline = "1 week (CRITICAL)"
            pass_intermediate = f"80%+ in 3 days, then {current_pass + pass_gap * 0.5:.1f}% in 1 week"
        elif current_pass < 85:
            pass_timeline = "2 weeks (HIGH priority)"
            pass_intermediate = f"{current_pass + pass_gap * 0.5:.1f}% in 1 week"
        elif current_pass < 95:
            pass_timeline = "3-4 weeks"
            pass_intermediate = f"{current_pass + pass_gap * 0.5:.1f}% in 2 weeks"
        else:
            pass_timeline = "Maintain current level"
            pass_intermediate = "Already at target"

        # Stability timeline based on flaky test count
        if current_flaky > 10:
            stability_timeline = "4-6 weeks (fix 2-3 tests/week)"
        elif current_flaky > 5:
            stability_timeline = "3-4 weeks (fix 1-2 tests/week)"
        elif current_flaky > 0:
            stability_timeline = "2-3 weeks (fix all)"
        else:
            stability_timeline = "Maintain 100% stability"

        # Flaky test timeline
        if current_flaky > 10:
            flaky_timeline = "6 weeks (reduce to 5 in 3 weeks, then 0 in 6 weeks)"
        elif current_flaky > 5:
            flaky_timeline = "4 weeks (incremental fixes)"
        elif current_flaky > 0:
            flaky_timeline = "2-3 weeks (achievable to fix all)"
        else:
            flaky_timeline = "Maintain zero flakiness"

        # Execution time optimization potential
        exec_improvement = min(0.3, max(0.15, current_flaky / 100))  # More flaky = more improvement potential
        target_exec = current_exec * (1 - exec_improvement)

        metrics = {
            'pass_rate': {
                'current': f"{current_pass:.1f}%",
                'target': "95%+",
                'timeline': pass_timeline,
                'how_to_measure': f"Track pass rate across last 20 builds (currently {total_runs} runs analyzed). Calculate: (passed_tests / total_tests) * 100 per build, then average."
            },
            'stability_score': {
                'current': f"{current_stability:.1f}/100",
                'target': "90+/100 (Excellent: 95+)",
                'timeline': stability_timeline,
                'how_to_measure': f"Calculate standard deviation of pass rates over 20 builds. Formula: 100 - (std_dev * 10). Lower variance = higher stability. Monitor weekly."
            },
            'flaky_tests': {
                'current': f"{current_flaky} flaky tests",
                'target': "0 (Zero tolerance)",
                'timeline': flaky_timeline,
                'how_to_measure': "Count tests with both passes AND fails in last 20 builds. Track flakiness_score = (min(passes,fails) / total_runs * 100). Monitor after each fix."
            },
            'execution_time': {
                'current': f"{current_exec:.1f}s per run",
                'target': f"{target_exec:.1f}s ({exec_improvement*100:.0f}% reduction)",
                'timeline': "4-6 weeks (incremental optimization)",
                'how_to_measure': f"Track average test suite duration from Jenkins/CI logs. Measure total_duration from build start to end. Target: optimize top 10 slowest tests first."
            },
            'quality_grade': {
                'current': f"Grade {current_grade} ({quality_score.get('overall_score', 0):.1f}/100)",
                'target': "Grade A (90+/100)",
                'timeline': "6-8 weeks (composite improvement)",
                'how_to_measure': "Composite score: Pass Rate (40%) + Stability (30%) + Performance (20%) + Maintainability (10%). Recalculate weekly to track progress."
            }
        }

        # Add intermediate milestones for critical metrics
        if current_pass < 95:
            metrics['pass_rate']['intermediate_milestone'] = pass_intermediate

        if current_flaky > 5:
            metrics['flaky_tests']['intermediate_milestone'] = f"Reduce to {current_flaky // 2} flaky tests in {(current_flaky // 4 + 1)} weeks"

        return metrics

    def _create_analysis_prompt(self, analysis: Dict[str, Any]) -> str:
            """Create prompt for AI analysis"""
            prompt = f"""
            Analyze the following Robot Framework test execution metrics and provide insights:
            
            ## Overall Statistics:
            - Total Runs: {analysis['total_runs']}
            - Average Pass Rate: {analysis['average_pass_rate']:.2f}%
            - Pass Rate Trend: {analysis['pass_rate_trend']}
            - Average Execution Time: {analysis['average_execution_time']:.2f}ms
            - Execution Time Trend: {analysis['execution_time_trend']}
            - Stability Score: {analysis['stability_score']}/100
            
            ## Most Failed Tests:
            {self._format_failed_tests(analysis['most_failed_tests'])}
            
            ## Flaky Tests:
            {self._format_flaky_tests(analysis['flaky_tests'])}
            
            ## Slowest Tests:
            {self._format_slowest_tests(analysis['slowest_tests'])}
            
            Please provide:
            1. Executive Summary: Brief overview of test health
            2. Key Concerns: Top 3-5 issues that need attention
            3. Improvement Recommendations: Specific actionable suggestions
            4. Predicted Impact: What will happen if issues aren't addressed
            5. Success Indicators: What metrics to monitor for improvement
            
            Format your response as JSON with these keys: executive_summary, key_concerns (array), recommendations (array), predicted_impact, success_indicators (array).
            """

            return prompt

    def _format_failed_tests(self, failed_tests: List[Dict]) -> str:
        """Format failed tests for prompt"""
        if not failed_tests:
            return "None"

        lines = []
        for test in failed_tests[:5]:
            lines.append(f"- {test['test']}: Failed {test['failure_count']} times ({test['failure_rate']:.1f}%)")

        return "\n".join(lines)

    def _format_flaky_tests(self, flaky_tests: List[Dict]) -> str:
        """Format flaky tests for prompt"""
        if not flaky_tests:
            return "None"

        lines = []
        for test in flaky_tests[:5]:
            lines.append(f"- {test['test']}: {test['flakiness_score']:.1f}% flaky ({test['passes']}P/{test['fails']}F)")

        return "\n".join(lines)

    def _format_slowest_tests(self, slowest_tests: List[Dict]) -> str:
            """Format slowest tests for prompt"""
            if not slowest_tests:
                return "None"

            lines = []
            for test in slowest_tests[:5]:
                lines.append(f"- {test['test']}: {test['avg_duration']:.2f}ms avg")

            return "\n".join(lines)

    def _generate_pattern_based_suggestion(self, test_name: str, failure_rate: float,
                                           failure_type: str, sample_messages: List[str],
                                           confidence: float, auto_healable: bool) -> Dict[str, Any]:
        """Generate accurate pattern-based suggestion based on failure type"""

        suggestion = {
            'test_name': test_name,
            'failure_rate': failure_rate,
            'failure_type': failure_type,
            'confidence': confidence,
            'auto_healable': auto_healable,
            'alternative_locators': [],
            'code_example': ''
        }

        # Type-specific recommendations
        if failure_type == 'LOCATOR_ISSUE':
            suggestion['root_cause'] = "Element locator is unreliable or element not immediately available when test runs"
            suggestion['recommendation'] = "Use more robust locator strategies (ID, data-testid, name) and add explicit waits for element visibility"
            suggestion['alternative_locators'] = [
                "id=element-id",
                "css=[data-testid='element']",
                "xpath=//button[@name='submit']",
                "name=elementName"
            ]
            suggestion['code_example'] = "Wait Until Element Is Visible    id=element-id    timeout=10s\nClick Element    id=element-id"

        elif failure_type == 'TIMEOUT':
            suggestion['root_cause'] = "Operation exceeds expected timeout - element/page loading slower than anticipated or network delays"
            suggestion['recommendation'] = "Increase timeout values, use explicit waits with proper conditions, and ensure network stability"
            suggestion['code_example'] = "Wait Until Element Is Visible    locator    timeout=30s\nWait Until Page Contains    text    timeout=20s"

        elif failure_type == 'STALE_ELEMENT':
            suggestion['root_cause'] = "Element reference became invalid due to page DOM changes, AJAX updates, or page refresh"
            suggestion['recommendation'] = "Re-locate element after page changes, add waits after DOM updates, use fresh locators each time"
            suggestion['code_example'] = "Wait Until Page Does Not Contain Element    loading-spinner\n${element}=    Get WebElement    locator\nClick Element    ${element}"

        elif failure_type == 'ASSERTION_FAILURE':
            suggestion['root_cause'] = "Expected value doesn't match actual value - data inconsistency or incorrect test expectations"
            suggestion['recommendation'] = "Verify expected values are correct, check data setup/state, ensure proper test sequence and dependencies"
            suggestion['code_example'] = "Wait Until Element Contains    locator    expected_text    timeout=10s\n${actual}=    Get Text    locator\nShould Be Equal    ${actual}    expected_value"

        elif failure_type == 'NETWORK_ERROR':
            suggestion['root_cause'] = "Network connectivity issues - server unreachable, DNS problems, or connection refused"
            suggestion['recommendation'] = "Check server availability, verify network configuration, add retry logic, ensure correct URLs"
            suggestion['code_example'] = "Wait Until Keyword Succeeds    3x    5s    Open Browser    ${URL}    ${BROWSER}"

        elif failure_type == 'PERMISSION_ERROR':
            suggestion['root_cause'] = "Access denied due to insufficient permissions or authentication failure"
            suggestion['recommendation'] = "Verify user credentials, check authentication tokens, ensure proper role/permission setup"

        elif failure_type == 'DATA_ISSUE':
            suggestion['root_cause'] = "Test data is missing, null, or invalid - data setup or cleanup issues"
            suggestion['recommendation'] = "Validate test data before test execution, ensure data setup in Suite Setup, add data validation steps"
            suggestion['code_example'] = "Run Keyword And Return Status    Page Should Contain Element    data-element\n${data}=    Get Text    data-element\nShould Not Be Empty    ${data}"

        elif failure_type == 'JAVASCRIPT_ERROR':
            suggestion['root_cause'] = "JavaScript errors on page interfering with test execution or element interaction"
            suggestion['recommendation'] = "Check browser console for JS errors, wait for JS to complete, consider using different browser"
            suggestion['code_example'] = "Wait For Condition    return document.readyState == 'complete'    timeout=10s"

        else:  # OTHER or unknown
            suggestion['root_cause'] = f"Test consistently failing {failure_rate:.0f}% of the time - requires detailed investigation of error messages"
            suggestion['recommendation'] = "Analyze error messages in detail, check recent application changes, review test logic and assertions, verify test environment stability"

        return suggestion

    def _categorize_failure_from_messages(self, error_messages: List[str], existing_type: str = 'Unknown') -> Dict[str, Any]:
        """Accurately categorize failure type from actual error messages with confidence calculation"""

        # Initialize scoring
        category_scores = {
            'LOCATOR_ISSUE': 0,
            'TIMEOUT': 0,
            'STALE_ELEMENT': 0,
            'ASSERTION_FAILURE': 0,
            'NETWORK_ERROR': 0,
            'PERMISSION_ERROR': 0,
            'DATA_ISSUE': 0,
            'JAVASCRIPT_ERROR': 0
        }

        # Define patterns with weights
        patterns = {
            'LOCATOR_ISSUE': [
                ('element not found', 3),
                ('no such element', 3),
                ('could not find element', 3),
                ('unable to locate', 3),
                ('locator', 2),
                ('selector', 2),
                ('xpath', 1),
                ('css', 1)
            ],
            'TIMEOUT': [
                ('timeout', 3),
                ('timed out', 3),
                ('time limit exceeded', 3),
                ('wait', 2),
                ('took too long', 2)
            ],
            'STALE_ELEMENT': [
                ('stale element', 3),
                ('stale', 2),
                ('no longer attached', 3),
                ('element is not attached', 3)
            ],
            'ASSERTION_FAILURE': [
                ('assertion', 3),
                ('expected', 2),
                ('actual', 2),
                ('should be', 2),
                ('should equal', 2),
                ('mismatch', 2)
            ],
            'NETWORK_ERROR': [
                ('connection', 3),
                ('network', 3),
                ('refused', 2),
                ('unreachable', 2),
                ('dns', 2)
            ],
            'PERMISSION_ERROR': [
                ('permission', 3),
                ('access denied', 3),
                ('forbidden', 2),
                ('unauthorized', 2)
            ],
            'DATA_ISSUE': [
                ('null', 2),
                ('undefined', 2),
                ('missing', 2),
                ('empty', 1),
                ('invalid data', 2)
            ],
            'JAVASCRIPT_ERROR': [
                ('javascript', 3),
                ('js error', 3),
                ('script error', 2)
            ]
        }

        # Analyze all error messages
        total_messages = len(error_messages) if error_messages else 0

        if total_messages > 0:
            for message in error_messages:
                if not message:
                    continue
                message_lower = message.lower()

                # Score each category
                for category, pattern_list in patterns.items():
                    for pattern, weight in pattern_list:
                        if pattern in message_lower:
                            category_scores[category] += weight

        # Find highest scoring category
        if total_messages > 0 and max(category_scores.values()) > 0:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category_name = best_category[0]
            score = best_category[1]

            # Calculate confidence (0-1 scale)
            # Higher score and more messages = higher confidence
            max_possible_score = total_messages * 3  # Maximum if all messages had highest weight pattern
            confidence = min(0.95, (score / max(1, max_possible_score)) + 0.3)  # Boost baseline, cap at 0.95

            # Determine if auto-healable
            auto_healable = category_name in ['LOCATOR_ISSUE', 'TIMEOUT', 'STALE_ELEMENT'] and confidence > 0.7

            return {
                'failure_type': category_name,
                'confidence': confidence,
                'auto_healable': auto_healable,
                'score': score
            }

        # Fallback: use existing type if available
        if existing_type and existing_type != 'Unknown':
            return {
                'failure_type': existing_type,
                'confidence': 0.5,  # Medium confidence for pre-categorized
                'auto_healable': existing_type in ['LOCATOR_ISSUE', 'TIMEOUT'],
                'score': 0
            }

        # Default
        return {
            'failure_type': 'OTHER',
            'confidence': 0.4,  # Low confidence for unknown
            'auto_healable': False,
            'score': 0
        }

    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse AI response into structured insights"""
        try:
            # Clean up response - remove markdown JSON blocks if present
            cleaned_response = response.strip()

            # Remove ```json and ``` markers
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove ```

            cleaned_response = cleaned_response.strip()

            # Try to parse as JSON
            insights = json.loads(cleaned_response)

            # Validate that we got the expected structure
            if not isinstance(insights, dict):
                raise ValueError("Response is not a dictionary")

            return insights

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            # Fallback: Use enhanced basic insights instead of raw response
            # This ensures we never show raw JSON to users
            return None  # Will trigger fallback to _generate_enhanced_basic_insights

    def _generate_healing_insights(self, failed_tests: List[Dict], flaky_tests: List[Dict],
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-powered healing insights for failed and flaky tests in real-time"""

        healing_insights = {
            'failed_test_suggestions': [],
            'flaky_test_suggestions': [],
            'summary': {}
        }

        if not self.azure_client or not AZURE_AVAILABLE:
            # Return basic insights without AI
            return self._generate_basic_healing_insights(failed_tests, flaky_tests)

        try:
            # Generate healing suggestions for ALL failed tests (not just top 5)
            for test in failed_tests:  # Process ALL failed tests
                test_name = test.get('test', 'Unknown')
                failure_rate = test.get('failure_rate', 0)
                failure_count = test.get('failure_count', 0)
                error_type = test.get('common_error_type', 'Unknown')
                sample_messages = test.get('sample_messages', [])
                all_messages = test.get('all_messages', sample_messages)

                # Accurately categorize the failure from actual error messages
                categorization = self._categorize_failure_from_messages(all_messages, error_type)
                accurate_failure_type = categorization['failure_type']
                base_confidence = categorization['confidence']
                is_auto_healable = categorization['auto_healable']

                # For high-priority tests (high failure rate), use AI for better suggestions
                # For others, use enhanced pattern-based suggestions
                use_ai = failure_count >= 3 or failure_rate > 50

                if use_ai and self.azure_client:
                    # Create focused prompt for this specific test
                    prompt = f"""Analyze this Robot Framework test failure and provide healing suggestions:

Test Name: {test_name}
Failure Rate: {failure_rate:.1f}%
Failed: {failure_count} times
Detected Failure Type: {accurate_failure_type}

Sample Error Messages:
{chr(10).join(f'- {msg[:200]}' for msg in sample_messages[:3] if msg)}

Provide:
1. Root cause analysis (1-2 sentences max)
2. Specific recommendation to fix (1-2 sentences max)
3. Alternative locators (if locator issue - provide 2-3 alternatives)
4. Code example (if applicable, Robot Framework syntax)

Return as JSON: {{"root_cause": "...", "recommendation": "...", "alternative_locators": ["...", "..."], "code_example": "..."}}"""

                    # Get AI suggestion
                    try:
                        response = self.azure_client.generate_response(
                            prompt=prompt,
                            max_tokens=400,
                            temperature=0.2  # Lower temperature for more consistent responses
                        )

                        # Parse response
                        suggestion = self._parse_ai_response(response)

                        if suggestion:
                            # Use AI response but keep our accurate categorization
                            suggestion['test_name'] = test_name
                            suggestion['failure_rate'] = failure_rate
                            suggestion['failure_type'] = accurate_failure_type
                            suggestion['confidence'] = base_confidence
                            suggestion['auto_healable'] = is_auto_healable
                            healing_insights['failed_test_suggestions'].append(suggestion)
                        else:
                            # AI parsing failed, use pattern-based
                            raise ValueError("AI response parsing failed")

                    except Exception as e:
                        logger.warning(f"Error getting AI suggestion for {test_name}: {e}, using pattern-based")
                        # Fall through to pattern-based suggestion below
                        use_ai = False

                if not use_ai:
                    # Use enhanced pattern-based suggestion with accurate categorization
                    pattern_suggestion = self._generate_pattern_based_suggestion(
                        test_name, failure_rate, accurate_failure_type,
                        sample_messages, base_confidence, is_auto_healable
                    )
                    healing_insights['failed_test_suggestions'].append(pattern_suggestion)

            # Generate healing suggestions for ALL flaky tests
            for test in flaky_tests:  # Process ALL flaky tests
                test_name = test.get('test', 'Unknown')
                flakiness_score = test.get('flakiness_score', 0)
                pattern = test.get('failure_pattern', 'Unknown')
                passes = test.get('passes', 0)
                fails = test.get('fails', 0)
                total_runs = passes + fails

                # Calculate confidence based on flakiness characteristics
                # Higher flakiness = more data points = higher confidence in diagnosis
                # Pattern recognition = higher confidence
                confidence = 0.5  # Base confidence
                if flakiness_score > 40:
                    confidence += 0.2  # High flakiness = clear pattern
                if pattern in ['Consecutive Failures', 'Alternating']:
                    confidence += 0.15  # Recognized pattern = more confidence
                if total_runs >= 10:
                    confidence += 0.1  # More data = better confidence
                confidence = min(0.95, confidence)  # Cap at 95%

                # Use AI for highly flaky tests (>30% flakiness)
                use_ai = flakiness_score > 30 and self.azure_client

                if use_ai:
                    prompt = f"""Analyze this flaky Robot Framework test and provide stabilization strategy:

Test Name: {test_name}
Flakiness Score: {flakiness_score:.1f}%
Pass/Fail Pattern: {passes} passes / {fails} fails in {total_runs} runs
Failure Pattern: {pattern}

Provide concise:
1. Why it's flaky (root cause in 1 sentence)
2. Stabilization strategy (specific steps in 1-2 sentences)
3. Recommended wait strategy

Return as JSON: {{"root_cause": "...", "recommendation": "...", "wait_strategy": "..."}}"""

                    try:
                        response = self.azure_client.generate_response(
                            prompt=prompt,
                            max_tokens=300,
                            temperature=0.2
                        )

                        suggestion = self._parse_ai_response(response)

                        if suggestion:
                            suggestion['test_name'] = test_name
                            suggestion['flakiness_score'] = flakiness_score
                            suggestion['confidence'] = confidence
                            suggestion['pattern'] = pattern
                            healing_insights['flaky_test_suggestions'].append(suggestion)
                        else:
                            raise ValueError("AI response parsing failed")

                    except Exception as e:
                        logger.warning(f"Error getting AI suggestion for flaky test {test_name}: {e}, using pattern-based")
                        use_ai = False

                if not use_ai:
                    # Pattern-based flaky test suggestion
                    if pattern == 'Consecutive Failures':
                        root_cause = "Test shows blocks of consecutive failures indicating environmental instability or state corruption"
                        recommendation = "Check for resource cleanup issues, verify environment stability, ensure proper test isolation and reset state between runs"
                    elif pattern == 'Alternating':
                        root_cause = "Test alternates between pass/fail suggesting race condition or timing sensitivity"
                        recommendation = "Add explicit synchronization, increase wait times for dynamic content, remove fixed sleeps and use conditional waits"
                    else:
                        root_cause = f"Test shows {flakiness_score:.0f}% flakiness - intermittent failures indicating timing issues, race conditions, or environmental instability"
                        recommendation = "Add explicit waits for all dynamic content, ensure proper test data setup/cleanup, check for shared state between tests"

                    healing_insights['flaky_test_suggestions'].append({
                        'test_name': test_name,
                        'root_cause': root_cause,
                        'recommendation': recommendation,
                        'wait_strategy': "Wait Until Element Is Visible    locator    timeout=15s\nWait Until Page Does Not Contain Element    loading-spinner    timeout=10s",
                        'confidence': confidence,
                        'flakiness_score': flakiness_score,
                        'pattern': pattern
                    })

            # Summary
            healing_insights['summary'] = {
                'total_analyzed': len(failed_tests) + len(flaky_tests),
                'suggestions_generated': len(healing_insights['failed_test_suggestions']) + len(healing_insights['flaky_test_suggestions']),
                'high_confidence_count': len([s for s in healing_insights['failed_test_suggestions'] + healing_insights['flaky_test_suggestions'] if s.get('confidence', 0) > 0.8])
            }

        except Exception as e:
            logger.error(f"Error generating healing insights: {e}")
            return self._generate_basic_healing_insights(failed_tests, flaky_tests)

        return healing_insights

    def _generate_basic_healing_insights(self, failed_tests: List[Dict], flaky_tests: List[Dict]) -> Dict[str, Any]:
        """Generate basic healing insights without AI - uses accurate categorization for ALL tests"""

        healing_insights = {
            'failed_test_suggestions': [],
            'flaky_test_suggestions': [],
            'summary': {}
        }

        # Process ALL failed tests with accurate categorization
        for test in failed_tests:
            test_name = test.get('test', 'Unknown')
            failure_rate = test.get('failure_rate', 0)
            error_type = test.get('common_error_type', 'Unknown')
            sample_messages = test.get('sample_messages', [])
            all_messages = test.get('all_messages', sample_messages)

            # Use accurate categorization
            categorization = self._categorize_failure_from_messages(all_messages, error_type)
            accurate_failure_type = categorization['failure_type']
            confidence = categorization['confidence']
            auto_healable = categorization['auto_healable']

            # Generate pattern-based suggestion
            suggestion = self._generate_pattern_based_suggestion(
                test_name, failure_rate, accurate_failure_type,
                sample_messages, confidence, auto_healable
            )

            healing_insights['failed_test_suggestions'].append(suggestion)

        # Process ALL flaky tests
        for test in flaky_tests:
            test_name = test.get('test', 'Unknown')
            flakiness_score = test.get('flakiness_score', 0)
            pattern = test.get('failure_pattern', 'Intermittent')
            passes = test.get('passes', 0)
            fails = test.get('fails', 0)
            total_runs = passes + fails

            # Calculate confidence
            confidence = 0.5
            if flakiness_score > 40:
                confidence += 0.2
            if pattern in ['Consecutive Failures', 'Alternating']:
                confidence += 0.15
            if total_runs >= 10:
                confidence += 0.1
            confidence = min(0.95, confidence)

            # Pattern-based recommendation
            if pattern == 'Consecutive Failures':
                root_cause = "Test shows blocks of consecutive failures indicating environmental instability or state corruption"
                recommendation = "Check for resource cleanup issues, verify environment stability, ensure proper test isolation"
            elif pattern == 'Alternating':
                root_cause = "Test alternates between pass/fail suggesting race condition or timing sensitivity"
                recommendation = "Add explicit synchronization, increase wait times, use conditional waits instead of sleeps"
            else:
                root_cause = f"Test shows {flakiness_score:.0f}% flakiness - intermittent failures indicating timing or environment issues"
                recommendation = "Add explicit waits for dynamic content, ensure proper test data setup/cleanup, check for shared state"

            healing_insights['flaky_test_suggestions'].append({
                'test_name': test_name,
                'root_cause': root_cause,
                'recommendation': recommendation,
                'wait_strategy': "Wait Until Element Is Visible    locator    timeout=15s\nWait Until Page Does Not Contain Element    loading-spinner",
                'confidence': confidence,
                'flakiness_score': flakiness_score,
                'pattern': pattern
            })

        # Calculate summary
        all_suggestions = healing_insights['failed_test_suggestions'] + healing_insights['flaky_test_suggestions']
        high_confidence = len([s for s in all_suggestions if s.get('confidence', 0) > 0.7])

        healing_insights['summary'] = {
            'total_analyzed': len(failed_tests) + len(flaky_tests),
            'suggestions_generated': len(all_suggestions),
            'high_confidence_count': high_confidence
        }

        return healing_insights

    def _generate_basic_insights(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic insights without AI"""
        insights = {
            'executive_summary': self._create_basic_summary(analysis),
            'key_concerns': self._create_basic_concerns(analysis),
            'recommendations': self._create_basic_recommendations(analysis),
            'predicted_impact': 'Continue monitoring test metrics',
            'success_indicators': [
                'Pass rate > 95%',
                'Stability score > 90',
                'No flaky tests',
                'Execution time stable or improving'
            ]
        }

        return insights

    def _create_basic_summary(self, analysis: Dict[str, Any]) -> str:
        """Create basic executive summary"""
        pass_rate = analysis['average_pass_rate']
        stability = analysis['stability_score']

        if pass_rate >= 95 and stability >= 90:
            return f"Test suite is healthy with {pass_rate:.1f}% pass rate and {stability:.1f} stability score."
        elif pass_rate >= 80:
            return f"Test suite needs attention: {pass_rate:.1f}% pass rate, {stability:.1f} stability score."
        else:
            return f"Test suite requires immediate attention: {pass_rate:.1f}% pass rate is below acceptable levels."

    def _create_basic_concerns(self, analysis: Dict[str, Any]) -> List[str]:
        """Create basic concerns list"""
        concerns = []

        if analysis['average_pass_rate'] < 95:
            concerns.append(f"Pass rate is {analysis['average_pass_rate']:.1f}% (target: 95%)")

        if analysis['stability_score'] < 90:
            concerns.append(f"Test stability is {analysis['stability_score']:.1f} (target: 90+)")

        if analysis['most_failed_tests']:
            concerns.append(f"{len(analysis['most_failed_tests'])} tests failing frequently")

        if analysis['flaky_tests']:
            concerns.append(f"{len(analysis['flaky_tests'])} flaky tests detected")

        if analysis['pass_rate_trend'] == 'degrading':
            concerns.append("Pass rate is trending downward")

        return concerns[:5] if concerns else ['No major concerns detected']

    def _create_basic_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Create basic recommendations"""
        recommendations = []

        if analysis['most_failed_tests']:
            recommendations.append("Fix or investigate frequently failing tests")

        if analysis['flaky_tests']:
            recommendations.append("Stabilize flaky tests to improve reliability")

        if analysis['execution_time_trend'] == 'degrading':
            recommendations.append("Optimize slow tests to reduce execution time")

        if analysis['stability_score'] < 90:
            recommendations.append("Improve test stability by addressing root causes")

        recommendations.append("Set up continuous monitoring of test metrics")

        return recommendations


def show_rf_dashboard_analytics():
    """Main UI for Robot Framework Dashboard Analytics"""

    st.title("🤖 Robot Framework Dashboard Analytics")
    st.markdown("""
    Analyze Robot Framework test results from Jenkins with AI-powered insights.
    Get predictive analytics, trend analysis, and improvement recommendations.
    """)

    # Configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            # Hardcoded Jenkins URL
            jenkins_url = "https://testeng1.qainfra.registeredsite.com:8080/"
            st.info(f"🔗 Jenkins URL: {jenkins_url}")

            username = st.text_input(
                "Jenkins Username",
                value=st.session_state.get('rf_jenkins_username', ''),
                placeholder="your-username"
            )

            # Authentication type toggle
            auth_type = st.radio(
                "Authentication Type",
                options=["API Token (Recommended)", "Password"],
                index=0 if st.session_state.get('rf_auth_type', 'token') == 'token' else 1,
                horizontal=True,
                help="API Token is more secure and recommended by Jenkins"
            )

        with col2:
            # Show either API token or password field based on selection
            if auth_type == "API Token (Recommended)":
                credential = st.text_input(
                    "Jenkins API Token",
                    type="password",
                    value=st.session_state.get('rf_jenkins_token', ''),
                    help="Generate from Jenkins > User > Configure > API Token",
                    key="api_token_input"
                )
                st.session_state.rf_auth_type = 'token'
            else:
                credential = st.text_input(
                    "Jenkins Password",
                    type="password",
                    value=st.session_state.get('rf_jenkins_token', ''),
                    help="Your Jenkins account password",
                    key="password_input"
                )
                st.session_state.rf_auth_type = 'password'

            max_builds = st.number_input(
                "Number of builds to analyze",
                min_value=5,
                max_value=100,
                value=50,
                help="How many recent builds to fetch"
            )

        # Save configuration
        if st.button("💾 Save Configuration"):
            st.session_state.rf_jenkins_url = jenkins_url
            st.session_state.rf_jenkins_username = username
            st.session_state.rf_jenkins_token = credential
            st.success(f"✅ Configuration saved! Using {auth_type}")

    # Azure OpenAI status
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        if AZURE_AVAILABLE:
            st.success("✅ Azure OpenAI is available for AI-powered insights")
        else:
            st.warning("⚠️ Azure OpenAI not available - using basic analytics")

    # Main analysis section
    if not all([jenkins_url, username, credential]):
        st.info("👆 Please configure Jenkins credentials above to get started")
        st.markdown("""
        ### Getting Started
        1. Enter your Jenkins URL (e.g., `https://jenkins.example.com`)
        2. Enter your Jenkins username
        3. Choose authentication type:
           - **API Token (Recommended)**: Generate from Jenkins > User > Configure > API Token
           - **Password**: Use your Jenkins account password
        4. Click "Save Configuration"
        5. Select a job and click "Analyze Test Results"
        """)
        return

    st.markdown("---")
    st.subheader("📊 Test Analysis")

    # Initialize clients
    try:
        rf_client = RobotFrameworkDashboardClient(jenkins_url, username, credential)
        azure_client = AzureOpenAIClient() if AZURE_AVAILABLE else None
        analyzer = RFDashboardAnalyzer(azure_client)
    except Exception as e:
        st.error(f"❌ Error initializing clients: {e}")
        return

    # Test connection button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔌 Test Connection"):
            with st.spinner("Testing connection to Jenkins..."):
                jobs = rf_client.get_jobs()
                if jobs:
                    st.success(f"✅ Successfully connected! Found {len(jobs)} jobs.")
                else:
                    st.error("❌ Could not connect to Jenkins or no jobs found. Check your credentials and URL.")
                    st.info("💡 Tip: Make sure you're using an API token, not a password")
                    return

    # Job selection
    with st.spinner("Fetching Jenkins jobs..."):
        jobs = rf_client.get_jobs()

    if not jobs:
        st.error("❌ No jobs found or unable to connect to Jenkins")
        st.markdown("""
        ### Troubleshooting:
        - Verify Jenkins URL is correct (include https:// or http://)
        - Ensure username is correct
        - Confirm you're using an API token, not your password
        - Check if you have permission to access Jenkins via API
        - Try clicking "Test Connection" button above
        """)
        return

    # Extract unique folders from jobs
    # Track both top-level folders and all folder paths for better filtering
    top_level_folders = set()
    all_folder_paths = set()

    for job in jobs:
        display_name = job.get('display_name', job.get('name', ''))

        # If display_name contains '/', it's in a folder
        if '/' in display_name:
            # Get the top-level folder (first part before /)
            folder_parts = display_name.split('/')
            top_level_folder = folder_parts[0]
            top_level_folders.add(top_level_folder)

            # Also track the full folder path (everything except last part which is job name)
            if len(folder_parts) > 1:
                full_folder_path = '/'.join(folder_parts[:-1])
                all_folder_paths.add(full_folder_path)

    # Sort folders alphabetically
    folders = sorted(list(top_level_folders))

    logger.info(f"Extracted folders: {folders}")
    logger.info(f"All folder paths: {sorted(list(all_folder_paths))}")

    # Folder selection filter
    col1, col2 = st.columns([1, 2])
    with col1:
        if folders:
            folder_options = ["All Folders"] + folders
            selected_folder = st.selectbox(
                "📁 Filter by Folder",
                options=folder_options,
                help="Select a specific Jenkins folder to filter jobs",
                key="folder_selector"
            )
        else:
            selected_folder = "All Folders"
            st.info("ℹ️ No folders found - all jobs are at root level")

    # Filter jobs by selected folder
    if selected_folder == "All Folders":
        filtered_jobs = jobs
    else:
        # Filter jobs whose display_name starts with the selected folder
        filtered_jobs = [job for job in jobs
                        if job.get('display_name', job.get('name', '')).startswith(f"{selected_folder}/")]

        logger.info(f"Filtered to {len(filtered_jobs)} jobs in folder '{selected_folder}'")

    # Just show all jobs without filtering or warnings
    # Users know what jobs they want to analyze
    rf_jobs = filtered_jobs

    if not rf_jobs:
        # No jobs at all
        st.error("❌ No jobs found to analyze")
        return

    # Create a mapping of display names to job info
    job_display_names = [job.get('display_name', job['name']) for job in rf_jobs]
    job_map = {job.get('display_name', job['name']): job for job in rf_jobs}

    # Display job count
    if selected_folder == "All Folders":
        st.info(f"📊 Found {len(job_display_names)} job(s) across all folders")
    else:
        st.info(f"📊 Found {len(job_display_names)} job(s) in folder '{selected_folder}'")

    selected_job_name = st.selectbox("Select Jenkins Job", job_display_names)

    if not selected_job_name:
        return

    # Get the full job info
    selected_job_info = job_map[selected_job_name]

    # Fetch and analyze builds
    if st.button("🔍 Analyze Test Results", type="primary"):
        with st.spinner(f"Fetching and analyzing {max_builds} builds from '{selected_job_name}'..."):
            builds = rf_client.get_job_builds(selected_job_info, max_builds)

            if not builds:
                st.error(f"❌ No builds found for job: {selected_job_name}")
                st.markdown("""
                ### Possible reasons:
                - Job name might contain special characters causing URL encoding issues
                - Job might not have any builds yet
                - You might not have permission to view builds
                - Check Jenkins logs for more details
                """)
                return

            st.success(f"✅ Found {len(builds)} builds for job '{selected_job_name}'")

            # Collect metrics from all builds
            metrics_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Create a container for detailed logs
            debug_expander = st.expander("🔍 Debug Information", expanded=False)
            debug_messages = []

            for idx, build in enumerate(builds):
                build_number = build['number']
                status_text.text(f"Processing build #{build_number}... ({idx + 1}/{len(builds)})")

                # Get Robot Framework test results
                robot_data = rf_client.get_build_test_results(selected_job_info, build_number)

                if robot_data:
                    debug_messages.append(f"✅ Build #{build_number}: RF data received, keys: {list(robot_data.keys())}")
                    metrics = analyzer.parse_robot_results(robot_data, build)
                    metrics_list.append(metrics)
                    debug_messages.append(f"   → Parsed: {metrics.total_tests} tests, {metrics.passed_tests} passed, {metrics.failed_tests} failed")
                else:
                    debug_messages.append(f"❌ Build #{build_number}: No RF results returned")
                    logger.info(f"No RF results for build #{build_number}")

                progress_bar.progress((idx + 1) / len(builds))

            # Show debug information
            with debug_expander:
                st.text("\n".join(debug_messages[-50:]))  # Show last 50 messages

            progress_bar.empty()
            status_text.empty()

            if not metrics_list:
                st.error("❌ No Robot Framework test results found in any builds")
                st.markdown("""
                ### This job might not have Robot Framework plugin installed or configured.
                
                To fix this:
                1. Ensure Robot Framework plugin is installed in Jenkins
                2. Configure your job to publish Robot Framework results
                3. Verify builds have completed successfully with test results
                
                ### Debug Information:
                - Attempted to fetch RF results from {len(builds)} build(s)
                - Check Jenkins console logs for details
                - Verify the Robot Framework plugin is publishing results to Jenkins
                """)
                return

            # Validate and report what was collected
            total_tests_found = sum(m.total_tests for m in metrics_list)
            builds_with_tests = len([m for m in metrics_list if m.total_tests > 0])

            # Show detailed success message
            if builds_with_tests == len(metrics_list):
                st.success(f"🎉 Successfully analyzed {len(metrics_list)} builds with {total_tests_found} total tests!")
            else:
                st.warning(f"⚠️ Analyzed {len(metrics_list)} builds, but only {builds_with_tests} had test results ({total_tests_found} total tests)")

            # Perform detailed analysis enrichment (fetch logs and XML for deeper insights)
            with st.spinner("🔍 Enriching analysis with detailed logs and reports..."):
                status_text.text("Fetching detailed error information from builds...")
                try:
                    enriched_metrics = analyzer.enrich_metrics_with_detailed_analysis(
                        metrics_list,
                        rf_client,
                        selected_job_info,
                        max_builds_to_analyze=min(5, len(metrics_list))  # Analyze top 5 builds in detail
                    )
                    st.success("✅ Detailed analysis enrichment complete!")
                    metrics_list = enriched_metrics
                except Exception as e:
                    logger.warning(f"Error during enrichment: {e}")
                    st.info("ℹ️ Continuing with basic analysis (detailed enrichment failed)")

            # Store in session state and clear old insights to force refresh
            st.session_state.rf_metrics = metrics_list
            st.session_state.rf_selected_job = selected_job_name
            st.session_state.rf_job_info = selected_job_info
            st.session_state.rf_client = rf_client
            # Clear old insights to force regeneration with new data
            if 'rf_insights' in st.session_state:
                del st.session_state.rf_insights

            # Show summary of what was found
            with st.expander("📊 Collection Summary", expanded=False):
                st.write(f"**Builds analyzed:** {len(metrics_list)}")
                st.write(f"**Builds with tests:** {builds_with_tests}")
                st.write(f"**Total tests collected:** {total_tests_found}")
                st.write(f"**Date range:** {min(m.timestamp for m in metrics_list).strftime('%Y-%m-%d')} to {max(m.timestamp for m in metrics_list).strftime('%Y-%m-%d')}")

                # Show per-build summary with accurate failure rate
                summary_data = []
                for m in metrics_list:
                    # Calculate failure rate for this build
                    failure_rate = (m.failed_tests / m.total_tests * 100) if m.total_tests > 0 else 0

                    summary_data.append({
                        'Build': m.build_number,
                        'Total Tests': m.total_tests,
                        'Passed': m.passed_tests,
                        'Failed': m.failed_tests,
                        'Pass Rate': f"{m.pass_rate:.1f}%",
                        'Failure Rate': f"{failure_rate:.1f}%",
                        'Exec Time (s)': f"{m.execution_time / 1000:.1f}"
                    })

                st.markdown("**Per-Build Metrics:**")
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

                # Show average calculation
                if summary_data:
                    avg_failure_rate = calculate_average_failure_rate_per_build(metrics_list)
                    st.info(f"📊 **Average Failure Rate across all builds:** {avg_failure_rate:.1f}%")

    # Display results if available
    if 'rf_metrics' in st.session_state and st.session_state.rf_metrics:
        display_analysis_results(st.session_state.rf_metrics, analyzer)


def calculate_average_failure_rate_per_build(metrics_list: List[RFTestMetrics]) -> float:
    """Calculate average failure rate at build level - correct calculation"""
    if not metrics_list:
        return 0.0

    build_failure_rates = []

    for metrics in metrics_list:
        if metrics.total_tests > 0:
            # Calculate failure rate for this specific build
            build_failure_rate = (metrics.failed_tests / metrics.total_tests) * 100
            build_failure_rates.append(build_failure_rate)

    # Average across all builds
    if build_failure_rates:
        return sum(build_failure_rates) / len(build_failure_rates)
    return 0.0

def display_analysis_results(metrics_list: List[RFTestMetrics], analyzer: RFDashboardAnalyzer):
    """Display analysis results with visualizations"""

    st.markdown("---")
    st.subheader("📈 Analysis Results")

    # Validate data before processing
    if not metrics_list:
        st.error("❌ No metrics data available for analysis")
        return

    # Check if metrics have valid data
    valid_metrics = [m for m in metrics_list if m.total_tests > 0]
    if not valid_metrics:
        st.error("❌ No valid test data found in the collected metrics")
        st.info("All builds returned 0 tests. This might indicate:")
        st.markdown("""
        - Robot Framework plugin not properly configured
        - Test results not being published
        - Incorrect job selected
        """)
        return

    # Use only valid metrics for analysis
    if len(valid_metrics) < len(metrics_list):
        st.warning(f"⚠️ Using {len(valid_metrics)} out of {len(metrics_list)} builds (others had no test data)")
        metrics_list = valid_metrics

    # Perform trend analysis
    with st.spinner("Analyzing trends..."):
        analysis = analyzer.analyze_trends(metrics_list)

    # Validate analysis results
    if not analysis or analysis.get('total_runs', 0) == 0:
        st.error("❌ Failed to generate analysis from the collected data")
        return

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pass_rate = analysis['average_pass_rate']
        pass_rate_target = 95.0
        pass_rate_delta = pass_rate - pass_rate_target
        st.metric(
            "Average Pass Rate",
            f"{pass_rate:.1f}%",
            delta=f"{pass_rate_delta:+.1f}% vs target (95%)",
            delta_color="normal" if pass_rate >= pass_rate_target else "inverse",
            help=f"Target: 95%+. Current: {pass_rate:.1f}%"
        )

    with col2:
        stability = analysis['stability_score']
        stability_target = 90.0
        stability_delta = stability - stability_target
        st.metric(
            "Stability Score",
            f"{stability:.1f}/100",
            delta=f"{stability_delta:+.1f} vs target (90)",
            delta_color="normal" if stability >= stability_target else "inverse",
            help=f"Stability measures consistency (0-100). Higher = more predictable. Target: 90+. Formula: 100 - variance of pass rates"
        )

    with col3:
        trend_emoji = "📈" if analysis['pass_rate_trend'] == 'improving' else "📉" if analysis['pass_rate_trend'] == 'degrading' else "➡️"
        trend_color = "normal" if analysis['pass_rate_trend'] == 'improving' else "inverse" if analysis['pass_rate_trend'] == 'degrading' else "off"
        st.metric(
            "Pass Rate Trend",
            trend_emoji,
            delta=analysis['pass_rate_trend'].title(),
            delta_color=trend_color,
            help="Linear regression trend analysis: improving (↑), degrading (↓), or stable (→)"
        )

    with col4:
        total_runs = analysis['total_runs']
        st.metric(
            "Total Runs Analyzed",
            total_runs,
            delta=f"{total_runs} builds",
            help=f"Analysis based on {total_runs} recent builds. More builds = more accurate insights (recommended: 30+)"
        )

    # Stability Score Explanation
    with st.expander("ℹ️ Understanding Stability Score & Metrics", expanded=False):
        st.markdown("### 📊 Metric Explanations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🎯 Pass Rate")
            st.markdown(f"""
            **Current:** {analysis['average_pass_rate']:.1f}%  
            **Target:** 95%+  
            **Status:** {'✅ Meeting target' if analysis['average_pass_rate'] >= 95 else '⚠️ Below target'}
            
            **What it means:**
            - Percentage of tests passing on average
            - Calculated across all analyzed builds
            - Higher is better
            
            **Interpretation:**
            - 95-100%: Excellent
            - 85-95%: Good, minor improvements needed
            - 75-85%: Acceptable, needs attention
            - <75%: Critical, immediate action required
            """)

            st.markdown("#### 📈 Pass Rate Trend")
            trend = analysis['pass_rate_trend']
            st.markdown(f"""
            **Current:** {trend.title()}
            
            **What it means:**
            - Statistical trend analysis using linear regression
            - Shows direction of quality over time
            
            **Interpretation:**
            - **Improving (📈):** Quality getting better - keep it up!
            - **Stable (➡️):** Consistent quality - maintain current practices
            - **Degrading (📉):** Quality declining - investigate immediately!
            
            **Technical:** Calculated using numpy.polyfit on pass rates
            """)

        with col2:
            st.markdown("#### 🔒 Stability Score")
            stability = analysis['stability_score']
            variance = 100 - stability
            st.markdown(f"""
            **Current:** {stability:.1f}/100  
            **Target:** 90+/100  
            **Variance:** {variance:.1f}  
            **Status:** {'✅ Stable' if stability >= 90 else '⚠️ Unstable'}
            
            **What it means:**
            - Measures consistency of test results
            - Based on variance in pass rates
            - Higher score = more predictable results
            
            **Formula:**
            ```
            Stability = max(0, 100 - variance(pass_rates))
            ```
            
            **Example:**
            - Pass rates: [95%, 94%, 96%, 95%, 95%]
            - Low variance → High stability (98/100)
            
            - Pass rates: [95%, 75%, 85%, 90%, 70%]
            - High variance → Low stability (45/100)
            
            **Interpretation:**
            - 90-100: Excellent - very predictable
            - 80-90: Good - mostly consistent
            - 70-80: Fair - some variation
            - <70: Poor - unreliable results
            
            **Why it matters:**
            - Low stability = unreliable test results
            - Can't trust pass/fail decisions
            - Indicates flaky tests or environment issues
            """)

        st.markdown("---")
        st.markdown("### 🎯 How to Improve Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Improve Pass Rate:**")
            st.markdown("""
            1. Fix failing tests
            2. Remove obsolete tests
            3. Update tests for changes
            4. Investigate root causes
            """)

        with col2:
            st.markdown("**Improve Stability:**")
            st.markdown("""
            1. Fix flaky tests
            2. Stabilize environment
            3. Improve wait strategies
            4. Remove race conditions
            """)

        with col3:
            st.markdown("**Reverse Degrading Trend:**")
            st.markdown("""
            1. Identify recent changes
            2. Review failed tests
            3. Check application quality
            4. Improve test maintenance
            """)

    # Visualizations
    st.markdown("### 📊 Trends Over Time")

    # Create pass rate trend chart
    df_metrics = pd.DataFrame([
        {
            'Build': m.build_number,
            'Pass Rate': m.pass_rate,
            'Failure Rate': (m.failed_tests / m.total_tests * 100) if m.total_tests > 0 else 0,
            'Execution Time': m.execution_time / 1000,  # Convert to seconds
            'Total Tests': m.total_tests,
            'Passed Tests': m.passed_tests,
            'Failed Tests': m.failed_tests,
            'Date': m.timestamp
        }
        for m in metrics_list
    ])

    # Sort by build number
    df_metrics = df_metrics.sort_values('Build')

    # Pass rate trend
    fig_pass_rate = px.line(
        df_metrics,
        x='Build',
        y='Pass Rate',
        title='Pass Rate Trend',
        markers=True
    )
    fig_pass_rate.add_hline(y=95, line_dash="dash", line_color="green", annotation_text="Target: 95%")
    st.plotly_chart(fig_pass_rate, use_container_width=True)

    # Execution time trend
    col1, col2 = st.columns(2)

    with col1:
        fig_exec_time = px.line(
            df_metrics,
            x='Build',
            y='Execution Time',
            title='Execution Time Trend (seconds)',
            markers=True
        )
        st.plotly_chart(fig_exec_time, use_container_width=True)

    with col2:
        fig_failed = px.bar(
            df_metrics,
            x='Build',
            y='Failed Tests',
            title='Failed Tests per Build',
            color='Failed Tests',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_failed, use_container_width=True)

    # Detailed build-level metrics table
    st.markdown("### 📋 Build-Level Metrics - Detailed View")
    with st.expander("📊 View Detailed Build Metrics Table", expanded=False):
        st.markdown("**Complete build-by-build breakdown with failure counts and rates:**")

        # Create detailed table with all metrics
        build_table_data = []
        for m in metrics_list:
            failure_rate = (m.failed_tests / m.total_tests * 100) if m.total_tests > 0 else 0

            build_table_data.append({
                'Build #': m.build_number,
                'Date': m.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Total': m.total_tests,
                'Passed': m.passed_tests,
                'Failed': m.failed_tests,
                'Pass Rate': f"{m.pass_rate:.1f}%",
                'Failure Rate': f"{failure_rate:.1f}%",
                'Exec Time (s)': f"{m.execution_time / 1000:.1f}"
            })

        df_build_table = pd.DataFrame(build_table_data)
        st.dataframe(df_build_table, use_container_width=True, hide_index=True)

        # Validation summary
        total_builds = len(build_table_data)
        total_all_tests = sum(m.total_tests for m in metrics_list)
        total_all_failed = sum(m.failed_tests for m in metrics_list)
        avg_failure_rate_calc = calculate_average_failure_rate_per_build(metrics_list)

        st.markdown("**Validation Summary:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Builds", total_builds)
        with col2:
            st.metric("Total Tests (All Builds)", total_all_tests)
        with col3:
            st.metric("Total Failures (All Builds)", total_all_failed)
        with col4:
            st.metric("Avg Failure Rate", f"{avg_failure_rate_calc:.1f}%")

        st.caption("💡 Failure Rate per build = (Failed Tests / Total Tests) × 100. Average is calculated across all builds.")

    # Most failed tests - Enhanced with detailed analysis
    st.markdown("### ❌ Most Failed Tests - Detailed Analysis")
    if analysis['most_failed_tests']:
        failed_count = len(analysis['most_failed_tests'])
        chronic_failures = [t for t in analysis['most_failed_tests'] if t.get('failure_rate', 0) > 70]

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Failing Tests", failed_count,
                     help="Tests that have failed at least once")
        with col2:
            st.metric("Chronic Failures (>70%)", len(chronic_failures),
                     help="Tests failing more than 70% of the time - likely broken")
        with col3:
            # Calculate correct average failure rate at build level
            avg_failure_rate = calculate_average_failure_rate_per_build(metrics_list)
            st.metric("Avg Failure Rate", f"{avg_failure_rate:.1f}%",
                     help="Average failure rate per build (failed_tests/total_tests per build, then averaged)")

        # Detailed table with actionable insights
        df_failed = pd.DataFrame(analysis['most_failed_tests'])

        # Add severity classification
        def get_severity(rate):
            if rate > 90: return "🔴 CRITICAL"
            elif rate > 70: return "🟠 HIGH"
            elif rate > 50: return "🟡 MEDIUM"
            else: return "🟢 LOW"

        df_failed['Severity'] = df_failed['failure_rate'].apply(get_severity)
        df_failed['Failure Rate'] = df_failed['failure_rate'].apply(lambda x: f"{x:.1f}%")
        df_failed['Failures'] = df_failed['failure_count']
        df_failed['Total Runs'] = df_failed['total_runs']

        # Display comprehensive table with Total Runs for transparency
        st.dataframe(
            df_failed[['Severity', 'test', 'Failures', 'Total Runs', 'Failure Rate']].rename(columns={'test': 'Test Name'}),
            use_container_width=True,
            height=300
        )

        st.caption("💡 **Failure Rate Calculation:** (Failures / Total Runs) × 100. 'Total Runs' = number of builds where this test actually ran.")

        # Visualize with color coding by severity
        fig_top_failures = px.bar(
            df_failed.head(10),
            x='failure_count',
            y='test',
            orientation='h',
            title='Top 10 Frequently Failing Tests',
            labels={'failure_count': 'Failures', 'test': 'Test Name'},
            color='failure_rate',
            color_continuous_scale='Reds',
            hover_data={'failure_rate': ':.1f%'}
        )
        st.plotly_chart(fig_top_failures, use_container_width=True)

        # Specific insights and recommendations
        st.markdown("#### 💡 Insights & Recommendations")

        if chronic_failures:
            st.error(f"**Critical Finding:** {len(chronic_failures)} tests are failing >70% of the time")
            st.markdown("**Recommended Actions:**")
            for idx, test in enumerate(chronic_failures[:3], 1):
                st.markdown(f"{idx}. **{test['test']}** (Failing {test['failure_rate']:.1f}%)")
                st.markdown(f"   - **Action:** Review test logic or disable if obsolete")
                st.markdown(f"   - **Impact:** Could improve pass rate by ~{test['failure_rate']/10:.1f}%")
                if test.get('sample_messages'):
                    with st.expander(f"View Error Messages for {test['test'][:50]}..."):
                        for msg in test['sample_messages'][:3]:
                            st.code(msg if msg else "No error message available", language="text")

        # Pattern analysis
        suite_patterns = {}
        for test in analysis['most_failed_tests']:
            for suite in test.get('suites', []):
                suite_patterns[suite] = suite_patterns.get(suite, 0) + 1

        if suite_patterns:
            st.markdown("#### 🔍 Failure Pattern by Suite")
            suite_data = [{'Suite': k, 'Failing Tests': v} for k, v in sorted(suite_patterns.items(), key=lambda x: x[1], reverse=True)]
            if suite_data:
                st.dataframe(pd.DataFrame(suite_data).head(5), use_container_width=True)
                st.info(f"**Suite '{suite_data[0]['Suite']}'** has the most failing tests ({suite_data[0]['Failing Tests']}) - investigate for common issues")
    else:
        st.success("✅ Excellent! No frequently failing tests detected!")

    # Flaky tests - Enhanced with detailed analysis
    st.markdown("### 🔄 Flaky Tests - Instability Analysis")
    if analysis['flaky_tests']:
        flaky_count = len(analysis['flaky_tests'])

        # Calculate impact metrics
        highly_flaky = [t for t in analysis['flaky_tests'] if t.get('flakiness_score', 0) > 40]
        total_flaky_runs = sum(t.get('total_runs', 0) for t in analysis['flaky_tests'])

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Flaky Tests", flaky_count,
                     help="Tests showing inconsistent pass/fail behavior")
        with col2:
            st.metric("Highly Flaky (>40%)", len(highly_flaky),
                     help="Tests with severe instability")
        with col3:
            avg_flakiness = sum(t.get('flakiness_score', 0) for t in analysis['flaky_tests']) / len(analysis['flaky_tests'])
            st.metric("Avg Flakiness Score", f"{avg_flakiness:.1f}%",
                     help="Average instability across all flaky tests")
        with col4:
            # Calculate reliability impact
            reliability_impact = flaky_count * 2  # Rough estimate
            st.metric("Reliability Impact", f"-{reliability_impact:.1f} pts",
                     delta=f"-{reliability_impact:.1f}",
                     delta_color="inverse",
                     help="Estimated impact on test suite reliability")

        # Explanation of flakiness score
        with st.expander("ℹ️ Understanding Flakiness Score", expanded=False):
            st.markdown("""
            **Flakiness Score** measures test instability:
            - **Formula:** `min(passes, fails) / total_runs * 100`
            - **>40%:** Highly flaky - test is unreliable
            - **20-40%:** Moderately flaky - needs investigation
            - **<20%:** Low flakiness - may be environment-related
            
            **Example:** A test with 6 passes and 4 fails out of 10 runs has:
            - Flakiness Score = `min(6, 4) / 10 * 100 = 40%`
            - This means 40% of runs show inconsistent behavior
            """)

        # Detailed table
        df_flaky = pd.DataFrame(analysis['flaky_tests'])

        # Add severity classification
        def get_flaky_severity(score):
            if score > 40: return "🔴 HIGH"
            elif score > 25: return "🟠 MEDIUM"
            else: return "🟡 LOW"

        df_flaky['Severity'] = df_flaky['flakiness_score'].apply(get_flaky_severity)
        df_flaky['Flakiness'] = df_flaky['flakiness_score'].apply(lambda x: f"{x:.1f}%")
        df_flaky['Pass/Fail Ratio'] = df_flaky.apply(lambda x: f"{x['passes']}P / {x['fails']}F", axis=1)
        df_flaky['Total Runs'] = df_flaky['total_runs']

        st.dataframe(
            df_flaky[['Severity', 'test', 'Flakiness', 'Pass/Fail Ratio', 'Total Runs']].rename(columns={'test': 'Test Name'}),
            use_container_width=True,
            height=300
        )

        # Visualization
        fig_flaky = px.bar(
            df_flaky.head(10),
            x='flakiness_score',
            y='test',
            orientation='h',
            title='Top 10 Flaky Tests (by Flakiness Score)',
            labels={'flakiness_score': 'Flakiness Score (%)', 'test': 'Test Name'},
            color='flakiness_score',
            color_continuous_scale='YlOrRd'
        )
        st.plotly_chart(fig_flaky, use_container_width=True)

        # Specific insights and recommendations
        st.markdown("#### 💡 Root Causes & Recommendations")

        if flaky_count > 10:
            st.error(f"**Critical:** {flaky_count} flaky tests indicate severe environment or test design issues")
            st.markdown("**Likely Root Causes:**")
            st.markdown("- 🌍 **Environment instability** - inconsistent test data, shared resources, or infrastructure issues")
            st.markdown("- ⏱️ **Timing issues** - race conditions, insufficient waits, or timeout problems")
            st.markdown("- 🔧 **Test design flaws** - improper cleanup, state dependencies, or order dependencies")
        elif flaky_count > 5:
            st.warning(f"**High Priority:** {flaky_count} flaky tests undermining test suite reliability")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Investigate top 5 flaky tests for common patterns")
            st.markdown("- Check for shared resources or global state")
            st.markdown("- Review wait strategies and timeout settings")
        else:
            st.info(f"**Manageable:** {flaky_count} flaky tests detected - prioritize fixing these")

        st.markdown("**Immediate Actions:**")
        for idx, test in enumerate(highly_flaky[:3], 1):
            st.markdown(f"{idx}. **{test['test']}** (Flakiness: {test['flakiness_score']:.1f}%)")
            st.markdown(f"   - **Pattern:** Passed {test['passes']} times, Failed {test['fails']} times")
            st.markdown(f"   - **Action:** Add proper waits, check for race conditions, verify cleanup")
            st.markdown(f"   - **Impact:** Fixing will improve stability by ~{test['flakiness_score']/5:.1f} points")

        # Flaky penalty calculation explanation
        with st.expander("📊 Flaky Penalty Impact on Quality Score", expanded=False):
            # Calculate actual penalty
            if flaky_count > 10:
                penalty_pct = 30
            elif flaky_count > 5:
                penalty_pct = 20
            elif flaky_count > 0:
                penalty_pct = 10
            else:
                penalty_pct = 0

            st.markdown(f"""
            **Current Flaky Penalty:** {penalty_pct}% reduction in reliability score
            
            **Penalty Calculation:**
            - 1-5 flaky tests: -10% reliability score
            - 6-10 flaky tests: -20% reliability score  
            - >10 flaky tests: -30% reliability score
            
            **Your Impact:** With {flaky_count} flaky tests, your reliability score is reduced by {penalty_pct}%
            
            **Potential Gain:** Fixing all flaky tests could improve your overall quality score by {penalty_pct * 0.4:.1f} points
            (Reliability = 40% of overall score)
            """)
    else:
        st.success("✅ Excellent! No flaky tests detected - test suite is stable and reliable!")

    # Slowest tests - Enhanced with performance analysis
    st.markdown("### 🐌 Slowest Tests - Performance Analysis")
    if analysis['slowest_tests']:
        # Calculate performance metrics
        df_slow = pd.DataFrame(analysis['slowest_tests'])
        df_slow['avg_duration_sec'] = df_slow['avg_duration'] / 1000

        very_slow = df_slow[df_slow['avg_duration_sec'] > 60]  # >1 minute
        slow = df_slow[(df_slow['avg_duration_sec'] > 30) & (df_slow['avg_duration_sec'] <= 60)]  # 30s-1min
        moderate = df_slow[(df_slow['avg_duration_sec'] > 10) & (df_slow['avg_duration_sec'] <= 30)]  # 10-30s

        total_time = df_slow['avg_duration_sec'].sum()
        avg_time = df_slow['avg_duration_sec'].mean()

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Very Slow Tests (>60s)", len(very_slow),
                     help="Tests taking more than 1 minute")
        with col2:
            st.metric("Slow Tests (30-60s)", len(slow),
                     help="Tests taking 30 seconds to 1 minute")
        with col3:
            st.metric("Total Time (Top 10)", f"{df_slow.head(10)['avg_duration_sec'].sum():.1f}s",
                     help="Combined execution time of 10 slowest tests")
        with col4:
            potential_saving = df_slow.head(10)['avg_duration_sec'].sum() * 0.25  # 25% optimization
            st.metric("Potential Saving (25%)", f"{potential_saving:.1f}s",
                     help="Time saved if tests optimized by 25%")

        # Performance categories
        def get_performance_category(seconds):
            if seconds > 60: return "🔴 VERY SLOW"
            elif seconds > 30: return "🟠 SLOW"
            elif seconds > 10: return "🟡 MODERATE"
            else: return "🟢 FAST"

        df_slow['Performance'] = df_slow['avg_duration_sec'].apply(get_performance_category)
        df_slow['Duration'] = df_slow['avg_duration_sec'].apply(lambda x: f"{x:.1f}s")
        df_slow['% of Total'] = (df_slow['avg_duration_sec'] / total_time * 100).apply(lambda x: f"{x:.1f}%")

        # Detailed table
        st.dataframe(
            df_slow[['Performance', 'test', 'Duration', '% of Total']].rename(columns={'test': 'Test Name'}).head(15),
            use_container_width=True,
            height=300
        )

        # Visualization with color coding
        fig_slow = px.bar(
            df_slow.head(10),
            x='avg_duration_sec',
            y='test',
            orientation='h',
            title='Top 10 Slowest Tests (seconds)',
            labels={'avg_duration_sec': 'Duration (seconds)', 'test': 'Test Name'},
            color='avg_duration_sec',
            color_continuous_scale='RdYlGn_r'  # Red for slow, green for fast
        )
        # Add reference lines
        fig_slow.add_vline(x=60, line_dash="dash", line_color="red", annotation_text="Very Slow (60s)")
        fig_slow.add_vline(x=30, line_dash="dash", line_color="orange", annotation_text="Slow (30s)")
        st.plotly_chart(fig_slow, use_container_width=True)

        # Specific insights and optimization recommendations
        st.markdown("#### 💡 Performance Insights & Optimization Opportunities")

        if len(very_slow) > 0:
            st.error(f"**Critical:** {len(very_slow)} tests taking >60 seconds each")
            st.markdown("**High-Priority Optimizations:**")
            for idx, row in very_slow.head(3).iterrows():
                test_name = row['test']
                duration = row['avg_duration_sec']
                st.markdown(f"{idx+1}. **{test_name}** - {duration:.1f}s")
                st.markdown(f"   - **Issue:** Excessive execution time likely due to:")
                st.markdown(f"      - Long waits or sleep statements")
                st.markdown(f"      - Heavy data processing")
                st.markdown(f"      - Multiple external API calls")
                st.markdown(f"      - Database operations without indexes")
                st.markdown(f"   - **Optimization:** Reduce waits, parallelize operations, optimize queries")
                st.markdown(f"   - **Impact:** Potential {duration * 0.3:.1f}s savings (30% improvement)")

        if len(slow) > 0:
            st.warning(f"**Medium Priority:** {len(slow)} tests taking 30-60 seconds")
            st.markdown("**Optimization Opportunities:**")
            st.markdown("- Review explicit wait times and reduce if excessive")
            st.markdown("- Check for unnecessary page reloads or navigation")
            st.markdown("- Optimize data setup/teardown operations")
            st.markdown("- Consider splitting long tests into smaller units")

        # Calculate optimization impact
        st.markdown("#### 📊 Optimization Impact Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Current State:**")
            st.write(f"- Top 10 tests: {df_slow.head(10)['avg_duration_sec'].sum():.1f}s combined")
            st.write(f"- Average test time: {avg_time:.1f}s")
            st.write(f"- Very slow tests (>60s): {len(very_slow)}")

        with col2:
            st.markdown("**After 25% Optimization:**")
            optimized_total = df_slow.head(10)['avg_duration_sec'].sum() * 0.75
            st.write(f"- Top 10 tests: {optimized_total:.1f}s combined")
            st.write(f"- Time saved per run: {potential_saving:.1f}s")
            st.write(f"- Annual saving (1000 runs): {potential_saving * 1000 / 3600:.1f} hours")

        # Specific optimization techniques
        with st.expander("🔧 Specific Optimization Techniques", expanded=False):
            st.markdown("""
            ### Common Performance Issues & Solutions:
            
            **1. Excessive Waits (Most Common)**
            - ❌ Problem: `Sleep 30s` or long implicit waits
            - ✅ Solution: Use explicit waits with conditions
            - 💰 Savings: 40-60% reduction
            
            **2. Redundant Operations**
            - ❌ Problem: Re-login for every test, repeated navigation
            - ✅ Solution: Session reuse, smart navigation
            - 💰 Savings: 20-30% reduction
            
            **3. Inefficient Data Setup**
            - ❌ Problem: Creating test data through UI
            - ✅ Solution: Use APIs for data setup
            - 💰 Savings: 50-70% reduction
            
            **4. Too Many Assertions**
            - ❌ Problem: Single test verifying 20+ things
            - ✅ Solution: Split into focused tests
            - 💰 Savings: Better parallelization
            
            **5. External Dependencies**
            - ❌ Problem: Multiple third-party API calls
            - ✅ Solution: Mock external services, use stubs
            - 💰 Savings: 30-50% reduction
            """)
    else:
        st.info("No significant slow tests detected - all tests are performing well!")

    # AI Insights Section - Enhanced
    st.markdown("---")
    st.markdown("## 🤖 AI-Powered Insights & Analytics")

    # Auto-generate insights on first display
    if 'rf_insights' not in st.session_state:
        with st.spinner("🧠 Generating comprehensive AI insights..."):
            insights = analyzer.generate_ai_insights(analysis, metrics_list)
            st.session_state.rf_insights = insights

    # Refresh button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("*Complete analysis with quality scoring, root cause analysis, and actionable recommendations*")
    with col3:
        if st.button("🔄 Refresh Insights", help="Regenerate AI insights with latest data"):
            with st.spinner("🧠 Regenerating insights..."):
                insights = analyzer.generate_ai_insights(analysis, metrics_list)
                st.session_state.rf_insights = insights
                st.success("✅ Insights updated!")

    # Display comprehensive insights
    if 'rf_insights' in st.session_state:
        insights = st.session_state.rf_insights

        # Quality Score Dashboard - Top Priority
        st.markdown("### 📊 Test Suite Quality Score")
        quality_score = insights.get('quality_score', {})

        if quality_score:
            # Overall grade with color coding
            grade = quality_score.get('grade', 'N/A')
            overall = quality_score.get('overall_score', 0)
            grade_desc = quality_score.get('grade_description', '')

            grade_colors = {
                'A': '#28a745',  # Green
                'B': '#5cb85c',  # Light Green
                'C': '#ffc107',  # Yellow
                'D': '#ff8c00',  # Orange
                'F': '#dc3545'   # Red
            }
            grade_color = grade_colors.get(grade, '#6c757d')

            # Create 5 columns for quality metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, {grade_color} 0%, {grade_color}cc 100%); 
                            border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h1 style='margin: 0; font-size: 72px; font-weight: bold;'>{grade}</h1>
                    <p style='margin: 5px 0 0 0; font-size: 14px;'>{grade_desc}</p>
                    <p style='margin: 5px 0 0 0; font-size: 20px; font-weight: bold;'>{overall:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                reliability = quality_score.get('reliability', 0)
                st.metric("🎯 Reliability", f"{reliability:.1f}/100",
                         help="Based on pass rate and stability")

            with col3:
                performance = quality_score.get('performance', 0)
                st.metric("⚡ Performance", f"{performance:.1f}/100",
                         help="Based on execution time")

            with col4:
                maintainability = quality_score.get('maintainability', 0)
                st.metric("🔧 Maintainability", f"{maintainability:.1f}/100",
                         help="Based on failure patterns")

            with col5:
                coverage = quality_score.get('coverage', 0)
                st.metric("📈 Coverage", f"{coverage:.1f}/100",
                         help="Based on test execution frequency")

            # Detailed breakdown in expander with comprehensive explanations
            with st.expander("📋 Quality Score Breakdown - Detailed Calculation", expanded=False):
                breakdown = quality_score.get('breakdown', {})
                if breakdown:
                    st.markdown("### How Your Quality Score is Calculated")

                    # Reliability Score Breakdown (40% weight)
                    st.markdown("#### 🎯 Reliability Score (40% of Overall)")
                    reliability_base = (breakdown.get('pass_rate', 0) * 0.65) + (breakdown.get('stability', 0) * 0.35)
                    flaky_penalty = breakdown.get('flaky_penalty', 0)
                    final_reliability = reliability - flaky_penalty if 'reliability' in locals() else quality_score.get('reliability', 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Components:**")
                        st.write(f"✓ Pass Rate: {breakdown.get('pass_rate', 0):.1f}% (65% weight)")
                        st.write(f"✓ Stability: {breakdown.get('stability', 0):.1f}/100 (35% weight)")
                        st.write(f"✓ Base Score: {reliability_base:.1f}")
                    with col2:
                        st.markdown("**Adjustments:**")
                        if flaky_penalty > 0:
                            st.write(f"⚠️ Flaky Test Penalty: -{flaky_penalty:.1f}")
                            st.write(f"✓ Final Reliability: {final_reliability:.1f}/100")
                        else:
                            st.write(f"✅ No flaky test penalty")
                            st.write(f"✓ Final Reliability: {reliability_base:.1f}/100")

                    st.markdown("---")

                    # Performance Score Breakdown (25% weight)
                    st.markdown("#### ⚡ Performance Score (25% of Overall)")
                    avg_exec = breakdown.get('avg_exec_time', 0)
                    time_per_test = breakdown.get('time_per_test', 0)
                    avg_test_count = breakdown.get('avg_test_count', 0)
                    perf_score = quality_score.get('performance', 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Performance Metrics:**")
                        st.write(f"• Avg Execution Time: {avg_exec:.1f}s")
                        st.write(f"• Time per Test: {time_per_test:.2f}s")
                        st.write(f"• Avg Test Count: {avg_test_count:.0f}")
                    with col2:
                        st.markdown("**Score Impact:**")
                        if time_per_test < 0.5:
                            st.success(f"✅ Excellent: <0.5s/test = {perf_score:.1f}/100")
                        elif time_per_test < 2:
                            st.info(f"ℹ️ Good: <2s/test = {perf_score:.1f}/100")
                        elif time_per_test < 5:
                            st.warning(f"⚠️ Acceptable: <5s/test = {perf_score:.1f}/100")
                        else:
                            st.error(f"🚨 Poor: {time_per_test:.1f}s/test = {perf_score:.1f}/100")

                    # Show execution time benchmarks
                    st.markdown("**📊 Performance Benchmarks:**")
                    st.markdown("""
                    **Time per Test (normalized):**
                    - <0.5s = 95 points (Excellent)
                    - 0.5-2s = 85 points (Good)
                    - 2-5s = 70 points (Acceptable)
                    - 5-10s = 55 points (Poor)
                    - \>10s = 40 points (Very Poor)
                    
                    💡 *Normalized by test count for fair comparison*
                    """)

                    st.markdown("---")

                    # Maintainability Score Breakdown (20% weight)
                    st.markdown("#### 🔧 Maintainability Score (20% of Overall)")
                    chronic_failures = breakdown.get('chronic_failures', 0)
                    total_failures = breakdown.get('total_failures', 0)
                    maint_score = quality_score.get('maintainability', 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Failure Analysis:**")
                        st.write(f"• Chronic Failures (>50%): {chronic_failures}")
                        st.write(f"• Total Failing Tests: {total_failures}")
                    with col2:
                        st.markdown("**Score Impact:**")
                        if chronic_failures > 10:
                            st.error(f"🚨 High chronic failures: {maint_score:.1f}/100")
                        elif chronic_failures > 5:
                            st.warning(f"⚠️ Moderate chronic failures: {maint_score:.1f}/100")
                        elif chronic_failures > 0:
                            st.info(f"ℹ️ Some chronic failures: {maint_score:.1f}/100")
                        else:
                            st.success(f"✅ No chronic failures: {maint_score:.1f}/100")

                    st.markdown("---")

                    # Coverage Score Breakdown (15% weight)
                    st.markdown("#### 📈 Coverage Score (15% of Overall)")
                    total_runs = breakdown.get('total_runs', 0)
                    coverage_score = quality_score.get('coverage', 0)
                    data_confidence = breakdown.get('data_confidence', 0)
                    execution_consistency = breakdown.get('execution_consistency', 0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Coverage Analysis:**")
                        st.write(f"• Total Builds Analyzed: {total_runs}")
                        st.write(f"• Data Confidence: {data_confidence:.1f}%")
                        st.write(f"• Execution Consistency: {execution_consistency:.1f}%")
                    with col2:
                        st.markdown("**Score Impact:**")
                        if coverage_score >= 80:
                            st.success(f"✅ Excellent coverage: {coverage_score:.1f}/100")
                        elif coverage_score >= 65:
                            st.info(f"ℹ️ Good coverage: {coverage_score:.1f}/100")
                        elif coverage_score >= 50:
                            st.warning(f"⚠️ Moderate coverage: {coverage_score:.1f}/100")
                        else:
                            st.error(f"🚨 Limited coverage: {coverage_score:.1f}/100")

                    st.caption("💡 Coverage = Execution Consistency × Data Confidence. High scores mean tests run consistently across builds with sufficient data.")

                    st.markdown("---")

                    # Overall Calculation
                    st.markdown("#### 📊 Overall Score Calculation")
                    st.markdown("""
                    **Formula:**
                    ```
                    Overall = (Reliability × 0.40) + (Performance × 0.25) + 
                              (Maintainability × 0.20) + (Coverage × 0.15)
                    ```
                    """)

                    calc_overall = (
                        quality_score.get('reliability', 0) * 0.40 +
                        quality_score.get('performance', 0) * 0.25 +
                        quality_score.get('maintainability', 0) * 0.20 +
                        quality_score.get('coverage', 0) * 0.15
                    )

                    st.markdown(f"""
                    **Your Calculation:**
                    ```
                    Overall = ({quality_score.get('reliability', 0):.1f} × 0.40) + 
                              ({quality_score.get('performance', 0):.1f} × 0.25) + 
                              ({quality_score.get('maintainability', 0):.1f} × 0.20) + 
                              ({quality_score.get('coverage', 0):.1f} × 0.15)
                            = {calc_overall:.1f} ≈ {quality_score.get('overall_score', 0):.1f}
                    ```
                    **Grade:** {quality_score.get('grade', 'N/A')} ({quality_score.get('grade_description', 'Unknown')})
                    """)

        # Executive Summary with improved styling and specific insights
        st.markdown("### 📋 Executive Summary")

        # ALWAYS use locally-generated summary to avoid AI truncation issues
        # Generate fresh summary from current data
        quality_score = insights.get('quality_score', {})
        grade = quality_score.get('grade', 'N/A')
        score = quality_score.get('overall_score', 0)
        grade_desc = quality_score.get('grade_description', '')
        pass_rate = analysis.get('average_pass_rate', 0)
        trend = analysis.get('pass_rate_trend', 'stable')
        stability = analysis.get('stability_score', 0)
        flaky_count = len(analysis.get('flaky_tests', []))
        failed_count = len(analysis.get('most_failed_tests', []))

        # Generate complete, non-truncated summary
        if grade in ['A', 'B'] and pass_rate >= 85:
            urgency = "LOW"
            status = f"Test suite is in {grade_desc.lower() if grade_desc else 'good'} condition"
        elif grade in ['C'] or (grade in ['B'] and pass_rate < 85):
            urgency = "MEDIUM"
            status = f"Test suite requires attention"
        else:
            urgency = "HIGH"
            status = f"Test suite needs immediate action"

        exec_summary = f"{status} (Grade {grade}, {score:.1f}/100). "
        exec_summary += f"Pass rate is {pass_rate:.1f}% and {trend}. "

        # Add specific concerns based on actual data
        if pass_rate < 75:
            exec_summary += f"CRITICAL: Low pass rate ({pass_rate:.1f}%) requires immediate investigation. "
        elif flaky_count > 10:
            exec_summary += f"CONCERN: {flaky_count} flaky tests undermining reliability. "
        elif flaky_count > 5:
            exec_summary += f"WARNING: {flaky_count} flaky tests detected. "

        if failed_count > 15:
            exec_summary += f"CONCERN: {failed_count} tests failing consistently. "
        elif failed_count > 10:
            exec_summary += f"NOTE: {failed_count} tests need attention. "

        if stability < 70:
            exec_summary += f"CRITICAL: Very low stability ({stability:.1f}/100) indicates environment issues. "
        elif stability < 85:
            exec_summary += f"WARNING: Moderate stability ({stability:.1f}/100) needs improvement. "

        if not any(keyword in exec_summary for keyword in ['CRITICAL', 'CONCERN', 'WARNING']):
            exec_summary += "Continue monitoring for stability. "

        exec_summary += f"Urgency: {urgency}."

        # Final cleanup (basic only since we generated it locally)
        exec_summary = str(exec_summary).strip()

        # Enhanced summary display with visual indicators
        # (quality_score already defined above)
        grade = quality_score.get('grade', 'N/A')
        overall = quality_score.get('overall_score', 0)

        # Determine status color
        if grade in ['A', 'B']:
            summary_color = "green"
            summary_emoji = "✅"
        elif grade == 'C':
            summary_color = "orange"
            summary_emoji = "⚠️"
        else:
            summary_color = "red"
            summary_emoji = "🚨"

        # Create comprehensive summary card with sanitized text
        st.markdown(f"""
        <div style='padding: 20px; background-color: rgba(0,0,0,0.05); border-left: 5px solid {summary_color}; border-radius: 5px;'>
            <h4 style='margin-top: 0;'>{summary_emoji} {exec_summary}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Add specific metrics summary
        col1, col2, col3 = st.columns(3)
        with col1:
            pass_rate = analysis.get('average_pass_rate', 0)
            pass_target = 95.0
            if pass_rate >= pass_target:
                st.success(f"✅ Pass Rate: {pass_rate:.1f}% (Meeting target)")
            elif pass_rate >= 80:
                st.warning(f"⚠️ Pass Rate: {pass_rate:.1f}% ({pass_target - pass_rate:.1f}% below target)")
            else:
                st.error(f"🚨 Pass Rate: {pass_rate:.1f}% (CRITICAL: {pass_target - pass_rate:.1f}% below target)")

        with col2:
            stability = analysis.get('stability_score', 0)
            stability_target = 90.0
            if stability >= stability_target:
                st.success(f"✅ Stability: {stability:.1f}/100 (Excellent)")
            elif stability >= 70:
                st.warning(f"⚠️ Stability: {stability:.1f}/100 (Needs improvement)")
            else:
                st.error(f"🚨 Stability: {stability:.1f}/100 (CRITICAL: Very unstable)")

        with col3:
            flaky_count = len(analysis.get('flaky_tests', []))
            if flaky_count == 0:
                st.success(f"✅ Flaky Tests: 0 (Perfect)")
            elif flaky_count <= 3:
                st.info(f"ℹ️ Flaky Tests: {flaky_count} (Manageable)")
            elif flaky_count <= 10:
                st.warning(f"⚠️ Flaky Tests: {flaky_count} (High priority)")
            else:
                st.error(f"🚨 Flaky Tests: {flaky_count} (CRITICAL: Severe instability)")

        # Critical Issues Section
        st.markdown("### 🚨 Critical Issues")
        critical_issues = insights.get('critical_issues', [])

        if critical_issues:
            # Show summary count with severity breakdown
            severity_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for issue in critical_issues:
                severity = issue.get('severity', 'MEDIUM').upper()
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            summary_parts = []
            if severity_counts['CRITICAL'] > 0:
                summary_parts.append(f"🔴 {severity_counts['CRITICAL']} Critical")
            if severity_counts['HIGH'] > 0:
                summary_parts.append(f"🟠 {severity_counts['HIGH']} High")
            if severity_counts['MEDIUM'] > 0:
                summary_parts.append(f"🟡 {severity_counts['MEDIUM']} Medium")

            if summary_parts:
                st.info(f"**Found {len(critical_issues)} issue(s):** {', '.join(summary_parts)}")

            for idx, issue in enumerate(critical_issues, 1):
                severity = issue.get('severity', 'MEDIUM').upper()
                severity_colors = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                severity_icon = severity_colors.get(severity, '⚪')

                # Get issue description - ensure it's meaningful
                issue_title = issue.get('issue', '').strip()
                if not issue_title or issue_title.lower() in ['unknown issue', 'unknown', 'unspecified issue']:
                    # Create a meaningful title from root_cause or impact
                    root_cause = issue.get('root_cause', '').strip()
                    if root_cause and root_cause not in ['Not identified', 'Unknown', '']:
                        issue_title = f"Issue: {root_cause[:60]}..."
                    else:
                        impact = issue.get('impact', '').strip()
                        if impact and impact not in ['Unknown', 'Unknown impact', '']:
                            issue_title = f"Impact: {impact[:60]}..."
                        else:
                            issue_title = f"Issue #{idx} - Details Below"

                with st.expander(f"{severity_icon} **[{severity}]** {issue_title}", expanded=idx <= 2):
                    root_cause = issue.get('root_cause', 'Not identified').strip()
                    impact = issue.get('impact', 'Unknown').strip()
                    evidence = issue.get('evidence', 'No evidence provided').strip()

                    # Enhanced display with better formatting
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("**🔍 Root Cause:**")
                    with col2:
                        st.markdown(root_cause if root_cause else '*Not identified*')

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("**💥 Impact:**")
                    with col2:
                        st.markdown(impact if impact else '*Unknown*')

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown("**📊 Evidence:**")
                    with col2:
                        st.markdown(evidence if evidence else '*No evidence provided*')

                    # Add recommended action if available
                    if 'recommended_action' in issue:
                        st.markdown(f"**✅ Recommended Action:** {issue['recommended_action']}")
        else:
            st.success("✅ No critical issues detected!")

        # Root Causes Analysis - Enhanced with specific explanations
        st.markdown("### 🔍 Root Cause Analysis - Why Tests Are Failing")
        root_causes = insights.get('root_causes', {})

        # Summary of root causes
        total_causes = sum(len(causes) for causes in root_causes.values())
        critical_causes = sum(len([c for c in causes if '[CRITICAL]' in c]) for causes in root_causes.values())

        if total_causes > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Root Causes", total_causes)
            with col2:
                st.metric("Critical Issues", critical_causes)
            with col3:
                most_affected = max(root_causes.items(), key=lambda x: len(x[1])) if root_causes else ('None', [])
                st.metric("Most Affected Category", most_affected[0].replace('_', ' ').title())

        if root_causes:
            tabs = st.tabs(["🌍 Environmental", "🧪 Test Design", "💻 Application", "🏗️ Infrastructure", "⏱️ Timing"])

            category_info = {
                'environmental': {
                    'tab': tabs[0],
                    'icon': '🌍',
                    'description': 'Test environment instability, configuration issues, and resource problems',
                    'common_fixes': [
                        'Stabilize test data and environment',
                        'Implement proper cleanup between tests',
                        'Use dedicated test environments',
                        'Avoid shared resources'
                    ]
                },
                'test_design': {
                    'tab': tabs[1],
                    'icon': '🧪',
                    'description': 'Poor test logic, invalid assertions, and maintenance issues',
                    'common_fixes': [
                        'Review and update test logic',
                        'Fix invalid assertions',
                        'Remove obsolete tests',
                        'Follow testing best practices'
                    ]
                },
                'application': {
                    'tab': tabs[2],
                    'icon': '💻',
                    'description': 'Application bugs, regressions, and feature changes',
                    'common_fixes': [
                        'Investigate application regressions',
                        'Update tests for feature changes',
                        'Report bugs to development team',
                        'Coordinate with developers'
                    ]
                },
                'infrastructure': {
                    'tab': tabs[3],
                    'icon': '🏗️',
                    'description': 'Service availability, database issues, and network problems',
                    'common_fixes': [
                        'Monitor service health',
                        'Check database connectivity',
                        'Verify network stability',
                        'Review resource allocation'
                    ]
                },
                'timing': {
                    'tab': tabs[4],
                    'icon': '⏱️',
                    'description': 'Timeout issues, race conditions, and synchronization problems',
                    'common_fixes': [
                        'Implement proper wait strategies',
                        'Add explicit waits for dynamic content',
                        'Review timeout values',
                        'Fix race conditions'
                    ]
                }
            }

            for category, info in category_info.items():
                with info['tab']:
                    st.markdown(f"**{info['icon']} About {category.replace('_', ' ').title()} Issues:**")
                    st.info(info['description'])

                    causes = root_causes.get(category, [])
                    if causes:
                        st.markdown(f"#### Identified Issues ({len(causes)})")

                        # Count by severity
                        critical = [c for c in causes if '[CRITICAL]' in c]
                        high = [c for c in causes if '[HIGH]' in c]
                        medium = [c for c in causes if '[MEDIUM]' in c]

                        if critical:
                            st.markdown("**🔴 Critical Issues:**")
                            for cause in critical:
                                st.error(cause)

                        if high:
                            st.markdown("**🟠 High Priority Issues:**")
                            for cause in high:
                                st.warning(cause)

                        if medium:
                            st.markdown("**🟡 Medium Priority Issues:**")
                            for cause in medium:
                                st.info(cause)

                        # Low priority (no severity tag)
                        low = [c for c in causes if not any(sev in c for sev in ['[CRITICAL]', '[HIGH]', '[MEDIUM]'])]
                        if low:
                            st.markdown("**🟢 Other Issues:**")
                            for cause in low:
                                st.write(f"• {cause}")

                        # Recommended fixes
                        st.markdown("#### 💡 Recommended Fixes")
                        for fix in info['common_fixes']:
                            st.write(f"✓ {fix}")
                    else:
                        st.success(f"✅ No {category.replace('_', ' ')} issues detected - this area is healthy!")
        else:
            st.success("✅ No root causes identified - test suite appears to be in good health!")

        # Pattern Detection - Enhanced with specific insights
        st.markdown("### 🔮 Pattern Detection & Anomalies - Trend Analysis")
        patterns = insights.get('patterns', {})

        # Summary of patterns
        total_patterns = sum(len(p) for p in patterns.values() if isinstance(p, list))

        if total_patterns > 0:
            st.info(f"📊 Detected {total_patterns} patterns across temporal, recurring, and anomaly categories")

        if patterns:
            # Temporal Patterns
            st.markdown("#### ⏰ Temporal Patterns - Time-Based Analysis")
            temporal = patterns.get('temporal', [])
            if temporal:
                st.markdown("""
                **What are Temporal Patterns?** Time-based trends showing when tests fail more frequently.
                These help identify environment or load issues tied to specific times or days.
                """)

                for idx, pattern in enumerate(temporal, 1):
                    st.info(f"📅 **Pattern {idx}:** {pattern}")

                with st.expander("ℹ️ Understanding Temporal Patterns"):
                    st.markdown("""
                    ### Why Temporal Patterns Matter:
                    
                    **Weekend vs Weekday Differences:**
                    - Higher weekend failure rates → Environment reset/data refresh issues
                    - Higher weekday failure rates → System load or concurrent execution problems
                    
                    **Business Hours vs Off-Hours:**
                    - Different pass rates → Shared resource contention
                    - Consistent differences → Environment configuration issues
                    
                    **What to Do:**
                    1. Investigate environment state during problem periods
                    2. Check for scheduled jobs or maintenance windows
                    3. Review test data setup and cleanup
                    4. Monitor resource utilization
                    """)
            else:
                st.success("✅ No temporal patterns detected - tests perform consistently across all time periods")

            st.markdown("---")

            # Recurring Issues
            st.markdown("#### 🔁 Recurring Issues - Systematic Problems")
            recurring = patterns.get('recurring', [])
            if recurring:
                st.markdown("""
                **What are Recurring Issues?** Tests failing consistently across multiple consecutive builds.
                These indicate systematic problems requiring immediate investigation.
                """)

                for idx, pattern in enumerate(recurring, 1):
                    # Extract build numbers if present
                    if 'builds' in pattern.lower():
                        st.error(f"🔄 **Issue {idx}:** {pattern}")
                        st.markdown("   **Action Required:** This is a chronic issue - investigate immediately")
                    else:
                        st.warning(f"🔄 **Issue {idx}:** {pattern}")

                with st.expander("ℹ️ Understanding Recurring Issues"):
                    st.markdown("""
                    ### Why Recurring Issues Matter:
                    
                    **Consecutive Failures (3+ builds):**
                    - Indicates broken test or broken functionality
                    - Not random - systematic problem
                    - Blocks release pipeline
                    
                    **Impact:**
                    - Wastes CI/CD resources
                    - Blocks deployments
                    - Erodes team confidence
                    - Creates technical debt
                    
                    **What to Do:**
                    1. Investigate root cause immediately
                    2. Fix or disable the failing test
                    3. Check for related application changes
                    4. Update test if functionality changed
                    5. Document the fix
                    """)
            else:
                st.success("✅ No recurring issues detected - failures appear to be isolated incidents")

            st.markdown("---")

            # Anomalies
            st.markdown("#### ⚠️ Anomalies - Statistical Outliers")
            anomalies = patterns.get('anomalies', [])
            if anomalies:
                st.markdown("""
                **What are Anomalies?** Builds with unusually high or low pass rates (>2 standard deviations from mean).
                These indicate special conditions or one-time events requiring investigation.
                """)

                for idx, anomaly in enumerate(anomalies, 1):
                    st.error(f"🎯 **Anomaly {idx}:** {anomaly}")

                st.markdown("**Analysis Method:** Statistical Z-score analysis (threshold: 2σ)")

                with st.expander("ℹ️ Understanding Anomalies"):
                    st.markdown("""
                    ### Why Anomalies Matter:
                    
                    **Unusually Low Pass Rate:**
                    - Environment issue during that build
                    - Bad code commit
                    - Infrastructure problem
                    - Data corruption
                    
                    **Unusually High Pass Rate:**
                    - Tests not running properly
                    - Missing test execution
                    - Configuration change
                    - False positives
                    
                    **Statistical Detection:**
                    ```
                    Z-score = (pass_rate - mean) / standard_deviation
                    Anomaly if |Z-score| > 2 (95% confidence)
                    ```
                    
                    **What to Do:**
                    1. Review build logs for that specific build
                    2. Check for environmental changes
                    3. Compare with adjacent builds
                    4. Investigate any code or config changes
                    5. Document findings
                    """)
            else:
                st.success("✅ No statistical anomalies detected - pass rates are consistent and predictable")

            # Correlation insights
            correlation = patterns.get('correlation', [])
            if correlation:
                st.markdown("---")
                st.markdown("#### 🔗 Correlation Analysis")
                st.markdown("**Correlated Patterns:** Relationships between different failure types")
                for corr in correlation:
                    st.info(f"🔗 {corr}")
        else:
            st.success("✅ No patterns detected - test execution is stable and consistent")

        # Self-Healing Insights Section - Analyze Current Test Failures
        st.markdown("---")
        st.markdown("### 🔧 AI-Powered Self-Healing Insights")

        # Extract failure information from current analysis
        failed_tests_list = analysis.get('most_failed_tests', [])
        flaky_tests_list = analysis.get('flaky_tests', [])

        if failed_tests_list or flaky_tests_list:
            # Show progress
            total_to_analyze = len(failed_tests_list) + len(flaky_tests_list)
            st.info(f"🔍 Analyzing **{len(failed_tests_list)} failed tests** and **{len(flaky_tests_list)} flaky tests** for healing opportunities...")

            # Analyze the current test failures in real-time
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text(f"🤖 Categorizing failures and generating healing suggestions for {total_to_analyze} tests...")

            # Generate healing insights for ALL tests
            healing_insights = analyzer._generate_healing_insights(
                failed_tests_list,  # ALL failed tests
                flaky_tests_list,   # ALL flaky tests
                analysis
            )

            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

            # Calculate accurate statistics
            failed_suggestions = healing_insights.get('failed_test_suggestions', [])
            flaky_suggestions = healing_insights.get('flaky_test_suggestions', [])
            all_suggestions = failed_suggestions + flaky_suggestions

            auto_healable_count = len([s for s in failed_suggestions if s.get('auto_healable', False)])
            high_confidence_count = len([s for s in all_suggestions if s.get('confidence', 0) > 0.7])
            critical_count = len([t for t in failed_tests_list if t.get('failure_rate', 0) > 70])

            # Display accurate metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Tests Analyzed", len(all_suggestions),
                         help=f"{len(failed_suggestions)} failed + {len(flaky_suggestions)} flaky")
            with col2:
                st.metric("Auto-Healable", auto_healable_count,
                         help="Tests that can be automatically fixed with high confidence")
            with col3:
                st.metric("High Confidence", high_confidence_count,
                         delta=f"{(high_confidence_count/max(1,len(all_suggestions))*100):.0f}%",
                         delta_color="normal",
                         help="Suggestions with >70% confidence")
            with col4:
                st.metric("Critical Failures", critical_count,
                         delta="Urgent" if critical_count > 0 else "None",
                         delta_color="inverse" if critical_count > 0 else "normal",
                         help="Tests failing >70% - need immediate attention")

            # Show healing suggestions for failed tests
            if healing_insights.get('failed_test_suggestions'):
                st.markdown("#### 💡 AI Healing Suggestions for Failed Tests")

                # Add filtering options
                col_filter1, col_filter2 = st.columns([1, 3])
                with col_filter1:
                    confidence_filter = st.selectbox(
                        "Filter by confidence:",
                        ["All", "High (>70%)", "Medium (50-70%)", "Low (<50%)"],
                        key="failed_confidence_filter"
                    )
                with col_filter2:
                    failure_type_filter = st.multiselect(
                        "Filter by failure type:",
                        ["LOCATOR_ISSUE", "TIMEOUT", "STALE_ELEMENT", "ASSERTION_FAILURE", "NETWORK_ERROR", "OTHER"],
                        default=[],
                        key="failure_type_filter"
                    )

                # Filter suggestions
                filtered_suggestions = healing_insights['failed_test_suggestions']

                if confidence_filter == "High (>70%)":
                    filtered_suggestions = [s for s in filtered_suggestions if s.get('confidence', 0) > 0.7]
                elif confidence_filter == "Medium (50-70%)":
                    filtered_suggestions = [s for s in filtered_suggestions if 0.5 <= s.get('confidence', 0) <= 0.7]
                elif confidence_filter == "Low (<50%)":
                    filtered_suggestions = [s for s in filtered_suggestions if s.get('confidence', 0) < 0.5]

                if failure_type_filter:
                    filtered_suggestions = [s for s in filtered_suggestions if s.get('failure_type', '') in failure_type_filter]

                if not filtered_suggestions:
                    st.info(f"No tests match the selected filters. Showing all {len(healing_insights['failed_test_suggestions'])} suggestions.")
                    filtered_suggestions = healing_insights['failed_test_suggestions']
                else:
                    st.info(f"Showing {len(filtered_suggestions)} of {len(healing_insights['failed_test_suggestions'])} suggestions")

                # Group by confidence for better organization
                high_conf = [s for s in filtered_suggestions if s.get('confidence', 0) > 0.7]
                med_conf = [s for s in filtered_suggestions if 0.5 <= s.get('confidence', 0) <= 0.7]
                low_conf = [s for s in filtered_suggestions if s.get('confidence', 0) < 0.5]

                # Display high confidence first
                if high_conf:
                    st.markdown("##### 🟢 High Confidence Suggestions (>70%)")
                    for suggestion in high_conf:
                        test_name = suggestion.get('test_name', 'Unknown')
                        confidence = suggestion.get('confidence', 0)

                        with st.expander(f"🟢 {test_name} - Confidence: {confidence:.0%}", expanded=False):
                            col_a, col_b = st.columns([2, 1])

                            with col_a:
                                st.markdown(f"**🔍 Root Cause:**")
                                st.write(suggestion.get('root_cause', 'Unknown'))

                                st.markdown(f"**💊 Recommended Fix:**")
                                st.write(suggestion.get('recommendation', 'No specific recommendation'))

                                if suggestion.get('alternative_locators'):
                                    st.markdown(f"**🎯 Alternative Locators:**")
                                    for loc in suggestion['alternative_locators'][:3]:
                                        st.code(loc, language="robot")

                                if suggestion.get('code_example'):
                                    st.markdown(f"**📝 Code Example:**")
                                    st.code(suggestion['code_example'], language="robot")

                            with col_b:
                                st.markdown(f"**📊 Details:**")
                                st.write(f"Failure Rate: {suggestion.get('failure_rate', 0):.1f}%")
                                st.write(f"Failure Type: {suggestion.get('failure_type', 'Unknown')}")
                                st.write(f"Auto-healable: {'✅ Yes' if suggestion.get('auto_healable') else '❌ No'}")

                # Medium confidence
                if med_conf:
                    st.markdown("##### 🟡 Medium Confidence Suggestions (50-70%)")
                    for suggestion in med_conf:
                        test_name = suggestion.get('test_name', 'Unknown')
                        confidence = suggestion.get('confidence', 0)

                        with st.expander(f"🟡 {test_name} - Confidence: {confidence:.0%}", expanded=False):
                            col_a, col_b = st.columns([2, 1])

                            with col_a:
                                st.markdown(f"**🔍 Root Cause:**")
                                st.write(suggestion.get('root_cause', 'Unknown'))

                                st.markdown(f"**💊 Recommended Fix:**")
                                st.write(suggestion.get('recommendation', 'No specific recommendation'))

                                if suggestion.get('alternative_locators'):
                                    st.markdown(f"**🎯 Alternative Locators:**")
                                    for loc in suggestion['alternative_locators'][:3]:
                                        st.code(loc, language="robot")

                                if suggestion.get('code_example'):
                                    st.markdown(f"**📝 Code Example:**")
                                    st.code(suggestion['code_example'], language="robot")

                            with col_b:
                                st.markdown(f"**📊 Details:**")
                                st.write(f"Failure Rate: {suggestion.get('failure_rate', 0):.1f}%")
                                st.write(f"Failure Type: {suggestion.get('failure_type', 'Unknown')}")
                                st.write(f"Auto-healable: {'✅ Yes' if suggestion.get('auto_healable') else '❌ No'}")

                # Low confidence
                if low_conf:
                    st.markdown("##### 🟠 Lower Confidence Suggestions (<50%)")
                    st.caption("These suggestions require manual investigation to confirm root cause")
                    for suggestion in low_conf:
                            test_name = suggestion.get('test_name', 'Unknown')
                            confidence = suggestion.get('confidence', 0)

                            with st.expander(f"🟠 {test_name} - Confidence: {confidence:.0%}", expanded=False):
                                col_a, col_b = st.columns([2, 1])

                                with col_a:
                                    st.markdown(f"**🔍 Root Cause:**")
                                    st.write(suggestion.get('root_cause', 'Unknown'))

                                    st.markdown(f"**💊 Recommended Fix:**")
                                    st.write(suggestion.get('recommendation', 'No specific recommendation'))

                                    if suggestion.get('alternative_locators'):
                                        st.markdown(f"**🎯 Alternative Locators:**")
                                        for loc in suggestion['alternative_locators'][:3]:
                                            st.code(loc, language="robot")

                                    if suggestion.get('code_example'):
                                        st.markdown(f"**📝 Code Example:**")
                                        st.code(suggestion['code_example'], language="robot")

                                with col_b:
                                    st.markdown(f"**📊 Details:**")
                                    st.write(f"Failure Rate: {suggestion.get('failure_rate', 0):.1f}%")
                                    st.write(f"Failure Type: {suggestion.get('failure_type', 'Unknown')}")
                                    st.write(f"Auto-healable: {'✅ Yes' if suggestion.get('auto_healable') else '❌ No'}")

            # Show healing suggestions for flaky tests
            if healing_insights.get('flaky_test_suggestions'):
                st.markdown("#### 🔄 AI Healing Suggestions for Flaky Tests")

                for suggestion in healing_insights['flaky_test_suggestions']:
                    test_name = suggestion.get('test_name', 'Unknown')
                    flakiness = suggestion.get('flakiness_score', 0)

                    with st.expander(f"⚠️ {test_name} - Flakiness: {flakiness:.1f}%", expanded=False):
                        st.markdown(f"**🔍 Why It's Flaky:**")
                        st.write(suggestion.get('root_cause', 'Unknown'))

                        st.markdown(f"**💊 Stabilization Strategy:**")
                        st.write(suggestion.get('recommendation', 'No specific recommendation'))

                        st.markdown(f"**⏱️ Wait Strategy:**")
                        st.code(suggestion.get('wait_strategy', 'Use explicit waits'), language="robot")

                        if suggestion.get('pattern'):
                            st.info(f"📈 Pattern: {suggestion['pattern']}")

            # Quick action summary
            st.markdown("#### ⚡ Quick Actions")

            quick_actions = []
            if critical_count > 0:
                quick_actions.append(f"🔴 Fix {critical_count} critical test(s) failing >70%")
            if len(flaky_tests_list) > 0:
                quick_actions.append(f"🔄 Stabilize {len(flaky_tests_list)} flaky test(s)")
            if auto_healable_count > 0:
                quick_actions.append(f"💡 Apply AI suggestions to {auto_healable_count} auto-healable test(s)")

            for action in quick_actions:
                st.write(f"• {action}")

            # Integration tip
            with st.expander("🚀 Advanced: Enable Automated Self-Healing", expanded=False):
                    st.markdown("""
                    For continuous self-healing, integrate the Azure Heal Listener with your CI/CD:
                    
                    **Jenkins Pipeline:**
                    ```groovy
                    stage('Test with Self-Healing') {
                        steps {
                            sh 'python scripts/listeners/run_with_healing.py tests/'
                        }
                    }
                    ```
                    
                    **Local Development:**
                    ```bash
                    python scripts/listeners/run_with_healing.py tests/ --auto-heal
                    ```
                    
                    This will automatically apply high-confidence fixes during test execution.
                    """)
        else:
            st.success("✅ No failed or flaky tests detected - All tests are passing!")
            st.info("💡 Self-healing insights will appear when test failures are detected.")

        # Actionable Items with Priority - Enhanced
        st.markdown("### ⚡ Actionable Recommendations - Prioritized Action Plan")

        immediate = insights.get('immediate_actions', [])
        short_term = insights.get('short_term_recommendations', [])
        long_term = insights.get('long_term_strategy', [])

        # Summary of action items
        total_actions = len(immediate) + len(short_term) + len(long_term)
        critical_actions = len([a for a in immediate if a.get('priority') == 'CRITICAL'])

        if total_actions > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Actions", total_actions)
            with col2:
                st.metric("Immediate (24h)", len(immediate), delta="High Priority")
            with col3:
                st.metric("Short-term (1 week)", len(short_term))
            with col4:
                st.metric("Long-term (1 month)", len(long_term))

        st.markdown("""
        **Action Priority System:**
        - 🔥 **Immediate (24h):** Critical issues affecting releases - start here
        - 📅 **Short-term (1 week):** Important improvements for stability
        - 🎯 **Long-term (1 month):** Strategic enhancements for excellence
        """)

        # Create tabs for different timeframes
        action_tabs = st.tabs(["🔥 Immediate (24h)", "📅 Short-term (1 week)", "🎯 Long-term (1 month)"])

        with action_tabs[0]:
            if immediate:
                for idx, action in enumerate(immediate, 1):
                    with st.container():
                        st.markdown(f"#### {idx}. {action.get('action', 'Unknown Action')}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            priority = action.get('priority', 'UNKNOWN')
                            priority_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                            st.markdown(f"**Priority:** {priority_color.get(priority, '⚪')} {priority}")
                        with col2:
                            st.markdown(f"**Effort:** {action.get('effort', 'Unknown')}")
                        with col3:
                            st.markdown(f"**Impact:** {action.get('estimated_impact', 'Unknown')}")

                        st.markdown(f"**Reason:** {action.get('reason', 'Not specified')}")

                        if action.get('tests'):
                            with st.expander("📝 Affected Tests"):
                                for test in action.get('tests', []):
                                    st.write(f"• {test}")
                        if action.get('details'):
                            with st.expander("ℹ️ Additional Details"):
                                for detail in action.get('details', []):
                                    st.write(f"• {detail}")

                        st.markdown("---")
            else:
                st.success("✅ No immediate actions required!")

        with action_tabs[1]:
            if short_term:
                for idx, action in enumerate(short_term, 1):
                    with st.container():
                        st.markdown(f"#### {idx}. {action.get('action', 'Unknown Action')}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            priority = action.get('priority', 'UNKNOWN')
                            priority_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                            st.markdown(f"**Priority:** {priority_color.get(priority, '⚪')} {priority}")
                        with col2:
                            st.markdown(f"**Effort:** {action.get('effort', 'Unknown')}")
                        with col3:
                            st.markdown(f"**Impact:** {action.get('estimated_impact', 'Unknown')}")

                        st.markdown(f"**Reason:** {action.get('reason', 'Not specified')}")

                        if action.get('tests'):
                            with st.expander("📝 Affected Tests"):
                                for test in action.get('tests', []):
                                    st.write(f"• {test}")
                        if action.get('details'):
                            with st.expander("ℹ️ Additional Details"):
                                for detail in action.get('details', []):
                                    st.write(f"• {detail}")

                        st.markdown("---")
            else:
                st.info("No short-term recommendations at this time")

        with action_tabs[2]:
            if long_term:
                for idx, action in enumerate(long_term, 1):
                    with st.container():
                        st.markdown(f"#### {idx}. {action.get('action', 'Unknown Action')}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            priority = action.get('priority', 'UNKNOWN')
                            priority_color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                            st.markdown(f"**Priority:** {priority_color.get(priority, '⚪')} {priority}")
                        with col2:
                            st.markdown(f"**Effort:** {action.get('effort', 'Unknown')}")
                        with col3:
                            st.markdown(f"**Impact:** {action.get('estimated_impact', 'Unknown')}")

                        st.markdown(f"**Reason:** {action.get('reason', 'Not specified')}")

                        if action.get('details'):
                            with st.expander("ℹ️ Additional Details"):
                                for detail in action.get('details', []):
                                    st.write(f"• {detail}")

                        st.markdown("---")
            else:
                st.info("No long-term strategic recommendations at this time")

        # Quick Wins Section
        st.markdown("### 🎯 Quick Wins (High ROI, Low Effort)")
        quick_wins = insights.get('quick_wins', [])

        # Defensive check: ensure quick_wins is a list
        if isinstance(quick_wins, str):
            logger.warning(f"quick_wins is a string, not a list. Value: {quick_wins}")
            quick_wins = []
        elif not isinstance(quick_wins, list):
            logger.warning(f"quick_wins has unexpected type: {type(quick_wins)}")
            quick_wins = []

        if quick_wins:
            for idx, win in enumerate(quick_wins, 1):
                # Defensive check: ensure win is a dict
                if not isinstance(win, dict):
                    logger.warning(f"Quick win item is not a dict: {type(win)}")
                    continue

                with st.container():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{idx}. {win.get('action', 'Unknown')}**")
                    with col2:
                        st.markdown(f"*ROI: {win.get('roi', 'Unknown')}*")

                    st.write(f"**Why:** {win.get('reason', 'Not specified')}")
                    st.write(f"**Potential Saving:** {win.get('potential_saving', 'Unknown')}")
                    st.write(f"**Effort Required:** {win.get('effort', 'Unknown')}")

                    if win.get('tests'):
                        with st.expander("📝 Target Tests"):
                            for test in win.get('tests', []):
                                st.write(f"• {test}")

                    st.markdown("---")
        else:
            st.info("No quick wins identified - test suite may already be optimized")

        # Success Metrics & Targets
        st.markdown("### 🎯 Success Metrics & Improvement Targets")
        success_metrics = insights.get('success_metrics', {})

        if success_metrics:
            st.markdown("""
            **How to use these metrics:**
            - 🎯 **Current**: Your starting point - where you are today
            - 🚀 **Target**: Your goal - where you want to be
            - 📅 **Timeline**: Realistic timeframe to achieve the target
            - 📊 **How to Measure**: Specific formula and method to track progress
            """)

            metrics_data = []
            for metric_name, metric_data in success_metrics.items():
                if isinstance(metric_data, dict):
                    metric_row = {
                        'Metric': metric_name.replace('_', ' ').title(),
                        'Current': metric_data.get('current', 'N/A'),
                        'Target': metric_data.get('target', 'N/A'),
                        'Timeline': metric_data.get('timeline', 'N/A'),
                        'How to Measure': metric_data.get('how_to_measure', 'N/A')
                    }
                    metrics_data.append(metric_row)

                    # Show intermediate milestone if available
                    if metric_data.get('intermediate_milestone'):
                        st.info(f"**{metric_name.replace('_', ' ').title()}** - Intermediate Milestone: {metric_data['intermediate_milestone']}")

            if metrics_data:
                df_metrics_targets = pd.DataFrame(metrics_data)
                st.dataframe(df_metrics_targets, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.markdown("**💡 Pro Tip:** Review these metrics weekly and adjust action plans based on progress. "
                          "Focus on metrics showing the least improvement first.")

        # Predicted Impact Timeline
        predicted_impact = insights.get('predicted_impact', {})
        if predicted_impact and isinstance(predicted_impact, dict):
            st.markdown("### ⚠️ Predicted Impact Timeline (If Issues Not Fixed)")

            timeline_items = []
            for period, impact in predicted_impact.items():
                if period not in ['error', 'message']:
                    timeline_items.append((period, impact))

            if timeline_items:
                for period, impact in timeline_items:
                    st.error(f"**{period}:** {impact}")
            else:
                st.warning("Impact prediction not available")

        # KPI Monitoring Dashboard
        st.markdown("### 📊 Key Performance Indicators (KPIs)")

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

        with kpi_col1:
            pass_rate = analysis.get('average_pass_rate', 0)
            pass_rate_target = 95.0
            pass_rate_delta = pass_rate - pass_rate_target
            st.metric(
                "Pass Rate Target",
                f"{pass_rate:.1f}%",
                delta=f"{pass_rate_delta:+.1f}%",
                delta_color="normal" if pass_rate >= pass_rate_target else "inverse"
            )

        with kpi_col2:
            stability = analysis.get('stability_score', 0)
            stability_target = 90.0
            stability_delta = stability - stability_target
            st.metric(
                "Stability Target",
                f"{stability:.1f}/100",
                delta=f"{stability_delta:+.1f}",
                delta_color="normal" if stability >= stability_target else "inverse"
            )

        with kpi_col3:
            flaky_count = len(analysis.get('flaky_tests', []))
            st.metric(
                "Flaky Tests Target",
                flaky_count,
                delta=f"{-flaky_count if flaky_count > 0 else 0}",
                delta_color="inverse" if flaky_count > 0 else "normal",
                help="Target: 0 flaky tests"
            )

        with kpi_col4:
            avg_exec = analysis.get('average_execution_time', 0) / 1000
            exec_trend = analysis.get('execution_time_trend', 'stable')
            trend_symbol = "↓" if exec_trend == 'improving' else "↑" if exec_trend == 'degrading' else "→"
            st.metric(
                "Execution Time",
                f"{avg_exec:.1f}s",
                delta=f"{trend_symbol} {exec_trend}",
                delta_color="normal" if exec_trend != 'degrading' else "inverse"
            )

    # Actions & Export Section
    st.markdown("---")
    st.subheader("⚡ Actions & Export")

    # Create tabs for different action categories
    action_export_tabs = st.tabs(["📥 Export", "🔄 Re-trigger Tests", "🎫 JIRA Integration", "📧 Notifications"])

    # Export Tab
    with action_export_tabs[0]:
        st.markdown("### Export Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📊 CSV Export")
            st.write("Export test metrics as CSV for further analysis in Excel or other tools")
            if st.button("Generate CSV", key="export_csv"):
                csv = df_metrics.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"rf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_csv"
                )
                st.success("✅ CSV generated successfully!")

        with col2:
            st.markdown("#### 📄 JSON Report")
            st.write("Export complete analysis including AI insights as JSON")
            if st.button("Generate Report", key="export_json"):
                report = create_analysis_report(analysis, st.session_state.get('rf_insights', {}))
                st.download_button(
                    label="📥 Download JSON Report",
                    data=report,
                    file_name=f"rf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
                st.success("✅ Report generated successfully!")

        with col3:
            st.markdown("#### 📝 PDF Summary")
            if PDF_AVAILABLE:
                st.write("Export comprehensive analysis report as PDF")

                if st.button("Generate PDF", key="export_pdf"):
                    with st.spinner("Generating PDF report..."):
                        try:
                            job_name = st.session_state.get('rf_selected_job', 'Robot Framework Job')
                            insights_data = st.session_state.get('rf_insights', {})

                            # Validate data before generating PDF
                            if not analysis:
                                st.error("❌ No analysis data available. Please run analysis first.")
                            elif not insights_data:
                                st.warning("⚠️ No AI insights available. Generating PDF with basic data only...")
                                # Create minimal insights structure
                                insights_data = {
                                    'executive_summary': 'Analysis completed. Detailed insights not yet generated.',
                                    'quality_score': {'grade': 'N/A', 'overall_score': 0},
                                    'critical_issues': [],
                                    'root_causes': [],
                                    'patterns': [],
                                    'recommendations': [],
                                    'quick_wins': []
                                }

                            logger.info(f"Generating PDF for job: {job_name}")
                            logger.info(f"Analysis has {len(analysis)} keys")
                            logger.info(f"Insights has {len(insights_data)} keys")

                            pdf_buffer = generate_pdf_report(analysis, insights_data, job_name)

                            if pdf_buffer:
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_buffer,
                                    file_name=f"rf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    key="download_pdf"
                                )
                                st.success("✅ PDF report generated successfully!")
                            else:
                                st.error("❌ Failed to generate PDF report")
                                st.info("💡 Check the console/terminal logs for detailed error information")
                                with st.expander("🔍 Troubleshooting Tips"):
                                    st.markdown("""
                                    **Common causes:**
                                    1. Missing insights data - Try regenerating AI insights first
                                    2. Invalid data structure - Check logs for type warnings
                                    3. Text encoding issues - Special characters in test names
                                    
                                    **Solutions:**
                                    - Click "🔄 Regenerate Insights" button
                                    - Check console logs for specific warnings
                                    - Try analyzing fewer builds (e.g., 20 instead of 50)
                                    """)

                        except Exception as e:
                            st.error(f"❌ Error generating PDF: {str(e)}")
                            logger.error(f"PDF generation error: {e}")
                            import traceback
                            error_trace = traceback.format_exc()
                            logger.error(error_trace)
                            with st.expander("🔍 Error Details", expanded=True):
                                st.code(error_trace, language="python")
                                st.markdown("""
                                **What to do:**
                                1. Copy the error above
                                2. Check if reportlab is installed: `pip install reportlab`
                                3. Try regenerating AI insights
                                4. If issue persists, export as JSON instead
                                """)
            else:
                st.write("PDF export requires reportlab library")
                st.info("Install with: pip install reportlab")
                if st.button("Generate PDF", key="export_pdf", disabled=True):
                    st.warning("PDF generation not available. Please install reportlab.")

    # Re-trigger Tests Tab
    with action_export_tabs[1]:
        st.markdown("### 🔄 Re-trigger Test Execution")
        st.info("Re-run tests in Jenkins for failed or flaky tests")

        # Get Jenkins client from session
        jenkins_url = st.session_state.get('rf_jenkins_url', '')
        username = st.session_state.get('rf_jenkins_username', '')
        credential = st.session_state.get('rf_jenkins_token', '')
        selected_job = st.session_state.get('rf_selected_job', '')

        if jenkins_url and username and credential and selected_job:
            st.markdown(f"**Current Job:** {selected_job}")

            # Re-trigger options
            trigger_option = st.radio(
                "Select what to re-trigger:",
                ["Full Test Suite", "Failed Tests Only", "Flaky Tests Only", "Specific Tests"],
                help="Choose which tests to re-run"
            )

            if trigger_option == "Specific Tests":
                # Let user select specific tests
                all_test_names = []
                if analysis.get('most_failed_tests'):
                    all_test_names.extend([t['test'] for t in analysis['most_failed_tests'][:10]])
                if analysis.get('flaky_tests'):
                    all_test_names.extend([t['test'] for t in analysis['flaky_tests'][:10]])

                selected_tests = st.multiselect(
                    "Select tests to re-trigger:",
                    options=list(set(all_test_names)),
                    help="Choose specific tests to re-run"
                )

            # Trigger parameters
            with st.expander("⚙️ Advanced Options"):
                trigger_params = st.text_area(
                    "Build Parameters (optional)",
                    placeholder="KEY1=value1\nKEY2=value2",
                    help="Enter build parameters, one per line"
                )

                priority = st.select_slider(
                    "Priority",
                    options=["Low", "Normal", "High"],
                    value="Normal",
                    help="Set build queue priority"
                )

            # Trigger button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🚀 Trigger Build", type="primary", key="trigger_build"):
                    with st.spinner("Triggering Jenkins build..."):
                        try:
                            # Parse parameters
                            params = {}
                            if trigger_params:
                                for line in trigger_params.split('\n'):
                                    if '=' in line:
                                        key, value = line.split('=', 1)
                                        params[key.strip()] = value.strip()

                            # Add test selection parameter if applicable
                            if trigger_option == "Failed Tests Only":
                                params['TEST_FILTER'] = 'failed'
                            elif trigger_option == "Flaky Tests Only":
                                params['TEST_FILTER'] = 'flaky'
                            elif trigger_option == "Specific Tests" and 'selected_tests' in locals():
                                params['TEST_NAMES'] = ','.join(selected_tests)

                            # Trigger the build (using existing client)
                            rf_client = RobotFrameworkDashboardClient(jenkins_url, username, credential)

                            # Build trigger URL
                            job_info = [j for j in rf_client.get_jobs() if j.get('display_name') == selected_job]
                            if job_info:
                                job_url = job_info[0].get('url', '')
                                trigger_url = f"{job_url}/buildWithParameters" if params else f"{job_url}/build"

                                response = rf_client.session.post(trigger_url, data=params, timeout=30)

                                if response.status_code in [200, 201]:
                                    st.success(f"✅ Build triggered successfully for {selected_job}!")
                                    st.info(f"**Trigger Type:** {trigger_option}")
                                    if params:
                                        st.info(f"**Parameters:** {params}")
                                    st.markdown("Check Jenkins for build progress.")
                                else:
                                    st.error(f"❌ Failed to trigger build. Status: {response.status_code}")
                                    st.error(f"Response: {response.text[:200]}")
                        except Exception as e:
                            st.error(f"❌ Error triggering build: {str(e)}")
                            st.info("Make sure you have permissions to trigger builds in Jenkins")

            with col2:
                st.caption("This will queue a new build in Jenkins with the selected options")
        else:
            st.warning("⚠️ Please configure Jenkins credentials and select a job first")
            st.info("Go back to Configuration section to set up Jenkins connection")

    # JIRA Integration Tab
    with action_export_tabs[2]:
        st.markdown("### 🎫 JIRA Issue Creation")
        st.info("Automatically create JIRA issues for failed or flaky tests")

        # JIRA Configuration
        with st.expander("⚙️ JIRA Configuration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                # Hardcoded JIRA URL
                jira_url = "https://newfold.atlassian.net/"
                st.info(f"🔗 JIRA URL: {jira_url}")

                jira_email = st.text_input(
                    "JIRA Email",
                    value=st.session_state.get('jira_email', ''),
                    placeholder="your-email@company.com",
                    key="jira_email_input"
                )
            with col2:
                jira_token = st.text_input(
                    "JIRA API Token",
                    type="password",
                    value=st.session_state.get('jira_token', ''),
                    help="Generate from JIRA Account Settings > Security > API Tokens",
                    key="jira_token_input"
                )
                jira_project = st.text_input(
                    "Project Key",
                    value=st.session_state.get('jira_project', ''),
                    placeholder="PROJ",
                    help="JIRA project key (e.g., PROJ, TEST, QA)",
                    key="jira_project_input"
                )

            if st.button("💾 Save JIRA Config", key="save_jira"):
                st.session_state.jira_url = jira_url
                st.session_state.jira_email = jira_email
                st.session_state.jira_token = jira_token
                st.session_state.jira_project = jira_project
                st.success("✅ JIRA configuration saved!")

        # Issue Creation
        if jira_url and jira_email and jira_token and jira_project:
            st.markdown("### Create Issues")

            issue_type = st.selectbox(
                "Select tests to create issues for:",
                ["Chronic Failures (>70%)", "All Failed Tests", "Flaky Tests (>40%)", "All Flaky Tests", "Custom Selection"],
                key="jira_issue_type"
            )

            # Get relevant tests based on selection
            if issue_type == "Chronic Failures (>70%)":
                relevant_tests = [t for t in analysis.get('most_failed_tests', []) if t.get('failure_rate', 0) > 70]
            elif issue_type == "All Failed Tests":
                relevant_tests = analysis.get('most_failed_tests', [])
            elif issue_type == "Flaky Tests (>40%)":
                relevant_tests = [t for t in analysis.get('flaky_tests', []) if t.get('flakiness_score', 0) > 40]
            elif issue_type == "All Flaky Tests":
                relevant_tests = analysis.get('flaky_tests', [])
            else:  # Custom Selection
                all_problem_tests = []
                if analysis.get('most_failed_tests'):
                    all_problem_tests.extend([{'type': 'Failed', **t} for t in analysis['most_failed_tests'][:10]])
                if analysis.get('flaky_tests'):
                    all_problem_tests.extend([{'type': 'Flaky', **t} for t in analysis['flaky_tests'][:10]])

                selected_for_jira = st.multiselect(
                    "Select tests:",
                    options=[f"[{t.get('type', 'Test')}] {t['test']}" for t in all_problem_tests],
                    key="jira_custom_tests"
                )
                relevant_tests = all_problem_tests  # Filter based on selection

            if relevant_tests:
                st.info(f"📊 {len(relevant_tests)} test(s) selected for JIRA issue creation")

                # Issue template
                issue_template = st.selectbox(
                    "Issue Type:",
                    ["Bug", "Task", "Story"],
                    key="jira_issue_template"
                )

                priority = st.select_slider(
                    "Priority:",
                    options=["Lowest", "Low", "Medium", "High", "Highest"],
                    value="Medium",
                    key="jira_priority"
                )

                # Create issues button
                if st.button("📝 Create JIRA Issues", type="primary", key="create_jira_issues"):
                    with st.spinner(f"Creating {len(relevant_tests)} JIRA issue(s)..."):
                        try:
                            from jira import JIRA

                            # Connect to JIRA
                            jira = JIRA(server=jira_url, basic_auth=(jira_email, jira_token))

                            created_issues = []
                            failed_issues = []

                            for test in relevant_tests[:10]:  # Limit to 10 to avoid overwhelming JIRA
                                try:
                                    test_name = test.get('test', 'Unknown Test')

                                    # Create issue description
                                    if 'failure_rate' in test:
                                        description = f"""
Test: {test_name}
Failure Rate: {test.get('failure_rate', 0):.1f}%
Failed: {test.get('failure_count', 0)} times
Status: Chronic Failure

Root Cause Analysis Needed:
- Review test logic and assertions
- Check for application changes
- Verify test data and environment
- Consider test maintenance or removal

Error Messages:
{chr(10).join(test.get('sample_messages', ['No error messages available'])[:3])}

Identified by: Robot Framework Dashboard Analytics
Job: {st.session_state.get('rf_selected_job', 'Unknown')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                                    else:  # Flaky test
                                        description = f"""
Test: {test_name}
Flakiness Score: {test.get('flakiness_score', 0):.1f}%
Pass/Fail Pattern: {test.get('passes', 0)} passes / {test.get('fails', 0)} fails
Status: Flaky/Unstable

Investigation Required:
- Add explicit waits for dynamic content
- Check for race conditions
- Review test environment stability
- Verify test data consistency

Identified by: Robot Framework Dashboard Analytics
Job: {st.session_state.get('rf_selected_job', 'Unknown')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

                                    # Create issue
                                    issue_dict = {
                                        'project': {'key': jira_project},
                                        'summary': f"[RF Test] {test_name[:80]}",
                                        'description': description,
                                        'issuetype': {'name': issue_template},
                                        'priority': {'name': priority}
                                    }

                                    new_issue = jira.create_issue(fields=issue_dict)
                                    created_issues.append(new_issue.key)

                                except Exception as e:
                                    failed_issues.append(f"{test_name}: {str(e)}")

                            # Show results
                            if created_issues:
                                st.success(f"✅ Created {len(created_issues)} JIRA issue(s)!")
                                for issue_key in created_issues:
                                    st.write(f"- [{issue_key}]({jira_url}/browse/{issue_key})")

                            if failed_issues:
                                st.warning(f"⚠️ Failed to create {len(failed_issues)} issue(s):")
                                for failure in failed_issues:
                                    st.write(f"- {failure}")

                        except ImportError:
                            st.error("❌ JIRA library not installed. Install with: pip install jira")
                        except Exception as e:
                            st.error(f"❌ Error creating JIRA issues: {str(e)}")
                            st.info("Check your JIRA credentials and permissions")
            else:
                st.info("No tests match the selected criteria")
        else:
            st.warning("⚠️ Please configure JIRA credentials above")

    # Notifications Tab
    with action_export_tabs[3]:
        st.markdown("### 📧 Send Notifications")
        st.info("Send analysis summary via email or notification system")

        notification_type = st.selectbox(
            "Notification Type:",
            ["Summary Only", "Detailed Report", "Critical Issues Only"],
            key="notification_type"
        )

        # Notification content preview
        insights = st.session_state.get('rf_insights', {})
        quality_score = insights.get('quality_score', {})

        st.markdown("#### Preview:")
        preview_content = f"""
**Robot Framework Test Analysis Summary**

Job: {st.session_state.get('rf_selected_job', 'Unknown')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Quality Score:** {quality_score.get('grade', 'N/A')} ({quality_score.get('overall_score', 0):.1f}/100)

**Key Metrics:**
- Pass Rate: {analysis.get('average_pass_rate', 0):.1f}%
- Stability: {analysis.get('stability_score', 0):.1f}/100
- Flaky Tests: {len(analysis.get('flaky_tests', []))}
- Failed Tests: {len(analysis.get('most_failed_tests', []))}

**Executive Summary:**
{insights.get('executive_summary', 'No summary available')}
"""

        if notification_type == "Critical Issues Only":
            critical_issues = insights.get('critical_issues', [])
            preview_content += f"\n\n**Critical Issues ({len(critical_issues)}):**\n"
            for issue in critical_issues[:3]:
                preview_content += f"\n- [{issue.get('severity', 'UNKNOWN')}] {issue.get('issue', 'Unknown')}"

        st.text_area("Notification Content:", preview_content, height=300, key="notification_preview")

        # Recipient selection
        recipients = st.text_input(
            "Recipients (comma-separated emails):",
            placeholder="user1@company.com, user2@company.com",
            key="notification_recipients"
        )

        # Send button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("📧 Send Notification", type="primary", key="send_notification"):
                if NOTIFICATIONS_AVAILABLE:
                    with st.spinner("Sending notification..."):
                        try:
                            # Determine status
                            if quality_score.get('grade', 'F') in ['A', 'B']:
                                status = "success"
                            elif quality_score.get('grade', 'F') == 'C':
                                status = "warning"
                            else:
                                status = "error"

                            notifications.add_notification(
                                module_name="rf_dashboard_analytics",
                                status=status,
                                message=f"RF Dashboard Analysis completed for {st.session_state.get('rf_selected_job', 'job')}",
                                details=preview_content
                            )

                            st.success("✅ Notification sent successfully!")
                            st.info(f"Sent to system notification center. Grade: {quality_score.get('grade', 'N/A')}")

                        except Exception as e:
                            st.error(f"❌ Error sending notification: {str(e)}")
                else:
                    st.error("❌ Notification system not available")
                    st.info("The notification module could not be imported. Check that it's installed and configured.")

        with col2:
            st.caption("Notification will be sent to the configured system and email recipients")


def _sanitize_text_for_pdf(text: str) -> str:
    """Sanitize text for PDF generation - remove problematic characters"""
    if not isinstance(text, str):
        text = str(text)

    # Replace common problematic characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')

    # Remove any non-printable characters except newlines
    text = ''.join(char if char.isprintable() or char == '\n' else ' ' for char in text)

    return text


def generate_pdf_report(analysis: Dict[str, Any], insights: Dict[str, Any],
                        job_name: str = "Robot Framework Job") -> Optional[BytesIO]:
    """Generate comprehensive PDF report with detailed analysis"""

    if not PDF_AVAILABLE:
        logger.error("PDF generation libraries not available. Install with: pip install reportlab")
        return None

    try:
        logger.info("Starting PDF generation...")
        logger.info(f"Analysis keys: {list(analysis.keys())}")
        logger.info(f"Insights keys: {list(insights.keys())}")

        # Validate required data
        if not analysis or len(analysis) == 0:
            logger.error("Analysis data is empty or has no keys")
            return None

        # Handle empty insights - create minimal structure
        if not insights or len(insights) == 0:
            logger.warning("Insights data is empty, creating minimal structure")
            insights = {
                'executive_summary': 'Test analysis completed. Detailed insights not yet generated.',
                'quality_score': {'grade': 'N/A', 'overall_score': 0},
                'critical_issues': [],
                'root_causes': [],
                'patterns': [],
                'recommendations': [],
                'quick_wins': []
            }

        # Ensure required keys exist in insights
        insights.setdefault('executive_summary', 'Analysis completed.')
        insights.setdefault('quality_score', {'grade': 'N/A', 'overall_score': 0})
        insights.setdefault('critical_issues', [])
        insights.setdefault('root_causes', [])
        insights.setdefault('patterns', [])
        insights.setdefault('recommendations', [])

        # Sanitize job name
        job_name = _sanitize_text_for_pdf(job_name)

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)

        # Container for the 'Flowable' objects
        elements = []

        logger.info("Creating PDF styles...")
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6,
            spaceBefore=6
        )
        normal_style = styles['Normal']

        # Title
        try:
            logger.info("Adding title...")
            elements.append(Paragraph(f"Robot Framework Test Analysis Report", title_style))
            elements.append(Paragraph(f"<b>Job:</b> {job_name}", normal_style))
            elements.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            elements.append(Spacer(1, 0.3*inch))
            logger.info("Title added successfully")
        except Exception as e:
            logger.error(f"Error adding title: {e}")
            raise

        # Executive Summary
        try:
            logger.info("Adding executive summary...")
            elements.append(Paragraph("Executive Summary", heading_style))
            quality_score = insights.get('quality_score', {})
            if not isinstance(quality_score, dict):
                quality_score = {'grade': 'N/A', 'overall_score': 0}
            exec_summary = _sanitize_text_for_pdf(insights.get('executive_summary', 'No summary available'))
            logger.info(f"Executive summary length: {len(exec_summary)} chars")
        except Exception as e:
            logger.error(f"Error preparing executive summary: {e}")
            raise

        # Summary table
        try:
            logger.info("Creating summary table...")
            summary_data = [
                ['Metric', 'Value', 'Status'],
                ['Quality Grade', str(quality_score.get('grade', 'N/A')), str(quality_score.get('grade', 'N/A'))],
                ['Overall Score', f"{quality_score.get('overall_score', 0):.1f}/100", ''],
                ['Average Pass Rate', f"{analysis.get('average_pass_rate', 0):.1f}%", ''],
                ['Stability Score', f"{analysis.get('stability_score', 0):.1f}/100", ''],
                ['Total Runs Analyzed', str(analysis.get('total_runs', 0)), ''],
            ]
            logger.info(f"Summary table has {len(summary_data)} rows")

            summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 0.2*inch))
            logger.info("Summary table added successfully")
        except Exception as e:
            logger.error(f"Error creating summary table: {e}")
            raise

        # Executive summary text
        try:
            logger.info("Adding executive summary paragraph...")
            elements.append(Paragraph(exec_summary.replace('\n', '<br/>'), normal_style))
            elements.append(Spacer(1, 0.3*inch))
            logger.info("Executive summary paragraph added successfully")
        except Exception as e:
            logger.error(f"Error adding executive summary paragraph: {e}")
            raise

        # Most Failed Tests
        logger.info("Adding most failed tests section...")
        elements.append(Paragraph("Most Failed Tests", heading_style))
        failed_tests = analysis.get('most_failed_tests', [])

        # Defensive check
        if not isinstance(failed_tests, list):
            logger.warning(f"most_failed_tests is not a list: {type(failed_tests)}")
            failed_tests = []

        if failed_tests:
            for i, test in enumerate(failed_tests[:5], 1):
                # Defensive check for test dict
                if not isinstance(test, dict):
                    logger.warning(f"Failed test item is not a dict: {type(test)}")
                    continue
                elements.append(Paragraph(f"<b>{i}. {test.get('test', 'Unknown')}</b>", subheading_style))

                # Test details table
                test_data = [
                    ['Metric', 'Value'],
                    ['Failure Count', str(test.get('failure_count', 0))],
                    ['Failure Rate', f"{test.get('failure_rate', 0):.1f}%"],
                    ['Common Error Type', test.get('common_error_type', 'Unknown')],
                    ['Failed in Builds', ', '.join(map(str, test.get('failed_in_builds', [])[:5]))],
                ]

                test_table = Table(test_data, colWidths=[2*inch, 3.5*inch])
                test_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(test_table)

                # Sample error messages
                if test.get('sample_messages'):
                    elements.append(Paragraph("<b>Sample Error Messages:</b>", normal_style))
                    for msg in test.get('sample_messages', [])[:2]:
                        if msg:
                            msg_text = msg[:200] + "..." if len(msg) > 200 else msg
                            elements.append(Paragraph(f"• {msg_text}", normal_style))

                elements.append(Spacer(1, 0.15*inch))
        else:
            elements.append(Paragraph("No failed tests found.", normal_style))

        elements.append(Spacer(1, 0.2*inch))

        # Flaky Tests
        elements.append(Paragraph("Flaky Tests Analysis", heading_style))
        flaky_tests = analysis.get('flaky_tests', [])

        if flaky_tests:
            for i, test in enumerate(flaky_tests[:5], 1):
                elements.append(Paragraph(f"<b>{i}. {test.get('test', 'Unknown')}</b>", subheading_style))

                # Flaky test details table
                flaky_data = [
                    ['Metric', 'Value'],
                    ['Flakiness Score', f"{test.get('flakiness_score', 0):.1f}%"],
                    ['Total Runs', str(test.get('total_runs', 0))],
                    ['Passes', str(test.get('passes', 0))],
                    ['Fails', str(test.get('fails', 0))],
                    ['Failure Pattern', test.get('failure_pattern', 'Unknown')],
                    ['Root Cause Hints', ', '.join(test.get('root_cause_hints', ['Unknown']))],
                ]

                flaky_table = Table(flaky_data, colWidths=[2*inch, 3.5*inch])
                flaky_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(flaky_table)
                elements.append(Spacer(1, 0.15*inch))
        else:
            elements.append(Paragraph("No flaky tests detected.", normal_style))

        elements.append(Spacer(1, 0.2*inch))

        # Slowest Tests
        elements.append(Paragraph("Slowest Tests", heading_style))
        slowest_tests = analysis.get('slowest_tests', [])

        if slowest_tests:
            slowest_data = [['Test Name', 'Avg Duration (ms)', 'Trend', 'Runs']]
            for test in slowest_tests[:10]:
                slowest_data.append([
                    test.get('test', 'Unknown')[:40],
                    f"{test.get('avg_duration', 0):.2f}",
                    test.get('trend', 'N/A'),
                    str(test.get('run_count', 0))
                ])

            slowest_table = Table(slowest_data, colWidths=[2.5*inch, 1.3*inch, 1*inch, 0.7*inch])
            slowest_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(slowest_table)
        else:
            elements.append(Paragraph("No performance data available.", normal_style))

        elements.append(PageBreak())

        # Root Cause Analysis
        logger.info("Adding root cause analysis section...")
        elements.append(Paragraph("Root Cause Analysis", heading_style))
        root_causes = insights.get('root_causes', [])

        # Defensive check - handle dict or non-list types
        if isinstance(root_causes, dict):
            logger.warning(f"root_causes is a dict, converting to list of values")
            root_causes = list(root_causes.values())
        elif not isinstance(root_causes, list):
            logger.warning(f"root_causes is not a list: {type(root_causes)}")
            root_causes = []

        if root_causes:
            for i, rc in enumerate(root_causes[:5], 1):
                # Defensive check for rc dict
                if not isinstance(rc, dict):
                    logger.warning(f"Root cause item is not a dict: {type(rc)}")
                    continue

                elements.append(Paragraph(f"<b>{i}. {rc.get('category', 'Unknown')}</b>", subheading_style))
                elements.append(Paragraph(f"<b>Impact:</b> {rc.get('impact', 'N/A')}", normal_style))
                elements.append(Paragraph(f"<b>Description:</b> {rc.get('description', 'N/A')}", normal_style))

                if rc.get('affected_tests'):
                    elements.append(Paragraph(f"<b>Affected Tests:</b>", normal_style))
                    affected = rc.get('affected_tests', [])
                    if isinstance(affected, list):
                        for test in affected[:3]:
                            elements.append(Paragraph(f"• {test}", normal_style))

                elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(Paragraph("No root cause analysis available.", normal_style))

        elements.append(Spacer(1, 0.2*inch))

        # Pattern Detection
        logger.info("Adding pattern detection section...")
        elements.append(Paragraph("Pattern Detection & Anomalies", heading_style))
        patterns = insights.get('patterns', [])

        # Defensive check - handle dict or non-list types
        if isinstance(patterns, dict):
            logger.warning(f"patterns is a dict, converting to list of values")
            patterns = list(patterns.values())
        elif not isinstance(patterns, list):
            logger.warning(f"patterns is not a list: {type(patterns)}")
            patterns = []

        if patterns:
            for pattern in patterns[:5]:
                # Defensive check for pattern dict
                if not isinstance(pattern, dict):
                    logger.warning(f"Pattern item is not a dict: {type(pattern)}")
                    continue

                elements.append(Paragraph(f"<b>Pattern:</b> {pattern.get('pattern', 'Unknown')}", subheading_style))
                elements.append(Paragraph(f"<b>Frequency:</b> {pattern.get('frequency', 'N/A')}", normal_style))
                elements.append(Paragraph(f"<b>Description:</b> {pattern.get('description', 'N/A')}", normal_style))
                elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(Paragraph("No patterns detected.", normal_style))

        elements.append(Spacer(1, 0.2*inch))

        # Recommendations
        logger.info("Adding recommendations section...")
        elements.append(Paragraph("Actionable Recommendations", heading_style))
        recommendations = insights.get('recommendations', [])

        # Defensive check - handle dict or non-list types
        if isinstance(recommendations, dict):
            logger.warning(f"recommendations is a dict, converting to list of values")
            recommendations = list(recommendations.values())
        elif not isinstance(recommendations, list):
            logger.warning(f"recommendations is not a list: {type(recommendations)}")
            recommendations = []

        if recommendations:
            for i, rec in enumerate(recommendations[:10], 1):
                # Defensive check for rec dict
                if not isinstance(rec, dict):
                    logger.warning(f"Recommendation item is not a dict: {type(rec)}")
                    continue

                priority = rec.get('priority', 'MEDIUM')
                color_map = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#3498db'}
                rec_color = color_map.get(priority, '#95a5a6')

                elements.append(Paragraph(
                    f"<b>{i}. [{priority}]</b> {rec.get('recommendation', 'No recommendation')}",
                    normal_style
                ))

                if rec.get('rationale'):
                    elements.append(Paragraph(f"<i>Rationale:</i> {rec.get('rationale', '')}", normal_style))

                elements.append(Spacer(1, 0.1*inch))
        else:
            elements.append(Paragraph("No recommendations available.", normal_style))

        # Build the PDF
        logger.info(f"Building PDF with {len(elements)} elements...")
        try:
            doc.build(elements)
            logger.info("PDF built successfully")
        except Exception as build_error:
            logger.error(f"Error building PDF document: {build_error}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        buffer.seek(0)
        logger.info("PDF generation complete")
        return buffer

    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_analysis_report(analysis: Dict[str, Any], insights: Dict[str, Any]) -> str:
    """Create a comprehensive analysis report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis': analysis,
        'ai_insights': insights,
        'summary': {
            'total_runs': analysis['total_runs'],
            'average_pass_rate': analysis['average_pass_rate'],
            'stability_score': analysis['stability_score'],
            'trends': {
                'pass_rate': analysis['pass_rate_trend'],
                'execution_time': analysis['execution_time_trend']
            }
        }
    }

    return json.dumps(report, indent=2)


# Wrapper function for main_ui.py integration
def show_ui():
    """Main UI entry point called by main_ui.py"""
    show_rf_dashboard_analytics()


# Main entry point
if __name__ == "__main__":
    show_rf_dashboard_analytics()

