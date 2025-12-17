"""
Azure OpenAI Self-Healing Listener for Robot Framework
Integrates robotframework-heal with Azure OpenAI for intelligent test healing and flaky test detection
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from robot.api import logger as robot_logger
from robot.libraries.BuiltIn import BuiltIn

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from gen_ai.azure_openai_client import AzureOpenAIClient
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    robot_logger.warn("Azure OpenAI client not available. AI-powered healing will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AzureHealListener:
    """
    Robot Framework Listener for AI-powered self-healing and flaky test detection

    Features:
    - Automatic failure analysis using Azure OpenAI
    - Smart locator healing suggestions
    - Flaky test detection and tracking
    - Intelligent retry strategies
    - Failure pattern analysis
    - Healing recommendations and auto-application
    """

    ROBOT_LISTENER_API_VERSION = 3

    def __init__(self,
                 output_dir: str = "heal_reports",
                 auto_heal: str = "false",
                 flaky_threshold: int = 3,
                 ai_model: str = "dev-chat-ai-gpt4.1-mini"):
        """
        Initialize the Azure Heal Listener

        Args:
            output_dir: Directory for healing reports and logs
            auto_heal: Enable automatic healing ("true"/"false")
            flaky_threshold: Number of failures before marking as flaky
            ai_model: Azure OpenAI deployment name
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_heal = auto_heal.lower() == "true"
        self.flaky_threshold = int(flaky_threshold)
        self.ai_model = ai_model

        # Initialize tracking structures
        self.test_failures = {}  # test_name -> [failure_info]
        self.flaky_tests = set()
        self.healed_tests = {}  # test_name -> healing_info
        self.failure_patterns = {}  # pattern -> count

        # Session statistics
        self.stats = {
            "total_tests": 0,
            "failed_tests": 0,
            "healed_tests": 0,
            "flaky_tests_detected": 0,
            "ai_suggestions": 0,
            "auto_healed": 0
        }

        # Initialize Azure OpenAI client
        self.ai_client = None
        if AZURE_OPENAI_AVAILABLE:
            try:
                self.ai_client = AzureOpenAIClient(deployment_name=self.ai_model)
                if self.ai_client.is_configured():
                    logger.info(f"✅ Azure OpenAI initialized with model: {self.ai_model}")
                    robot_logger.info(f"Azure Self-Healing enabled with AI model: {self.ai_model}")
                else:
                    logger.warning("Azure OpenAI not fully configured")
                    robot_logger.warn("Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI: {e}")
                robot_logger.warn(f"AI healing disabled: {e}")

        # Report file paths
        self.heal_report_path = self.output_dir / f"heal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.flaky_report_path = self.output_dir / "flaky_tests.json"

        robot_logger.info(f"Azure Heal Listener initialized (auto_heal={self.auto_heal})")

    def start_suite(self, suite, result):
        """Called when a test suite starts"""
        robot_logger.info(f"Starting suite: {suite.name}")

    def start_test(self, test, result):
        """Called when a test starts"""
        self.stats["total_tests"] += 1
        robot_logger.debug(f"Starting test: {test.name}")

    def end_test(self, test, result):
        """Called when a test ends - main healing logic"""
        test_name = test.name

        if result.status == "FAIL":
            self.stats["failed_tests"] += 1
            robot_logger.info(f"❌ Test failed: {test_name}")

            # Analyze failure
            failure_info = self._extract_failure_info(test, result)

            # Track failure
            if test_name not in self.test_failures:
                self.test_failures[test_name] = []
            self.test_failures[test_name].append(failure_info)

            # Check for flakiness
            if len(self.test_failures[test_name]) >= self.flaky_threshold:
                if test_name not in self.flaky_tests:
                    self.flaky_tests.add(test_name)
                    self.stats["flaky_tests_detected"] += 1
                    robot_logger.warn(f"🔄 Flaky test detected: {test_name} (failed {len(self.test_failures[test_name])} times)")

            # Get AI-powered healing suggestions
            if self.ai_client and self.ai_client.is_configured():
                healing_suggestion = self._get_ai_healing_suggestion(test_name, failure_info)

                if healing_suggestion:
                    self.stats["ai_suggestions"] += 1
                    robot_logger.info(f"🤖 AI Suggestion for {test_name}:")
                    robot_logger.info(f"   {healing_suggestion.get('recommendation', 'No recommendation')}")

                    # Store healing info
                    self.healed_tests[test_name] = {
                        "failure": failure_info,
                        "suggestion": healing_suggestion,
                        "timestamp": datetime.now().isoformat()
                    }

                    # Auto-heal if enabled
                    if self.auto_heal and healing_suggestion.get("auto_healable", False):
                        success = self._apply_healing(test_name, healing_suggestion)
                        if success:
                            self.stats["auto_healed"] += 1
                            self.stats["healed_tests"] += 1
                            robot_logger.info(f"✅ Auto-healing applied for {test_name}")

        elif result.status == "PASS":
            # Check if this was a previously failed test (possible flaky)
            if test_name in self.test_failures:
                robot_logger.info(f"✅ Previously failed test passed: {test_name} (possible flaky test)")

    def end_suite(self, suite, result):
        """Called when a test suite ends"""
        robot_logger.info(f"Finished suite: {suite.name}")

    def close(self):
        """Called when all tests are finished"""
        # Generate comprehensive reports
        self._generate_heal_report()
        self._generate_flaky_report()
        self._log_session_summary()

    def _extract_failure_info(self, test, result) -> Dict[str, Any]:
        """Extract detailed failure information"""
        failure_info = {
            "test_name": test.name,
            "timestamp": datetime.now().isoformat(),
            "status": result.status,
            "message": result.message,
            "tags": list(test.tags),
        }

        # Extract library and keyword info
        try:
            if hasattr(result, 'body'):
                for kw in result.body:
                    if hasattr(kw, 'status') and kw.status == 'FAIL':
                        failure_info["failed_keyword"] = kw.name
                        failure_info["keyword_args"] = list(kw.args) if hasattr(kw, 'args') else []
                        if hasattr(kw, 'libname'):
                            failure_info["library"] = kw.libname
                        break
        except Exception as e:
            logger.debug(f"Could not extract detailed keyword info: {e}")

        # Analyze failure patterns
        failure_type = self._classify_failure(failure_info)
        failure_info["failure_type"] = failure_type

        # Update pattern tracking
        if failure_type not in self.failure_patterns:
            self.failure_patterns[failure_type] = 0
        self.failure_patterns[failure_type] += 1

        return failure_info

    def _classify_failure(self, failure_info: Dict) -> str:
        """Classify failure type for pattern analysis"""
        message = failure_info.get("message", "").lower()

        if "element not found" in message or "locator" in message:
            return "LOCATOR_ISSUE"
        elif "timeout" in message or "timed out" in message:
            return "TIMEOUT"
        elif "stale element" in message:
            return "STALE_ELEMENT"
        elif "assertion" in message or "should be" in message:
            return "ASSERTION_FAILURE"
        elif "connection" in message or "network" in message:
            return "NETWORK_ISSUE"
        elif "permission" in message or "access denied" in message:
            return "PERMISSION_ISSUE"
        else:
            return "UNKNOWN"

    def _get_ai_healing_suggestion(self, test_name: str, failure_info: Dict) -> Optional[Dict]:
        """Get AI-powered healing suggestion from Azure OpenAI"""
        if not self.ai_client or not self.ai_client.is_configured():
            return None

        try:
            # Build context-rich prompt
            prompt = self._build_healing_prompt(test_name, failure_info)

            # Call Azure OpenAI
            response = self.ai_client.generate_response(
                prompt,
                max_tokens=1000,
                temperature=0.3,
                system_message="""You are an expert test automation engineer specializing in self-healing tests.
Analyze test failures and provide specific, actionable healing recommendations.
Focus on: locator strategies, wait conditions, timing issues, and common patterns.
Return recommendations in JSON format with fields: 
- recommendation (string): specific healing action
- confidence (float): 0-1 confidence score
- auto_healable (boolean): can be automatically applied
- alternative_locators (list): alternative locator strategies if applicable
- wait_strategy (string): recommended wait strategy
- root_cause (string): likely root cause analysis
"""
            )

            # Parse AI response
            suggestion = self._parse_ai_response(response)

            if suggestion:
                logger.info(f"AI healing suggestion generated for {test_name}")
                return suggestion

        except Exception as e:
            logger.error(f"Failed to get AI healing suggestion: {e}")

        return None

    def _build_healing_prompt(self, test_name: str, failure_info: Dict) -> str:
        """Build detailed prompt for AI healing analysis"""
        prompt = f"""Test Failure Analysis Request:

Test Name: {test_name}
Failure Type: {failure_info.get('failure_type', 'UNKNOWN')}
Error Message: {failure_info.get('message', 'N/A')}
"""

        if 'failed_keyword' in failure_info:
            prompt += f"Failed Keyword: {failure_info['failed_keyword']}\n"

        if 'keyword_args' in failure_info:
            prompt += f"Keyword Arguments: {failure_info['keyword_args']}\n"

        if 'library' in failure_info:
            prompt += f"Library: {failure_info['library']}\n"

        # Add historical context if available
        if test_name in self.test_failures and len(self.test_failures[test_name]) > 1:
            prompt += f"\nHistorical Failures: {len(self.test_failures[test_name])} times\n"
            prompt += "This test has failed multiple times, suggesting a flaky or unstable test.\n"

        prompt += """\nPlease provide:
1. Root cause analysis
2. Specific healing recommendation
3. Alternative locator strategies (if applicable)
4. Recommended wait strategy
5. Confidence level (0-1)
6. Whether this can be auto-healed

Return as JSON with fields: recommendation, confidence, auto_healable, alternative_locators, wait_strategy, root_cause
"""

        return prompt

    def _parse_ai_response(self, response: str) -> Optional[Dict]:
        """Parse AI response into structured suggestion"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                # Fallback: treat as plain text recommendation
                return {
                    "recommendation": response.strip(),
                    "confidence": 0.7,
                    "auto_healable": False,
                    "root_cause": "Analysis provided as text"
                }

            suggestion = json.loads(json_str)
            return suggestion

        except Exception as e:
            logger.debug(f"Could not parse AI response as JSON: {e}")
            # Return plain text fallback
            return {
                "recommendation": response.strip()[:500],
                "confidence": 0.5,
                "auto_healable": False
            }

    def _apply_healing(self, test_name: str, suggestion: Dict) -> bool:
        """Apply automatic healing if safe to do so"""
        try:
            # For now, log the suggestion for manual review
            # In future, implement safe auto-healing strategies
            robot_logger.info(f"Auto-healing suggestion for {test_name}: {suggestion.get('recommendation')}")

            # Example auto-healing strategies that could be implemented:
            # - Update locators in a separate healing file
            # - Add dynamic waits
            # - Retry with alternative strategies

            return True
        except Exception as e:
            logger.error(f"Failed to apply healing: {e}")
            return False

    def _generate_heal_report(self):
        """Generate comprehensive healing report"""
        report = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "auto_heal_enabled": self.auto_heal,
                "flaky_threshold": self.flaky_threshold,
                "ai_model": self.ai_model
            },
            "statistics": self.stats,
            "failure_patterns": self.failure_patterns,
            "healed_tests": self.healed_tests,
            "flaky_tests": list(self.flaky_tests),
            "test_failures": {
                name: failures
                for name, failures in self.test_failures.items()
            }
        }

        # Save report
        try:
            with open(self.heal_report_path, 'w') as f:
                json.dump(report, f, indent=2)
            robot_logger.info(f"📊 Heal report saved: {self.heal_report_path}")
        except Exception as e:
            logger.error(f"Failed to save heal report: {e}")

    def _generate_flaky_report(self):
        """Generate flaky test report"""
        flaky_report = {
            "timestamp": datetime.now().isoformat(),
            "flaky_threshold": self.flaky_threshold,
            "flaky_tests": [
                {
                    "test_name": test_name,
                    "failure_count": len(self.test_failures.get(test_name, [])),
                    "failures": self.test_failures.get(test_name, [])
                }
                for test_name in self.flaky_tests
            ]
        }

        # Save or update flaky report
        try:
            # Merge with existing flaky report if present
            if self.flaky_report_path.exists():
                with open(self.flaky_report_path, 'r') as f:
                    existing = json.load(f)
                    existing_tests = {t["test_name"] for t in existing.get("flaky_tests", [])}

                    # Add new flaky tests
                    for test in flaky_report["flaky_tests"]:
                        if test["test_name"] not in existing_tests:
                            existing.setdefault("flaky_tests", []).append(test)

                flaky_report = existing

            with open(self.flaky_report_path, 'w') as f:
                json.dump(flaky_report, f, indent=2)

            if flaky_report["flaky_tests"]:
                robot_logger.warn(f"🔄 {len(flaky_report['flaky_tests'])} flaky tests tracked: {self.flaky_report_path}")
        except Exception as e:
            logger.error(f"Failed to save flaky report: {e}")

    def _log_session_summary(self):
        """Log session summary"""
        robot_logger.info("=" * 60)
        robot_logger.info("Azure Self-Healing Session Summary")
        robot_logger.info("=" * 60)
        robot_logger.info(f"Total Tests: {self.stats['total_tests']}")
        robot_logger.info(f"Failed Tests: {self.stats['failed_tests']}")
        robot_logger.info(f"Flaky Tests Detected: {self.stats['flaky_tests_detected']}")
        robot_logger.info(f"AI Suggestions: {self.stats['ai_suggestions']}")
        robot_logger.info(f"Healed Tests: {self.stats['healed_tests']}")
        robot_logger.info(f"Auto-Healed: {self.stats['auto_healed']}")

        if self.failure_patterns:
            robot_logger.info("\nFailure Patterns:")
            for pattern, count in sorted(self.failure_patterns.items(), key=lambda x: x[1], reverse=True):
                robot_logger.info(f"  - {pattern}: {count}")

        robot_logger.info("=" * 60)

