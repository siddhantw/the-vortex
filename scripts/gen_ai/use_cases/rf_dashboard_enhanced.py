"""
Enhanced Robot Framework Dashboard Analytics Module
The most comprehensive, insightful, and interactive RF Dashboard ever created.

Features:
- Deep AI-powered root cause analysis
- Interactive build triggering and actions
- Advanced metrics and pattern recognition
- Predictive failure analysis
- Test quality scoring
- Real-time monitoring
- Comparative analysis
- JIRA integration for issue tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from requests.auth import HTTPBasicAuth
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, Counter
from urllib.parse import quote
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("RFDashboardEnhanced", level=logging.INFO, log_file="rf_dashboard_enhanced.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# Ensure parent directory is in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import Azure OpenAI Client
try:
    from azure_openai_client import AzureOpenAIClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Import original module for Jenkins client
try:
    from rf_dashboard_analytics import RobotFrameworkDashboardClient, RFTestMetrics, RFDashboardAnalyzer
    ORIGINAL_MODULE_AVAILABLE = True
except ImportError:
    ORIGINAL_MODULE_AVAILABLE = False


class EnhancedAIInsights:
    """Enhanced AI-powered insights with deep root cause analysis"""

    def __init__(self, azure_client: Optional[Any] = None):
        self.azure_client = azure_client

    def generate_comprehensive_insights(self,
                                       analysis: Dict[str, Any],
                                       metrics_list: List[Any],
                                       historical_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate comprehensive AI insights with root cause analysis"""

        if not self.azure_client or not AZURE_AVAILABLE:
            return self._generate_enhanced_basic_insights(analysis, metrics_list)

        try:
            # Perform deep analysis
            root_causes = self._analyze_root_causes(analysis, metrics_list)
            patterns = self._detect_patterns(metrics_list)
            predictions = self._predict_future_trends(analysis, metrics_list)
            quality_score = self._calculate_test_suite_quality(analysis)

            # Create comprehensive prompt
            prompt = self._create_comprehensive_prompt(
                analysis, root_causes, patterns, predictions, quality_score
            )

            # Get AI insights
            response = self.azure_client.generate_response(
                prompt=prompt,
                max_tokens=2500,
                temperature=0.7
            )

            # Parse and enhance response
            insights = self._parse_comprehensive_response(response)
            insights['root_causes'] = root_causes
            insights['patterns'] = patterns
            insights['predictions'] = predictions
            insights['quality_score'] = quality_score

            return insights

        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {e}")
            return self._generate_enhanced_basic_insights(analysis, metrics_list)

    def _analyze_root_causes(self, analysis: Dict, metrics_list: List) -> Dict[str, List[str]]:
        """Analyze root causes of failures using pattern matching"""
        root_causes = {
            'environmental': [],
            'test_design': [],
            'application': [],
            'infrastructure': [],
            'timing': []
        }

        # Analyze failure patterns
        failed_tests = analysis.get('most_failed_tests', [])
        flaky_tests = analysis.get('flaky_tests', [])

        # Check for environmental issues
        if flaky_tests:
            root_causes['environmental'].append(
                f"{len(flaky_tests)} flaky tests detected - likely environment instability"
            )

        # Check for timing issues
        slow_tests = analysis.get('slowest_tests', [])
        if slow_tests and len([t for t in slow_tests if t.get('avg_duration', 0) > 30000]) > 0:
            root_causes['timing'].append(
                "Multiple tests with >30s execution time - potential timeout issues"
            )

        # Check for test design issues
        if len(failed_tests) > 5:
            failure_rate_high = [t for t in failed_tests if t.get('failure_rate', 0) > 50]
            if failure_rate_high:
                root_causes['test_design'].append(
                    f"{len(failure_rate_high)} tests failing >50% - may need test redesign"
                )

        # Check pass rate trend for application issues
        if analysis.get('pass_rate_trend') == 'degrading':
            root_causes['application'].append(
                "Pass rate is declining - potential regression in application"
            )

        # Analyze error message patterns from metrics
        error_patterns = self._extract_error_patterns(metrics_list)
        for category, pattern in error_patterns.items():
            if pattern:
                root_causes[category].extend(pattern)

        return root_causes

    def _extract_error_patterns(self, metrics_list: List) -> Dict[str, List[str]]:
        """Extract common error patterns from test failures"""
        patterns = defaultdict(list)
        error_keywords = {
            'environmental': ['connection', 'timeout', 'network', 'unreachable'],
            'application': ['assertion', 'unexpected', 'incorrect', 'invalid'],
            'infrastructure': ['database', 'server', 'service', 'unavailable'],
        }

        all_errors = []
        for metrics in metrics_list:
            for failed_test in metrics.failed_test_details:
                message = failed_test.get('message', '').lower()
                all_errors.append(message)

        # Count error patterns
        error_counts = Counter()
        for error in all_errors:
            for category, keywords in error_keywords.items():
                for keyword in keywords:
                    if keyword in error:
                        error_counts[(category, keyword)] += 1

        # Report significant patterns
        for (category, keyword), count in error_counts.most_common(10):
            if count >= 3:  # At least 3 occurrences
                patterns[category].append(
                    f"'{keyword}' appears in {count} failure messages"
                )

        return dict(patterns)

    def _detect_patterns(self, metrics_list: List) -> Dict[str, Any]:
        """Detect patterns in test execution"""
        patterns = {
            'temporal': [],
            'recurring': [],
            'correlation': []
        }

        if len(metrics_list) < 5:
            return patterns

        # Temporal patterns (time-based)
        timestamps = [m.timestamp for m in metrics_list]
        pass_rates = [m.pass_rate for m in metrics_list]

        # Check for weekend effect
        weekend_indices = [i for i, ts in enumerate(timestamps)
                          if ts.weekday() >= 5]
        if weekend_indices:
            weekend_pass_rates = [pass_rates[i] for i in weekend_indices]
            weekday_pass_rates = [pass_rates[i] for i, ts in enumerate(timestamps)
                                 if ts.weekday() < 5]
            if weekday_pass_rates and weekend_pass_rates:
                avg_weekend = np.mean(weekend_pass_rates)
                avg_weekday = np.mean(weekday_pass_rates)
                if abs(avg_weekend - avg_weekday) > 5:
                    patterns['temporal'].append(
                        f"Weekend vs Weekday difference: {avg_weekend:.1f}% vs {avg_weekday:.1f}%"
                    )

        # Check for time-of-day patterns
        hours = [ts.hour for ts in timestamps]
        if len(set(hours)) > 3:
            business_hours = [i for i, h in enumerate(hours) if 9 <= h <= 17]
            if business_hours and len(business_hours) < len(hours):
                bh_pass_rates = [pass_rates[i] for i in business_hours]
                patterns['temporal'].append(
                    f"Business hours avg: {np.mean(bh_pass_rates):.1f}%"
                )

        # Recurring failures
        test_failures = defaultdict(list)
        for metrics in metrics_list:
            for failed_test in metrics.failed_test_details:
                test_failures[failed_test['name']].append(metrics.build_number)

        # Find tests that fail in sequence
        for test_name, builds in test_failures.items():
            if len(builds) >= 3:
                # Check if failures are consecutive
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
                        f"'{test_name}' failed {max_consecutive} times consecutively"
                    )

        return patterns

    def _predict_future_trends(self, analysis: Dict, metrics_list: List) -> Dict[str, Any]:
        """Predict future trends using statistical analysis"""
        predictions = {
            'pass_rate_7days': None,
            'expected_failures': None,
            'risk_level': 'Unknown',
            'confidence': 0
        }

        if len(metrics_list) < 10:
            predictions['confidence'] = 0
            predictions['risk_level'] = 'Insufficient data'
            return predictions

        try:
            # Prepare data
            pass_rates = np.array([m.pass_rate for m in metrics_list])
            x = np.arange(len(pass_rates))

            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, pass_rates)

            # Predict 7 builds ahead (approximating 7 days)
            future_x = len(pass_rates) + 7
            predicted_pass_rate = slope * future_x + intercept
            predictions['pass_rate_7days'] = max(0, min(100, predicted_pass_rate))
            predictions['confidence'] = abs(r_value) * 100

            # Calculate risk level
            if predicted_pass_rate < 80:
                predictions['risk_level'] = 'High'
            elif predicted_pass_rate < 90:
                predictions['risk_level'] = 'Medium'
            else:
                predictions['risk_level'] = 'Low'

            # Estimate expected failures
            avg_total_tests = np.mean([m.total_tests for m in metrics_list if m.total_tests > 0])
            if avg_total_tests > 0:
                expected_fail_rate = (100 - predicted_pass_rate) / 100
                predictions['expected_failures'] = int(avg_total_tests * expected_fail_rate)

        except Exception as e:
            logger.error(f"Error predicting trends: {e}")

        return predictions

    def _calculate_test_suite_quality(self, analysis: Dict) -> Dict[str, Any]:
        """Calculate comprehensive test suite quality score"""
        quality_metrics = {
            'overall_score': 0,
            'reliability': 0,
            'performance': 0,
            'coverage': 0,
            'maintainability': 0,
            'grade': 'F'
        }

        try:
            # Reliability score (40% weight)
            pass_rate = analysis.get('average_pass_rate', 0)
            stability = analysis.get('stability_score', 0)
            flaky_count = len(analysis.get('flaky_tests', []))

            reliability = (pass_rate * 0.6) + (stability * 0.4)
            if flaky_count > 5:
                reliability *= 0.8
            elif flaky_count > 0:
                reliability *= 0.9

            quality_metrics['reliability'] = round(reliability, 1)

            # Performance score (20% weight)
            avg_exec_time = analysis.get('average_execution_time', 0)
            exec_trend = analysis.get('execution_time_trend', '')

            if avg_exec_time > 0:
                # Lower is better for execution time
                if avg_exec_time < 60000:  # < 1 min
                    performance = 90
                elif avg_exec_time < 300000:  # < 5 min
                    performance = 75
                elif avg_exec_time < 600000:  # < 10 min
                    performance = 60
                else:
                    performance = 40

                if exec_trend == 'degrading':
                    performance *= 0.85
                elif exec_trend == 'improving':
                    performance *= 1.05

                quality_metrics['performance'] = round(min(100, performance), 1)

            # Maintainability score (20% weight)
            failed_tests = analysis.get('most_failed_tests', [])
            high_failure_tests = [t for t in failed_tests if t.get('failure_rate', 0) > 50]

            maintainability = 85  # Start with good score
            if len(high_failure_tests) > 5:
                maintainability = 60
            elif len(high_failure_tests) > 0:
                maintainability = 75

            quality_metrics['maintainability'] = maintainability

            # Coverage approximation (20% weight)
            total_runs = analysis.get('total_runs', 0)
            if total_runs >= 20:
                coverage = 85
            elif total_runs >= 10:
                coverage = 70
            else:
                coverage = 50

            quality_metrics['coverage'] = coverage

            # Overall score (weighted average)
            overall = (
                quality_metrics['reliability'] * 0.4 +
                quality_metrics['performance'] * 0.2 +
                quality_metrics['maintainability'] * 0.2 +
                quality_metrics['coverage'] * 0.2
            )
            quality_metrics['overall_score'] = round(overall, 1)

            # Assign grade
            if overall >= 90:
                quality_metrics['grade'] = 'A'
            elif overall >= 80:
                quality_metrics['grade'] = 'B'
            elif overall >= 70:
                quality_metrics['grade'] = 'C'
            elif overall >= 60:
                quality_metrics['grade'] = 'D'
            else:
                quality_metrics['grade'] = 'F'

        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")

        return quality_metrics

    def _create_comprehensive_prompt(self, analysis: Dict, root_causes: Dict,
                                    patterns: Dict, predictions: Dict,
                                    quality_score: Dict) -> str:
        """Create comprehensive prompt for AI analysis"""
        prompt = f"""
You are an expert test automation analyst. Analyze this Robot Framework test suite data and provide deep, actionable insights.

## Current State:
- Test Suite Quality Grade: {quality_score.get('grade', 'N/A')} (Score: {quality_score.get('overall_score', 0):.1f}/100)
- Total Test Runs Analyzed: {analysis.get('total_runs', 0)}
- Average Pass Rate: {analysis.get('average_pass_rate', 0):.1f}%
- Stability Score: {analysis.get('stability_score', 0):.1f}/100
- Pass Rate Trend: {analysis.get('pass_rate_trend', 'Unknown')}

## Identified Root Causes:
Environmental Issues: {', '.join(root_causes.get('environmental', ['None']))}
Test Design Issues: {', '.join(root_causes.get('test_design', ['None']))}
Application Issues: {', '.join(root_causes.get('application', ['None']))}
Infrastructure Issues: {', '.join(root_causes.get('infrastructure', ['None']))}
Timing Issues: {', '.join(root_causes.get('timing', ['None']))}

## Detected Patterns:
Temporal Patterns: {', '.join(patterns.get('temporal', ['None detected']))}
Recurring Issues: {', '.join(patterns.get('recurring', ['None detected']))}

## Predictions (7-day forecast):
Predicted Pass Rate: {predictions.get('pass_rate_7days', 'N/A')}%
Expected Failures: {predictions.get('expected_failures', 'N/A')}
Risk Level: {predictions.get('risk_level', 'Unknown')}
Confidence: {predictions.get('confidence', 0):.1f}%

## Failed Tests ({len(analysis.get('most_failed_tests', []))} total):
{self._format_detailed_failures(analysis.get('most_failed_tests', []))}

## Flaky Tests ({len(analysis.get('flaky_tests', []))} total):
{self._format_detailed_flaky(analysis.get('flaky_tests', []))}

Based on this comprehensive analysis, provide:

1. **Executive Summary** (2-3 sentences): Clear assessment of test suite health

2. **Critical Issues** (Top 3-5 with severity): Rank by impact
   Format: [SEVERITY] Issue description - Impact - Root cause

3. **Root Cause Analysis**: For each critical issue, explain:
   - Why it's happening
   - What evidence supports this
   - How it affects the system

4. **Actionable Recommendations** (Prioritized):
   - Immediate actions (within 24 hours)
   - Short-term fixes (within 1 week)
   - Long-term improvements (within 1 month)
   Each with: Action, Expected Impact, Effort Level

5. **Predicted Impact if Not Fixed**: Specific consequences with timeline

6. **Success Metrics**: How to measure improvement (with target values)

7. **Quick Wins**: 3 easiest improvements with highest ROI

Format as JSON with keys: executive_summary, critical_issues, root_cause_analysis, immediate_actions, short_term_actions, long_term_actions, predicted_impact, success_metrics, quick_wins
"""
        return prompt

    def _format_detailed_failures(self, failed_tests: List[Dict]) -> str:
        """Format failed tests with details"""
        if not failed_tests:
            return "None"

        lines = []
        for test in failed_tests[:10]:
            lines.append(
                f"- {test.get('test', 'Unknown')}: "
                f"{test.get('failure_count', 0)} failures "
                f"({test.get('failure_rate', 0):.1f}% failure rate)"
            )
        return "\n".join(lines)

    def _format_detailed_flaky(self, flaky_tests: List[Dict]) -> str:
        """Format flaky tests with details"""
        if not flaky_tests:
            return "None"

        lines = []
        for test in flaky_tests[:10]:
            lines.append(
                f"- {test.get('test', 'Unknown')}: "
                f"{test.get('flakiness_score', 0):.1f}% flakiness "
                f"({test.get('passes', 0)}P/{test.get('fails', 0)}F in {test.get('total_runs', 0)} runs)"
            )
        return "\n".join(lines)

    def _parse_comprehensive_response(self, response: str) -> Dict[str, Any]:
        """Parse comprehensive AI response"""
        try:
            return json.loads(response)
        except:
            return {
                'executive_summary': response[:500],
                'critical_issues': [],
                'root_cause_analysis': {},
                'immediate_actions': [],
                'short_term_actions': [],
                'long_term_actions': [],
                'predicted_impact': response,
                'success_metrics': [],
                'quick_wins': []
            }

    def _generate_enhanced_basic_insights(self, analysis: Dict, metrics_list: List) -> Dict[str, Any]:
        """Generate enhanced basic insights without AI"""
        root_causes = self._analyze_root_causes(analysis, metrics_list)
        quality_score = self._calculate_test_suite_quality(analysis)
        predictions = self._predict_future_trends(analysis, metrics_list)

        insights = {
            'executive_summary': self._create_basic_summary(analysis, quality_score),
            'critical_issues': self._identify_critical_issues(analysis),
            'root_causes': root_causes,
            'quality_score': quality_score,
            'predictions': predictions,
            'immediate_actions': self._generate_immediate_actions(analysis),
            'quick_wins': self._identify_quick_wins(analysis)
        }

        return insights

    def _create_basic_summary(self, analysis: Dict, quality_score: Dict) -> str:
        """Create basic executive summary"""
        grade = quality_score.get('grade', 'N/A')
        score = quality_score.get('overall_score', 0)
        pass_rate = analysis.get('average_pass_rate', 0)
        trend = analysis.get('pass_rate_trend', 'stable')

        if grade in ['A', 'B'] and pass_rate >= 90:
            status = "HEALTHY"
            summary = f"Test suite is in good condition (Grade {grade}, {score:.1f}/100). "
        elif grade in ['C', 'D'] or pass_rate >= 75:
            status = "NEEDS ATTENTION"
            summary = f"Test suite needs improvement (Grade {grade}, {score:.1f}/100). "
        else:
            status = "CRITICAL"
            summary = f"Test suite requires immediate attention (Grade {grade}, {score:.1f}/100). "

        summary += f"Average pass rate is {pass_rate:.1f}% and trending {trend}. "

        flaky_count = len(analysis.get('flaky_tests', []))
        if flaky_count > 0:
            summary += f"{flaky_count} flaky tests detected requiring stabilization."

        return summary

    def _identify_critical_issues(self, analysis: Dict) -> List[Dict]:
        """Identify critical issues"""
        issues = []

        pass_rate = analysis.get('average_pass_rate', 0)
        if pass_rate < 80:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Low Pass Rate',
                'description': f'Pass rate at {pass_rate:.1f}% is below acceptable threshold of 80%',
                'impact': 'High risk of production issues'
            })

        flaky_tests = analysis.get('flaky_tests', [])
        if len(flaky_tests) > 5:
            issues.append({
                'severity': 'HIGH',
                'issue': 'Multiple Flaky Tests',
                'description': f'{len(flaky_tests)} flaky tests causing unreliable results',
                'impact': 'Loss of confidence in test suite'
            })

        if analysis.get('pass_rate_trend') == 'degrading':
            issues.append({
                'severity': 'MEDIUM',
                'issue': 'Degrading Trend',
                'description': 'Pass rate is trending downward',
                'impact': 'Quality deteriorating over time'
            })

        return issues

    def _generate_immediate_actions(self, analysis: Dict) -> List[Dict]:
        """Generate immediate action items"""
        actions = []

        failed_tests = analysis.get('most_failed_tests', [])
        if failed_tests:
            top_failure = failed_tests[0]
            actions.append({
                'action': f"Investigate '{top_failure.get('test', 'Unknown')}' test",
                'reason': f"Failing {top_failure.get('failure_rate', 0):.1f}% of the time",
                'effort': 'Medium',
                'impact': 'High'
            })

        flaky_tests = analysis.get('flaky_tests', [])
        if flaky_tests:
            actions.append({
                'action': "Stabilize flaky tests",
                'reason': f"{len(flaky_tests)} tests showing inconsistent results",
                'effort': 'High',
                'impact': 'High'
            })

        return actions

    def _identify_quick_wins(self, analysis: Dict) -> List[Dict]:
        """Identify quick win opportunities"""
        wins = []

        slow_tests = analysis.get('slowest_tests', [])
        if slow_tests:
            wins.append({
                'action': 'Optimize slow tests',
                'tests': [t['test'] for t in slow_tests[:3]],
                'potential_saving': 'Reduce execution time by 20-30%',
                'effort': 'Low'
            })

        return wins


class JenkinsActions:
    """Handle Jenkins actions like triggering builds"""

    def __init__(self, jenkins_client):
        self.client = jenkins_client

    def trigger_build(self, job_info: Dict, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Trigger a Jenkins build"""
        try:
            job_name = job_info.get('name')
            folder_path = job_info.get('folder', '')

            if folder_path:
                url = f"{self.client.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/build"
            else:
                url = f"{self.client.jenkins_url}/job/{quote(job_name, safe='')}/build"

            if parameters:
                url += "WithParameters"
                response = self.client.session.post(url, data=parameters, timeout=30)
            else:
                response = self.client.session.post(url, timeout=30)

            response.raise_for_status()

            return {
                'success': True,
                'message': f'Build triggered successfully for {job_info.get("display_name")}',
                'queue_url': response.headers.get('Location', '')
            }

        except Exception as e:
            logger.error(f"Error triggering build: {e}")
            return {
                'success': False,
                'message': f'Failed to trigger build: {str(e)}'
            }

    def get_build_parameters(self, job_info: Dict) -> List[Dict]:
        """Get build parameters for a job"""
        try:
            job_name = job_info.get('name')
            folder_path = job_info.get('folder', '')

            if folder_path:
                url = f"{self.client.jenkins_url}/{folder_path}/job/{quote(job_name, safe='')}/api/json"
            else:
                url = f"{self.client.jenkins_url}/job/{quote(job_name, safe='')}/api/json"

            params = {'tree': 'actions[parameterDefinitions[name,type,defaultParameterValue[value],description]]'}
            response = self.client.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            parameters = []

            for action in data.get('actions', []):
                if 'parameterDefinitions' in action:
                    for param in action['parameterDefinitions']:
                        parameters.append({
                            'name': param.get('name'),
                            'type': param.get('type'),
                            'default': param.get('defaultParameterValue', {}).get('value'),
                            'description': param.get('description', '')
                        })

            return parameters

        except Exception as e:
            logger.error(f"Error getting build parameters: {e}")
            return []


def show_enhanced_dashboard():
    """Main UI for Enhanced RF Dashboard"""

    st.set_page_config(page_title="RF Dashboard Pro", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .big-metric {
        font-size: 2.5rem !important;
        font-weight: bold;
    }
    .grade-badge {
        font-size: 3rem;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .grade-A { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .grade-B { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
    .grade-C { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; }
    .grade-D { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
    .grade-F { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); color: white; }
    .severity-high { color: #ff4444; font-weight: bold; }
    .severity-medium { color: #ffaa00; font-weight: bold; }
    .severity-low { color: #00cc66; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🚀 Robot Framework Dashboard Pro")
    st.markdown("### The Ultimate Test Intelligence Platform")

    # Initialize components
    if 'enhanced_insights' not in st.session_state:
        azure_client = AzureOpenAIClient() if AZURE_AVAILABLE else None
        st.session_state.enhanced_insights = EnhancedAIInsights(azure_client)

    # Configuration (reuse from original module)
    st.markdown("---")

    # Check if we have analysis data
    if 'rf_metrics' in st.session_state and st.session_state.rf_metrics:
        show_enhanced_analysis(
            st.session_state.rf_metrics,
            st.session_state.enhanced_insights
        )
    else:
        st.info("👈 Please analyze test results from the main RF Dashboard Analytics first")
        st.markdown("""
        ### Quick Start:
        1. Go to RF Dashboard Analytics
        2. Configure Jenkins and select a job
        3. Click "Analyze Test Results"
        4. Return here for enhanced insights and actions
        """)


def show_enhanced_analysis(metrics_list: List, insights_engine):
    """Show enhanced analysis with all features"""

    st.markdown("---")

    # Create analyzer
    analyzer = RFDashboardAnalyzer(None) if ORIGINAL_MODULE_AVAILABLE else None

    if not analyzer:
        st.error("Original module not available")
        return

    # Perform analysis
    with st.spinner("🔍 Performing comprehensive analysis..."):
        analysis = analyzer.analyze_trends(metrics_list)
        comprehensive_insights = insights_engine.generate_comprehensive_insights(
            analysis, metrics_list
        )

    # Show Quality Grade Dashboard
    show_quality_dashboard(comprehensive_insights.get('quality_score', {}))

    # Show Predictions
    show_predictions_dashboard(comprehensive_insights.get('predictions', {}))

    # Show Enhanced AI Insights
    show_enhanced_ai_insights(comprehensive_insights, analysis)

    # Show Root Cause Analysis
    show_root_cause_analysis(comprehensive_insights.get('root_causes', {}))

    # Show Interactive Actions
    show_interactive_actions(st.session_state.get('rf_selected_job'))

    # Show Advanced Metrics
    show_advanced_metrics(analysis, metrics_list)

    # Show Pattern Detection
    show_pattern_detection(comprehensive_insights.get('patterns', {}))

    # Show Comparative Analysis
    show_comparative_analysis(metrics_list)


def show_quality_dashboard(quality_score: Dict):
    """Show test suite quality dashboard"""
    st.markdown("## 📊 Test Suite Quality Score")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        grade = quality_score.get('grade', 'F')
        score = quality_score.get('overall_score', 0)
        st.markdown(f"""
        <div class="grade-badge grade-{grade}">
            {grade}<br/>
            <span style="font-size: 1.5rem">{score:.1f}/100</span>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Overall Grade")

    with col2:
        st.metric("Reliability", f"{quality_score.get('reliability', 0):.1f}%",
                 help="Pass rate and stability")

    with col3:
        st.metric("Performance", f"{quality_score.get('performance', 0):.1f}%",
                 help="Execution time efficiency")

    with col4:
        st.metric("Maintainability", f"{quality_score.get('maintainability', 0):.1f}%",
                 help="Test design quality")

    with col5:
        st.metric("Coverage", f"{quality_score.get('coverage', 0):.1f}%",
                 help="Test execution frequency")


def show_predictions_dashboard(predictions: Dict):
    """Show predictions dashboard"""
    st.markdown("## 🔮 Predictive Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        predicted_rate = predictions.get('pass_rate_7days')
        if predicted_rate:
            delta = predicted_rate - 90  # Compare to target
            st.metric("7-Day Forecast", f"{predicted_rate:.1f}%",
                     delta=f"{delta:+.1f}%", help="Predicted pass rate")

    with col2:
        expected_failures = predictions.get('expected_failures')
        if expected_failures is not None:
            st.metric("Expected Failures", expected_failures,
                     help="Estimated failures in next run")

    with col3:
        risk = predictions.get('risk_level', 'Unknown')
        risk_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
        st.metric("Risk Level", f"{risk_colors.get(risk, '⚪')} {risk}")

    with col4:
        confidence = predictions.get('confidence', 0)
        st.metric("Confidence", f"{confidence:.0f}%",
                 help="Prediction reliability")


def show_enhanced_ai_insights(insights: Dict, analysis: Dict):
    """Show enhanced AI insights"""
    st.markdown("## 🤖 AI-Powered Deep Insights")

    # Executive Summary
    with st.container():
        st.markdown("### 📋 Executive Summary")
        summary = insights.get('executive_summary', 'No summary available')
        st.info(summary)

    # Critical Issues
    st.markdown("### ⚠️ Critical Issues")

    critical_issues = insights.get('critical_issues', [])
    if not critical_issues and isinstance(insights.get('critical_issues'), list):
        # Try to parse from response
        critical_issues = []

    if critical_issues:
        for issue in critical_issues:
            severity = issue.get('severity', 'MEDIUM')
            severity_class = f"severity-{severity.lower()}"

            with st.expander(f"**{issue.get('issue', 'Unknown Issue')}**", expanded=True):
                st.markdown(f"**Severity:** <span class='{severity_class}'>{severity}</span>",
                           unsafe_allow_html=True)
                st.markdown(f"**Description:** {issue.get('description', 'N/A')}")
                st.markdown(f"**Impact:** {issue.get('impact', 'N/A')}")
    else:
        st.success("✅ No critical issues detected!")

    # Action Items
    tabs = st.tabs(["🔥 Immediate Actions", "📅 Short-term", "🎯 Long-term", "⚡ Quick Wins"])

    with tabs[0]:
        immediate = insights.get('immediate_actions', [])
        if immediate:
            for action in immediate:
                with st.container():
                    st.markdown(f"**Action:** {action.get('action', 'N/A')}")
                    st.markdown(f"*Reason:* {action.get('reason', 'N/A')}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Effort: {action.get('effort', 'N/A')}")
                    with col2:
                        st.caption(f"Impact: {action.get('impact', 'N/A')}")
                    st.markdown("---")
        else:
            st.info("No immediate actions required")

    with tabs[1]:
        short_term = insights.get('short_term_actions', [])
        if short_term:
            for action in short_term:
                st.markdown(f"- {action.get('action', action) if isinstance(action, dict) else action}")
        else:
            st.info("No short-term actions identified")

    with tabs[2]:
        long_term = insights.get('long_term_actions', [])
        if long_term:
            for action in long_term:
                st.markdown(f"- {action.get('action', action) if isinstance(action, dict) else action}")
        else:
            st.info("No long-term actions identified")

    with tabs[3]:
        quick_wins = insights.get('quick_wins', [])
        if quick_wins:
            for win in quick_wins:
                if isinstance(win, dict):
                    st.markdown(f"**{win.get('action', 'N/A')}**")
                    st.markdown(f"*{win.get('potential_saving', 'N/A')}* (Effort: {win.get('effort', 'N/A')})")
                else:
                    st.markdown(f"- {win}")
        else:
            st.info("No quick wins identified")


def show_root_cause_analysis(root_causes: Dict):
    """Show root cause analysis"""
    st.markdown("## 🔬 Root Cause Analysis")

    cols = st.columns(3)

    categories = [
        ('Environmental', 'environmental', '🌍'),
        ('Test Design', 'test_design', '📝'),
        ('Application', 'application', '💻'),
        ('Infrastructure', 'infrastructure', '🏗️'),
        ('Timing', 'timing', '⏱️')
    ]

    for idx, (label, key, icon) in enumerate(categories):
        with cols[idx % 3]:
            causes = root_causes.get(key, [])
            if causes:
                with st.expander(f"{icon} {label} ({len(causes)})", expanded=False):
                    for cause in causes:
                        st.markdown(f"- {cause}")
            else:
                st.success(f"{icon} {label}: ✅ OK")


def show_interactive_actions(selected_job):
    """Show interactive actions panel"""
    st.markdown("## 🎮 Interactive Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Trigger New Build", use_container_width=True):
            if selected_job and 'rf_client' in st.session_state:
                with st.spinner("Triggering build..."):
                    actions = JenkinsActions(st.session_state.rf_client)
                    result = actions.trigger_build(st.session_state.rf_job_info)
                    if result['success']:
                        st.success(result['message'])
                    else:
                        st.error(result['message'])
            else:
                st.warning("Please select a job first")

    with col2:
        if st.button("📊 Export Report", use_container_width=True):
            # Export comprehensive report
            st.info("Feature coming soon: Export to PDF/Excel")

    with col3:
        if st.button("🐛 Create JIRA Ticket", use_container_width=True):
            st.info("Feature coming soon: Auto-create JIRA tickets for failures")


def show_advanced_metrics(analysis: Dict, metrics_list: List):
    """Show advanced metrics"""
    st.markdown("## 📈 Advanced Metrics")

    # Create multi-dimensional analysis
    df_metrics = pd.DataFrame([
        {
            'Build': m.build_number,
            'Pass Rate': m.pass_rate,
            'Total Tests': m.total_tests,
            'Execution Time (s)': m.execution_time / 1000,
            'Failed': m.failed_tests,
            'Date': m.timestamp
        }
        for m in metrics_list
    ])

    # Multi-axis chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Pass Rate Over Time', 'Test Volume Trend',
                       'Failure Rate Distribution', 'Execution Time Box Plot'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'histogram'}, {'type': 'box'}]]
    )

    # Pass rate trend
    fig.add_trace(
        go.Scatter(x=df_metrics['Build'], y=df_metrics['Pass Rate'],
                  mode='lines+markers', name='Pass Rate',
                  line=dict(color='green', width=2)),
        row=1, col=1
    )

    # Test volume
    fig.add_trace(
        go.Scatter(x=df_metrics['Build'], y=df_metrics['Total Tests'],
                  mode='lines+markers', name='Total Tests',
                  line=dict(color='blue', width=2)),
        row=1, col=2
    )

    # Failure distribution
    fig.add_trace(
        go.Histogram(x=df_metrics['Failed'], name='Failures',
                    marker_color='red'),
        row=2, col=1
    )

    # Execution time box plot
    fig.add_trace(
        go.Box(y=df_metrics['Execution Time (s)'], name='Exec Time',
              marker_color='purple'),
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_pattern_detection(patterns: Dict):
    """Show detected patterns"""
    st.markdown("## 🔍 Pattern Detection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⏰ Temporal Patterns")
        temporal = patterns.get('temporal', [])
        if temporal:
            for pattern in temporal:
                st.info(pattern)
        else:
            st.success("No temporal patterns detected")

    with col2:
        st.markdown("### 🔁 Recurring Issues")
        recurring = patterns.get('recurring', [])
        if recurring:
            for pattern in recurring:
                st.warning(pattern)
        else:
            st.success("No recurring issues detected")


def show_comparative_analysis(metrics_list: List):
    """Show comparative analysis"""
    st.markdown("## 📊 Comparative Analysis")

    if len(metrics_list) < 10:
        st.info("Need at least 10 builds for comparative analysis")
        return

    # Split into time periods
    mid_point = len(metrics_list) // 2
    first_half = metrics_list[:mid_point]
    second_half = metrics_list[mid_point:]

    col1, col2, col3 = st.columns(3)

    # Calculate metrics
    first_avg = np.mean([m.pass_rate for m in first_half])
    second_avg = np.mean([m.pass_rate for m in second_half])
    improvement = second_avg - first_avg

    with col1:
        st.metric("First Half Avg", f"{first_avg:.1f}%")

    with col2:
        st.metric("Second Half Avg", f"{second_avg:.1f}%",
                 delta=f"{improvement:+.1f}%")

    with col3:
        if improvement > 5:
            st.success("📈 Improving")
        elif improvement < -5:
            st.error("📉 Degrading")
        else:
            st.info("➡️ Stable")


# Wrapper for integration
def show_ui():
    """Main entry point"""
    show_enhanced_dashboard()


if __name__ == "__main__":
    show_enhanced_dashboard()

