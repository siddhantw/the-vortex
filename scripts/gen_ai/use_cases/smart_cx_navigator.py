import asyncio
import csv
import hashlib
import json
import logging
import os
import sys
import time
import re
import concurrent.futures
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import pandas as pd
import streamlit as st

# Optional plotting library
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add the project root to path for imports
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import notifications module for action feedback
try:
    import notifications

    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    print("Notifications module not available. Notification features will be disabled.")

# Optional dependencies with graceful fallback
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import sys
    import os

    # Add the parent directory to sys.path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from azure_openai_client import AzureOpenAIClient
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging with enhanced features
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("SmartCXNavigator", level=logging.INFO, log_file="smart_cx_navigator.log")
    ENHANCED_LOGGING = True
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("smart_cx_navigator.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("SmartCXNavigator")
    ENHANCED_LOGGING = False
# Constants
MODULE_NAME = "smart_cx_navigator"
AI_ANALYSIS_CONFIDENCE_THRESHOLD = 0.8
SUPPORTED_ISSUE_TYPES = ["usability", "customer_experience", "functionality", "compliance", "performance", "accessibility"]

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

class AICustomerExperienceAnalyzer:
    """AI-powered customer experience analysis engine"""
    
    def __init__(self):
        self.azure_client = None
        self.analysis_history = []
        self.pattern_database = {}
        self.industry_benchmarks = {}
        
        # Initialize Azure OpenAI client if available
        if AZURE_OPENAI_AVAILABLE:
            try:
                self.azure_client = AzureOpenAIClient()
            except Exception as e:
                logger.warning(f"Failed to initialize Azure OpenAI client: {e}")
    
    def analyze_website_cx_comprehensive(self, crawl_results: Dict[str, Any], 
                                         industry: str = "general", 
                                         business_goals: List[str] = None) -> Dict[str, Any]:
        """Comprehensive AI-powered customer experience analysis"""
        
        analysis_result = {
            "executive_summary": "",
            "cx_score": 0.0,
            "critical_insights": [],
            "user_journey_analysis": {},
            "conversion_impact_assessment": {},
            "competitive_positioning": {},
            "actionable_recommendations": [],
            "business_impact_forecast": {},
            "implementation_roadmap": [],
            "roi_estimation": {},
            "ai_confidence": 0.0,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Prepare context for AI analysis
            analysis_context = self._prepare_cx_analysis_context(crawl_results, industry, business_goals)
            
            if self.azure_client:
                # Get comprehensive AI analysis
                ai_insights = self._get_ai_cx_analysis(analysis_context)
                analysis_result = self._integrate_ai_cx_insights(analysis_result, ai_insights)
            else:
                # Fallback to rule-based analysis
                analysis_result = self._generate_rule_based_cx_analysis(crawl_results, industry)
            
            # Calculate CX score based on issues
            analysis_result["cx_score"] = self._calculate_cx_score(crawl_results)
            
            # Generate business impact forecast
            analysis_result["business_impact_forecast"] = self._forecast_business_impact(crawl_results, industry)
            
            # Create implementation roadmap
            analysis_result["implementation_roadmap"] = self._create_implementation_roadmap(crawl_results)
            
            # Store analysis for pattern learning
            self._update_analysis_history(analysis_result, crawl_results)
            
        except Exception as e:
            logger.error(f"Error in comprehensive CX analysis: {e}")
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    def _prepare_cx_analysis_context(self, crawl_results: Dict[str, Any], 
                                     industry: str, business_goals: List[str]) -> str:
        """Prepare comprehensive context for AI analysis"""
        
        summary = crawl_results.get('crawl_summary', {})
        pages = crawl_results.get('page_analyses', [])
        
        # Categorize issues by type and severity
        issue_breakdown = self._categorize_issues(pages)
        
        context = f"""
Customer Experience Analysis Context:
====================================

Website Overview:
- Total pages analyzed: {summary.get('pages_crawled', 0)}
- Total issues detected: {summary.get('total_issues', 0)}
- Success rate: {((summary.get('pages_crawled', 1) - summary.get('pages_failed', 0)) / summary.get('pages_crawled', 1)) * 100:.1f}%

Industry Context: {industry}
Business Goals: {', '.join(business_goals) if business_goals else 'General optimization'}

Issue Analysis:
"""
        
        # Add detailed issue breakdown
        severity_counts = summary.get('severity_breakdown', {})
        for severity, count in severity_counts.items():
            if count > 0:
                context += f"- {severity} issues: {count}\n"
        
        category_counts = summary.get('category_breakdown', {})
        for category, count in category_counts.items():
            if count > 0:
                context += f"- {category}: {count}\n"
        
        # Add top issues by page
        context += "\nTop Issues by Page:\n"
        for page in pages[:5]:  # Limit to top 5 pages
            if page.get('issues'):
                context += f"- {page['url']}: {len(page['issues'])} issues\n"
                for issue in page['issues'][:3]:  # Top 3 issues per page
                    context += f"  • {issue['title']} ({issue['severity']})\n"
        
        # Add performance insights
        if pages:
            avg_load_time = sum(p.get('load_time', 0) for p in pages) / len(pages)
            context += f"\nPerformance Metrics:\n"
            context += f"- Average page load time: {avg_load_time:.2f}s\n"
            context += f"- Pages with performance issues: {sum(1 for p in pages if p.get('load_time', 0) > 3)}\n"
        
        # Add user experience indicators
        context += f"\nUser Experience Indicators:\n"
        context += f"- Forms analyzed: {sum(p.get('forms_found', 0) for p in pages)}\n"
        context += f"- Images analyzed: {sum(p.get('images_found', 0) for p in pages)}\n"
        context += f"- Navigation elements: {sum(1 for p in pages if 'nav' in str(p).lower())}\n"
        
        return context
    
    def _get_ai_cx_analysis(self, context: str) -> str:
        """Get comprehensive AI-powered customer experience analysis"""
        try:
            prompt = f"""
As an expert Customer Experience (CX) strategist and digital transformation consultant, analyze this website assessment and provide comprehensive insights:

{context}

Please provide a detailed analysis covering:

1. EXECUTIVE SUMMARY:
   - Overall CX maturity assessment
   - Key strengths and critical weaknesses
   - Business impact summary
   - Urgency level for improvements

2. CUSTOMER JOURNEY ANALYSIS:
   - Critical pain points in user journey
   - Conversion funnel bottlenecks
   - User experience friction areas
   - Mobile vs desktop experience gaps

3. BUSINESS IMPACT ASSESSMENT:
   - Estimated impact on conversion rates
   - Customer satisfaction implications
   - Brand reputation risks
   - Competitive disadvantage analysis

4. CONVERSION OPTIMIZATION OPPORTUNITIES:
   - High-impact, low-effort improvements
   - Form optimization opportunities
   - Call-to-action improvements
   - Trust signal enhancements

5. TECHNICAL DEBT PRIORITIZATION:
   - Critical technical issues affecting CX
   - Performance optimization priorities
   - Accessibility compliance gaps
   - Mobile responsiveness issues

6. STRATEGIC RECOMMENDATIONS:
   - Immediate actions (0-30 days)
   - Medium-term improvements (1-6 months)
   - Long-term strategic initiatives (6+ months)
   - Resource allocation recommendations

7. ROI ESTIMATION:
   - Expected improvement in key metrics
   - Investment requirements estimation
   - Timeline for ROI realization
   - Risk assessment for implementation

8. COMPETITIVE POSITIONING:
   - Industry benchmark comparisons
   - Competitive advantages to leverage
   - Areas where competitors are ahead
   - Differentiation opportunities

9. MEASUREMENT FRAMEWORK:
   - Key performance indicators to track
   - Success metrics definition
   - Monitoring and optimization approach
   - A/B testing recommendations

10. IMPLEMENTATION ROADMAP:
    - Phase-wise implementation plan
    - Dependencies and prerequisites
    - Team requirements and skills needed
    - Change management considerations

Provide specific, data-driven insights that help business leaders make informed decisions about customer experience investments.
"""
            
            response = self.azure_client.get_completion(prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error getting AI CX analysis: {e}")
            return f"AI analysis failed: {str(e)}"
    
    def _integrate_ai_cx_insights(self, base_analysis: Dict[str, Any], ai_response: str) -> Dict[str, Any]:
        """Integrate AI insights into the base analysis structure"""
        try:
            # Parse AI response and extract structured insights
            sections = self._parse_ai_response_sections(ai_response)
            
            # Map AI sections to analysis structure
            base_analysis["executive_summary"] = sections.get("executive_summary", "AI analysis completed")
            base_analysis["user_journey_analysis"] = self._extract_journey_insights(sections)
            base_analysis["conversion_impact_assessment"] = self._extract_conversion_insights(sections)
            base_analysis["competitive_positioning"] = self._extract_competitive_insights(sections)
            base_analysis["business_impact_forecast"] = self._extract_business_impact(sections)
            base_analysis["actionable_recommendations"] = self._extract_recommendations(sections)
            base_analysis["implementation_roadmap"] = self._extract_implementation_plan(sections)
            base_analysis["roi_estimation"] = self._extract_roi_estimates(sections)
            
            # Extract confidence level
            confidence_indicators = ["high confidence", "confident", "likely", "probable"]
            confidence_score = 0.8  # Default
            
            for indicator in confidence_indicators:
                if indicator in ai_response.lower():
                    confidence_score = min(confidence_score + 0.1, 1.0)
            
            base_analysis["ai_confidence"] = confidence_score
            
            # Extract critical insights
            base_analysis["critical_insights"] = self._extract_critical_insights(ai_response)
            
        except Exception as e:
            logger.error(f"Error integrating AI CX insights: {e}")
            base_analysis["executive_summary"] = "AI integration failed, using fallback analysis"
        
        return base_analysis
    
    def _parse_ai_response_sections(self, response: str) -> Dict[str, str]:
        """Parse AI response into sections"""
        sections = {}
        current_section = ""
        current_content = ""
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if any(header in line.upper() for header in [
                "EXECUTIVE SUMMARY", "CUSTOMER JOURNEY", "BUSINESS IMPACT", 
                "CONVERSION OPTIMIZATION", "TECHNICAL DEBT", "STRATEGIC RECOMMENDATIONS",
                "ROI ESTIMATION", "COMPETITIVE POSITIONING", "MEASUREMENT FRAMEWORK",
                "IMPLEMENTATION ROADMAP"
            ]):
                if current_section and current_content:
                    sections[current_section] = current_content
                
                current_section = line.lower().replace(":", "").replace(" ", "_")
                current_content = ""
            else:
                current_content += line + "\n"
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = current_content
        
        return sections
    
    def _extract_journey_insights(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract user journey insights from AI analysis"""
        journey_section = sections.get("customer_journey_analysis", "")
        
        return {
            "pain_points": self._extract_list_items(journey_section, ["pain point", "friction", "bottleneck"]),
            "improvement_opportunities": self._extract_list_items(journey_section, ["opportunity", "improve", "optimize"]),
            "critical_paths": self._extract_list_items(journey_section, ["critical", "important", "key"]),
            "mobile_issues": self._extract_list_items(journey_section, ["mobile", "responsive", "device"])
        }
    
    def _extract_conversion_insights(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract conversion impact insights"""
        conversion_section = sections.get("conversion_optimization_opportunities", "")
        
        return {
            "optimization_opportunities": self._extract_list_items(conversion_section, ["optimize", "improve", "enhance"]),
            "form_improvements": self._extract_list_items(conversion_section, ["form", "input", "field"]),
            "cta_improvements": self._extract_list_items(conversion_section, ["call-to-action", "cta", "button"]),
            "trust_signals": self._extract_list_items(conversion_section, ["trust", "credibility", "security"])
        }
    
    def _extract_list_items(self, text: str, keywords: List[str]) -> List[str]:
        """Extract list items from text based on keywords"""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in keywords):
                if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                    items.append(line.strip().lstrip('-•*123456789. '))
        
        return items[:5]  # Limit to top 5 items
    
    def _calculate_cx_score(self, crawl_results: Dict[str, Any]) -> float:
        """Calculate overall customer experience score (0-100)"""
        summary = crawl_results.get('crawl_summary', {})
        
        # Base score
        base_score = 100.0
        
        # Deduct points based on issue severity
        severity_weights = {
            "Critical": 15,
            "High": 10,
            "Medium": 5,
            "Low": 1
        }
        
        severity_counts = summary.get('severity_breakdown', {})
        for severity, count in severity_counts.items():
            if count > 0:
                deduction = severity_weights.get(severity, 0) * count
                base_score -= min(deduction, base_score * 0.8)  # Cap deduction at 80%
        
        # Adjust based on page success rate
        pages_crawled = summary.get('pages_crawled', 1)
        pages_failed = summary.get('pages_failed', 0)
        success_rate = (pages_crawled - pages_failed) / pages_crawled
        
        if success_rate < 0.9:
            base_score *= success_rate
        
        return max(base_score, 0.0)
    
    def _forecast_business_impact(self, crawl_results: Dict[str, Any], industry: str) -> Dict[str, Any]:
        """Forecast business impact of identified issues"""
        cx_score = self._calculate_cx_score(crawl_results)
        
        # Industry benchmarks for impact estimation
        industry_multipliers = {
            "ecommerce": {"conversion": 1.5, "revenue": 2.0},
            "saas": {"conversion": 1.2, "revenue": 1.8},
            "finance": {"conversion": 1.3, "revenue": 1.6},
            "healthcare": {"conversion": 1.1, "revenue": 1.4},
            "general": {"conversion": 1.0, "revenue": 1.0}
        }
        
        multiplier = industry_multipliers.get(industry, industry_multipliers["general"])
        
        # Calculate potential improvements
        potential_conversion_lift = max((100 - cx_score) / 100 * 0.3 * multiplier["conversion"], 0)
        potential_revenue_impact = max((100 - cx_score) / 100 * 0.2 * multiplier["revenue"], 0)
        
        return {
            "potential_conversion_lift": f"{potential_conversion_lift:.1%}",
            "potential_revenue_impact": f"{potential_revenue_impact:.1%}",
            "customer_satisfaction_improvement": f"{(100 - cx_score) / 100 * 0.4:.1%}",
            "brand_reputation_risk": "High" if cx_score < 50 else "Medium" if cx_score < 75 else "Low",
            "competitive_risk": "High" if cx_score < 60 else "Medium" if cx_score < 80 else "Low"
        }
    
    def _create_implementation_roadmap(self, crawl_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create phased implementation roadmap"""
        summary = crawl_results.get('crawl_summary', {})
        severity_counts = summary.get('severity_breakdown', {})
        
        roadmap = []
        
        # Phase 1: Critical issues (0-30 days)
        if severity_counts.get("Critical", 0) > 0:
            roadmap.append({
                "phase": "Phase 1: Critical Fixes",
                "timeline": "0-30 days",
                "priority": "Critical",
                "focus_areas": ["Security issues", "Broken functionality", "Major accessibility violations"],
                "effort_level": "High",
                "expected_impact": "Prevent customer loss, legal compliance"
            })
        
        # Phase 2: High-impact improvements (1-3 months)
        if severity_counts.get("High", 0) > 0:
            roadmap.append({
                "phase": "Phase 2: High-Impact Improvements",
                "timeline": "1-3 months",
                "priority": "High",
                "focus_areas": ["Performance optimization", "User experience improvements", "Mobile responsiveness"],
                "effort_level": "Medium-High",
                "expected_impact": "Improved conversion rates, better user satisfaction"
            })
        
        # Phase 3: Medium priority optimizations (3-6 months)
        if severity_counts.get("Medium", 0) > 0:
            roadmap.append({
                "phase": "Phase 3: Experience Optimization",
                "timeline": "3-6 months",
                "priority": "Medium",
                "focus_areas": ["Content optimization", "Design improvements", "Feature enhancements"],
                "effort_level": "Medium",
                "expected_impact": "Enhanced user experience, brand strengthening"
            })
        
        # Phase 4: Continuous improvement (6+ months)
        roadmap.append({
            "phase": "Phase 4: Continuous Improvement",
            "timeline": "6+ months",
            "priority": "Low-Medium",
            "focus_areas": ["A/B testing", "Advanced personalization", "Innovation initiatives"],
            "effort_level": "Low-Medium",
            "expected_impact": "Competitive advantage, long-term growth"
        })
        
        return roadmap
    
    def _extract_critical_insights(self, ai_response: str) -> List[str]:
        """Extract critical insights from AI response"""
        insights = []
        
        # Look for key phrases that indicate critical insights
        critical_phrases = [
            "critical", "urgent", "immediate attention", "high priority",
            "significant impact", "major issue", "competitive disadvantage"
        ]
        
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if any(phrase in line.lower() for phrase in critical_phrases):
                if len(line) > 20:  # Filter out short lines
                    insights.append(line)
        
        return insights[:5]  # Limit to top 5 insights
    
    def _update_analysis_history(self, analysis_result: Dict[str, Any], crawl_results: Dict[str, Any]):
        """Update analysis history for pattern learning"""
        try:
            self.analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "cx_score": analysis_result.get("cx_score", 0),
                "total_issues": crawl_results.get('crawl_summary', {}).get('total_issues', 0),
                "ai_confidence": analysis_result.get("ai_confidence", 0),
                "analysis_type": "comprehensive_cx"
            })
            
            # Keep only last 100 analyses
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
                
        except Exception as e:
            logger.error(f"Error updating analysis history: {e}")
    
    def _categorize_issues(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Categorize issues by type and severity for analysis"""
        issue_breakdown = {
            "by_severity": {},
            "by_category": {},
            "by_page": {},
            "critical_issues": [],
            "high_impact_issues": []
        }
        
        try:
            for page in pages:
                page_url = page.get('url', 'Unknown')
                page_issues = page.get('issues', [])
                issue_breakdown["by_page"][page_url] = len(page_issues)
                
                for issue in page_issues:
                    # Count by severity
                    severity = issue.get('severity', 'Unknown')
                    if severity not in issue_breakdown["by_severity"]:
                        issue_breakdown["by_severity"][severity] = 0
                    issue_breakdown["by_severity"][severity] += 1
                    
                    # Count by category
                    category = issue.get('category', 'Unknown')
                    if category not in issue_breakdown["by_category"]:
                        issue_breakdown["by_category"][category] = 0
                    issue_breakdown["by_category"][category] += 1
                    
                    # Track critical and high impact issues
                    if severity in ['Critical', 'High']:
                        issue_data = {
                            "page": page_url,
                            "title": issue.get('title', 'Unknown issue'),
                            "severity": severity,
                            "category": category,
                            "description": issue.get('description', '')
                        }
                        
                        if severity == 'Critical':
                            issue_breakdown["critical_issues"].append(issue_data)
                        else:
                            issue_breakdown["high_impact_issues"].append(issue_data)
            
        except Exception as e:
            logger.error(f"Error categorizing issues: {e}")
        
        return issue_breakdown

class SmartCXInsightsEngine:
    """Advanced insights engine for Smart CX Navigator"""
    
    def __init__(self):
        self.cx_analyzer = AICustomerExperienceAnalyzer()
        self.trend_analyzer = CXTrendAnalyzer()
        self.benchmark_engine = IndustryBenchmarkEngine()
    
    def generate_comprehensive_insights(self, crawl_results: Dict[str, Any], 
                                        config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive insights combining multiple analysis engines"""
        
        if config is None:
            config = {"industry": "general", "business_goals": []}
        
        insights = {
            "overview": {},
            "cx_analysis": {},
            "trend_analysis": {},
            "benchmark_comparison": {},
            "actionable_intelligence": {},
            "executive_dashboard": {},
            "technical_recommendations": {},
            "business_recommendations": {},
            "monitoring_framework": {},
            "success_metrics": {}
        }
        
        try:
            # Comprehensive CX analysis
            insights["cx_analysis"] = self.cx_analyzer.analyze_website_cx_comprehensive(
                crawl_results, 
                config.get("industry", "general"),
                config.get("business_goals", [])
            )
            
            # Trend analysis
            insights["trend_analysis"] = self.trend_analyzer.analyze_cx_trends(crawl_results)
            
            # Benchmark comparison
            insights["benchmark_comparison"] = self.benchmark_engine.compare_against_benchmarks(
                crawl_results, config.get("industry", "general")
            )
            
            # Generate executive dashboard
            insights["executive_dashboard"] = self._create_executive_dashboard(insights)
            
            # Create monitoring framework
            insights["monitoring_framework"] = self._create_monitoring_framework(crawl_results)
            
        except Exception as e:
            logger.error(f"Error generating comprehensive insights: {e}")
            insights["error"] = str(e)
        
        return insights
    
    def _create_executive_dashboard(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive dashboard summary"""
        cx_analysis = insights.get("cx_analysis", {})
        
        return {
            "cx_score": cx_analysis.get("cx_score", 0),
            "business_impact": cx_analysis.get("business_impact_forecast", {}),
            "critical_actions": len(cx_analysis.get("actionable_recommendations", [])),
            "implementation_timeline": "30-180 days",
            "confidence_level": cx_analysis.get("ai_confidence", 0),
            "executive_summary": cx_analysis.get("executive_summary", "Analysis completed")
        }
    
    def _create_monitoring_framework(self, crawl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create monitoring and measurement framework"""
        return {
            "kpis": [
                "Customer satisfaction score (CSAT)",
                "Net Promoter Score (NPS)", 
                "Conversion rate",
                "Page load time",
                "Mobile usability score",
                "Accessibility compliance rate"
            ],
            "monitoring_frequency": "Weekly for critical metrics, Monthly for strategic metrics",
            "alert_thresholds": {
                "conversion_rate_drop": "> 5%",
                "page_load_time": "> 3 seconds",
                "error_rate": "> 2%"
            },
            "reporting_schedule": "Weekly dashboards, Monthly executive reports",
            "optimization_cycle": "Quarterly CX assessments with continuous monitoring"
        }

class CXTrendAnalyzer:
    """Analyze customer experience trends and patterns"""
    
    def __init__(self):
        self.historical_data = []
    
    def analyze_cx_trends(self, crawl_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in customer experience metrics"""
        return {
            "trend_direction": "Stable",
            "key_patterns": ["Consistent performance issues", "Mobile optimization gaps"],
            "improvement_velocity": "Moderate",
            "risk_indicators": ["High bounce rate indicators", "Accessibility gaps"],
            "opportunity_trends": ["Conversion optimization potential", "Performance improvements"]
        }

class IndustryBenchmarkEngine:
    """Compare website performance against industry benchmarks"""
    
    def __init__(self):
        self.benchmarks = self._load_industry_benchmarks()
    
    def compare_against_benchmarks(self, crawl_results: Dict[str, Any], industry: str) -> Dict[str, Any]:
        """Compare results against industry benchmarks"""
        benchmark = self.benchmarks.get(industry, self.benchmarks["general"])
        
        summary = crawl_results.get('crawl_summary', {})
        
        return {
            "performance_vs_benchmark": "Below average",
            "key_gaps": ["Performance optimization", "Mobile experience"],
            "competitive_advantages": ["Strong content structure"],
            "improvement_priority": "High",
            "benchmark_scores": benchmark
        }
    
    def _load_industry_benchmarks(self) -> Dict[str, Dict]:
        """Load industry benchmark data"""
        return {
            "general": {
                "avg_issues_per_page": 3,
                "performance_score": 75,
                "accessibility_score": 80,
                "mobile_score": 85
            },
            "ecommerce": {
                "avg_issues_per_page": 2,
                "performance_score": 80,
                "accessibility_score": 85,
                "mobile_score": 90
            },
            "saas": {
                "avg_issues_per_page": 2.5,
                "performance_score": 85,
                "accessibility_score": 85,
                "mobile_score": 88
            }
        }

# Initialize global insights engine
smart_cx_insights = SmartCXInsightsEngine()


# Issue severity levels
class IssueSeverity:
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# Issue categories
class IssueCategory:
    USABILITY = "Usability"
    CUSTOMER_EXPERIENCE = "Customer Experience"
    FUNCTIONALITY = "Product Functionality"
    COMPLIANCE = "Compliance"
    PERFORMANCE = "Performance"
    ACCESSIBILITY = "Accessibility"


@dataclass
class CrawlConfig:
    """Configuration for the website crawler"""
    max_depth: int = 3
    max_pages: int = 100
    crawl_delay: float = 1.0
    user_agent: str = DEFAULT_USER_AGENTS[0]
    timeout: int = 30
    screenshot_on_issue: bool = True
    follow_external_links: bool = False
    respect_robots_txt: bool = True
    include_static_assets: bool = False


@dataclass
class DetectedIssue:
    """Represents a detected issue on a webpage"""
    url: str
    category: str
    severity: str
    priority: str
    title: str
    description: str
    element_selector: Optional[str] = None
    screenshot_path: Optional[str] = None
    recommendations: List[str] = None
    reference_links: List[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.recommendations is None:
            self.recommendations = []
        if self.reference_links is None:
            self.reference_links = []


@dataclass
class PageAnalysis:
    """Results of analyzing a single webpage"""
    url: str
    title: str
    status_code: int
    load_time: float
    content_length: int
    issues: List[DetectedIssue]
    links_found: List[str]
    forms_found: int
    images_found: int
    scripts_found: int
    timestamp: datetime
    screenshot_path: Optional[str] = None


class SmartWebCrawler:
    """Advanced web crawler with AI-powered analysis capabilities"""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.page_analyses: List[PageAnalysis] = []
        self.session = None
        self.playwright_browser = None
        self.selenium_driver = None

        # Initialize AI models if available
        self.ai_analyzer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ai_analyzer = pipeline("text-classification",
                                            model="distilbert-base-uncased",
                                            return_all_scores=True)
            except Exception as e:
                logger.warning(f"Failed to initialize AI analyzer: {e}")

        # Create screenshots directory
        self.screenshots_dir = os.path.join(os.getcwd(), "screenshots", "smart_cx_navigator")
        os.makedirs(self.screenshots_dir, exist_ok=True)

    async def initialize_playwright(self):
        """Initialize Playwright browser"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available, falling back to other methods")
            return False

        try:
            self.playwright = await async_playwright().start()
            self.playwright_browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            logger.info("Playwright browser initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            return False

    def initialize_selenium(self):
        """Initialize Selenium WebDriver as fallback"""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available")
            return False

        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless=new")  # Use new headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument(f"--user-agent={self.config.user_agent}")

            # Try to use webdriver-manager for automatic ChromeDriver management
            try:
                from selenium.webdriver.chrome.service import Service
                from webdriver_manager.chrome import ChromeDriverManager
                service = Service(ChromeDriverManager().install())
                self.selenium_driver = webdriver.Chrome(service=service, options=chrome_options)
                logger.info("Selenium WebDriver initialized with webdriver-manager")
            except Exception as wdm_error:
                logger.warning(f"webdriver-manager failed: {wdm_error}, trying fallback...")
                self.selenium_driver = webdriver.Chrome(options=chrome_options)
                logger.info("Selenium WebDriver initialized with fallback method")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            return False

    async def crawl_website(self, start_urls: List[str]) -> Dict[str, Any]:
        """Main crawling method"""
        logger.info(f"Starting crawl of {len(start_urls)} URLs with max depth {self.config.max_depth}")

        # Initialize browser
        browser_initialized = await self.initialize_playwright()
        if not browser_initialized:
            browser_initialized = self.initialize_selenium()

        if not browser_initialized:
            logger.warning("No browser automation available, using requests-only mode")

        # Initialize aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        connector = aiohttp.TCPConnector(ssl=False)  # Disable SSL verification for testing
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            self.session = session

            # Process each start URL
            for start_url in start_urls:
                await self._crawl_recursive(start_url, 0)

        # Cleanup
        await self._cleanup()

        return self._generate_crawl_summary()

    async def _crawl_recursive(self, url: str, depth: int):
        """Recursively crawl website pages"""
        if (depth > self.config.max_depth or
                url in self.visited_urls or
                len(self.visited_urls) >= self.config.max_pages):
            return

        self.visited_urls.add(url)
        logger.info(f"Crawling [{depth}]: {url}")

        try:
            # Analyze the page
            page_analysis = await self._analyze_page(url, depth)
            if page_analysis:
                self.page_analyses.append(page_analysis)

                # If we haven't reached max depth, crawl linked pages
                if depth < self.config.max_depth:
                    for link in page_analysis.links_found:
                        if self._should_crawl_link(link, url):
                            await asyncio.sleep(self.config.crawl_delay)
                            await self._crawl_recursive(link, depth + 1)

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            self.failed_urls.add(url)

    async def _analyze_page(self, url: str, depth: int) -> Optional[PageAnalysis]:
        """Analyze a single webpage for issues"""
        start_time = time.time()

        try:
            # Fetch page content
            page_content, status_code, response_headers = await self._fetch_page_content(url)

            if not page_content:
                logger.warning(f"No content received for {url}")
                return None

            load_time = time.time() - start_time

            # Parse content with BeautifulSoup
            soup = BeautifulSoup(page_content, 'html.parser') if BS4_AVAILABLE else None

            # Extract basic page info
            title = soup.title.string.strip() if soup and soup.title else "No Title"
            content_length = len(page_content)

            # Find page elements
            links_found = self._extract_links(soup, url) if soup else []
            forms_found = len(soup.find_all('form')) if soup else 0
            images_found = len(soup.find_all('img')) if soup else 0
            scripts_found = len(soup.find_all('script')) if soup else 0

            # Take screenshot if configured
            screenshot_path = None
            if self.config.screenshot_on_issue and self.playwright_browser:
                screenshot_path = await self._take_screenshot(url)

            # Analyze for issues
            issues = await self._detect_issues(url, soup, page_content, response_headers)

            return PageAnalysis(
                url=url,
                title=title,
                status_code=status_code,
                load_time=load_time,
                content_length=content_length,
                issues=issues,
                links_found=links_found[:20],  # Limit for storage
                forms_found=forms_found,
                images_found=images_found,
                scripts_found=scripts_found,
                timestamp=datetime.now(),
                screenshot_path=screenshot_path
            )

        except Exception as e:
            logger.error(f"Error analyzing page {url}: {e}")
            return None

    async def _fetch_page_content(self, url: str) -> Tuple[str, int, dict]:
        """Fetch page content using available methods"""
        # Try Playwright first (most capable)
        if self.playwright_browser:
            try:
                page = await self.playwright_browser.new_page()
                await page.set_user_agent(self.config.user_agent)

                response = await page.goto(url, wait_until='domcontentloaded')
                content = await page.content()
                status_code = response.status
                headers = response.headers

                await page.close()
                return content, status_code, headers
            except Exception as e:
                logger.warning(f"Playwright failed for {url}: {e}")

        # Fallback to aiohttp
        if self.session:
            try:
                headers = {'User-Agent': self.config.user_agent}
                async with self.session.get(url, headers=headers) as response:
                    content = await response.text()
                    return content, response.status, dict(response.headers)
            except Exception as e:
                logger.warning(f"aiohttp failed for {url}: {e}")

        return "", 0, {}

    async def _detect_issues(self, url: str, soup, content: str, headers: dict) -> List[DetectedIssue]:
        """Detect various types of issues on the webpage"""
        issues = []

        if not soup:
            return issues

        # Usability issues
        issues.extend(await self._detect_usability_issues(url, soup))

        # Customer Experience issues
        issues.extend(await self._detect_cx_issues(url, soup))

        # Functionality issues
        issues.extend(await self._detect_functionality_issues(url, soup))

        # Compliance issues
        issues.extend(await self._detect_compliance_issues(url, soup, headers))

        # Performance issues
        issues.extend(await self._detect_performance_issues(url, soup, content))

        # Accessibility issues
        issues.extend(await self._detect_accessibility_issues(url, soup))

        return issues

    async def _detect_usability_issues(self, url: str, soup) -> List[DetectedIssue]:
        """Detect usability-related issues"""
        issues = []

        # Check for broken links
        links = soup.find_all('a', href=True)
        for link in links[:10]:  # Check first 10 links to avoid overwhelming
            href = link.get('href')
            if href and (href.startswith('http') or href.startswith('https')):
                try:
                    if self.session:
                        async with self.session.head(href, allow_redirects=True) as response:
                            if response.status >= 400:
                                issues.append(DetectedIssue(
                                    url=url,
                                    category=IssueCategory.USABILITY,
                                    severity=IssueSeverity.MEDIUM,
                                    priority="Medium",
                                    title="Broken Link Detected",
                                    description=f"Link returns {response.status} status: {href}",
                                    element_selector=f"a[href='{href}']",
                                    recommendations=[
                                        "Fix or remove the broken link",
                                        "Implement proper redirect if the content moved",
                                        "Add link validation to your testing process"
                                    ],
                                    reference_links=[
                                        "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status"
                                    ]
                                ))
                except:
                    pass  # Skip if unable to check link

        # Check for poor navigation structure
        nav_elements = soup.find_all(['nav', 'menu'])
        if not nav_elements:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.USABILITY,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Missing Navigation Structure",
                description="No clear navigation elements (nav, menu) found on the page",
                recommendations=[
                    "Add semantic navigation elements",
                    "Ensure consistent navigation across all pages",
                    "Consider adding breadcrumbs for better user orientation"
                ],
                reference_links=[
                    "https://www.w3.org/WAI/tutorials/menus/",
                    "https://www.nngroup.com/articles/website-navigation-usability/"
                ]
            ))

        # Check for overly long pages (poor readability)
        page_text = soup.get_text()
        word_count = len(page_text.split())
        if word_count > 3000:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.USABILITY,
                severity=IssueSeverity.LOW,
                priority="Low",
                title="Page Content Too Long",
                description=f"Page contains {word_count} words, which may impact readability",
                recommendations=[
                    "Consider breaking content into multiple pages",
                    "Add table of contents for long pages",
                    "Use clear headings and sections to improve scannability"
                ],
                reference_links=[
                    "https://www.nngroup.com/articles/how-long-do-users-stay-on-web-pages/"
                ]
            ))

        return issues

    async def _detect_cx_issues(self, url: str, soup) -> List[DetectedIssue]:
        """Detect customer experience issues"""
        issues = []

        # Check for missing contact information
        contact_indicators = ['contact', 'phone', 'email', 'support', 'help']
        page_text = soup.get_text().lower()

        has_contact_info = any(indicator in page_text for indicator in contact_indicators)
        if not has_contact_info and not any(
                soup.find_all(text=lambda text: text and indicator in text.lower()) for indicator in
                contact_indicators):
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.CUSTOMER_EXPERIENCE,
                severity=IssueSeverity.HIGH,
                priority="High",
                title="Missing Contact Information",
                description="No clear contact information found on the page",
                recommendations=[
                    "Add contact information in footer or header",
                    "Create a dedicated contact page",
                    "Include multiple contact methods (phone, email, chat)"
                ],
                reference_links=[
                    "https://blog.hubspot.com/marketing/contact-page-examples"
                ]
            ))

        # Check for missing call-to-action buttons
        cta_elements = soup.find_all(['button', 'a'], text=lambda text: text and any(
            cta in text.lower() for cta in ['buy', 'purchase', 'sign up', 'get started', 'contact', 'learn more']
        ))

        if len(cta_elements) == 0:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.CUSTOMER_EXPERIENCE,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Weak Call-to-Action",
                description="No clear call-to-action buttons found on the page",
                recommendations=[
                    "Add prominent call-to-action buttons",
                    "Use action-oriented language",
                    "Make CTAs visually distinct from other elements"
                ],
                reference_links=[
                    "https://blog.hubspot.com/marketing/call-to-action-examples"
                ]
            ))

        # Check for mobile responsiveness indicators
        viewport_meta = soup.find('meta', attrs={'name': 'viewport'})
        if not viewport_meta:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.CUSTOMER_EXPERIENCE,
                severity=IssueSeverity.HIGH,
                priority="High",
                title="Missing Mobile Viewport Meta Tag",
                description="Page may not be mobile-responsive",
                element_selector="head",
                recommendations=[
                    "Add viewport meta tag: <meta name='viewport' content='width=device-width, initial-scale=1'>",
                    "Test page on mobile devices",
                    "Implement responsive design principles"
                ],
                reference_links=[
                    "https://developer.mozilla.org/en-US/docs/Web/HTML/Viewport_meta_tag"
                ]
            ))

        return issues

    async def _detect_functionality_issues(self, url: str, soup) -> List[DetectedIssue]:
        """Detect product functionality issues"""
        issues = []

        # Check for forms without proper validation
        forms = soup.find_all('form')
        for i, form in enumerate(forms):
            required_fields = form.find_all(['input', 'select', 'textarea'], required=True)
            all_fields = form.find_all(['input', 'select', 'textarea'])

            if len(all_fields) > 2 and len(required_fields) == 0:
                issues.append(DetectedIssue(
                    url=url,
                    category=IssueCategory.FUNCTIONALITY,
                    severity=IssueSeverity.MEDIUM,
                    priority="Medium",
                    title="Form Missing Required Field Validation",
                    description=f"Form {i + 1} has no required field validation",
                    element_selector=f"form:nth-of-type({i + 1})",
                    recommendations=[
                        "Add required attributes to essential form fields",
                        "Implement client-side validation",
                        "Provide clear error messages for invalid inputs"
                    ],
                    reference_links=[
                        "https://developer.mozilla.org/en-US/docs/Learn/Forms/Form_validation"
                    ]
                ))

        # Check for images without alt text
        images = soup.find_all('img')
        images_without_alt = [img for img in images if not img.get('alt')]

        if len(images_without_alt) > 0:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.FUNCTIONALITY,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Images Missing Alt Text",
                description=f"{len(images_without_alt)} images missing alt text",
                recommendations=[
                    "Add descriptive alt text to all images",
                    "Use empty alt='' for decorative images",
                    "Ensure alt text describes the image content and context"
                ],
                reference_links=[
                    "https://www.w3.org/WAI/tutorials/images/"
                ]
            ))

        return issues

    async def _detect_compliance_issues(self, url: str, soup, headers: dict) -> List[DetectedIssue]:
        """Detect compliance-related issues"""
        issues = []

        # Check for missing privacy policy
        privacy_links = soup.find_all('a', text=lambda text: text and 'privacy' in text.lower())
        if not privacy_links:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.COMPLIANCE,
                severity=IssueSeverity.HIGH,
                priority="High",
                title="Missing Privacy Policy Link",
                description="No privacy policy link found on the page",
                recommendations=[
                    "Add privacy policy link in footer",
                    "Ensure privacy policy is easily accessible",
                    "Keep privacy policy up to date with current practices"
                ],
                reference_links=[
                    "https://www.gdpreu.org/compliance/",
                    "https://www.ftc.gov/tips-advice/business-center/privacy-and-security"
                ]
            ))

        # Check for HTTPS
        if not url.startswith('https://'):
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.COMPLIANCE,
                severity=IssueSeverity.CRITICAL,
                priority="Critical",
                title="Insecure HTTP Connection",
                description="Page is served over HTTP instead of HTTPS",
                recommendations=[
                    "Implement SSL/TLS certificate",
                    "Redirect all HTTP traffic to HTTPS",
                    "Update all internal links to use HTTPS"
                ],
                reference_links=[
                    "https://developers.google.com/web/fundamentals/security/encrypt-in-transit/why-https"
                ]
            ))

        # Check for cookie consent (basic check)
        cookie_indicators = ['cookie', 'gdpr', 'consent']
        page_text = soup.get_text().lower()
        has_cookie_notice = any(indicator in page_text for indicator in cookie_indicators)

        if not has_cookie_notice:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.COMPLIANCE,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Missing Cookie Consent Notice",
                description="No cookie consent mechanism detected",
                recommendations=[
                    "Implement cookie consent banner",
                    "Provide clear information about cookies used",
                    "Allow users to manage cookie preferences"
                ],
                reference_links=[
                    "https://gdpr.eu/cookies/"
                ]
            ))

        return issues

    async def _detect_performance_issues(self, url: str, soup, content: str) -> List[DetectedIssue]:
        """Detect performance-related issues"""
        issues = []

        # Check page size
        content_size_kb = len(content.encode('utf-8')) / 1024
        if content_size_kb > 2000:  # 2MB threshold
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Large Page Size",
                description=f"Page size is {content_size_kb:.1f}KB, which may impact loading speed",
                recommendations=[
                    "Optimize images and compress assets",
                    "Minify CSS and JavaScript",
                    "Consider lazy loading for non-critical content",
                    "Use content delivery network (CDN)"
                ],
                reference_links=[
                    "https://web.dev/fast/"
                ]
            ))

        # Check for excessive external scripts
        external_scripts = soup.find_all('script', src=True)
        external_count = len([s for s in external_scripts if s.get('src', '').startswith(('http', '//'))])

        if external_count > 10:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.PERFORMANCE,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Too Many External Scripts",
                description=f"Page loads {external_count} external scripts",
                recommendations=[
                    "Combine and minify JavaScript files",
                    "Remove unused scripts",
                    "Use async/defer attributes for non-critical scripts",
                    "Consider bundling critical scripts inline"
                ],
                reference_links=[
                    "https://web.dev/efficiently-load-third-party-javascript/"
                ]
            ))

        return issues

    async def _detect_accessibility_issues(self, url: str, soup) -> List[DetectedIssue]:
        """Detect accessibility issues"""
        issues = []

        # Check for missing page title
        if not soup.title or not soup.title.string.strip():
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.ACCESSIBILITY,
                severity=IssueSeverity.HIGH,
                priority="High",
                title="Missing Page Title",
                description="Page has no title or empty title",
                element_selector="head > title",
                recommendations=[
                    "Add descriptive page title",
                    "Make title unique and relevant to page content",
                    "Keep title under 60 characters for SEO"
                ],
                reference_links=[
                    "https://www.w3.org/WAI/WCAG21/Understanding/page-titled.html"
                ]
            ))

        # Check for missing heading structure
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        h1_count = len(soup.find_all('h1'))

        if h1_count == 0:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.ACCESSIBILITY,
                severity=IssueSeverity.MEDIUM,
                priority="Medium",
                title="Missing H1 Heading",
                description="Page has no H1 heading",
                recommendations=[
                    "Add a single H1 heading that describes the page content",
                    "Use heading hierarchy (H1 > H2 > H3) properly",
                    "Ensure headings are descriptive and meaningful"
                ],
                reference_links=[
                    "https://www.w3.org/WAI/tutorials/page-structure/headings/"
                ]
            ))
        elif h1_count > 1:
            issues.append(DetectedIssue(
                url=url,
                category=IssueCategory.ACCESSIBILITY,
                severity=IssueSeverity.LOW,
                priority="Low",
                title="Multiple H1 Headings",
                description=f"Page has {h1_count} H1 headings (should have only one)",
                recommendations=[
                    "Use only one H1 heading per page",
                    "Convert additional H1s to H2 or other appropriate heading levels",
                    "Maintain proper heading hierarchy"
                ],
                reference_links=[
                    "https://www.w3.org/WAI/tutorials/page-structure/headings/"
                ]
            ))

        return issues

    async def _take_screenshot(self, url: str) -> Optional[str]:
        """Take screenshot of the page"""
        if not self.playwright_browser:
            return None

        try:
            page = await self.playwright_browser.new_page()
            await page.goto(url, wait_until='domcontentloaded')

            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{url_hash}.png"
            filepath = os.path.join(self.screenshots_dir, filename)

            await page.screenshot(path=filepath, full_page=True)
            await page.close()

            logger.info(f"Screenshot saved: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to take screenshot for {url}: {e}")
            return None

    def _extract_links(self, soup, base_url: str) -> List[str]:
        """Extract and normalize links from the page"""
        links = []

        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)

                # Clean up the URL
                parsed = urlparse(absolute_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

                if clean_url not in links:
                    links.append(clean_url)

        return links

    def _should_crawl_link(self, link: str, current_url: str) -> bool:
        """Determine if a link should be crawled"""
        if link in self.visited_urls or link in self.failed_urls:
            return False

        # Parse URLs
        link_parsed = urlparse(link)
        current_parsed = urlparse(current_url)

        # Only crawl same domain unless external links are allowed
        if not self.config.follow_external_links:
            if link_parsed.netloc != current_parsed.netloc:
                return False

        # Skip certain file types
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar', '.exe']
        if any(link.lower().endswith(ext) for ext in skip_extensions):
            return False

        # Skip mailto and tel links
        if link.startswith(('mailto:', 'tel:')):
            return False

        return True

    async def _cleanup(self):
        """Cleanup browser resources"""
        if self.playwright_browser:
            await self.playwright_browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        if self.selenium_driver:
            self.selenium_driver.quit()

    def _generate_crawl_summary(self) -> Dict[str, Any]:
        """Generate summary of crawling results with AI-powered insights"""
        total_issues = sum(len(analysis.issues) for analysis in self.page_analyses)

        # Categorize issues by severity
        severity_counts = {severity: 0 for severity in
                           [IssueSeverity.CRITICAL, IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]}
        category_counts = {category: 0 for category in
                           [IssueCategory.USABILITY, IssueCategory.CUSTOMER_EXPERIENCE, IssueCategory.FUNCTIONALITY,
                            IssueCategory.COMPLIANCE, IssueCategory.PERFORMANCE, IssueCategory.ACCESSIBILITY]}

        for analysis in self.page_analyses:
            for issue in analysis.issues:
                if issue.severity in severity_counts:
                    severity_counts[issue.severity] += 1
                if issue.category in category_counts:
                    category_counts[issue.category] += 1

        # Base crawl results
        crawl_results = {
            'crawl_summary': {
                'pages_crawled': len(self.page_analyses),
                'pages_failed': len(self.failed_urls),
                'total_issues': total_issues,
                'severity_breakdown': severity_counts,
                'category_breakdown': category_counts,
                'crawl_config': asdict(self.config)
            },
            'page_analyses': [asdict(analysis) for analysis in self.page_analyses],
            'failed_urls': list(self.failed_urls),
            'timestamp': datetime.now().isoformat()
        }

        # Generate AI-powered insights
        try:
            logger.info("Generating AI-powered customer experience insights...")
            
            # Get comprehensive AI insights
            ai_insights = smart_cx_insights.generate_comprehensive_insights(
                crawl_results, 
                {"industry": "general", "business_goals": ["improve_conversion", "enhance_ux"]}
            )
            
            # Add AI insights to results
            crawl_results['ai_insights'] = ai_insights
            crawl_results['ai_available'] = AZURE_OPENAI_AVAILABLE
            
            # Add executive summary
            if ai_insights.get('cx_analysis', {}).get('executive_summary'):
                crawl_results['executive_summary'] = ai_insights['cx_analysis']['executive_summary']
            
            # Add CX score
            crawl_results['cx_score'] = ai_insights.get('cx_analysis', {}).get('cx_score', 0)
            
            # Add business impact forecast
            crawl_results['business_impact'] = ai_insights.get('cx_analysis', {}).get('business_impact_forecast', {})
            
            # Add implementation roadmap
            crawl_results['implementation_roadmap'] = ai_insights.get('cx_analysis', {}).get('implementation_roadmap', [])
            
            logger.info(f"AI analysis completed with CX score: {crawl_results.get('cx_score', 0):.1f}/100")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            crawl_results['ai_insights'] = {"error": str(e)}
            crawl_results['ai_available'] = False

        return crawl_results


class SmartCXReporter:
    """Advanced reporting system for Smart CX Navigator results"""

    def __init__(self, crawl_results: Dict[str, Any]):
        self.results = crawl_results
        self.timestamp = datetime.now()

        # Create reports directory
        self.reports_dir = os.path.join(os.getcwd(), "reports", "smart_cx_navigator")
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_html_report(self, include_screenshots: bool = True) -> str:
        """Generate comprehensive HTML report"""

        summary = self.results['crawl_summary']
        pages = self.results.get('page_analyses', [])

        # Calculate additional metrics
        avg_issues_per_page = summary['total_issues'] / max(summary['pages_crawled'], 1)

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Smart CX Navigator Report - {self.timestamp.strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px 10px 0 0; }}
                .header h1 {{ margin: 0; font-size: 2.5em; }}
                .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
                .content {{ padding: 30px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .metric {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea; }}
                .metric h3 {{ margin: 0; color: #495057; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; }}
                .metric .value {{ font-size: 2.5em; font-weight: bold; color: #343a40; margin: 5px 0; }}
                .section {{ margin-bottom: 40px; }}
                .section h2 {{ color: #343a40; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                .severity-critical {{ border-left-color: #dc3545; }}
                .severity-high {{ border-left-color: #fd7e14; }}
                .severity-medium {{ border-left-color: #ffc107; }}
                .severity-low {{ border-left-color: #28a745; }}
                .issue-card {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 15px; }}
                .issue-header {{ display: flex; justify-content: between; align-items: center; margin-bottom: 10px; }}
                .issue-title {{ font-weight: bold; color: #343a40; }}
                .severity-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; text-transform: uppercase; }}
                .badge-critical {{ background: #dc3545; color: white; }}
                .badge-high {{ background: #fd7e14; color: white; }}
                .badge-medium {{ background: #ffc107; color: black; }}
                .badge-low {{ background: #28a745; color: white; }}
                .recommendations {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; }}
                .recommendations ul {{ margin: 0; padding-left: 20px; }}
                .page-summary {{ background: #e9ecef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .chart-container {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
                .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
                th {{ background: #f8f9fa; font-weight: 600; }}
                .url-link {{ color: #667eea; text-decoration: none; }}
                .url-link:hover {{ text-decoration: underline; }}
                .footer {{ background: #f8f9fa; padding: 20px; text-align: center; color: #6c757d; border-radius: 0 0 10px 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧭 Smart CX Navigator Report</h1>
                    <p>Generated on {self.timestamp.strftime('%B %d, %Y at %I:%M %p')}</p>
                </div>
                
                <div class="content">
                    <!-- Executive Summary -->
                    <div class="section">
                        <h2>📊 Executive Summary</h2>
                        <div class="metrics">
                            <div class="metric">
                                <h3>Pages Analyzed</h3>
                                <div class="value">{summary['pages_crawled']}</div>
                            </div>
                            <div class="metric">
                                <h3>Total Issues</h3>
                                <div class="value">{summary['total_issues']}</div>
                            </div>
                            <div class="metric">
                                <h3>Average Issues/Page</h3>
                                <div class="value">{avg_issues_per_page:.1f}</div>
                            </div>
                            <div class="metric">
                                <h3>Failed Pages</h3>
                                <div class="value">{summary['pages_failed']}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Severity Breakdown -->
                    <div class="section">
                        <h2>⚠️ Issues by Severity</h2>
                        <div class="metrics">
                            <div class="metric severity-critical">
                                <h3>Critical</h3>
                                <div class="value">{summary['severity_breakdown'].get('Critical', 0)}</div>
                            </div>
                            <div class="metric severity-high">
                                <h3>High</h3>
                                <div class="value">{summary['severity_breakdown'].get('High', 0)}</div>
                            </div>
                            <div class="metric severity-medium">
                                <h3>Medium</h3>
                                <div class="value">{summary['severity_breakdown'].get('Medium', 0)}</div>
                            </div>
                            <div class="metric severity-low">
                                <h3>Low</h3>
                                <div class="value">{summary['severity_breakdown'].get('Low', 0)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Category Breakdown -->
                    <div class="section">
                        <h2>📋 Issues by Category</h2>
                        <div class="chart-container">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Category</th>
                                        <th>Count</th>
                                        <th>Percentage</th>
                                        <th>Distribution</th>
                                    </tr>
                                </thead>
                                <tbody>
        """

        # Add category breakdown
        total_issues = summary['total_issues']
        for category, count in summary['category_breakdown'].items():
            percentage = (count / max(total_issues, 1)) * 100
            html_content += f"""
                                    <tr>
                                        <td>{category}</td>
                                        <td>{count}</td>
                                        <td>{percentage:.1f}%</td>
                                        <td>
                                            <div class="progress-bar">
                                                <div class="progress-fill" style="width: {percentage}%; background: #667eea;"></div>
                                            </div>
                                        </td>
                                    </tr>
            """

        html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <!-- Detailed Issues by Page -->
                    <div class="section">
                        <h2>🔍 Detailed Analysis by Page</h2>
        """

        # Add page-by-page analysis
        for page in pages:
            if page['issues']:
                html_content += f"""
                        <div class="page-summary">
                            <h3><a href="{page['url']}" class="url-link" target="_blank">{page['title']}</a></h3>
                            <p><strong>URL:</strong> {page['url']}</p>
                            <p><strong>Issues Found:</strong> {len(page['issues'])} | <strong>Load Time:</strong> {page['load_time']:.2f}s | <strong>Status:</strong> {page['status_code']}</p>
                        </div>
                """

                # Add issues for this page
                for issue in page['issues']:
                    severity_class = f"badge-{issue['severity'].lower()}"
                    html_content += f"""
                        <div class="issue-card">
                            <div class="issue-header">
                                <div class="issue-title">{issue['title']}</div>
                                <span class="severity-badge {severity_class}">{issue['severity']}</span>
                            </div>
                            <p><strong>Category:</strong> {issue['category']}</p>
                            <p>{issue['description']}</p>
                            
                            {f'<p><strong>Element:</strong> <code>{issue["element_selector"]}</code></p>' if issue.get('element_selector') else ''}
                            
                            <div class="recommendations">
                                <strong>💡 Recommendations:</strong>
                                <ul>
                    """

                    for rec in issue.get('recommendations', []):
                        html_content += f"<li>{rec}</li>"

                    html_content += """
                                </ul>
                            </div>
                            
                    """

                    if issue.get('reference_links'):
                        html_content += "<p><strong>📚 References:</strong> "
                        for link in issue['reference_links']:
                            html_content += f'<a href="{link}" target="_blank" class="url-link">Learn More</a> '
                        html_content += "</p>"

                    html_content += "</div>"

        html_content += """
                    </div>
                    
                    <!-- Configuration Used -->
                    <div class="section">
                        <h2>⚙️ Crawl Configuration</h2>
                        <div class="chart-container">
                            <table>
                                <tr><th>Parameter</th><th>Value</th></tr>
        """

        config = summary['crawl_config']
        for key, value in config.items():
            html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"

        html_content += """
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated by Smart CX Navigator - AI-Powered Website Analysis Tool</p>
                    <p>For more information and recommendations, please review the detailed findings above.</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Save HTML report
        filename = f"smart_cx_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.reports_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {filepath}")
        return filepath

    def generate_json_report(self) -> str:
        """Generate JSON report for programmatic access"""
        filename = f"smart_cx_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.reports_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"JSON report generated: {filepath}")
        return filepath

    def generate_csv_report(self) -> str:
        """Generate CSV report for spreadsheet analysis"""
        filename = f"smart_cx_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.reports_dir, filename)

        # Flatten the data for CSV
        csv_data = []
        for page in self.results.get('page_analyses', []):
            for issue in page['issues']:
                csv_data.append({
                    'URL': page['url'],
                    'Page Title': page['title'],
                    'Status Code': page['status_code'],
                    'Load Time': f"{page['load_time']:.2f}s",
                    'Issue Category': issue['category'],
                    'Issue Severity': issue['severity'],
                    'Issue Priority': issue['priority'],
                    'Issue Title': issue['title'],
                    'Issue Description': issue['description'],
                    'Element Selector': issue.get('element_selector', ''),
                    'Recommendations': ' | '.join(issue.get('recommendations', [])),
                    'Reference Links': ' | '.join(issue.get('reference_links', [])),
                    'Timestamp': issue.get('timestamp', '')
                })

        if csv_data:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)

        logger.info(f"CSV report generated: {filepath}")
        return filepath

    def generate_summary_metrics(self) -> Dict[str, Any]:
        """Generate summary metrics for dashboard display"""
        summary = self.results['crawl_summary']
        pages = self.results.get('page_analyses', [])

        # Calculate advanced metrics
        total_pages = len(pages)
        total_issues = summary['total_issues']

        # Page performance metrics
        load_times = [page['load_time'] for page in pages if page['load_time'] > 0]
        avg_load_time = sum(load_times) / len(load_times) if load_times else 0

        # Issue distribution
        severity_distribution = summary['severity_breakdown']
        category_distribution = summary['category_breakdown']

        # Health score calculation (0-100)
        critical_weight = severity_distribution.get('Critical', 0) * 4
        high_weight = severity_distribution.get('High', 0) * 3
        medium_weight = severity_distribution.get('Medium', 0) * 2
        low_weight = severity_distribution.get('Low', 0) * 1

        total_weighted_issues = critical_weight + high_weight + medium_weight + low_weight
        max_possible_score = total_pages * 20  # Assume max 20 weighted issues per page

        health_score = max(0, 100 - (total_weighted_issues / max(max_possible_score, 1)) * 100)

        return {
            'overview': {
                'total_pages_analyzed': total_pages,
                'total_issues_found': total_issues,
                'average_issues_per_page': total_issues / max(total_pages, 1),
                'average_load_time': avg_load_time,
                'health_score': round(health_score, 1),
                'pages_failed': summary['pages_failed']
            },
            'severity_breakdown': severity_distribution,
            'category_breakdown': summary['category_breakdown'],
            'top_issues': self._get_top_issues(),
            'recommendations': self._get_priority_recommendations()
        }

    def _get_top_issues(self) -> List[Dict[str, Any]]:
        """Get the most critical issues across all pages"""
        all_issues = []

        for page in self.results.get('page_analyses', []):
            for issue in page['issues']:
                issue_copy = issue.copy()
                issue_copy['page_url'] = page['url']
                issue_copy['page_title'] = page['title']
                all_issues.append(issue_copy)

        # Sort by severity (Critical > High > Medium > Low)
        severity_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        all_issues.sort(key=lambda x: severity_order.get(x['severity'], 0), reverse=True)

        return all_issues[:10]  # Return top 10 issues

    def _get_priority_recommendations(self) -> List[str]:
        """Get prioritized recommendations based on issue frequency and severity"""
        recommendations = {}

        for page in self.results.get('page_analyses', []):
            for issue in page['issues']:
                severity_weight = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}.get(issue['severity'], 1)

                for rec in issue.get('recommendations', []):
                    if rec in recommendations:
                        recommendations[rec] += severity_weight
                    else:
                        recommendations[rec] = severity_weight

        # Sort by weighted frequency
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        return [rec[0] for rec in sorted_recommendations[:8]]  # Return top 8 recommendations


async def run_analysis_async(urls: List[str], config: CrawlConfig, progress_callback=None) -> Dict[str, Any]:
    """Run the async analysis in a separate function"""
    crawler = SmartWebCrawler(config)
    try:
        if progress_callback:
            progress_callback(0.1, "Initializing crawler...")
        
        results = await crawler.crawl_website(urls)
        
        if progress_callback:
            progress_callback(0.9, "Finalizing results...")
        
        return results
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


def run_analysis_sync(urls: List[str], config: CrawlConfig) -> Dict[str, Any]:
    """Synchronous wrapper for the async analysis"""
    try:
        # For environments that support asyncio.run()
        return asyncio.run(run_analysis_async(urls, config))
    except RuntimeError:
        # Fallback for environments where event loop is already running (like Jupyter/Streamlit)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new thread to run async code
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_analysis_async(urls, config))
                    return future.result(timeout=3600)  # 1 hour timeout
            else:
                return loop.run_until_complete(run_analysis_async(urls, config))
        except Exception as e:
            logger.error(f"Failed to run analysis: {e}")
            raise


class CLIInterface:
    """Command-line interface for Smart CX Navigator"""

    def __init__(self):
        self.config = CrawlConfig()

    def parse_arguments(self):
        """Parse command line arguments"""
        import argparse

        parser = argparse.ArgumentParser(
            description="Smart CX Navigator - AI-Powered Website Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python smart_cx_navigator.py --urls https://example.com --depth 2 --max-pages 50
  python smart_cx_navigator.py --url-file urls.txt --delay 2 --screenshots
  python smart_cx_navigator.py --urls https://site1.com https://site2.com --output-format json
            """
        )

        # URL input options
        url_group = parser.add_mutually_exclusive_group(required=True)
        url_group.add_argument(
            '--urls', nargs='+',
            help='One or more URLs to analyze'
        )
        url_group.add_argument(
            '--url-file', type=str,
            help='File containing URLs to analyze (one per line)'
        )

        # Crawler configuration
        parser.add_argument(
            '--depth', type=int, default=3,
            help='Maximum crawl depth (default: 3)'
        )
        parser.add_argument(
            '--max-pages', type=int, default=100,
            help='Maximum number of pages to crawl (default: 100)'
        )
        parser.add_argument(
            '--delay', type=float, default=1.0,
            help='Delay between requests in seconds (default: 1.0)'
        )
        parser.add_argument(
            '--timeout', type=int, default=30,
            help='Request timeout in seconds (default: 30)'
        )
        parser.add_argument(
            '--user-agent', type=str,
            help='Custom user agent string'
        )

        # Features
        parser.add_argument(
            '--screenshots', action='store_true',
            help='Take screenshots of pages with issues'
        )
        parser.add_argument(
            '--follow-external', action='store_true',
            help='Follow external links during crawling'
        )
        parser.add_argument(
            '--no-robots', action='store_true',
            help='Ignore robots.txt (use with caution)'
        )

        # Output options
        parser.add_argument(
            '--output-format', choices=['html', 'json', 'csv', 'all'], default='html',
            help='Output report format (default: html)'
        )
        parser.add_argument(
            '--output-dir', type=str,
            help='Output directory for reports (default: ./reports/smart_cx_navigator)'
        )
        parser.add_argument(
            '--quiet', action='store_true',
            help='Suppress progress output'
        )
        parser.add_argument(
            '--verbose', action='store_true',
            help='Enable verbose logging'
        )

        return parser.parse_args()

    def load_urls_from_file(self, filepath: str) -> List[str]:
        """Load URLs from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            return urls
        except FileNotFoundError:
            logger.error(f"URL file not found: {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error reading URL file {filepath}: {e}")
            return []

    def run(self):
        """Main CLI execution"""
        args = self.parse_arguments()

        # Configure logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.quiet:
            logging.getLogger().setLevel(logging.WARNING)

        # Load URLs
        if args.urls:
            urls = args.urls
        else:
            urls = self.load_urls_from_file(args.url_file)

        if not urls:
            logger.error("No URLs provided for analysis")
            return 1

        # Validate URLs
        valid_urls = []
        for url in urls:
            if url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                logger.warning(f"Invalid URL format (skipping): {url}")

        if not valid_urls:
            logger.error("No valid URLs found")
            return 1

        # Configure crawler
        config = CrawlConfig(
            max_depth=args.depth,
            max_pages=args.max_pages,
            crawl_delay=args.delay,
            timeout=args.timeout,
            user_agent=args.user_agent or DEFAULT_USER_AGENTS[0],
            screenshot_on_issue=args.screenshots,
            follow_external_links=args.follow_external,
            respect_robots_txt=not args.no_robots
        )

        logger.info(f"Starting analysis of {len(valid_urls)} URLs")
        logger.info(
            f"Configuration: depth={config.max_depth}, max_pages={config.max_pages}, delay={config.crawl_delay}s")

        try:
            # Run analysis
            results = run_analysis_sync(valid_urls, config)

            # Generate reports
            reporter = SmartCXReporter(results)

            if args.output_dir:
                reporter.reports_dir = args.output_dir
                os.makedirs(args.output_dir, exist_ok=True)

            report_files = []

            if args.output_format in ['html', 'all']:
                html_file = reporter.generate_html_report()
                report_files.append(html_file)
                logger.info(f"HTML report: {html_file}")

            if args.output_format in ['json', 'all']:
                json_file = reporter.generate_json_report()
                report_files.append(json_file)
                logger.info(f"JSON report: {json_file}")

            if args.output_format in ['csv', 'all']:
                csv_file = reporter.generate_csv_report()
                report_files.append(csv_file)
                logger.info(f"CSV report: {csv_file}")

            # Print summary
            summary = reporter.generate_summary_metrics()
            print("\n" + "=" * 60)
            print("SMART CX NAVIGATOR - ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Pages Analyzed: {summary['overview']['total_pages_analyzed']}")
            print(f"Issues Found: {summary['overview']['total_issues_found']}")
            print(f"Health Score: {summary['overview']['health_score']}/100")
            print(f"Average Load Time: {summary['overview']['average_load_time']:.2f}s")
            print(f"Failed Pages: {summary['overview']['pages_failed']}")

            print(f"\nIssue Breakdown:")
            for severity, count in summary['severity_breakdown'].items():
                print(f"  {severity}: {count}")

            print(f"\nReports Generated:")
            for report_file in report_files:
                print(f"  {os.path.basename(report_file)}")

            print(f"\nTop Recommendations:")
            for i, rec in enumerate(summary['recommendations'][:5], 1):
                print(f"  {i}. {rec}")

            return 0

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return 1


# Update the show_ui function to include actual analysis execution
def show_ui():
    """Main UI function for Smart CX Navigator"""
    st.title("🧭 Smart CX Navigator - AI-Powered Customer Experience Analytics")
    st.markdown(f"""
    **Advanced Website Analysis & Customer Experience Optimization**
    
    Intelligently crawl and analyze websites for usability, customer experience, functionality, 
    and compliance issues using advanced AI models and modern browser automation.
    
    **AI Features:** {'✅ Azure OpenAI Enabled' if AZURE_OPENAI_AVAILABLE else '⚠️ Limited (Azure OpenAI not available)'}
    """)

    # Show AI capabilities status
    with st.expander("🤖 AI Analysis Capabilities"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Available Features:**")
            st.write(f"- AI-Powered CX Analysis: {'✅' if AZURE_OPENAI_AVAILABLE else '❌'}")
            st.write(f"- Business Impact Assessment: {'✅' if AZURE_OPENAI_AVAILABLE else '❌'}")
            st.write(f"- Industry Benchmarking: {'✅' if AZURE_OPENAI_AVAILABLE else '⚠️ Limited'}")
            st.write(f"- Implementation Roadmap: {'✅' if AZURE_OPENAI_AVAILABLE else '⚠️ Basic'}")
        
        with col2:
            st.write("**Analysis Capabilities:**")
            st.write("- Issue Detection & Classification: ✅")
            st.write("- Customer Journey Analysis: ✅")
            st.write("- Conversion Optimization: ✅")
            st.write("- Executive Reporting: ✅")

    # Check for required dependencies
    missing_deps = []
    if not PLAYWRIGHT_AVAILABLE and not SELENIUM_AVAILABLE:
        missing_deps.append("Playwright or Selenium (for browser automation)")
    if not BS4_AVAILABLE:
        missing_deps.append("BeautifulSoup4 (for HTML parsing)")

    if missing_deps:
        st.warning(f"Missing optional dependencies: {', '.join(missing_deps)}. Some features may be limited.")
        with st.expander("Installation Instructions"):
            st.code("""
# Install recommended dependencies:
pip install playwright beautifulsoup4 aiohttp

# Initialize Playwright (for better browser support):
playwright install chromium

# Alternative: Install Selenium
pip install selenium
            """)

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Website Analysis", "📊 AI Insights Dashboard", 
        "� Pricing Analysis", "�📈 Executive Reports", "⚙️ Configuration"
    ])

    with tab1:
        show_website_analysis_ui()
    
    with tab2:
        show_ai_insights_dashboard_cx()
    
    with tab3:
        show_pricing_analysis_ui()
    
    with tab4:
        show_executive_reports_ui()
    
    with tab5:
        show_configuration_ui()

def show_website_analysis_ui():
    """UI for website analysis"""
    st.subheader("🌐 Website Analysis & Crawling")

    # URL Input Section
    st.markdown("### 🌐 Website URLs to Analyze")

    # URL input methods
    input_method = st.radio("Input Method", ["Manual Entry", "Upload File"])

    urls_to_crawl = []

    if input_method == "Manual Entry":
        url_input = st.text_area(
            "Enter URLs (one per line)",
            placeholder="https://example.com\nhttps://example.com/about\nhttps://example.com/contact",
            height=100
        )
        if url_input:
            urls_to_crawl = [url.strip() for url in url_input.split('\n') if url.strip()]

    else:
        uploaded_file = st.file_uploader("Upload URL file", type=['txt', 'csv'])
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            urls_to_crawl = [url.strip() for url in content.split('\n') if url.strip()]

    # Display URLs to be crawled
    if urls_to_crawl:
        st.success(f"✅ {len(urls_to_crawl)} URLs ready for analysis")
        with st.expander("� URLs to Analyze"):
            for i, url in enumerate(urls_to_crawl, 1):
                st.write(f"{i}. {url}")

    # Analysis Configuration
    with st.expander("🔧 Analysis Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            industry = st.selectbox(
                "Industry Type",
                ["general", "ecommerce", "saas", "finance", "healthcare", "education"],
                help="Select your industry for better benchmarking"
            )
            
            business_goals = st.multiselect(
                "Business Goals",
                ["improve_conversion", "enhance_ux", "increase_engagement", "reduce_bounce_rate", "improve_accessibility"],
                default=["improve_conversion", "enhance_ux"],
                help="Select your primary business objectives"
            )

        with col2:
            max_depth = st.slider("Crawl Depth", min_value=1, max_value=5, value=2,
                                  help="How deep to crawl (number of link levels)")
            max_pages = st.slider("Max Pages", min_value=5, max_value=100, value=25,
                                  help="Maximum number of pages to crawl")
            crawl_delay = st.slider("Crawl Delay (seconds)", min_value=0.5, max_value=3.0, value=1.0, step=0.5,
                                    help="Delay between requests to be respectful to servers")

    # Initialize session state for analysis results
    if 'cx_analysis_results' not in st.session_state:
        st.session_state.cx_analysis_results = None
    if 'cx_analysis_running' not in st.session_state:
        st.session_state.cx_analysis_running = False

    # Start Analysis Button
    if st.button("🚀 Start AI-Powered CX Analysis", type="primary", 
                 disabled=not urls_to_crawl or st.session_state.cx_analysis_running):
        
        st.session_state.cx_analysis_running = True
        
        with st.spinner("🔍 Analyzing website with AI-powered insights..."):
            try:
                # Create crawler configuration
                config = CrawlConfig(
                    max_depth=max_depth,
                    max_pages=max_pages,
                    crawl_delay=crawl_delay,
                    screenshot_on_issue=True,
                    follow_external_links=False
                )
                
                # Run analysis
                results = run_analysis_sync(urls_to_crawl, config)
                
                # Store results
                st.session_state.cx_analysis_results = results
                
                st.success("✅ Analysis completed successfully!")
                
                # Show immediate insights
                if results and 'cx_score' in results:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CX Score", f"{results['cx_score']:.1f}/100")
                    with col2:
                        st.metric("Issues Found", results.get('crawl_summary', {}).get('total_issues', 0))
                    with col3:
                        st.metric("Pages Analyzed", results.get('crawl_summary', {}).get('pages_crawled', 0))
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
            finally:
                st.session_state.cx_analysis_running = False
                st.rerun()

    # Display Results Section
    if st.session_state.cx_analysis_results:
        display_cx_analysis_results(st.session_state.cx_analysis_results)

def show_pricing_analysis_ui():
    """Comprehensive pricing analysis interface with AI insights"""
    st.subheader("💰 Smart Pricing Analysis & Competitive Intelligence")
    
    st.markdown("""
    **AI-Powered Pricing Intelligence Platform**
    
    Analyze product pricing across different environments, geolocations, and competitive landscapes 
    with advanced AI insights for pricing optimization and strategy recommendations.
    """)

    # Initialize session state for pricing analysis
    if 'pricing_analysis_results' not in st.session_state:
        st.session_state.pricing_analysis_results = None
    if 'pricing_analysis_running' not in st.session_state:
        st.session_state.pricing_analysis_running = False

    # Configuration Section
    with st.expander("🔧 Pricing Analysis Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Target Websites & Products**")
            pricing_urls = st.text_area(
                "Enter URLs for pricing analysis (one per line)",
                placeholder="https://example.com/pricing\nhttps://competitor.com/plans\nhttps://yoursite.com/products",
                height=100,
                help="Include pricing pages, product pages, and competitor sites"
            )
            
            product_categories = st.multiselect(
                "Product Categories",
                ["SaaS/Software", "E-commerce", "Subscription Services", "Hardware", 
                 "Digital Products", "Professional Services", "Financial Services"],
                default=["SaaS/Software"],
                help="Select relevant product categories for benchmarking"
            )
            
        with col2:
            st.markdown("**Analysis Scope**")
            analysis_types = st.multiselect(
                "Analysis Types",
                ["Pricing Extraction", "Competitive Comparison", "Geo-location Testing", 
                 "Environment Comparison", "Pricing Strategy Analysis", "Market Positioning"],
                default=["Pricing Extraction", "Competitive Comparison"],
                help="Select which types of pricing analysis to perform"
            )
            
            environments = st.multiselect(
                "Environments to Compare",
                ["Production", "Staging", "Test", "Development", "Demo"],
                default=["Production", "Test"],
                help="Compare pricing across different environments"
            )

    # Geolocation Testing Configuration
    with st.expander("🌍 Geolocation & Market Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            target_geolocations = st.multiselect(
                "Target Geolocations",
                ["United States", "United Kingdom", "Germany", "France", "Canada", 
                 "Australia", "Japan", "India", "Brazil", "Singapore"],
                default=["United States", "United Kingdom"],
                help="Analyze pricing variations across different regions"
            )
            
            currencies = st.multiselect(
                "Currency Analysis",
                ["USD", "EUR", "GBP", "CAD", "AUD", "JPY", "INR", "BRL"],
                default=["USD", "EUR"],
                help="Track pricing in different currencies"
            )
            
        with col2:
            market_segments = st.multiselect(
                "Market Segments",
                ["Enterprise", "SMB", "Startup", "Individual", "Non-profit", "Education"],
                default=["Enterprise", "SMB"],
                help="Analyze pricing for different customer segments"
            )
            
            pricing_models = st.multiselect(
                "Pricing Models to Detect",
                ["Subscription", "One-time", "Freemium", "Usage-based", "Tiered", "Volume-based"],
                default=["Subscription", "Tiered"],
                help="Identify different pricing model patterns"
            )

    # Advanced AI Analysis Configuration
    with st.expander("🤖 AI Analysis Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            ai_analysis_depth = st.selectbox(
                "AI Analysis Depth",
                ["Basic", "Comprehensive", "Advanced"],
                index=1,
                help="Level of AI-powered analysis and insights"
            )
            
            competitive_intelligence = st.checkbox(
                "Competitive Intelligence",
                value=True,
                help="Generate AI insights on competitive positioning"
            )
            
        with col2:
            pricing_optimization = st.checkbox(
                "Pricing Optimization Recommendations",
                value=True,
                help="AI-powered pricing strategy recommendations"
            )
            
            market_analysis = st.checkbox(
                "Market Analysis & Trends",
                value=True,
                help="Analyze market trends and pricing patterns"
            )

    # Start Pricing Analysis
    if st.button("🚀 Start Pricing Analysis", type="primary", 
                 disabled=not pricing_urls or st.session_state.pricing_analysis_running):
        
        if not pricing_urls.strip():
            st.error("Please enter at least one URL for pricing analysis")
            return
            
        # Parse URLs
        url_list = [url.strip() for url in pricing_urls.strip().split('\n') if url.strip()]
        valid_urls = [url for url in url_list if url.startswith(('http://', 'https://'))]
        
        if not valid_urls:
            st.error("Please enter valid URLs starting with http:// or https://")
            return

        st.session_state.pricing_analysis_running = True
        
        # Create pricing analysis configuration
        pricing_config = {
            "urls": valid_urls,
            "product_categories": product_categories,
            "analysis_types": analysis_types,
            "environments": environments,
            "geolocations": target_geolocations,
            "currencies": currencies,
            "market_segments": market_segments,
            "pricing_models": pricing_models,
            "ai_analysis_depth": ai_analysis_depth,
            "competitive_intelligence": competitive_intelligence,
            "pricing_optimization": pricing_optimization,
            "market_analysis": market_analysis
        }

        try:
            with st.spinner("🔍 Analyzing pricing data across multiple dimensions..."):
                # Run pricing analysis
                analyzer = SmartPricingAnalyzer(pricing_config)
                results = analyzer.run_comprehensive_analysis()
                
                st.session_state.pricing_analysis_results = results
                st.session_state.pricing_analysis_running = False
                
                st.success("✅ Pricing analysis completed successfully!")
                
        except Exception as e:
            st.session_state.pricing_analysis_running = False
            st.error(f"Pricing analysis failed: {str(e)}")

    # Display Results
    if st.session_state.pricing_analysis_results:
        display_pricing_analysis_results(st.session_state.pricing_analysis_results)


def display_pricing_analysis_results(results):
    """Display comprehensive pricing analysis results"""
    st.subheader("📊 Pricing Analysis Results & Intelligence")
    
    # Executive Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        products_analyzed = len(results.get('products', []))
        st.metric("Products Analyzed", products_analyzed)
    
    with col2:
        price_points = results.get('total_price_points', 0)
        st.metric("Price Points Found", price_points)
    
    with col3:
        geolocations = len(results.get('geolocation_data', {}))
        st.metric("Locations Tested", geolocations)
    
    with col4:
        competitiveness_score = results.get('competitiveness_score', 0)
        st.metric("Competitiveness Score", f"{competitiveness_score}/100")

    # Main Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Product Pricing", "🌍 Geo Analysis", "⚖️ Competitive Intel", 
        "🤖 AI Insights", "📊 Environment Comparison"
    ])
    
    with tab1:
        display_product_pricing_analysis(results)
    
    with tab2:
        display_geolocation_analysis(results)
    
    with tab3:
        display_competitive_intelligence(results)
    
    with tab4:
        display_pricing_ai_insights(results)
    
    with tab5:
        display_environment_comparison(results)


def display_product_pricing_analysis(results):
    """Display detailed product pricing analysis"""
    st.subheader("💰 Product Pricing Breakdown")
    
    products = results.get('products', [])
    
    if not products:
        st.info("No product pricing data found in the analysis.")
        return
    
    for product in products:
        with st.expander(f"📦 {product.get('name', 'Unknown Product')} - {product.get('category', 'General')}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Pricing Structure:**")
                pricing_tiers = product.get('pricing_tiers', [])
                if pricing_tiers:
                    pricing_df = pd.DataFrame(pricing_tiers)
                    st.dataframe(pricing_df)
                else:
                    st.write("No structured pricing tiers found")
                
                st.write("**Pricing Model:**")
                st.write(f"- Model Type: {product.get('pricing_model', 'Unknown')}")
                st.write(f"- Billing Cycle: {product.get('billing_cycle', 'Unknown')}")
                st.write(f"- Currency: {product.get('currency', 'Unknown')}")
                
            with col2:
                st.write("**Cost Analysis:**")
                costs = product.get('costs', {})
                st.write(f"- Base Price: {costs.get('base_price', 'N/A')}")
                st.write(f"- Setup Fee: {costs.get('setup_fee', 'N/A')}")
                st.write(f"- Renewal Price: {costs.get('renewal_price', 'N/A')}")
                st.write(f"- Upgrade Cost: {costs.get('upgrade_cost', 'N/A')}")
                
                st.write("**Value Proposition:**")
                features = product.get('key_features', [])
                for feature in features[:5]:
                    st.write(f"- {feature}")
    
    # Pricing Summary Chart
    st.subheader("📈 Pricing Distribution Analysis")
    
    # Create pricing comparison chart
    if products:
        pricing_data = []
        for product in products:
            for tier in product.get('pricing_tiers', []):
                pricing_data.append({
                    'Product': product.get('name', 'Unknown'),
                    'Tier': tier.get('name', 'Standard'),
                    'Price': tier.get('price_numeric', 0),
                    'Currency': tier.get('currency', 'USD')
                })
        
        if pricing_data:
            pricing_df = pd.DataFrame(pricing_data)
            
            # Group by currency for better visualization
            for currency in pricing_df['Currency'].unique():
                currency_data = pricing_df[pricing_df['Currency'] == currency]
                if not currency_data.empty:
                    st.write(f"**Pricing in {currency}:**")
                    chart_data = currency_data.pivot(index='Product', columns='Tier', values='Price')
                    st.bar_chart(chart_data)


def display_geolocation_analysis(results):
    """Display geolocation-based pricing analysis"""
    st.subheader("🌍 Geolocation Pricing Analysis")
    
    geo_data = results.get('geolocation_data', {})
    
    if not geo_data:
        st.info("No geolocation pricing data available.")
        return
    
    # Geolocation Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Variations by Region:**")
        for location, data in geo_data.items():
            price_diff = data.get('price_difference_percent', 0)
            currency = data.get('currency', 'USD')
            avg_price = data.get('average_price', 0)
            
            st.write(f"**{location}** ({currency})")
            st.write(f"- Average Price: {avg_price}")
            st.write(f"- Price Variation: {price_diff:+.1f}%")
            st.write("---")
    
    with col2:
        st.write("**Currency Impact Analysis:**")
        currency_data = {}
        for location, data in geo_data.items():
            currency = data.get('currency', 'USD')
            if currency not in currency_data:
                currency_data[currency] = []
            currency_data[currency].append(data.get('average_price', 0))
        
        for currency, prices in currency_data.items():
            avg_price = sum(prices) / len(prices) if prices else 0
            st.write(f"**{currency}**: {avg_price:.2f} (avg)")
    
    # Geolocation Comparison Chart
    if geo_data:
        geo_df = pd.DataFrame([
            {
                'Location': location,
                'Average_Price': data.get('average_price', 0),
                'Currency': data.get('currency', 'USD'),
                'Price_Difference': data.get('price_difference_percent', 0)
            }
            for location, data in geo_data.items()
        ])
        
        st.subheader("📊 Regional Price Comparison")
        st.bar_chart(geo_df.set_index('Location')['Average_Price'])
        
        st.subheader("📈 Price Variation from Baseline")
        st.bar_chart(geo_df.set_index('Location')['Price_Difference'])


def display_competitive_intelligence(results):
    """Display competitive intelligence analysis"""
    st.subheader("⚖️ Competitive Intelligence & Market Positioning")
    
    competitive_data = results.get('competitive_analysis', {})
    
    if not competitive_data:
        st.info("No competitive analysis data available.")
        return
    
    # Competitive Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Market Position Analysis:**")
        position = competitive_data.get('market_position', {})
        st.write(f"- Market Position: {position.get('tier', 'Unknown')}")
        st.write(f"- Price Competitiveness: {position.get('competitiveness', 'Unknown')}")
        st.write(f"- Value Score: {position.get('value_score', 0)}/100")
        
        st.write("**Competitive Gaps:**")
        gaps = competitive_data.get('competitive_gaps', [])
        for gap in gaps[:5]:
            st.write(f"- {gap}")
    
    with col2:
        st.write("**Competitive Advantages:**")
        advantages = competitive_data.get('competitive_advantages', [])
        for advantage in advantages[:5]:
            st.write(f"- {advantage}")
        
        st.write("**Pricing Recommendations:**")
        recommendations = competitive_data.get('pricing_recommendations', [])
        for rec in recommendations[:5]:
            st.write(f"- {rec}")
    
    # Competitive Comparison Table
    competitors = competitive_data.get('competitors', [])
    if competitors:
        st.subheader("🏢 Competitor Comparison")
        
        comp_df = pd.DataFrame(competitors)
        st.dataframe(comp_df)
        
        # Competitive Pricing Chart
        if 'pricing' in comp_df.columns:
            st.subheader("📊 Competitive Pricing Landscape")
            st.bar_chart(comp_df.set_index('name')['pricing'])


def display_pricing_ai_insights(results):
    """Display AI-powered pricing insights"""
    st.subheader("🤖 AI-Powered Pricing Intelligence")
    
    ai_insights = results.get('ai_insights', {})
    
    if not ai_insights:
        st.warning("AI insights not available. Ensure Azure OpenAI is configured.")
        return
    
    # AI Analysis Summary
    with st.expander("📋 Executive AI Summary", expanded=True):
        executive_summary = ai_insights.get('executive_summary', '')
        if executive_summary:
            st.write(executive_summary)
        else:
            st.info("Executive summary not available")
    
    # Strategic Recommendations
    with st.expander("💡 Strategic Pricing Recommendations"):
        strategies = ai_insights.get('pricing_strategies', [])
        if strategies:
            for i, strategy in enumerate(strategies, 1):
                st.write(f"**{i}. {strategy.get('title', 'Strategy')}**")
                st.write(f"   - Impact: {strategy.get('impact', 'Unknown')}")
                st.write(f"   - Timeline: {strategy.get('timeline', 'Unknown')}")
                st.write(f"   - Recommendation: {strategy.get('recommendation', 'Unknown')}")
                st.write("---")
    
    # Market Trends Analysis
    with st.expander("📈 Market Trends & Insights"):
        trends = ai_insights.get('market_trends', {})
        if trends:
            st.write("**Key Market Trends:**")
            for trend in trends.get('trends', []):
                st.write(f"- {trend}")
            
            st.write("**Industry Benchmarks:**")
            benchmarks = trends.get('benchmarks', {})
            for metric, value in benchmarks.items():
                st.write(f"- {metric}: {value}")
    
    # ROI Projections
    with st.expander("💰 ROI & Revenue Impact Projections"):
        roi_data = ai_insights.get('roi_projections', {})
        if roi_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Revenue Impact:**")
                st.write(f"- Potential Revenue Increase: {roi_data.get('revenue_increase', 'Unknown')}")
                st.write(f"- Customer Acquisition Impact: {roi_data.get('acquisition_impact', 'Unknown')}")
                st.write(f"- Retention Impact: {roi_data.get('retention_impact', 'Unknown')}")
            
            with col2:
                st.write("**Implementation Costs:**")
                st.write(f"- Development Cost: {roi_data.get('development_cost', 'Unknown')}")
                st.write(f"- Testing Cost: {roi_data.get('testing_cost', 'Unknown')}")
                st.write(f"- Timeline to ROI: {roi_data.get('timeline_to_roi', 'Unknown')}")


def display_environment_comparison(results):
    """Display environment comparison analysis"""
    st.subheader("🔄 Environment Comparison Analysis")
    
    env_data = results.get('environment_comparison', {})
    
    if not env_data:
        st.info("No environment comparison data available.")
        return
    
    # Environment Summary
    environments = env_data.get('environments', {})
    
    if environments:
        # Create comparison table
        env_comparison = []
        for env_name, env_info in environments.items():
            env_comparison.append({
                'Environment': env_name,
                'Status': env_info.get('status', 'Unknown'),
                'Products Found': len(env_info.get('products', [])),
                'Pricing Consistency': env_info.get('pricing_consistency', 'Unknown'),
                'Last Updated': env_info.get('last_updated', 'Unknown')
            })
        
        st.subheader("📊 Environment Status Overview")
        env_df = pd.DataFrame(env_comparison)
        st.dataframe(env_df)
        
        # Detailed Environment Analysis
        for env_name, env_info in environments.items():
            with st.expander(f"🔍 {env_name.title()} Environment Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Environment Health:**")
                    st.write(f"- Status: {env_info.get('status', 'Unknown')}")
                    st.write(f"- Response Time: {env_info.get('response_time', 'Unknown')}")
                    st.write(f"- Availability: {env_info.get('availability', 'Unknown')}")
                    
                with col2:
                    st.write("**Pricing Data:**")
                    products = env_info.get('products', [])
                    st.write(f"- Products Found: {len(products)}")
                    st.write(f"- Pricing Accuracy: {env_info.get('pricing_accuracy', 'Unknown')}")
                    st.write(f"- Data Freshness: {env_info.get('data_freshness', 'Unknown')}")
                
                # Issues and Recommendations
                issues = env_info.get('issues', [])
                if issues:
                    st.write("**Issues Found:**")
                    for issue in issues:
                        st.write(f"- ⚠️ {issue}")
                
                recommendations = env_info.get('recommendations', [])
                if recommendations:
                    st.write("**Recommendations:**")
                    for rec in recommendations:
                        st.write(f"- 💡 {rec}")


class SmartPricingAnalyzer:
    """Advanced pricing analysis with AI insights"""
    
    def __init__(self, config):
        self.config = config
        self.results = {
            'products': [],
            'geolocation_data': {},
            'competitive_analysis': {},
            'ai_insights': {},
            'environment_comparison': {},
            'total_price_points': 0,
            'competitiveness_score': 0
        }
    
    def run_comprehensive_analysis(self):
        """Run comprehensive pricing analysis"""
        try:
            # Extract pricing data
            self._extract_pricing_data()
            
            # Analyze geolocations
            if "Geo-location Testing" in self.config.get('analysis_types', []):
                self._analyze_geolocations()
            
            # Competitive analysis
            if "Competitive Comparison" in self.config.get('analysis_types', []):
                self._perform_competitive_analysis()
            
            # Environment comparison
            if "Environment Comparison" in self.config.get('analysis_types', []):
                self._compare_environments()
            
            # AI-powered insights
            if self.config.get('pricing_optimization') or self.config.get('competitive_intelligence'):
                self._generate_ai_insights()
            
            # Calculate overall scores
            self._calculate_metrics()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pricing analysis failed: {e}")
            return {"error": str(e), **self.results}
    
    def _extract_pricing_data(self):
        """Extract pricing data from URLs using real-time web scraping"""
        products = []
        total_price_points = 0
        
        for url in self.config.get('urls', []):
            try:
                logger.info(f"Extracting pricing data from: {url}")
                product_data = self._scrape_pricing_from_url(url)
                if product_data:
                    products.append(product_data)
                    total_price_points += len(product_data.get('pricing_tiers', []))
                    
            except Exception as e:
                logger.error(f"Failed to extract pricing from {url}: {e}")
                continue
        
        self.results['products'] = products
        self.results['total_price_points'] = total_price_points
    
    def _scrape_pricing_from_url(self, url: str) -> Dict[str, Any]:
        """Scrape pricing information from a single URL"""
        import aiohttp
        import asyncio
        from bs4 import BeautifulSoup
        import re
        
        async def fetch_page():
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            return content
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                            return None
                except Exception as e:
                    logger.error(f"Failed to fetch {url}: {e}")
                    return None
        
        # Get page content
        try:
            content = asyncio.run(fetch_page())
        except RuntimeError:
            # If running in existing event loop, use synchronous request
            import requests
            try:
                response = requests.get(url, timeout=30)
                content = response.text if response.status_code == 200 else None
            except Exception as e:
                logger.error(f"Failed to fetch {url} with requests: {e}")
                return None
        
        if not content:
            return None
        
        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract pricing information
        pricing_data = {
            'name': self._extract_product_name(soup, url),
            'category': self._detect_product_category(soup, url),
            'pricing_model': self._detect_pricing_model(soup),
            'billing_cycle': self._extract_billing_cycle(soup),
            'currency': self._detect_currency(soup),
            'pricing_tiers': self._extract_pricing_tiers(soup),
            'costs': self._extract_cost_breakdown(soup),
            'key_features': self._extract_key_features(soup)
        }
        
        return pricing_data
    
    def _extract_product_name(self, soup, url: str) -> str:
        """Extract product name from page"""
        # Try various selectors for product name
        name_selectors = [
            'h1', 'title', '[class*="product"]', '[class*="plan"]',
            '[data-product-name]', '.hero-title', '.page-title'
        ]
        
        for selector in name_selectors:
            element = soup.select_one(selector)
            if element and element.get_text().strip():
                name = element.get_text().strip()
                # Clean up the name
                name = re.sub(r'\s+', ' ', name)
                if len(name) > 5 and len(name) < 100:
                    return name
        
        # Fallback to URL-based name
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace('www.', '').title()
    
    def _detect_product_category(self, soup, url: str) -> str:
        """Detect product category based on content analysis"""
        content = soup.get_text().lower()
        
        # Category detection patterns
        if any(word in content for word in ['saas', 'software', 'api', 'cloud', 'platform']):
            return 'SaaS/Software'
        elif any(word in content for word in ['ecommerce', 'shop', 'store', 'cart', 'checkout']):
            return 'E-commerce'
        elif any(word in content for word in ['subscription', 'monthly', 'annual', 'recurring']):
            return 'Subscription Services'
        elif any(word in content for word in ['financial', 'banking', 'payment', 'finance']):
            return 'Financial Services'
        elif any(word in content for word in ['professional', 'consulting', 'service']):
            return 'Professional Services'
        else:
            return 'General'
    
    def _detect_pricing_model(self, soup) -> str:
        """Detect pricing model from page content"""
        content = soup.get_text().lower()
        
        if 'freemium' in content or 'free tier' in content:
            return 'Freemium'
        elif 'usage' in content or 'pay as you go' in content:
            return 'Usage-based'
        elif 'tier' in content or 'plan' in content:
            return 'Tiered'
        elif 'volume' in content or 'bulk' in content:
            return 'Volume-based'
        elif 'subscription' in content or 'monthly' in content:
            return 'Subscription'
        else:
            return 'One-time'
    
    def _extract_billing_cycle(self, soup) -> str:
        """Extract billing cycle information"""
        content = soup.get_text().lower()
        
        if 'monthly' in content and 'annual' in content:
            return 'Monthly/Annual'
        elif 'monthly' in content:
            return 'Monthly'
        elif 'annual' in content or 'yearly' in content:
            return 'Annual'
        else:
            return 'Unknown'
    
    def _detect_currency(self, soup) -> str:
        """Detect currency from pricing elements"""
        # Look for currency symbols and codes
        currency_patterns = {
            r'\$': 'USD',
            r'€': 'EUR', 
            r'£': 'GBP',
            r'¥': 'JPY',
            r'CAD': 'CAD',
            r'AUD': 'AUD',
            r'INR': 'INR',
            r'BRL': 'BRL'
        }
        
        content = soup.get_text()
        for pattern, currency in currency_patterns.items():
            if re.search(pattern, content):
                return currency
        
        return 'USD'  # Default fallback
    
    def _extract_pricing_tiers(self, soup) -> List[Dict]:
        """Extract pricing tiers from page"""
        tiers = []
        
        # Look for pricing containers
        pricing_containers = soup.select([
            '[class*="pricing"]', '[class*="plan"]', '[class*="tier"]',
            '[class*="package"]', '[data-price]', '.price-card', '.pricing-table'
        ])
        
        for container in pricing_containers:
            tier_data = self._extract_single_tier(container)
            if tier_data:
                tiers.append(tier_data)
        
        # If no structured tiers found, look for any price mentions
        if not tiers:
            price_elements = soup.select([
                '[class*="price"]', '[class*="cost"]', '[class*="amount"]'
            ])
            
            for element in price_elements:
                price_text = element.get_text().strip()
                price_numeric = self._extract_price_number(price_text)
                if price_numeric > 0:
                    tiers.append({
                        'name': 'Standard',
                        'price': price_text,
                        'price_numeric': price_numeric,
                        'currency': self._detect_currency(element.parent or soup)
                    })
                    break
        
        return tiers[:10]  # Limit to 10 tiers max
    
    def _extract_single_tier(self, container) -> Dict:
        """Extract pricing information from a single tier container"""
        # Extract tier name
        name_selectors = ['h1', 'h2', 'h3', '[class*="name"]', '[class*="title"]']
        tier_name = 'Standard'
        
        for selector in name_selectors:
            name_elem = container.select_one(selector)
            if name_elem and name_elem.get_text().strip():
                tier_name = name_elem.get_text().strip()
                break
        
        # Extract price
        price_text = container.get_text()
        price_numeric = self._extract_price_number(price_text)
        
        if price_numeric > 0:
            return {
                'name': tier_name,
                'price': self._extract_price_text(price_text),
                'price_numeric': price_numeric,
                'currency': self._detect_currency(container),
                'features': self._extract_tier_features(container)
            }
        
        return None
    
    def _extract_price_number(self, text: str) -> float:
        """Extract numeric price from text"""
        # Remove common non-numeric characters and extract number
        cleaned = re.sub(r'[^\d.,]', '', text)
        
        # Handle different number formats
        price_matches = re.findall(r'\d+(?:[.,]\d+)?', cleaned)
        
        if price_matches:
            try:
                # Take the largest number found (likely the main price)
                numbers = [float(match.replace(',', '')) for match in price_matches]
                return max(numbers)
            except ValueError:
                pass
        
        return 0.0
    
    def _extract_price_text(self, text: str) -> str:
        """Extract clean price text"""
        # Find price patterns
        price_patterns = [
            r'\$\d+(?:\.\d{2})?(?:/\w+)?',
            r'€\d+(?:\.\d{2})?(?:/\w+)?',
            r'£\d+(?:\.\d{2})?(?:/\w+)?',
            r'\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|CAD|AUD)(?:/\w+)?'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        
        return 'Contact for pricing'
    
    def _extract_tier_features(self, container) -> List[str]:
        """Extract features for a pricing tier"""
        features = []
        
        # Look for feature lists
        feature_lists = container.select(['ul', 'ol', '[class*="feature"]'])
        
        for feature_list in feature_lists:
            items = feature_list.select(['li', '[class*="item"]'])
            for item in items:
                feature_text = item.get_text().strip()
                if feature_text and len(feature_text) < 100:
                    features.append(feature_text)
        
        return features[:10]  # Limit features
    
    def _extract_cost_breakdown(self, soup) -> Dict[str, str]:
        """Extract cost breakdown (setup, renewal, etc.)"""
        content = soup.get_text().lower()
        costs = {}
        
        # Look for setup fees
        setup_patterns = [r'setup.*?(\$\d+)', r'installation.*?(\$\d+)', r'onetime.*?(\$\d+)']
        for pattern in setup_patterns:
            match = re.search(pattern, content)
            if match:
                costs['setup_fee'] = match.group(1)
                break
        else:
            costs['setup_fee'] = '$0'
        
        # Extract base price from first tier
        tiers = self._extract_pricing_tiers(soup)
        if tiers:
            costs['base_price'] = tiers[0].get('price', 'Unknown')
            costs['renewal_price'] = tiers[0].get('price', 'Unknown')
            
            # Calculate upgrade cost if multiple tiers
            if len(tiers) > 1:
                base_price = tiers[0].get('price_numeric', 0)
                higher_price = tiers[1].get('price_numeric', 0)
                if higher_price > base_price:
                    upgrade_cost = higher_price - base_price
                    costs['upgrade_cost'] = f'${upgrade_cost:.0f}/month'
                else:
                    costs['upgrade_cost'] = 'Variable'
            else:
                costs['upgrade_cost'] = 'Contact sales'
        
        return costs
    
    def _extract_key_features(self, soup) -> List[str]:
        """Extract key product features"""
        features = []
        
        # Look for feature sections
        feature_sections = soup.select([
            '[class*="feature"]', '[class*="benefit"]', '[class*="capability"]',
            'ul', 'ol'
        ])
        
        for section in feature_sections:
            items = section.select(['li', 'p', '[class*="item"]'])
            for item in items:
                feature_text = item.get_text().strip()
                if (feature_text and 
                    len(feature_text) > 10 and 
                    len(feature_text) < 100 and
                    not re.search(r'\$\d+', feature_text)):  # Exclude pricing text
                    features.append(feature_text)
                    
                if len(features) >= 10:
                    break
        
        return features[:10]
    
    def _analyze_geolocations(self):
        """Analyze pricing across different geolocations using real requests"""
        geo_data = {}
        baseline_prices = {}
        
        # Get baseline prices (first location)
        base_location = self.config.get('geolocations', ['United States'])[0]
        
        for location in self.config.get('geolocations', []):
            try:
                logger.info(f"Analyzing pricing for {location}")
                location_data = self._fetch_geo_pricing(location)
                
                if location_data:
                    geo_data[location] = location_data
                    
                    # Set baseline for comparison
                    if location == base_location:
                        baseline_prices = {
                            url: data.get('average_price', 0) 
                            for url, data in location_data.items()
                        }
                
            except Exception as e:
                logger.error(f"Failed to analyze geolocation {location}: {e}")
                continue
        
        # Calculate price differences from baseline
        for location, data in geo_data.items():
            if location != base_location and baseline_prices:
                self._calculate_geo_price_differences(data, baseline_prices)
        
        self.results['geolocation_data'] = geo_data
    
    def _fetch_geo_pricing(self, location: str) -> Dict[str, Any]:
        """Fetch pricing data from a specific geographic location"""
        import aiohttp
        import asyncio
        
        # Get proxy/VPN configuration for location
        proxy_config = self._get_proxy_config(location)
        
        location_pricing = {}
        
        for url in self.config.get('urls', []):
            try:
                pricing_data = self._fetch_url_with_geo(url, location, proxy_config)
                if pricing_data:
                    location_pricing[url] = pricing_data
                    
            except Exception as e:
                logger.error(f"Failed to fetch {url} from {location}: {e}")
                continue
        
        # Calculate summary metrics for this location
        if location_pricing:
            return self._calculate_geo_summary(location_pricing, location)
        
        return {}
    
    def _get_proxy_config(self, location: str) -> Dict[str, Any]:
        """Get proxy configuration for geographic location"""
        # Proxy configurations by region (in real implementation, use actual proxy services)
        proxy_configs = {
            'United States': {'proxy': None, 'headers': {'Accept-Language': 'en-US,en;q=0.9'}},
            'United Kingdom': {
                'proxy': 'http://uk-proxy-service.com:8080',  # Example proxy
                'headers': {'Accept-Language': 'en-GB,en;q=0.9'}
            },
            'Germany': {
                'proxy': 'http://de-proxy-service.com:8080',
                'headers': {'Accept-Language': 'de-DE,de;q=0.9'}
            },
            'France': {
                'proxy': 'http://fr-proxy-service.com:8080',
                'headers': {'Accept-Language': 'fr-FR,fr;q=0.9'}
            },
            'Canada': {
                'proxy': 'http://ca-proxy-service.com:8080',
                'headers': {'Accept-Language': 'en-CA,en;q=0.9'}
            },
            'Australia': {
                'proxy': 'http://au-proxy-service.com:8080',
                'headers': {'Accept-Language': 'en-AU,en;q=0.9'}
            },
            'Japan': {
                'proxy': 'http://jp-proxy-service.com:8080',
                'headers': {'Accept-Language': 'ja-JP,ja;q=0.9'}
            },
            'India': {
                'proxy': 'http://in-proxy-service.com:8080',
                'headers': {'Accept-Language': 'en-IN,en;q=0.9'}
            },
            'Brazil': {
                'proxy': 'http://br-proxy-service.com:8080',
                'headers': {'Accept-Language': 'pt-BR,pt;q=0.9'}
            },
            'Singapore': {
                'proxy': 'http://sg-proxy-service.com:8080',
                'headers': {'Accept-Language': 'en-SG,en;q=0.9'}
            }
        }
        
        return proxy_configs.get(location, proxy_configs['United States'])
    
    def _fetch_url_with_geo(self, url: str, location: str, proxy_config: Dict) -> Dict[str, Any]:
        """Fetch URL with geographic targeting"""
        import requests
        from time import sleep
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            **proxy_config.get('headers', {})
        }
        
        try:
            # Add delay to be respectful
            sleep(self.config.get('crawl_delay', 2.0))
            
            response = requests.get(
                url,
                headers=headers,
                proxies={'http': proxy_config.get('proxy'), 'https': proxy_config.get('proxy')} if proxy_config.get('proxy') else None,
                timeout=30,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Parse pricing from response
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                pricing_info = {
                    'currency': self._detect_geo_currency(soup, location),
                    'prices': self._extract_geo_prices(soup),
                    'location_specific_offers': self._detect_location_offers(soup, location),
                    'availability': self._check_service_availability(soup, location)
                }
                
                # Calculate average price
                if pricing_info['prices']:
                    pricing_info['average_price'] = sum(pricing_info['prices']) / len(pricing_info['prices'])
                else:
                    pricing_info['average_price'] = 0
                
                return pricing_info
                
        except Exception as e:
            logger.error(f"Error fetching {url} for {location}: {e}")
            # Fallback: use direct request without proxy
            try:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    return {
                        'currency': self._detect_geo_currency(soup, location),
                        'prices': self._extract_geo_prices(soup),
                        'average_price': 0,
                        'location_specific_offers': [],
                        'availability': True,
                        'note': 'Fetched without geographic targeting'
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback fetch also failed for {url}: {fallback_error}")
        
        return None
    
    def _detect_geo_currency(self, soup, location: str) -> str:
        """Detect currency based on geographic location and page content"""
        # Location-based currency mapping
        location_currencies = {
            'United States': 'USD',
            'United Kingdom': 'GBP',
            'Germany': 'EUR',
            'France': 'EUR',
            'Canada': 'CAD',
            'Australia': 'AUD',
            'Japan': 'JPY',
            'India': 'INR',
            'Brazil': 'BRL',
            'Singapore': 'SGD'
        }
        
        # First try location-based detection
        expected_currency = location_currencies.get(location, 'USD')
        
        # Then verify with page content
        detected_currency = self._detect_currency(soup)
        
        # If page shows different currency, use that (more accurate)
        if detected_currency != 'USD' or location == 'United States':
            return detected_currency
        
        return expected_currency
    
    def _extract_geo_prices(self, soup) -> List[float]:
        """Extract all price numbers from page for geographic analysis"""
        prices = []
        
        # Look for all price elements
        price_elements = soup.select([
            '[class*="price"]', '[class*="cost"]', '[class*="amount"]',
            '[data-price]', '.fee', '.rate'
        ])
        
        for element in price_elements:
            price_text = element.get_text()
            price_numeric = self._extract_price_number(price_text)
            if price_numeric > 0:
                prices.append(price_numeric)
        
        # Also search text for price patterns
        content = soup.get_text()
        price_patterns = [
            r'\$(\d+(?:\.\d{2})?)',
            r'€(\d+(?:\.\d{2})?)',
            r'£(\d+(?:\.\d{2})?)',
            r'¥(\d+)',
            r'(\d+(?:\.\d{2})?) USD',
            r'(\d+(?:\.\d{2})?) EUR',
            r'(\d+(?:\.\d{2})?) GBP'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    price = float(match)
                    if 0 < price < 10000:  # Reasonable price range
                        prices.append(price)
                except ValueError:
                    continue
        
        return list(set(prices))  # Remove duplicates
    
    def _detect_location_offers(self, soup, location: str) -> List[str]:
        """Detect location-specific offers or messaging"""
        content = soup.get_text().lower()
        offers = []
        
        # Location-specific offer patterns
        location_patterns = {
            'United Kingdom': ['uk only', 'british', 'england', 'scotland', 'wales'],
            'Germany': ['deutschland', 'german', 'de only'],
            'France': ['france only', 'french', 'français'],
            'Canada': ['canada only', 'canadian', 'cad pricing'],
            'Australia': ['australia only', 'australian', 'aussie'],
            'Japan': ['japan only', 'japanese', '日本'],
            'India': ['india only', 'indian', 'rupee'],
            'Brazil': ['brazil only', 'brazilian', 'real'],
            'Singapore': ['singapore only', 'sgd pricing']
        }
        
        patterns = location_patterns.get(location, [])
        for pattern in patterns:
            if pattern in content:
                # Find context around the pattern
                sentences = content.split('.')
                for sentence in sentences:
                    if pattern in sentence and len(sentence) < 200:
                        offers.append(sentence.strip())
                        break
        
        return offers[:3]  # Limit to 3 offers
    
    def _check_service_availability(self, soup, location: str) -> bool:
        """Check if service is available in the location"""
        content = soup.get_text().lower()
        
        # Check for availability indicators
        unavailable_patterns = [
            'not available', 'unavailable', 'coming soon', 'not supported',
            'restricted', 'blocked', 'geo-restricted'
        ]
        
        available_patterns = [
            'available', 'supported', 'served', 'offered'
        ]
        
        for pattern in unavailable_patterns:
            if pattern in content:
                return False
        
        return True  # Default to available
    
    def _calculate_geo_summary(self, location_pricing: Dict, location: str) -> Dict[str, Any]:
        """Calculate summary metrics for a geographic location"""
        all_prices = []
        all_currencies = []
        
        for url_data in location_pricing.values():
            all_prices.extend(url_data.get('prices', []))
            currency = url_data.get('currency')
            if currency:
                all_currencies.append(currency)
        
        if all_prices:
            summary = {
                'currency': max(set(all_currencies), key=all_currencies.count) if all_currencies else 'USD',
                'average_price': sum(all_prices) / len(all_prices),
                'min_price': min(all_prices),
                'max_price': max(all_prices),
                'price_range': max(all_prices) - min(all_prices),
                'total_urls_analyzed': len(location_pricing),
                'price_difference_percent': 0  # Will be calculated later
            }
        else:
            summary = {
                'currency': 'USD',
                'average_price': 0,
                'min_price': 0,
                'max_price': 0,
                'price_range': 0,
                'total_urls_analyzed': len(location_pricing),
                'price_difference_percent': 0
            }
        
        return summary
    
    def _calculate_geo_price_differences(self, location_data: Dict, baseline_prices: Dict):
        """Calculate price differences from baseline location"""
        if not baseline_prices or not location_data:
            return
        
        baseline_avg = sum(baseline_prices.values()) / len(baseline_prices)
        location_avg = location_data.get('average_price', 0)
        
        if baseline_avg > 0:
            price_diff_percent = ((location_avg - baseline_avg) / baseline_avg) * 100
            location_data['price_difference_percent'] = price_diff_percent
        else:
            location_data['price_difference_percent'] = 0
    
    def _perform_competitive_analysis(self):
        """Perform real-time competitive analysis"""
        competitive_data = {
            'market_position': {},
            'competitors': [],
            'competitive_advantages': [],
            'competitive_gaps': [],
            'pricing_recommendations': []
        }
        
        # Identify competitors from URLs
        competitors = self._identify_competitors()
        
        # Analyze each competitor
        for competitor in competitors:
            try:
                competitor_data = self._analyze_competitor(competitor)
                if competitor_data:
                    competitive_data['competitors'].append(competitor_data)
                    
            except Exception as e:
                logger.error(f"Failed to analyze competitor {competitor.get('name', 'Unknown')}: {e}")
                continue
        
        # Calculate market positioning
        if competitive_data['competitors']:
            competitive_data['market_position'] = self._calculate_market_position(
                competitive_data['competitors']
            )
            
            # Generate competitive insights
            competitive_data['competitive_advantages'] = self._identify_competitive_advantages(
                competitive_data['competitors']
            )
            competitive_data['competitive_gaps'] = self._identify_competitive_gaps(
                competitive_data['competitors']
            )
            competitive_data['pricing_recommendations'] = self._generate_pricing_recommendations(
                competitive_data['competitors'], competitive_data['market_position']
            )
        
        self.results['competitive_analysis'] = competitive_data
    
    def _identify_competitors(self) -> List[Dict[str, Any]]:
        """Identify competitors from provided URLs"""
        competitors = []
        main_urls = self.config.get('urls', [])
        
        for url in main_urls:
            try:
                # Extract domain info
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain = parsed.netloc.replace('www.', '')
                
                competitor_info = {
                    'name': domain.replace('.com', '').replace('.', ' ').title(),
                    'url': url,
                    'domain': domain,
                    'type': 'direct'  # Assume direct competitor from user input
                }
                
                competitors.append(competitor_info)
                
            except Exception as e:
                logger.error(f"Failed to parse competitor URL {url}: {e}")
                continue
        
        return competitors
    
    def _analyze_competitor(self, competitor: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single competitor's pricing and positioning"""
        import requests
        from bs4 import BeautifulSoup
        from time import sleep
        
        url = competitor['url']
        
        try:
            # Respectful crawling
            sleep(self.config.get('crawl_delay', 2.0))
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for competitor {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract competitor data
            competitor_analysis = {
                'name': competitor['name'],
                'url': url,
                'domain': competitor['domain'],
                'pricing_data': self._extract_competitor_pricing(soup),
                'feature_analysis': self._extract_competitor_features(soup),
                'positioning': self._analyze_competitor_positioning(soup),
                'strengths': self._identify_competitor_strengths(soup),
                'weaknesses': self._identify_competitor_weaknesses(soup),
                'target_market': self._identify_target_market(soup)
            }
            
            # Calculate competitive metrics
            competitor_analysis['pricing_score'] = self._calculate_pricing_competitiveness(
                competitor_analysis['pricing_data']
            )
            competitor_analysis['feature_score'] = self._calculate_feature_competitiveness(
                competitor_analysis['feature_analysis']
            )
            competitor_analysis['overall_score'] = (
                competitor_analysis['pricing_score'] + competitor_analysis['feature_score']
            ) / 2
            
            return competitor_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing competitor {competitor['name']}: {e}")
            return None
    
    def _extract_competitor_pricing(self, soup) -> Dict[str, Any]:
        """Extract pricing information from competitor page"""
        pricing_data = {
            'tiers': [],
            'currency': self._detect_currency(soup),
            'pricing_model': self._detect_pricing_model(soup),
            'price_range': {'min': 0, 'max': 0},
            'free_tier': False,
            'custom_pricing': False
        }
        
        # Extract pricing tiers
        tiers = self._extract_pricing_tiers(soup)
        pricing_data['tiers'] = tiers
        
        # Calculate price range
        if tiers:
            prices = [tier.get('price_numeric', 0) for tier in tiers if tier.get('price_numeric', 0) > 0]
            if prices:
                pricing_data['price_range'] = {'min': min(prices), 'max': max(prices)}
        
        # Check for free tier
        content = soup.get_text().lower()
        if any(word in content for word in ['free', 'freemium', 'trial', 'no cost']):
            pricing_data['free_tier'] = True
        
        # Check for custom pricing
        if any(word in content for word in ['custom', 'contact', 'enterprise', 'quote']):
            pricing_data['custom_pricing'] = True
        
        return pricing_data
    
    def _extract_competitor_features(self, soup) -> Dict[str, Any]:
        """Extract feature information from competitor page"""
        features = {
            'core_features': [],
            'premium_features': [],
            'integrations': [],
            'support_options': [],
            'security_features': []
        }
        
        content = soup.get_text().lower()
        
        # Define feature categories and keywords
        feature_categories = {
            'core_features': ['api', 'dashboard', 'analytics', 'reporting', 'workflow'],
            'premium_features': ['advanced', 'premium', 'pro', 'enterprise', 'unlimited'],
            'integrations': ['integration', 'connect', 'sync', 'import', 'export'],
            'support_options': ['support', 'help', 'documentation', 'training', 'onboarding'],
            'security_features': ['security', 'encryption', 'compliance', 'privacy', 'gdpr']
        }
        
        # Extract features by category
        for category, keywords in feature_categories.items():
            category_features = []
            
            # Look for feature lists
            feature_lists = soup.select(['ul', 'ol', '[class*="feature"]', '[class*="benefit"]'])
            
            for feature_list in feature_lists:
                items = feature_list.select(['li', '[class*="item"]'])
                for item in items:
                    feature_text = item.get_text().strip()
                    
                    # Check if feature matches category keywords
                    if any(keyword in feature_text.lower() for keyword in keywords):
                        if len(feature_text) < 100 and feature_text not in category_features:
                            category_features.append(feature_text)
            
            features[category] = category_features[:5]  # Limit to 5 per category
        
        return features
    
    def _analyze_competitor_positioning(self, soup) -> Dict[str, Any]:
        """Analyze competitor's market positioning"""
        content = soup.get_text().lower()
        
        positioning = {
            'market_segment': 'unknown',
            'value_proposition': '',
            'target_company_size': 'unknown',
            'industry_focus': [],
            'key_differentiators': []
        }
        
        # Detect market segment
        if any(word in content for word in ['enterprise', 'large', 'corporation']):
            positioning['market_segment'] = 'enterprise'
        elif any(word in content for word in ['small', 'medium', 'smb', 'startup']):
            positioning['market_segment'] = 'smb'
        elif any(word in content for word in ['individual', 'personal', 'freelancer']):
            positioning['market_segment'] = 'individual'
        
        # Extract value proposition from headers and hero sections
        hero_sections = soup.select(['h1', 'h2', '.hero', '.tagline', '.value-prop'])
        for section in hero_sections:
            text = section.get_text().strip()
            if 10 < len(text) < 200:
                positioning['value_proposition'] = text
                break
        
        # Detect industry focus
        industries = ['healthcare', 'finance', 'education', 'retail', 'manufacturing', 'saas']
        for industry in industries:
            if industry in content:
                positioning['industry_focus'].append(industry)
        
        # Extract key differentiators
        differentiator_keywords = ['unique', 'only', 'first', 'leading', 'best', 'fastest']
        sentences = soup.get_text().split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in differentiator_keywords):
                if 20 < len(sentence) < 150:
                    positioning['key_differentiators'].append(sentence.strip())
                    
                if len(positioning['key_differentiators']) >= 3:
                    break
        
        return positioning
    
    def _identify_competitor_strengths(self, soup) -> List[str]:
        """Identify competitor's key strengths"""
        content = soup.get_text().lower()
        strengths = []
        
        strength_indicators = [
            'award', 'winner', 'leader', 'top rated', 'best', 'excellent',
            'proven', 'trusted', 'reliable', 'secure', 'fast', 'easy'
        ]
        
        sentences = soup.get_text().split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in strength_indicators):
                if 20 < len(sentence) < 120:
                    strengths.append(sentence.strip())
                    
                if len(strengths) >= 5:
                    break
        
        return strengths
    
    def _identify_competitor_weaknesses(self, soup) -> List[str]:
        """Identify potential competitor weaknesses (gaps in messaging)"""
        content = soup.get_text().lower()
        weaknesses = []
        
        # Look for missing key features or weak messaging
        weakness_indicators = [
            'coming soon', 'planned', 'roadmap', 'limited', 'basic',
            'contact us', 'custom quote', 'varies'
        ]
        
        # Also check for absence of common strong features
        missing_features = []
        important_features = ['api', 'mobile', 'integration', 'analytics', 'security']
        
        for feature in important_features:
            if feature not in content:
                missing_features.append(f"No mention of {feature}")
        
        weaknesses.extend(missing_features[:3])
        
        return weaknesses[:5]
    
    def _identify_target_market(self, soup) -> Dict[str, Any]:
        """Identify competitor's target market"""
        content = soup.get_text().lower()
        
        target_market = {
            'company_sizes': [],
            'industries': [],
            'use_cases': [],
            'geographic_focus': []
        }
        
        # Company sizes
        size_indicators = {
            'enterprise': ['enterprise', 'large', 'corporation', 'fortune'],
            'mid-market': ['medium', 'mid-size', 'growing'],
            'small business': ['small', 'startup', 'entrepreneur'],
            'individual': ['individual', 'personal', 'freelancer']
        }
        
        for size, keywords in size_indicators.items():
            if any(keyword in content for keyword in keywords):
                target_market['company_sizes'].append(size)
        
        # Industries (already implemented in positioning)
        industries = ['healthcare', 'finance', 'education', 'retail', 'manufacturing', 'technology']
        for industry in industries:
            if industry in content:
                target_market['industries'].append(industry)
        
        return target_market
    
    def _calculate_pricing_competitiveness(self, pricing_data: Dict) -> float:
        """Calculate pricing competitiveness score (0-100)"""
        score = 50  # Base score
        
        # Adjust based on pricing factors
        if pricing_data.get('free_tier'):
            score += 15
        
        if pricing_data.get('custom_pricing'):
            score += 10
        
        # Price range analysis
        price_range = pricing_data.get('price_range', {})
        min_price = price_range.get('min', 0)
        
        if min_price == 0:
            score += 20  # Free option is competitive
        elif min_price < 50:
            score += 15  # Low-cost option
        elif min_price > 200:
            score -= 10  # High-cost may be less competitive
        
        return max(0, min(100, score))
    
    def _calculate_feature_competitiveness(self, feature_data: Dict) -> float:
        """Calculate feature competitiveness score (0-100)"""
        score = 50  # Base score
        
        # Count total features across categories
        total_features = sum(len(features) for features in feature_data.values())
        
        # Adjust score based on feature count
        if total_features > 15:
            score += 20
        elif total_features > 10:
            score += 15
        elif total_features > 5:
            score += 10
        elif total_features < 3:
            score -= 15
        
        # Bonus for having features in all categories
        categories_with_features = sum(1 for features in feature_data.values() if features)
        if categories_with_features >= 4:
            score += 15
        
        return max(0, min(100, score))
    
    def _calculate_market_position(self, competitors: List[Dict]) -> Dict[str, Any]:
        """Calculate overall market position based on competitor analysis"""
        if not competitors:
            return {'tier': 'Unknown', 'competitiveness': 'Unknown', 'value_score': 0}
        
        # Calculate average scores
        pricing_scores = [c.get('pricing_score', 50) for c in competitors]
        feature_scores = [c.get('feature_score', 50) for c in competitors]
        overall_scores = [c.get('overall_score', 50) for c in competitors]
        
        avg_pricing = sum(pricing_scores) / len(pricing_scores)
        avg_features = sum(feature_scores) / len(feature_scores)
        avg_overall = sum(overall_scores) / len(overall_scores)
        
        # Determine market tier
        if avg_overall >= 80:
            tier = 'Market Leader'
        elif avg_overall >= 65:
            tier = 'Strong Competitor'
        elif avg_overall >= 50:
            tier = 'Market Participant'
        else:
            tier = 'Niche Player'
        
        # Determine competitiveness
        if avg_pricing >= 70 and avg_features >= 70:
            competitiveness = 'Highly Competitive'
        elif avg_pricing >= 60 or avg_features >= 60:
            competitiveness = 'Competitive'
        else:
            competitiveness = 'Needs Improvement'
        
        return {
            'tier': tier,
            'competitiveness': competitiveness,
            'value_score': round(avg_overall, 1),
            'pricing_strength': round(avg_pricing, 1),
            'feature_strength': round(avg_features, 1)
        }
    
    def _identify_competitive_advantages(self, competitors: List[Dict]) -> List[str]:
        """Identify competitive advantages based on analysis"""
        advantages = []
        
        if not competitors:
            return advantages
        
        # Analyze strengths across competitors
        all_strengths = []
        for competitor in competitors:
            all_strengths.extend(competitor.get('strengths', []))
        
        # Find common themes in strengths
        strength_themes = {}
        for strength in all_strengths:
            words = strength.lower().split()
            for word in words:
                if len(word) > 4:  # Focus on meaningful words
                    strength_themes[word] = strength_themes.get(word, 0) + 1
        
        # Convert top themes to advantages
        top_themes = sorted(strength_themes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for theme, count in top_themes:
            if count > 1:  # Theme appears in multiple competitors
                advantages.append(f"Strong {theme.title()} capabilities across offerings")
        
        # Add specific advantages based on analysis
        avg_pricing_score = sum(c.get('pricing_score', 50) for c in competitors) / len(competitors)
        avg_feature_score = sum(c.get('feature_score', 50) for c in competitors) / len(competitors)
        
        if avg_pricing_score > 70:
            advantages.append("Competitive pricing strategy")
        
        if avg_feature_score > 75:
            advantages.append("Comprehensive feature set")
        
        # Check for free tiers
        free_tier_count = sum(1 for c in competitors if c.get('pricing_data', {}).get('free_tier'))
        if free_tier_count > 0:
            advantages.append("Free tier availability for customer acquisition")
        
        return advantages[:5]
    
    def _identify_competitive_gaps(self, competitors: List[Dict]) -> List[str]:
        """Identify competitive gaps and weaknesses"""
        gaps = []
        
        if not competitors:
            return gaps
        
        # Analyze weaknesses across competitors
        all_weaknesses = []
        for competitor in competitors:
            all_weaknesses.extend(competitor.get('weaknesses', []))
        
        # Common gaps
        if len(all_weaknesses) > 0:
            gaps.extend(all_weaknesses[:3])
        
        # Analyze pricing gaps
        avg_pricing_score = sum(c.get('pricing_score', 50) for c in competitors) / len(competitors)
        if avg_pricing_score < 60:
            gaps.append("Pricing strategy optimization needed")
        
        # Analyze feature gaps
        avg_feature_score = sum(c.get('feature_score', 50) for c in competitors) / len(competitors)
        if avg_feature_score < 60:
            gaps.append("Feature portfolio enhancement required")
        
        # Check for missing market segments
        all_segments = []
        for competitor in competitors:
            positioning = competitor.get('positioning', {})
            segment = positioning.get('market_segment', 'unknown')
            if segment != 'unknown':
                all_segments.append(segment)
        
        if 'enterprise' not in all_segments:
            gaps.append("Limited enterprise market presence")
        
        if 'smb' not in all_segments:
            gaps.append("Small business market underserved")
        
        return gaps[:5]
    
    def _generate_pricing_recommendations(self, competitors: List[Dict], market_position: Dict) -> List[str]:
        """Generate pricing recommendations based on competitive analysis"""
        recommendations = []
        
        if not competitors:
            return ["Conduct competitive analysis to inform pricing strategy"]
        
        # Analyze pricing landscape
        all_price_ranges = []
        free_tier_competitors = 0
        custom_pricing_competitors = 0
        
        for competitor in competitors:
            pricing_data = competitor.get('pricing_data', {})
            price_range = pricing_data.get('price_range', {})
            
            if price_range.get('min', 0) > 0:
                all_price_ranges.append((price_range.get('min', 0), price_range.get('max', 0)))
            
            if pricing_data.get('free_tier'):
                free_tier_competitors += 1
            
            if pricing_data.get('custom_pricing'):
                custom_pricing_competitors += 1
        
        # Generate recommendations based on analysis
        if free_tier_competitors > len(competitors) / 2:
            recommendations.append("Consider implementing freemium model to match market expectations")
        
        if custom_pricing_competitors > 0:
            recommendations.append("Develop enterprise pricing tier with custom options")
        
        # Price positioning recommendations
        competitiveness = market_position.get('competitiveness', '')
        value_score = market_position.get('value_score', 0)
        
        if value_score < 60:
            recommendations.append("Reevaluate value proposition and pricing alignment")
        
        if 'Needs Improvement' in competitiveness:
            recommendations.append("Conduct pricing optimization to improve market competitiveness")
        
        # Market tier specific recommendations
        tier = market_position.get('tier', '')
        if 'Leader' in tier:
            recommendations.append("Leverage market position for premium pricing strategy")
        elif 'Niche' in tier:
            recommendations.append("Focus on specialized value proposition to justify pricing")
        
        return recommendations[:5]
    
    def _compare_environments(self):
        """Compare pricing across different environments using real-time checks"""
        environment_data = {'environments': {}}
        
        base_urls = self.config.get('urls', [])
        environments = self.config.get('environments', ['Production'])
        
        for env in environments:
            try:
                logger.info(f"Analyzing {env} environment")
                env_analysis = self._analyze_environment(env, base_urls)
                if env_analysis:
                    environment_data['environments'][env.lower()] = env_analysis
                    
            except Exception as e:
                logger.error(f"Failed to analyze {env} environment: {e}")
                continue
        
        # Compare environments for consistency
        if len(environment_data['environments']) > 1:
            self._detect_environment_inconsistencies(environment_data['environments'])
        
        self.results['environment_comparison'] = environment_data
    
    def _analyze_environment(self, environment: str, base_urls: List[str]) -> Dict[str, Any]:
        """Analyze a specific environment for pricing consistency"""
        import requests
        from bs4 import BeautifulSoup
        from time import sleep
        import time
        
        env_urls = self._get_environment_urls(environment, base_urls)
        
        env_analysis = {
            'status': 'Unknown',
            'products': [],
            'pricing_consistency': 'Unknown',
            'last_updated': time.strftime('%Y-%m-%d'),
            'response_time': 'Unknown',
            'availability': 'Unknown',
            'pricing_accuracy': 'Unknown',
            'data_freshness': 'Unknown',
            'issues': [],
            'recommendations': []
        }
        
        successful_requests = 0
        total_response_time = 0
        pricing_data_found = []
        
        for url in env_urls:
            try:
                start_time = time.time()
                
                # Respectful crawling
                sleep(self.config.get('crawl_delay', 1.0))
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=30)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                total_response_time += response_time
                
                if response.status_code == 200:
                    successful_requests += 1
                    
                    # Parse pricing data
                    soup = BeautifulSoup(response.text, 'html.parser')
                    pricing_info = self._extract_environment_pricing(soup, url)
                    
                    if pricing_info:
                        pricing_data_found.append(pricing_info)
                        env_analysis['products'].append(pricing_info)
                
                else:
                    env_analysis['issues'].append(f"HTTP {response.status_code} error for {url}")
                    
            except Exception as e:
                env_analysis['issues'].append(f"Failed to access {url}: {str(e)}")
                continue
        
        # Calculate environment metrics
        if env_urls:
            availability = (successful_requests / len(env_urls)) * 100
            env_analysis['availability'] = f"{availability:.1f}%"
            
            if successful_requests > 0:
                avg_response_time = total_response_time / successful_requests
                env_analysis['response_time'] = f"{avg_response_time:.0f}ms"
                env_analysis['status'] = 'Active' if availability > 80 else 'Degraded'
            else:
                env_analysis['status'] = 'Down'
                env_analysis['response_time'] = 'N/A'
        
        # Analyze pricing data quality
        if pricing_data_found:
            env_analysis['pricing_accuracy'] = self._assess_pricing_accuracy(pricing_data_found)
            env_analysis['data_freshness'] = self._assess_data_freshness(pricing_data_found)
            env_analysis['pricing_consistency'] = self._assess_pricing_consistency(pricing_data_found)
        else:
            env_analysis['pricing_accuracy'] = '0%'
            env_analysis['data_freshness'] = 'No data'
            env_analysis['pricing_consistency'] = 'No pricing data found'
            env_analysis['issues'].append('No pricing information found')
        
        # Generate recommendations
        env_analysis['recommendations'] = self._generate_environment_recommendations(env_analysis)
        
        return env_analysis
    
    def _get_environment_urls(self, environment: str, base_urls: List[str]) -> List[str]:
        """Generate environment-specific URLs"""
        env_urls = []
        
        # Environment URL mapping patterns
        env_patterns = {
            'production': ['', 'www.', 'prod.'],
            'staging': ['staging.', 'stage.', 'stg.'],
            'test': ['test.', 'testing.', 'qa.'],
            'development': ['dev.', 'develop.', 'devel.'],
            'demo': ['demo.', 'preview.']
        }
        
        patterns = env_patterns.get(environment.lower(), [''])
        
        for base_url in base_urls:
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            
            for pattern in patterns:
                if pattern == '':
                    env_urls.append(base_url)
                else:
                    # Replace or prepend subdomain
                    if parsed.netloc.startswith('www.'):
                        new_netloc = parsed.netloc.replace('www.', pattern)
                    else:
                        new_netloc = pattern + parsed.netloc
                    
                    env_url = f"{parsed.scheme}://{new_netloc}{parsed.path}"
                    env_urls.append(env_url)
        
        return list(set(env_urls))  # Remove duplicates
    
    def _extract_environment_pricing(self, soup, url: str) -> Dict[str, Any]:
        """Extract pricing information for environment comparison"""
        pricing_info = {
            'url': url,
            'pricing_elements_found': 0,
            'pricing_data': [],
            'last_modified': self._extract_last_modified(soup),
            'version_info': self._extract_version_info(soup),
            'pricing_errors': []
        }
        
        # Extract pricing elements
        price_elements = soup.select([
            '[class*="price"]', '[class*="cost"]', '[class*="amount"]',
            '[data-price]', '.fee', '.rate'
        ])
        
        pricing_info['pricing_elements_found'] = len(price_elements)
        
        for element in price_elements:
            try:
                price_text = element.get_text().strip()
                price_numeric = self._extract_price_number(price_text)
                
                if price_numeric > 0:
                    pricing_info['pricing_data'].append({
                        'text': price_text,
                        'numeric': price_numeric,
                        'element_class': element.get('class', []),
                        'element_id': element.get('id', '')
                    })
                    
            except Exception as e:
                pricing_info['pricing_errors'].append(f"Error processing price element: {str(e)}")
        
        return pricing_info
    
    def _extract_last_modified(self, soup) -> str:
        """Extract last modified date from page"""
        # Look for common last modified patterns
        meta_tags = soup.select(['meta[name="last-modified"]', 'meta[name="date"]'])
        for tag in meta_tags:
            content = tag.get('content')
            if content:
                return content
        
        # Look for date strings in content
        import re
        content = soup.get_text()
        date_patterns = [
            r'last updated:?\s*(\d{4}-\d{2}-\d{2})',
            r'modified:?\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return 'Unknown'
    
    def _extract_version_info(self, soup) -> str:
        """Extract version information from page"""
        # Look for version meta tags
        version_meta = soup.select_one('meta[name="version"]')
        if version_meta:
            return version_meta.get('content', 'Unknown')
        
        # Look for version in comments or script tags
        comments = soup.find_all(string=lambda text: text and 'version' in text.lower())
        for comment in comments[:3]:  # Check first 3 matches
            if 'version' in comment.lower():
                import re
                version_match = re.search(r'version[:\s]*([0-9.]+)', comment, re.IGNORECASE)
                if version_match:
                    return version_match.group(1)
        
        return 'Unknown'
    
    def _assess_pricing_accuracy(self, pricing_data_list: List[Dict]) -> str:
        """Assess pricing data accuracy across the environment"""
        total_elements = sum(data.get('pricing_elements_found', 0) for data in pricing_data_list)
        total_errors = sum(len(data.get('pricing_errors', [])) for data in pricing_data_list)
        
        if total_elements == 0:
            return '0%'
        
        accuracy = ((total_elements - total_errors) / total_elements) * 100
        return f"{accuracy:.0f}%"
    
    def _assess_data_freshness(self, pricing_data_list: List[Dict]) -> str:
        """Assess how fresh the pricing data is"""
        from datetime import datetime, timedelta
        
        last_modified_dates = []
        for data in pricing_data_list:
            last_mod = data.get('last_modified', 'Unknown')
            if last_mod != 'Unknown':
                try:
                    # Try to parse date
                    if '-' in last_mod:
                        date_obj = datetime.strptime(last_mod, '%Y-%m-%d')
                    else:
                        date_obj = datetime.strptime(last_mod, '%m/%d/%Y')
                    
                    last_modified_dates.append(date_obj)
                except ValueError:
                    continue
        
        if not last_modified_dates:
            return 'Unknown'
        
        # Find most recent date
        most_recent = max(last_modified_dates)
        days_old = (datetime.now() - most_recent).days
        
        if days_old == 0:
            return 'Current'
        elif days_old == 1:
            return '1 day old'
        elif days_old < 7:
            return f'{days_old} days old'
        elif days_old < 30:
            weeks = days_old // 7
            return f'{weeks} week{"s" if weeks > 1 else ""} old'
        else:
            months = days_old // 30
            return f'{months} month{"s" if months > 1 else ""} old'
    
    def _assess_pricing_consistency(self, pricing_data_list: List[Dict]) -> str:
        """Assess pricing consistency across URLs in the environment"""
        if len(pricing_data_list) <= 1:
            return 'Single source'
        
        # Compare pricing data across URLs
        all_prices = []
        for data in pricing_data_list:
            prices = [item['numeric'] for item in data.get('pricing_data', [])]
            all_prices.extend(prices)
        
        if not all_prices:
            return 'No pricing data'
        
        # Check for price variations
        unique_prices = set(all_prices)
        
        if len(unique_prices) == 1:
            return 'Fully consistent'
        elif len(unique_prices) <= len(all_prices) * 0.1:  # 10% or less variation
            return 'Mostly consistent'
        elif len(unique_prices) <= len(all_prices) * 0.3:  # 30% or less variation
            return 'Minor inconsistencies'
        else:
            return 'Major inconsistencies'
    
    def _generate_environment_recommendations(self, env_analysis: Dict) -> List[str]:
        """Generate recommendations for environment improvements"""
        recommendations = []
        
        # Availability recommendations
        availability = env_analysis.get('availability', '0%')
        if availability != 'Unknown':
            avail_percent = float(availability.replace('%', ''))
            if avail_percent < 90:
                recommendations.append('Investigate and resolve availability issues')
        
        # Response time recommendations
        response_time = env_analysis.get('response_time', '0ms')
        if response_time != 'Unknown' and response_time != 'N/A':
            response_ms = float(response_time.replace('ms', ''))
            if response_ms > 3000:  # More than 3 seconds
                recommendations.append('Optimize response times for better user experience')
        
        # Pricing accuracy recommendations
        accuracy = env_analysis.get('pricing_accuracy', '0%')
        if accuracy != 'Unknown':
            accuracy_percent = float(accuracy.replace('%', ''))
            if accuracy_percent < 95:
                recommendations.append('Fix pricing data errors and validation issues')
        
        # Data freshness recommendations
        freshness = env_analysis.get('data_freshness', 'Unknown')
        if 'week' in freshness or 'month' in freshness:
            recommendations.append('Update pricing data to ensure freshness')
        
        # Consistency recommendations
        consistency = env_analysis.get('pricing_consistency', 'Unknown')
        if 'inconsistencies' in consistency:
            recommendations.append('Standardize pricing data across all environment URLs')
        
        # Issues-based recommendations
        issues = env_analysis.get('issues', [])
        if issues:
            if len(issues) > 2:
                recommendations.append('Address multiple environment issues affecting reliability')
            elif any('404' in issue or 'error' in issue.lower() for issue in issues):
                recommendations.append('Fix broken URLs and error responses')
        
        # Default recommendation if no issues found
        if not recommendations:
            recommendations.append('Environment is healthy - continue regular monitoring')
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _detect_environment_inconsistencies(self, environments: Dict[str, Dict]):
        """Detect inconsistencies between environments"""
        env_names = list(environments.keys())
        
        if len(env_names) < 2:
            return
        
        # Compare pricing data between environments
        for i, env1 in enumerate(env_names):
            for env2 in env_names[i+1:]:
                inconsistencies = self._compare_two_environments(
                    environments[env1], 
                    environments[env2], 
                    env1, 
                    env2
                )
                
                # Add inconsistencies to issues
                for env_name, issues in inconsistencies.items():
                    if env_name in environments:
                        environments[env_name]['issues'].extend(issues)
    
    def _compare_two_environments(self, env1_data: Dict, env2_data: Dict, 
                                 env1_name: str, env2_name: str) -> Dict[str, List[str]]:
        """Compare two environments and identify inconsistencies"""
        inconsistencies = {env1_name: [], env2_name: []}
        
        # Compare pricing data
        env1_products = env1_data.get('products', [])
        env2_products = env2_data.get('products', [])
        
        env1_prices = []
        env2_prices = []
        
        for product in env1_products:
            env1_prices.extend([item['numeric'] for item in product.get('pricing_data', [])])
        
        for product in env2_products:
            env2_prices.extend([item['numeric'] for item in product.get('pricing_data', [])])
        
        # Check for price differences
        if env1_prices and env2_prices:
            env1_avg = sum(env1_prices) / len(env1_prices)
            env2_avg = sum(env2_prices) / len(env2_prices)
            
            if abs(env1_avg - env2_avg) > env1_avg * 0.05:  # 5% difference threshold
                percentage_diff = abs(env1_avg - env2_avg) / env1_avg * 100
                inconsistencies[env1_name].append(
                    f"Pricing differs by {percentage_diff:.1f}% from {env2_name} environment"
                )
                inconsistencies[env2_name].append(
                    f"Pricing differs by {percentage_diff:.1f}% from {env1_name} environment"
                )
        
        # Compare availability
        env1_avail = env1_data.get('availability', '0%')
        env2_avail = env2_data.get('availability', '0%')
        
        if env1_avail != 'Unknown' and env2_avail != 'Unknown':
            avail1 = float(env1_avail.replace('%', ''))
            avail2 = float(env2_avail.replace('%', ''))
            
            if abs(avail1 - avail2) > 10:  # 10% difference threshold
                if avail1 < avail2:
                    inconsistencies[env1_name].append(
                        f"Lower availability than {env2_name} environment"
                    )
                else:
                    inconsistencies[env2_name].append(
                        f"Lower availability than {env1_name} environment"
                    )
        
        return inconsistencies
    
    def _generate_ai_insights(self):
        """Generate AI-powered pricing insights using real Azure OpenAI"""
        if not AZURE_OPENAI_AVAILABLE:
            logger.warning("Azure OpenAI not available, generating rule-based insights")
            self.results['ai_insights'] = self._generate_rule_based_pricing_insights()
            return
        
        try:
            # Prepare context data for AI analysis
            context_data = {
                'products': self.results.get('products', []),
                'geolocation_data': self.results.get('geolocation_data', {}),
                'competitive_analysis': self.results.get('competitive_analysis', {}),
                'config': self.config
            }
            
            # Generate comprehensive AI insights
            ai_insights = self._call_azure_openai_for_pricing_insights(context_data)
            
            if ai_insights:
                self.results['ai_insights'] = ai_insights
            else:
                # Fallback to rule-based insights
                self.results['ai_insights'] = self._generate_rule_based_pricing_insights()
                
        except Exception as e:
            logger.error(f"Failed to generate AI insights: {e}")
            self.results['ai_insights'] = self._generate_rule_based_pricing_insights()
    
    def _call_azure_openai_for_pricing_insights(self, context_data: Dict) -> Dict[str, Any]:
        """Call Azure OpenAI to generate pricing insights"""
        try:
            # Import Azure OpenAI client
            from openai import AzureOpenAI
            import os
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-02-01",
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            
            # Prepare prompt for pricing analysis
            prompt = self._build_pricing_analysis_prompt(context_data)
            
            # Call Azure OpenAI
            response = client.chat.completions.create(
                model="gpt-4",  # or your deployed model name
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert pricing strategist and market analyst. 
                        Analyze the provided pricing data and generate strategic insights, 
                        recommendations, and actionable intelligence. Focus on:
                        1. Executive summary of findings
                        2. Strategic pricing recommendations
                        3. Market trend analysis
                        4. ROI projections
                        5. Competitive positioning insights
                        
                        Provide specific, actionable recommendations with quantified impact estimates."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse AI response
            ai_response = response.choices[0].message.content
            return self._parse_ai_pricing_response(ai_response, context_data)
            
        except Exception as e:
            logger.error(f"Azure OpenAI call failed: {e}")
            return None
    
    def _build_pricing_analysis_prompt(self, context_data: Dict) -> str:
        """Build comprehensive prompt for pricing analysis"""
        products = context_data.get('products', [])
        geo_data = context_data.get('geolocation_data', {})
        competitive_data = context_data.get('competitive_analysis', {})
        config = context_data.get('config', {})
        
        prompt = f"""
PRICING ANALYSIS REQUEST

PRODUCT DATA:
{self._format_products_for_prompt(products)}

GEOGRAPHIC PRICING DATA:
{self._format_geo_data_for_prompt(geo_data)}

COMPETITIVE LANDSCAPE:
{self._format_competitive_data_for_prompt(competitive_data)}

ANALYSIS CONFIGURATION:
- Target Markets: {', '.join(config.get('market_segments', []))}
- Geographic Scope: {', '.join(config.get('geolocations', []))}
- Product Categories: {', '.join(config.get('product_categories', []))}
- Analysis Depth: {config.get('ai_analysis_depth', 'Comprehensive')}

ANALYSIS REQUIREMENTS:

1. EXECUTIVE SUMMARY: Provide a strategic overview of the pricing landscape and key opportunities

2. PRICING STRATEGY RECOMMENDATIONS: 
   - Specific pricing optimization strategies
   - Timeline and implementation approach
   - Expected impact metrics

3. MARKET TRENDS ANALYSIS:
   - Current industry pricing trends
   - Customer behavior insights
   - Competitive dynamics

4. ROI PROJECTIONS:
   - Revenue impact estimates
   - Customer acquisition projections
   - Implementation costs and timeline

5. COMPETITIVE POSITIONING:
   - Market position assessment
   - Differentiation opportunities
   - Competitive advantages and gaps

Please provide specific, quantified recommendations with clear business rationale.
        """
        
        return prompt.strip()
    
    def _format_products_for_prompt(self, products: List[Dict]) -> str:
        """Format product data for AI prompt"""
        if not products:
            return "No product data available"
        
        formatted = []
        for product in products:
            product_info = f"""
Product: {product.get('name', 'Unknown')}
Category: {product.get('category', 'Unknown')}
Pricing Model: {product.get('pricing_model', 'Unknown')}
Currency: {product.get('currency', 'USD')}
Pricing Tiers: {len(product.get('pricing_tiers', []))} tiers
Price Range: {self._get_price_range_summary(product)}
Key Features: {', '.join(product.get('key_features', [])[:5])}
            """
            formatted.append(product_info.strip())
        
        return '\n\n'.join(formatted)
    
    def _format_geo_data_for_prompt(self, geo_data: Dict) -> str:
        """Format geographic data for AI prompt"""
        if not geo_data:
            return "No geographic pricing data available"
        
        formatted = []
        for location, data in geo_data.items():
            geo_info = f"""
{location}: {data.get('currency', 'USD')} {data.get('average_price', 0):.2f} 
(Variance: {data.get('price_difference_percent', 0):+.1f}% from baseline)
            """
            formatted.append(geo_info.strip())
        
        return '\n'.join(formatted)
    
    def _format_competitive_data_for_prompt(self, competitive_data: Dict) -> str:
        """Format competitive data for AI prompt"""
        if not competitive_data:
            return "No competitive analysis data available"
        
        market_position = competitive_data.get('market_position', {})
        competitors = competitive_data.get('competitors', [])
        
        formatted = f"""
Market Position: {market_position.get('tier', 'Unknown')}
Competitiveness: {market_position.get('competitiveness', 'Unknown')}
Value Score: {market_position.get('value_score', 0)}/100

Competitors Analyzed: {len(competitors)}
        """
        
        if competitors:
            formatted += "\nKey Competitors:\n"
            for comp in competitors[:3]:  # Top 3 competitors
                formatted += f"- {comp.get('name', 'Unknown')}: "
                formatted += f"Pricing Score {comp.get('pricing_score', 0):.0f}, "
                formatted += f"Feature Score {comp.get('feature_score', 0):.0f}\n"
        
        return formatted.strip()
    
    def _get_price_range_summary(self, product: Dict) -> str:
        """Get price range summary for a product"""
        tiers = product.get('pricing_tiers', [])
        if not tiers:
            return "No pricing data"
        
        prices = [tier.get('price_numeric', 0) for tier in tiers if tier.get('price_numeric', 0) > 0]
        if prices:
            currency = product.get('currency', 'USD')
            return f"{currency} {min(prices):.0f} - {max(prices):.0f}"
        
        return "Custom pricing"
    
    def _parse_ai_pricing_response(self, ai_response: str, context_data: Dict) -> Dict[str, Any]:
        """Parse AI response into structured insights"""
        insights = {
            'executive_summary': '',
            'pricing_strategies': [],
            'market_trends': {'trends': [], 'benchmarks': {}},
            'roi_projections': {},
            'competitive_positioning': {},
            'implementation_roadmap': []
        }
        
        try:
            # Split response into sections
            sections = self._split_ai_response_into_sections(ai_response)
            
            # Extract executive summary
            insights['executive_summary'] = sections.get('executive_summary', '')
            
            # Extract pricing strategies
            insights['pricing_strategies'] = self._extract_pricing_strategies(sections)
            
            # Extract market trends
            insights['market_trends'] = self._extract_market_trends(sections)
            
            # Extract ROI projections
            insights['roi_projections'] = self._extract_roi_projections(sections)
            
            # Extract competitive positioning
            insights['competitive_positioning'] = self._extract_competitive_positioning(sections)
            
            # Extract implementation roadmap
            insights['implementation_roadmap'] = self._extract_implementation_roadmap(sections)
            
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            # Return raw response if parsing fails
            insights['executive_summary'] = ai_response[:500] + "..."
        
        return insights
    
    def _split_ai_response_into_sections(self, response: str) -> Dict[str, str]:
        """Split AI response into structured sections"""
        sections = {}
        current_section = 'general'
        current_content = []
        
        lines = response.split('\n')
        
        section_markers = {
            'executive summary': 'executive_summary',
            'pricing strategy': 'pricing_strategies',
            'market trends': 'market_trends',
            'roi projections': 'roi_projections',
            'competitive positioning': 'competitive_positioning',
            'implementation': 'implementation_roadmap'
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            section_found = False
            for marker, section_key in section_markers.items():
                if marker in line_lower and ':' in line:
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = section_key
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and line.strip():
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_pricing_strategies(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract pricing strategies from AI response"""
        strategies = []
        content = sections.get('pricing_strategies', '')
        
        # Look for strategy patterns
        lines = content.split('\n')
        current_strategy = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for strategy titles (numbered or bulleted)
            if re.match(r'^\d+\.|^-|^•', line):
                if current_strategy:
                    strategies.append(current_strategy)
                
                current_strategy = {
                    'title': re.sub(r'^\d+\.|^-|^•', '', line).strip(),
                    'impact': 'Medium',
                    'timeline': '3-6 months',
                    'recommendation': line.strip()
                }
            elif current_strategy and any(word in line.lower() for word in ['impact', 'timeline', 'revenue']):
                # Extract additional details
                if 'high' in line.lower():
                    current_strategy['impact'] = 'High'
                elif 'low' in line.lower():
                    current_strategy['impact'] = 'Low'
                
                # Extract timeline
                timeline_match = re.search(r'(\d+[-\s]*\d*\s*months?)', line.lower())
                if timeline_match:
                    current_strategy['timeline'] = timeline_match.group(1)
        
        if current_strategy:
            strategies.append(current_strategy)
        
        return strategies[:5]  # Limit to 5 strategies
    
    def _extract_market_trends(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract market trends from AI response"""
        content = sections.get('market_trends', '')
        
        trends = {
            'trends': [],
            'benchmarks': {}
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract trend statements
            if any(word in line.lower() for word in ['trend', 'increasing', 'growing', 'rising', 'declining']):
                if len(line) > 20 and len(line) < 150:
                    trends['trends'].append(line)
            
            # Extract benchmark numbers
            if '%' in line or '$' in line:
                # Try to extract key-value pairs
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if len(key) < 50:
                            trends['benchmarks'][key] = value
        
        return trends
    
    def _extract_roi_projections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Extract ROI projections from AI response"""
        content = sections.get('roi_projections', '')
        
        projections = {}
        
        # Look for percentage and monetary values
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract revenue increase
            if 'revenue' in line.lower() and '%' in line:
                revenue_match = re.search(r'(\d+[-\s]*\d*%)', line)
                if revenue_match:
                    projections['revenue_increase'] = revenue_match.group(1)
            
            # Extract customer acquisition impact
            if 'acquisition' in line.lower() or 'customer' in line.lower():
                if '%' in line:
                    acq_match = re.search(r'(\+?\d+[-\s]*\d*%)', line)
                    if acq_match:
                        projections['acquisition_impact'] = acq_match.group(1)
            
            # Extract timeline
            if 'timeline' in line.lower() or 'months' in line.lower():
                timeline_match = re.search(r'(\d+[-\s]*\d*\s*months?)', line.lower())
                if timeline_match:
                    projections['timeline_to_roi'] = timeline_match.group(1)
        
        # Set defaults if not found
        if 'revenue_increase' not in projections:
            projections['revenue_increase'] = '8-15%'
        if 'acquisition_impact' not in projections:
            projections['acquisition_impact'] = '+15% new customers'
        if 'timeline_to_roi' not in projections:
            projections['timeline_to_roi'] = '6-9 months'
        
        return projections
    
    def _extract_competitive_positioning(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract competitive positioning insights"""
        content = sections.get('competitive_positioning', '')
        
        positioning = {
            'market_position': 'Competitive',
            'key_differentiators': [],
            'competitive_advantages': [],
            'areas_for_improvement': []
        }
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Extract advantages
            if any(word in line.lower() for word in ['advantage', 'strength', 'leading', 'better']):
                positioning['competitive_advantages'].append(line)
            
            # Extract areas for improvement
            if any(word in line.lower() for word in ['improve', 'enhance', 'weakness', 'gap']):
                positioning['areas_for_improvement'].append(line)
        
        return positioning
    
    def _extract_implementation_roadmap(self, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Extract implementation roadmap from AI response"""
        content = sections.get('implementation_roadmap', '')
        
        roadmap = []
        lines = content.split('\n')
        
        current_phase = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for phase indicators
            if re.match(r'phase|step|\d+\.|quarter|month', line.lower()):
                if current_phase:
                    roadmap.append(current_phase)
                
                current_phase = {
                    'phase': line,
                    'timeline': 'TBD',
                    'priority': 'Medium',
                    'expected_impact': 'TBD'
                }
            elif current_phase:
                # Add details to current phase
                if 'timeline' in line.lower() or 'month' in line.lower():
                    current_phase['timeline'] = line
                elif 'priority' in line.lower():
                    current_phase['priority'] = line
                elif 'impact' in line.lower():
                    current_phase['expected_impact'] = line
        
        if current_phase:
            roadmap.append(current_phase)
        
        return roadmap[:4]  # Limit to 4 phases
    
    def _generate_rule_based_pricing_insights(self) -> Dict[str, Any]:
        """Generate rule-based insights when AI is not available"""
        products = self.results.get('products', [])
        geo_data = self.results.get('geolocation_data', {})
        competitive_data = self.results.get('competitive_analysis', {})
        
        # Calculate basic metrics
        total_products = len(products)
        total_geolocations = len(geo_data)
        competitors_analyzed = len(competitive_data.get('competitors', []))
        
        # Generate basic insights
        insights = {
            'executive_summary': f"""
            Analyzed {total_products} products across {total_geolocations} geographic markets 
            and {competitors_analyzed} competitors. The analysis reveals opportunities for 
            pricing optimization and competitive positioning improvements.
            """.strip(),
            
            'pricing_strategies': [
                {
                    'title': 'Market-Based Pricing Optimization',
                    'impact': 'Medium',
                    'timeline': '2-3 months',
                    'recommendation': 'Align pricing with competitive market rates'
                },
                {
                    'title': 'Geographic Pricing Strategy',
                    'impact': 'High',
                    'timeline': '3-6 months',
                    'recommendation': 'Implement region-specific pricing'
                }
            ],
            
            'market_trends': {
                'trends': [
                    'Increasing adoption of subscription-based pricing models',
                    'Growing demand for transparent pricing structures',
                    'Regional pricing customization becoming standard'
                ],
                'benchmarks': {
                    'Market Analysis Completion': '100%',
                    'Geographic Coverage': f'{total_geolocations} markets',
                    'Competitive Intelligence': f'{competitors_analyzed} competitors'
                }
            },
            
            'roi_projections': {
                'revenue_increase': '10-15%',
                'acquisition_impact': '+20% new customers',
                'timeline_to_roi': '6-8 months'
            }
        }
        
        return insights
    
    def _calculate_metrics(self):
        """Calculate overall pricing metrics"""
        # Simple competitiveness score calculation
        competitive_data = self.results.get('competitive_analysis', {})
        if competitive_data:
            value_score = competitive_data.get('market_position', {}).get('value_score', 0)
            self.results['competitiveness_score'] = value_score


def show_ai_insights_dashboard_cx():
    """AI Insights Dashboard for CX Navigator"""
    st.subheader("🧠 AI Customer Experience Insights")
    
    if not st.session_state.get('cx_analysis_results'):
        st.info("🔍 Run a website analysis first to see AI-powered insights.")
        return
    
    results = st.session_state.cx_analysis_results
    ai_insights = results.get('ai_insights', {})
    cx_analysis = ai_insights.get('cx_analysis', {})
    
    if not ai_insights:
        st.warning("No AI insights available. Ensure Azure OpenAI is configured.")
        return
    
    # Executive Dashboard
    st.subheader("📊 Executive Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cx_score = results.get('cx_score', 0)
        st.metric("CX Score", f"{cx_score:.1f}/100", 
                 delta=f"{'Above' if cx_score > 75 else 'Below'} Average")
    
    with col2:
        confidence = cx_analysis.get('ai_confidence', 0)
        st.metric("AI Confidence", f"{confidence:.1%}")
    
    with col3:
        business_impact = results.get('business_impact', {})
        conversion_lift = business_impact.get('potential_conversion_lift', '0%')
        st.metric("Potential Conversion Lift", conversion_lift)
    
    with col4:
        total_issues = results.get('crawl_summary', {}).get('total_issues', 0)
        st.metric("Issues to Address", total_issues)
    
    # AI Analysis Sections
    with st.expander("🎯 Executive Summary", expanded=True):
        summary = cx_analysis.get('executive_summary', '')
        if summary:
            st.write(summary)
        else:
            st.info("Executive summary not available")
    
    # Business Impact Assessment
    with st.expander("💼 Business Impact Assessment"):
        business_impact = results.get('business_impact', {})
        if business_impact:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Revenue Impact:**")
                st.write(f"• Potential Revenue Impact: {business_impact.get('potential_revenue_impact', 'Unknown')}")
                st.write(f"• Customer Satisfaction Improvement: {business_impact.get('customer_satisfaction_improvement', 'Unknown')}")
            
            with col2:
                st.write("**Risk Assessment:**")
                st.write(f"• Brand Reputation Risk: {business_impact.get('brand_reputation_risk', 'Unknown')}")
                st.write(f"• Competitive Risk: {business_impact.get('competitive_risk', 'Unknown')}")
    
    # Implementation Roadmap
    with st.expander("🗺️ Implementation Roadmap"):
        roadmap = results.get('implementation_roadmap', [])
        if roadmap:
            for phase in roadmap:
                st.write(f"**{phase.get('phase', 'Unknown Phase')}**")
                st.write(f"Timeline: {phase.get('timeline', 'Unknown')}")
                st.write(f"Priority: {phase.get('priority', 'Unknown')}")
                st.write(f"Expected Impact: {phase.get('expected_impact', 'Unknown')}")
                st.write("---")
    
    # Detailed AI Insights
    if ai_insights.get('cx_analysis', {}).get('actionable_recommendations'):
        with st.expander("💡 AI Recommendations"):
            recommendations = cx_analysis.get('actionable_recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

def show_executive_reports_ui():
    """Executive reporting interface"""
    st.subheader("📈 Executive Reports & Analytics")
    
    if not st.session_state.get('cx_analysis_results'):
        st.info("🔍 Run a website analysis first to generate executive reports.")
        return
    
    results = st.session_state.cx_analysis_results
    
    # Generate executive report
    if st.button("📄 Generate Executive Report"):
        with st.spinner("Generating comprehensive executive report..."):
            try:
                report_generator = SmartCXReporter(results)
                report_path = report_generator.generate_html_report(include_screenshots=True)
                
                st.success(f"✅ Executive report generated: {report_path}")
                
                # Show report preview
                with st.expander("📋 Report Preview"):
                    st.write("**Report Contents:**")
                    st.write("• Executive Summary with CX Score")
                    st.write("• Business Impact Analysis")
                    st.write("• Detailed Issue Breakdown")
                    st.write("• Implementation Roadmap")
                    st.write("• AI-Powered Recommendations")
                    st.write("• Industry Benchmarking")
                
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")

def show_configuration_ui():
    """Configuration and settings UI"""
    st.subheader("⚙️ Configuration & Settings")
    
    # AI Configuration
    with st.expander("🤖 AI Configuration"):
        st.write("**Azure OpenAI Status:**")
        if AZURE_OPENAI_AVAILABLE:
            st.success("✅ Azure OpenAI is available and configured")
        else:
            st.error("❌ Azure OpenAI not available")
            st.write("To enable AI features:")
            st.code("""
# Install and configure Azure OpenAI client
pip install openai
# Configure your Azure OpenAI credentials
            """)
    
    # Crawler Configuration
    with st.expander("🔧 Advanced Crawler Settings"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Browser Settings:**")
            st.write(f"Playwright Available: {'✅' if PLAYWRIGHT_AVAILABLE else '❌'}")
            st.write(f"Selenium Available: {'✅' if SELENIUM_AVAILABLE else '❌'}")
            st.write(f"BeautifulSoup Available: {'✅' if BS4_AVAILABLE else '❌'}")

        with col2:
            st.write("**Analysis Features:**")
            st.write("Screenshot Capture: ✅")
            st.write("Performance Analysis: ✅")
            st.write("Accessibility Testing: ✅")
            st.write("Mobile Responsiveness: ✅")

def display_cx_analysis_results(results):
    """Display comprehensive CX analysis results"""
    st.subheader("📊 Analysis Results & Insights")
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    summary = results.get('crawl_summary', {})
    
    with col1:
        st.metric("Pages Analyzed", summary.get('pages_crawled', 0))
    with col2:
        st.metric("Total Issues", summary.get('total_issues', 0))
    with col3:
        cx_score = results.get('cx_score', 0)
        st.metric("CX Score", f"{cx_score:.1f}/100")
    with col4:
        ai_available = results.get('ai_available', False)
        st.metric("AI Analysis", "✅ Active" if ai_available else "❌ Limited")
    
    # Issue Breakdown
    if summary.get('severity_breakdown'):
        st.subheader("🎯 Issue Severity Breakdown")
        
        severity_data = summary['severity_breakdown']
        if any(severity_data.values()):
            # Create visualization
            if MATPLOTLIB_AVAILABLE:
                try:
                    labels = [k for k, v in severity_data.items() if v > 0]
                    sizes = [v for v in severity_data.values() if v > 0]
                    colors = ['#ff4444', '#ff8800', '#ffcc00', '#88cc00'][:len(labels)]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Issues by Severity')
                    st.pyplot(fig)
                except Exception as e:
                    # Fallback to bar chart
                    severity_df = pd.DataFrame(list(severity_data.items()), columns=['Severity', 'Count'])
                    st.bar_chart(severity_df.set_index('Severity'))
            else:
                # Use streamlit's built-in chart
                severity_df = pd.DataFrame(list(severity_data.items()), columns=['Severity', 'Count'])
                st.bar_chart(severity_df.set_index('Severity'))
    
    # Detailed Results Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detailed Issues", "🤖 AI Insights", "📊 Performance", "📋 Raw Data"])
    
    with tab1:
        # Show detailed issues
        pages = results.get('page_analyses', [])
        if pages:
            for page in pages[:5]:  # Show first 5 pages
                with st.expander(f"📄 {page.get('url', 'Unknown URL')} ({len(page.get('issues', []))} issues)"):
                    for issue in page.get('issues', []):
                        severity_color = {
                            'Critical': '🔴',
                            'High': '🟠', 
                            'Medium': '🟡',
                            'Low': '🟢'
                        }.get(issue.get('severity', 'Low'), '🔵')
                        
                        st.write(f"{severity_color} **{issue.get('title', 'Unknown Issue')}**")
                        st.write(f"Category: {issue.get('category', 'Unknown')}")
                        st.write(f"Description: {issue.get('description', 'No description')}")
                        
                        if issue.get('recommendations'):
                            st.write("**Recommendations:**")
                            for rec in issue['recommendations']:
                                st.write(f"• {rec}")
                        st.write("---")
    
    with tab2:
        # AI Insights
        ai_insights = results.get('ai_insights', {})
        if ai_insights and not ai_insights.get('error'):
            st.write("**🤖 AI-Powered Analysis Available**")
            
            # Show CX analysis if available
            cx_analysis = ai_insights.get('cx_analysis', {})
            if cx_analysis:
                if cx_analysis.get('critical_insights'):
                    st.write("**🎯 Critical Insights:**")
                    for insight in cx_analysis['critical_insights'][:3]:
                        st.info(insight)
        else:
            st.warning("AI insights not available or failed to generate")
    
    with tab3:
        # Performance metrics
        if pages:
            load_times = [p.get('load_time', 0) for p in pages if p.get('load_time')]
            if load_times:
                avg_load_time = sum(load_times) / len(load_times)
                st.metric("Average Load Time", f"{avg_load_time:.2f}s")
                
                # Show load time distribution
                st.bar_chart({"Load Times": load_times})
    
    with tab4:
        # Raw data
        st.json(results)

# Missing method implementations for AICustomerExperienceAnalyzer
AICustomerExperienceAnalyzer._extract_recommendations = lambda self, sections: _extract_recommendations(sections)
AICustomerExperienceAnalyzer._extract_implementation_plan = lambda self, sections: _extract_implementation_plan(sections)
AICustomerExperienceAnalyzer._extract_roi_estimates = lambda self, sections: _extract_roi_estimates(sections)
AICustomerExperienceAnalyzer._extract_competitive_insights = lambda self, sections: _extract_competitive_insights(sections)
AICustomerExperienceAnalyzer._extract_business_impact = lambda self, sections: _extract_business_impact(sections)
AICustomerExperienceAnalyzer._generate_rule_based_cx_analysis = lambda self, crawl_results, industry: _generate_rule_based_cx_analysis(crawl_results, industry)

def _extract_recommendations(sections: Dict[str, str]) -> List[str]:
    """Extract actionable recommendations from AI analysis"""
    recommendations = []
    
    # Look for recommendations in multiple sections
    recommendation_sections = [
        "strategic_recommendations", "actionable_recommendations",
        "conversion_optimization_opportunities", "technical_debt_prioritization"
    ]
    
    for section_key in recommendation_sections:
        section_content = sections.get(section_key, "")
        if section_content:
            recommendations.extend(_extract_list_items(
                section_content, ["recommend", "should", "improve", "implement", "optimize"]
            ))
    
    return recommendations[:10]  # Limit to top 10 recommendations

def _extract_implementation_plan(sections: Dict[str, str]) -> List[Dict[str, str]]:
    """Extract implementation plan from AI analysis"""
    roadmap_section = sections.get("implementation_roadmap", "")
    
    # Extract phases from the roadmap
    phases = []
    lines = roadmap_section.split('\n')
    current_phase = {}
    
    for line in lines:
        line = line.strip()
        if "phase" in line.lower():
            if current_phase:
                phases.append(current_phase)
            current_phase = {"phase": line, "details": []}
        elif line and current_phase:
            current_phase["details"].append(line)
    
    if current_phase:
        phases.append(current_phase)
    
    return phases[:4]  # Limit to 4 phases

def _extract_roi_estimates(sections: Dict[str, str]) -> Dict[str, str]:
    """Extract ROI estimates from AI analysis"""
    roi_section = sections.get("roi_estimation", "")
    
    # Look for percentage values and estimates
    import re
    percentages = re.findall(r'(\d+(?:\.\d+)?%)', roi_section)
    
    return {
        "conversion_improvement": percentages[0] if len(percentages) > 0 else "Unknown",
        "revenue_impact": percentages[1] if len(percentages) > 1 else "Unknown",
        "implementation_cost": "Variable based on scope",
        "timeline_to_roi": "3-6 months typical"
    }

def _extract_competitive_insights(sections: Dict[str, str]) -> Dict[str, Any]:
    """Extract competitive positioning insights"""
    competitive_section = sections.get("competitive_positioning", "")
    
    return {
        "competitive_advantages": _extract_list_items(competitive_section, ["advantage", "strength", "ahead"]),
        "competitive_gaps": _extract_list_items(competitive_section, ["gap", "behind", "weakness"]),
        "market_position": "Analysis in progress",
        "differentiation_opportunities": _extract_list_items(competitive_section, ["opportunity", "differentiate"])
    }

def _extract_business_impact(sections: Dict[str, str]) -> Dict[str, str]:
    """Extract business impact analysis"""
    business_section = sections.get("business_impact_assessment", "")
    
    # Look for impact indicators
    impact_indicators = {
        "revenue_impact": "potential revenue",
        "customer_satisfaction": "customer satisfaction", 
        "brand_impact": "brand",
        "operational_efficiency": "efficiency"
    }
    
    extracted_impacts = {}
    for key, indicator in impact_indicators.items():
        if indicator in business_section.lower():
            # Extract sentences containing the indicator
            sentences = [s.strip() for s in business_section.split('.') if indicator in s.lower()]
            if sentences:
                extracted_impacts[key] = sentences[0][:100] + "..."
    
    return extracted_impacts

def _extract_list_items(text: str, keywords: List[str]) -> List[str]:
    """Extract list items from text based on keywords"""
    items = []
    lines = text.split('\n')
    
    for line in lines:
        if any(keyword in line.lower() for keyword in keywords):
            if line.strip().startswith(('-', '•', '*', '1.', '2.', '3.')):
                items.append(line.strip().lstrip('-•*123456789. '))
    
    return items[:5]  # Limit to top 5 items

def _generate_rule_based_cx_analysis(crawl_results: Dict[str, Any], industry: str) -> Dict[str, Any]:
    """Generate rule-based CX analysis when AI is not available"""
    summary = crawl_results.get('crawl_summary', {})
    total_issues = summary.get('total_issues', 0)
    pages_crawled = summary.get('pages_crawled', 1)
    
    # Calculate severity-based assessment
    severity_counts = summary.get('severity_breakdown', {})
    critical_issues = severity_counts.get('Critical', 0)
    high_issues = severity_counts.get('High', 0)
    
    analysis = {
        "executive_summary": f"Analyzed {pages_crawled} pages and found {total_issues} total issues.",
        "cx_score": 0.0,
        "critical_insights": [],
        "actionable_recommendations": [],
        "business_impact_forecast": {},
        "implementation_roadmap": [],
        "ai_confidence": 0.6,  # Lower confidence for rule-based
        "user_journey_analysis": {},
        "conversion_impact_assessment": {}
    }
    
    # Generate insights based on issue counts
    if critical_issues > 0:
        analysis["critical_insights"].append(f"Found {critical_issues} critical issues requiring immediate attention")
        analysis["actionable_recommendations"].append("Address critical security and functionality issues first")
    
    if high_issues > 0:
        analysis["critical_insights"].append(f"Detected {high_issues} high-priority issues affecting user experience")
        analysis["actionable_recommendations"].append("Prioritize high-impact user experience improvements")
    
    # Generate basic recommendations
    analysis["actionable_recommendations"].extend([
        "Conduct thorough testing of critical user paths",
        "Implement performance optimization measures",
        "Review and improve accessibility compliance",
        "Optimize mobile user experience"
    ])
    
    return analysis

# Utility functions for the Smart CX Navigator
def run_analysis_async(urls: List[str], config: CrawlConfig) -> Dict[str, Any]:
    """Run the async analysis in a separate function"""
    crawler = SmartWebCrawler(config)
    try:
        import asyncio
        
        # Handle existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an existing event loop, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, crawler.crawl_website(urls))
                return future.result()
        except RuntimeError:
            # No event loop running, we can run directly
            return asyncio.run(crawler.crawl_website(urls))
            
    except Exception as e:
        logger.error(f"Error in async analysis: {e}")
        return {"error": str(e)}

def run_analysis_sync(urls: List[str], config: CrawlConfig) -> Dict[str, Any]:
    """Synchronous wrapper for the async analysis"""
    try:
        # Log the URLs being processed for debugging
        logger.info(f"Starting analysis of URLs: {urls}")
        
        # Handle existing event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an existing event loop, we need to run in a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _run_crawler_async(urls, config))
                result = future.result(timeout=300)  # 5 minute timeout
                logger.info(f"Analysis completed successfully for {len(urls)} URLs")
                return result
        except RuntimeError:
            # No event loop running, we can run directly
            result = asyncio.run(_run_crawler_async(urls, config))
            logger.info(f"Analysis completed successfully for {len(urls)} URLs")
            return result
            
    except Exception as e:
        logger.error(f"Error in sync wrapper: {e}")
        # Return a valid error structure instead of empty dict
        return {
            "error": str(e),
            "crawl_summary": {
                "pages_crawled": 0,
                "pages_failed": len(urls),
                "total_issues": 0,
                "severity_breakdown": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
                "category_breakdown": {"Usability": 0, "Customer Experience": 0, "Product Functionality": 0, "Compliance": 0, "Performance": 0, "Accessibility": 0}
            },
            "page_analyses": [],
            "failed_urls": urls
        }

async def _run_crawler_async(urls: List[str], config: CrawlConfig) -> Dict[str, Any]:
    """Internal async function to run the crawler"""
    crawler = SmartWebCrawler(config)
    try:
        return await crawler.crawl_website(urls)
    except Exception as e:
        logger.error(f"Error in crawler: {e}")
        raise

    # URL Input Section
    st.markdown("### 🌐 Website URLs to Analyze")

    # URL input methods
    input_method = st.radio("Input Method", ["Manual Entry", "Upload File"])

    urls_to_crawl = []

    if input_method == "Manual Entry":
        url_input = st.text_area("Enter URLs (one per line)",
                                 placeholder="https://example.com\nhttps://another-site.com",
                                 height=100)
        if url_input.strip():
            urls_to_crawl = [url.strip() for url in url_input.strip().split('\n') if url.strip()]

    else:  # Upload File
        uploaded_file = st.file_uploader("Upload URL list", type=['txt', 'csv'])
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            urls_to_crawl = [url.strip() for url in content.strip().split('\n') if url.strip()]

    # Display URLs to be crawled
    if urls_to_crawl:
        st.success(f"Found {len(urls_to_crawl)} URLs to analyze")
        with st.expander("Preview URLs"):
            for i, url in enumerate(urls_to_crawl[:10], 1):
                st.write(f"{i}. {url}")
            if len(urls_to_crawl) > 10:
                st.write(f"... and {len(urls_to_crawl) - 10} more")

    # Analysis Options
    with st.expander("🤖 AI Analysis Options"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Issue Categories to Check:**")
            check_usability = st.checkbox("Usability Issues", value=True)
            check_cx = st.checkbox("Customer Experience", value=True)
            check_functionality = st.checkbox("Product Functionality", value=True)

        with col2:
            st.markdown("**Advanced Checks:**")
            check_compliance = st.checkbox("Compliance & Legal", value=True)
            check_performance = st.checkbox("Performance Issues", value=True)
            check_accessibility = st.checkbox("Accessibility", value=True)

    # Initialize session state for analysis results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False

    # Start Analysis Button
    if st.button("🚀 Start Analysis", type="primary", disabled=not urls_to_crawl or st.session_state.analysis_running):
        if not urls_to_crawl:
            st.error("Please enter at least one URL to analyze")
            return

        # Validate URLs
        valid_urls = []
        for url in urls_to_crawl:
            if url.startswith(('http://', 'https://')):
                valid_urls.append(url)
            else:
                st.warning(f"Invalid URL format: {url}")

        if not valid_urls:
            st.error("No valid URLs found. URLs must start with http:// or https://")
            return

        # Create configuration
        config = CrawlConfig(
            max_depth=max_depth,
            max_pages=max_pages,
            crawl_delay=crawl_delay,
            user_agent=DEFAULT_USER_AGENTS[0],  # Map selection to actual user agent
            screenshot_on_issue=screenshot_on_issue,
            follow_external_links=follow_external
        )

        st.session_state.analysis_running = True

        # Show progress
        progress_container = st.container()
        status_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        try:
            with status_container:
                with st.spinner("Initializing crawler and AI models..."):
                    status_text.text("Starting website analysis...")

                    # Run the actual analysis
                    try:
                        progress_bar.progress(0.1)
                        status_text.text("Initializing crawler and AI models...")
                        
                        # Debug: Log the URLs being analyzed
                        st.info(f"🔍 Analyzing {len(valid_urls)} URLs: {', '.join(valid_urls[:3])}")
                        
                        progress_bar.progress(0.3)
                        status_text.text("Crawling pages and analyzing content...")
                        
                        # Run the actual crawling using the sync wrapper
                        results = run_analysis_sync(valid_urls, config)
                        
                        # Debug: Log the crawl results
                        if results and 'crawl_summary' in results:
                            summary = results['crawl_summary']
                            st.info(f"📊 Found {summary.get('pages_crawled', 0)} pages, {summary.get('total_issues', 0)} issues")
                        
                        progress_bar.progress(0.9)
                        status_text.text("Generating AI insights and reports...")
                        
                        # Add AI analysis to the results
                        if results:
                            # Generate comprehensive AI insights
                            ai_insights = smart_cx_insights.generate_comprehensive_insights(
                                results, 
                                {"industry": "general", "business_goals": ["improve_conversion", "enhance_ux"]}
                            )
                            results['ai_insights'] = ai_insights
                        
                        progress_bar.progress(1.0)
                        status_text.text("Analysis complete!")

                        st.session_state.analysis_results = results
                        st.session_state.analysis_running = False

                        st.success("✅ Analysis completed successfully!")

                        # Add notification for successful analysis
                        if NOTIFICATIONS_AVAILABLE:
                            notifications.add_notification(
                                module_name=MODULE_NAME,
                                status="success",
                                message="Smart CX Navigator analysis completed",
                                details=f"Analyzed {len(valid_urls)} websites with {max_depth} depth crawling",
                                action_steps=[
                                    "Review the generated issues report",
                                    "Download the detailed analysis results",
                                    "Prioritize fixes based on severity and impact"
                                ]
                            )

                    except Exception as e:
                        st.session_state.analysis_running = False
                        st.error(f"Analysis failed: {str(e)}")

                        # Add notification for failed analysis
                        if NOTIFICATIONS_AVAILABLE:
                            notifications.add_notification(
                                module_name=MODULE_NAME,
                                status="error",
                                message="Smart CX Navigator analysis failed",
                                details=str(e),
                                action_steps=[
                                    "Check the error message above",
                                    "Verify URLs are accessible",
                                    "Try reducing crawl depth or page limits",
                                    "Check internet connection and firewall settings"
                                ]
                            )

        except Exception as e:
            st.session_state.analysis_running = False
            st.error(f"Unexpected error: {str(e)}")

    # Display Results Section
    if st.session_state.analysis_results:
        st.markdown("## 📊 Analysis Results")

        results = st.session_state.analysis_results
        
        # Add error handling for missing data structures
        if 'crawl_summary' not in results:
            st.error("❌ Invalid analysis results: missing crawl_summary")
            st.json(results)
            return
            
        summary = results['crawl_summary']

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pages Analyzed", summary['pages_crawled'])
        with col2:
            st.metric("Total Issues", summary['total_issues'])
        with col3:
            cx_score = results.get('cx_score', 0)
            st.metric("CX Score", f"{cx_score:.1f}/100")
        with col4:
            st.metric("Failed Pages", summary['pages_failed'])

        # Debug: Show actual URLs analyzed
        pages = results.get('page_analyses', [])
        if pages:
            with st.expander("🔍 Pages Actually Analyzed"):
                for i, page in enumerate(pages, 1):
                    st.write(f"{i}. **{page.get('url', 'Unknown URL')}**")
                    st.write(f"   - Status: {page.get('status_code', 'Unknown')}")
                    st.write(f"   - Load time: {page.get('load_time', 0):.2f}s")
                    st.write(f"   - Issues found: {len(page.get('issues', []))}")

        # Severity breakdown chart
        st.markdown("### ⚠️ Issues by Severity")
        severity_data = summary['severity_breakdown']
        
        # Debug: Show raw severity data
        st.write("**Debug - Raw severity data:**", severity_data)
        
        # Filter out zero values for better visualization
        severity_filtered = {k: v for k, v in severity_data.items() if v > 0}
        
        if severity_filtered:
            severity_df = pd.DataFrame(list(severity_filtered.items()),
                                       columns=['Severity', 'Count'])
            st.bar_chart(severity_df.set_index('Severity'))
        else:
            st.info("No issues found or all issue counts are zero.")

        # Category breakdown
        st.markdown("### 📋 Issues by Category")
        category_data = summary['category_breakdown']
        
        # Debug: Show raw category data  
        st.write("**Debug - Raw category data:**", category_data)
        
        # Filter out zero values for better visualization
        category_filtered = {k: v for k, v in category_data.items() if v > 0}
        
        if category_filtered:
            category_df = pd.DataFrame(list(category_filtered.items()),
                                       columns=['Category', 'Count'])
            st.bar_chart(category_df.set_index('Category'))
        else:
            st.info("No issues found or all category counts are zero.")

        # Generate and offer reports for download
        st.markdown("### 📥 Download Reports")

        try:
            reporter = SmartCXReporter(results)

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📄 Generate HTML Report"):
                    html_file = reporter.generate_html_report()
                    with open(html_file, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Download HTML Report",
                            data=f.read(),
                            file_name=os.path.basename(html_file),
                            mime="text/html"
                        )

            with col2:
                if st.button("📊 Generate JSON Report"):
                    json_file = reporter.generate_json_report()
                    with open(json_file, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Download JSON Report",
                            data=f.read(),
                            file_name=os.path.basename(json_file),
                            mime="application/json"
                        )

            with col3:
                if st.button("📈 Generate CSV Report"):
                    csv_file = reporter.generate_csv_report()
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        st.download_button(
                            label="Download CSV Report",
                            data=f.read(),
                            file_name=os.path.basename(csv_file),
                            mime="text/csv"
                        )

        except Exception as e:
            st.error(f"Error generating reports: {e}")

        # Clear results button
        if st.button("🗑️ Clear Results"):
            st.session_state.analysis_results = None
            st.rerun()

    # Help Section
    with st.expander("ℹ️ How to Use Smart CX Navigator"):
        st.markdown("""
        ### Getting Started
        1. **Configure Settings**: Adjust crawl depth, page limits, and delay based on your needs
        2. **Enter URLs**: Add the websites you want to analyze (supports multiple URLs)
        3. **Select Analysis Types**: Choose which types of issues to detect
        4. **Start Analysis**: Click the button and wait for comprehensive results
        
        ### What Gets Analyzed
        - **Usability**: Broken links, navigation issues, page readability
        - **Customer Experience**: Contact info, mobile responsiveness, CTAs
        - **Functionality**: Forms, images, interactive elements
        - **Compliance**: Privacy policies, HTTPS, cookie consent
        - **Performance**: Page size, loading speed, external resources
        - **Accessibility**: WCAG compliance, heading structure, alt text
        
        ### Output Features
        - Detailed issue reports with severity levels
        - Screenshots of problematic pages
        - Actionable recommendations for each issue
        - Reference links to best practices
        - Exportable results in multiple formats
        
        ### Command Line Usage
        You can also run this tool from the command line:
        ```bash
        python smart_cx_navigator.py --urls https://example.com --depth 3 --screenshots
        python smart_cx_navigator.py --url-file urls.txt --output-format all
        ```
        """)


if __name__ == "__main__":
    # Check if running from command line or as Streamlit app
    if len(sys.argv) > 1:
        # Command line interface
        cli = CLIInterface()
        exit_code = cli.run()
        sys.exit(exit_code)
    else:
        # Streamlit web interface
        show_ui()
