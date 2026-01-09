import base64
import re
import json
import os
import datetime
import sys
import time
import logging
import urllib.parse
import requests
import hashlib
import fnmatch
import io
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlunparse
import concurrent.futures
from threading import Lock

import cv2
import numpy as np
import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("IntelligentTestData", level=logging.INFO, log_file="intelligent_test_data.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# Selenium imports for web automation
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Beautiful Soup for HTML parsing
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Azure OpenAI Integration
try:
    # Add the path to import azure_openai_client
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from azure_openai_client import AzureOpenAIClient
    AI_AVAILABLE = True
    azure_openai_client = AzureOpenAIClient()
except (ImportError, ValueError) as e:
    AI_AVAILABLE = False
    azure_openai_client = None
    print(f"Warning: Azure OpenAI client not available: {e}")

# Import notifications module for sending notifications
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False


if __name__ == "__main__":
    st.set_page_config(page_title="Intelligent Test Data Generator", page_icon="🧪", layout="wide")

# --- Data Classes for Structure ---

@dataclass
class WebsiteAnalysis:
    """Structure for website analysis results"""
    url: str
    forms: List[Dict]
    fields: List[Dict]
    security_considerations: List[str]
    accessibility_issues: List[str]
    performance_notes: List[str]
    ai_insights: Optional[str] = None

@dataclass
@dataclass
class FormField:
    """Enhanced structure for form field information"""
    name: str
    field_type: str
    label: str
    placeholder: str
    required: bool
    validation_rules: List[str]
    constraints: Dict[str, Any]
    ai_analysis: Optional[str] = None
    suggested_test_data: Optional[Dict] = None

@dataclass
class TestScenario:
    """Structure for test scenarios"""
    scenario_id: str
    title: str
    description: str
    test_type: str  # positive, negative, boundary, security, accessibility
    priority: str  # high, medium, low
    test_data: Any
    expected_result: str
    automation_script: Optional[str] = None

# --- Advanced Website Traversal and Analysis ---

class WebsiteTraverser:
    """Advanced website traversal and form detection"""

    def __init__(self):
        self.visited_urls = set()
        self.analysis_cache = {}
        self.rate_limit_delay = 2  # seconds between requests
        self.max_depth = 3
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def traverse_website(self, base_url: str, max_pages: int = 50) -> List[WebsiteAnalysis]:
        """Traverse website and analyze all forms"""
        results = []
        urls_to_visit = [base_url]
        visited_count = 0
        
        st.info(f"🕷️ Starting website traversal from: {base_url}")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while urls_to_visit and visited_count < max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            try:
                status_text.text(f"Analyzing: {current_url}")
                progress_bar.progress((visited_count + 1) / max_pages)
                
                analysis = self.analyze_page(current_url)
                if analysis:
                    results.append(analysis)
                
                # Discover new URLs
                new_urls = self.discover_links(current_url, base_url)
                urls_to_visit.extend(new_urls)
                
                self.visited_urls.add(current_url)
                visited_count += 1
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error analyzing {current_url}: {e}")
                
        status_text.text(f"✅ Analysis complete! Processed {visited_count} pages")
        return results

    def analyze_page(self, url: str) -> Optional[WebsiteAnalysis]:
        """Analyze a single page for forms and testable elements"""
        try:
            # Check cache first
            cache_key = hashlib.md5(url.encode()).hexdigest()
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract forms
            forms = self.extract_forms(soup, url)
            
            # Extract all potential input fields
            fields = self.extract_all_fields(soup)
            
            # Security analysis
            security_issues = self.analyze_security(soup, response.headers)
            
            # Accessibility analysis
            accessibility_issues = self.analyze_accessibility(soup)
            
            # Performance notes
            performance_notes = self.analyze_performance(response)
            
            analysis = WebsiteAnalysis(
                url=url,
                forms=forms,
                fields=fields,
                security_considerations=security_issues,
                accessibility_issues=accessibility_issues,
                performance_notes=performance_notes
            )
            
            # Add AI insights if available
            if AI_AVAILABLE:
                analysis.ai_insights = self.get_ai_insights(analysis)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze page {url}: {e}")
            return None

    def extract_forms(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Extract all forms from the page"""
        forms = []
        
        for form_element in soup.find_all('form'):
            form_data = {
                'action': form_element.get('action', ''),
                'method': form_element.get('method', 'GET').upper(),
                'id': form_element.get('id', ''),
                'class': form_element.get('class', []),
                'fields': [],
                'buttons': []
            }
            
            # Extract form fields
            for input_elem in form_element.find_all(['input', 'select', 'textarea']):
                field = self.extract_field_info(input_elem)
                if field:
                    form_data['fields'].append(field)
            
            # Extract buttons
            for button in form_element.find_all(['button', 'input']):
                if button.get('type') in ['submit', 'button'] or button.name == 'button':
                    form_data['buttons'].append({
                        'type': button.get('type', 'button'),
                        'text': button.get_text(strip=True) or button.get('value', ''),
                        'id': button.get('id', ''),
                        'class': button.get('class', [])
                    })
            
            if form_data['fields']:  # Only include forms with fields
                forms.append(form_data)
        
        return forms

    def extract_field_info(self, element) -> Optional[Dict]:
        """Extract detailed information about a form field"""
        field_type = element.get('type', element.name).lower()
        
        if field_type in ['hidden', 'submit', 'button', 'reset']:
            return None
        
        return {
            'name': element.get('name', ''),
            'id': element.get('id', ''),
            'type': field_type,
            'label': self.find_label_for_field(element),
            'placeholder': element.get('placeholder', ''),
            'required': element.has_attr('required'),
            'pattern': element.get('pattern', ''),
            'min_length': element.get('minlength', ''),
            'max_length': element.get('maxlength', ''),
            'min': element.get('min', ''),
            'max': element.get('max', ''),
            'step': element.get('step', ''),
            'autocomplete': element.get('autocomplete', ''),
            'aria_label': element.get('aria-label', ''),
            'aria_describedby': element.get('aria-describedby', ''),
            'class': element.get('class', []),
            'data_attributes': {k: v for k, v in element.attrs.items() if k.startswith('data-')}
        }

    def find_label_for_field(self, element) -> str:
        """Find the label associated with a form field"""
        # Check for explicit label association
        field_id = element.get('id')
        if field_id:
            label = element.find_parent().find('label', {'for': field_id})
            if label:
                return label.get_text(strip=True)
        
        # Check for implicit label (field inside label)
        parent_label = element.find_parent('label')
        if parent_label:
            return parent_label.get_text(strip=True)
        
        # Check for nearby text (heuristic)
        prev_sibling = element.find_previous_sibling()
        if prev_sibling and prev_sibling.get_text(strip=True):
            return prev_sibling.get_text(strip=True)
        
        # Check parent's previous sibling
        parent = element.find_parent()
        if parent:
            prev = parent.find_previous_sibling()
            if prev and prev.get_text(strip=True):
                return prev.get_text(strip=True)
        
        return ''

    def extract_all_fields(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract all interactive fields from the page"""
        fields = []
        
        for element in soup.find_all(['input', 'select', 'textarea', 'button']):
            field_info = self.extract_field_info(element)
            if field_info:
                fields.append(field_info)
        
        return fields

    def analyze_security(self, soup: BeautifulSoup, headers: Dict) -> List[str]:
        """Analyze page for security considerations"""
        issues = []
        
        # Check for HTTPS
        if not headers.get('strict-transport-security'):
            issues.append("Missing HSTS header - consider enforcing HTTPS")
        
        # Check for CSP
        if not headers.get('content-security-policy'):
            issues.append("Missing Content Security Policy header")
        
        # Check for password fields without proper attributes
        for pwd_field in soup.find_all('input', {'type': 'password'}):
            if not pwd_field.get('autocomplete'):
                issues.append("Password field missing autocomplete attribute")
        
        # Check for forms without CSRF protection indicators
        forms = soup.find_all('form')
        for form in forms:
            if form.get('method', '').upper() == 'POST':
                csrf_found = form.find('input', {'name': re.compile(r'csrf|token', re.I)})
                if not csrf_found:
                    issues.append("POST form may be missing CSRF protection")
        
        return issues

    def analyze_accessibility(self, soup: BeautifulSoup) -> List[str]:
        """Analyze page for accessibility issues"""
        issues = []
        
        # Check for missing alt attributes on images
        images_without_alt = soup.find_all('img', alt=False)
        if images_without_alt:
            issues.append(f"{len(images_without_alt)} images missing alt text")
        
        # Check for form fields without labels
        unlabeled_fields = 0
        for field in soup.find_all(['input', 'select', 'textarea']):
            if field.get('type') not in ['hidden', 'submit', 'button']:
                if not (field.get('aria-label') or field.get('id') and 
                       soup.find('label', {'for': field.get('id')})):
                    unlabeled_fields += 1
        
        if unlabeled_fields:
            issues.append(f"{unlabeled_fields} form fields missing proper labels")
        
        # Check for missing page title
        if not soup.find('title') or not soup.find('title').get_text(strip=True):
            issues.append("Page missing or empty title")
        
        # Check for missing main landmark
        if not soup.find('main') and not soup.find(attrs={'role': 'main'}):
            issues.append("Page missing main landmark")
        
        return issues

    def analyze_performance(self, response) -> List[str]:
        """Analyze performance characteristics"""
        notes = []
        
        # Response time (basic)
        if hasattr(response, 'elapsed'):
            response_time = response.elapsed.total_seconds()
            if response_time > 3:
                notes.append(f"Slow response time: {response_time:.2f}s")
        
        # Content size
        content_length = len(response.content)
        if content_length > 1024 * 1024:  # 1MB
            notes.append(f"Large page size: {content_length / 1024:.1f}KB")
        
        # Check for compression
        if not response.headers.get('content-encoding'):
            notes.append("Response not compressed")
        
        return notes

    def discover_links(self, url: str, base_url: str) -> List[str]:
        """Discover new URLs to analyze"""
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            links = []
            base_domain = urlparse(base_url).netloc
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                parsed_url = urlparse(full_url)
                
                # Only follow same-domain links
                if parsed_url.netloc == base_domain:
                    # Remove fragments and query params for deduplication
                    clean_url = urlunparse((
                        parsed_url.scheme,
                        parsed_url.netloc,
                        parsed_url.path,
                        '', '', ''
                    ))
                    
                    if clean_url not in self.visited_urls and clean_url != url:
                        links.append(clean_url)
            
            return links[:10]  # Limit to prevent explosion
            
        except Exception as e:
            logger.error(f"Error discovering links from {url}: {e}")
            return []

    def get_ai_insights(self, analysis: WebsiteAnalysis) -> str:
        """Get AI insights about the page analysis"""
        if not AI_AVAILABLE:
            return "AI analysis not available"
        
        try:
            prompt = f"""
            Analyze this webpage for comprehensive testing insights:
            
            URL: {analysis.url}
            Forms found: {len(analysis.forms)}
            Total fields: {len(analysis.fields)}
            Security issues: {len(analysis.security_considerations)}
            Accessibility issues: {len(analysis.accessibility_issues)}
            
            Form details: {json.dumps(analysis.forms[:2], indent=2)}
            
            Please provide:
            1. Key testing priorities for this page
            2. Critical test scenarios to focus on
            3. Potential edge cases specific to this form structure
            4. Security testing recommendations
            5. Accessibility testing priorities
            
            Keep the response concise but actionable.
            """
            
            return azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=800,
                temperature=0.7
            )
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return f"AI analysis error: {str(e)}"

# --- AI-Powered Test Data Generator ---

class AITestDataGenerator:
    """AI-powered intelligent test data generation"""
    
    def __init__(self):
        self.field_patterns = {}
        self.context_cache = {}
    
    def generate_intelligent_test_data(self, field: FormField, context: Dict = None) -> Dict:
        """Generate contextually aware test data using AI"""
        if not AI_AVAILABLE:
            return self.generate_fallback_test_data(field)
        
        try:
            context_info = context or {}
            
            prompt = f"""
            Generate comprehensive test data for this form field:
            
            Field Details:
            - Name: {field.name}
            - Type: {field.field_type}
            - Label: {field.label}
            - Placeholder: {field.placeholder}
            - Required: {field.required}
            - Validation Rules: {field.validation_rules}
            - Constraints: {json.dumps(field.constraints, indent=2)}
            
            Context: {json.dumps(context_info, indent=2)}
            
            Generate test data in this JSON format:
            {{
                "valid_data": ["list of 5-7 valid inputs"],
                "invalid_data": ["list of 5-7 invalid inputs"],
                "boundary_data": ["list of 3-5 boundary cases"],
                "edge_cases": ["list of 3-5 edge cases"],
                "security_tests": ["list of 3-5 security-focused inputs"],
                "accessibility_tests": ["list of accessibility-focused test scenarios"],
                "localization_tests": ["list of internationalization test inputs"],
                "performance_tests": ["list of performance-focused test inputs"]
            }}
            
            Consider:
            - Real-world usage patterns
            - Industry-specific requirements
            - Cultural and linguistic variations
            - Modern attack vectors
            - Accessibility standards
            - Performance implications
            
            Provide only the JSON response.
            """
            
            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            # Parse AI response
            try:
                test_data = json.loads(response)
                return test_data
            except json.JSONDecodeError:
                # Fallback if AI doesn't return valid JSON
                return self.generate_fallback_test_data(field)
                
        except Exception as e:
            logger.error(f"Error generating AI test data: {e}")
            return self.generate_fallback_test_data(field)
    
    def generate_fallback_test_data(self, field: FormField) -> Dict:
        """Fallback test data generation when AI is unavailable"""
        # Convert tuple result to dictionary format
        valid_data, invalid_data, boundary_data, edge_cases, exploits = generate_test_data(field.field_type)
        
        return {
            'valid_data': valid_data,
            'invalid_data': invalid_data,
            'boundary_data': boundary_data,
            'edge_cases': edge_cases,
            'security_tests': exploits,
            'accessibility_tests': [],
            'localization_tests': [],
            'ai_insights': "AI insights not available in fallback mode",
            'test_scenarios': []
        }
    
    def generate_cross_field_scenarios(self, fields: List[FormField]) -> List[TestScenario]:
        """Generate scenarios that test interactions between fields"""
        scenarios = []
        
        if not AI_AVAILABLE:
            return self.generate_basic_cross_field_scenarios(fields)
        
        try:
            field_summary = []
            for field in fields:
                field_summary.append({
                    'name': field.name,
                    'type': field.field_type,
                    'label': field.label,
                    'required': field.required
                })
            
            prompt = f"""
            Generate cross-field test scenarios for this form with {len(fields)} fields:
            
            Fields: {json.dumps(field_summary, indent=2)}
            
            Create scenarios that test:
            1. Field dependencies and relationships
            2. Conditional validation rules
            3. Data consistency across fields
            4. Form completion workflows
            5. Error handling combinations
            
            Return 8-10 scenarios in this JSON format:
            {{
                "scenarios": [
                    {{
                        "id": "unique_id",
                        "title": "scenario title",
                        "description": "detailed description",
                        "type": "positive|negative|boundary|security",
                        "priority": "high|medium|low",
                        "steps": ["step 1", "step 2", "..."],
                        "expected_result": "expected outcome"
                    }}
                ]
            }}
            """
            
            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            )
            
            scenario_data = json.loads(response)
            
            for scenario_json in scenario_data.get('scenarios', []):
                scenario = TestScenario(
                    scenario_id=scenario_json.get('id', ''),
                    title=scenario_json.get('title', ''),
                    description=scenario_json.get('description', ''),
                    test_type=scenario_json.get('type', 'positive'),
                    priority=scenario_json.get('priority', 'medium'),
                    test_data=scenario_json.get('steps', []),
                    expected_result=scenario_json.get('expected_result', '')
                )
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating cross-field scenarios: {e}")
            return self.generate_basic_cross_field_scenarios(fields)
    
    def generate_basic_cross_field_scenarios(self, fields: List[FormField]) -> List[TestScenario]:
        """Basic cross-field scenarios without AI"""
        scenarios = []
        field_types = [f.field_type for f in fields]
        field_names = [f.name for f in fields]
        
        # Basic scenarios based on field combinations
        if any(f.required for f in fields):
            scenarios.append(TestScenario(
                scenario_id="req_fields_empty",
                title="Submit form with required fields empty",
                description="Test form validation when required fields are not filled",
                test_type="negative",
                priority="high",
                test_data=[
                    "Navigate to the form",
                    "Leave all required fields empty",
                    "Click submit button",
                    "Verify validation errors appear"
                ],
                expected_result="Form shows validation errors and prevents submission"
            ))
            
            # Test filling only some required fields
            scenarios.append(TestScenario(
                scenario_id="partial_req_fields",
                title="Submit form with only some required fields filled",
                description="Test partial completion of required fields",
                test_type="negative",
                priority="high",
                test_data=[
                    "Fill only first required field",
                    "Leave other required fields empty",
                    "Attempt to submit",
                    "Verify specific field validation errors"
                ],
                expected_result="Form shows errors for empty required fields only"
            ))
        
        # Password confirmation scenario
        if "password" in field_types and any("confirm" in f.name.lower() for f in fields):
            scenarios.append(TestScenario(
                scenario_id="password_mismatch",
                title="Password and confirmation mismatch",
                description="Test password confirmation validation",
                test_type="negative",
                priority="high",
                test_data=[
                    "Enter valid password in password field",
                    "Enter different password in confirmation field",
                    "Submit form",
                    "Verify mismatch error"
                ],
                expected_result="Form shows password mismatch error"
            ))
        
        # Email uniqueness (if registration form)
        if "email" in field_types and any("name" in f.field_type for f in fields):
            scenarios.append(TestScenario(
                scenario_id="duplicate_email",
                title="Registration with existing email",
                description="Test duplicate email handling in registration",
                test_type="negative",
                priority="medium",
                test_data=[
                    "Fill form with existing email address",
                    "Complete all other required fields",
                    "Submit form",
                    "Verify duplicate email error"
                ],
                expected_result="System shows email already exists error"
            ))
        
        # Age validation with date of birth
        if "age" in field_types and "date" in field_types:
            scenarios.append(TestScenario(
                scenario_id="age_dob_mismatch",
                title="Age and date of birth mismatch",
                description="Test consistency between age and date of birth fields",
                test_type="negative",
                priority="medium",
                test_data=[
                    "Enter age: 25",
                    "Enter birth date that indicates age 30",
                    "Submit form",
                    "Verify consistency error"
                ],
                expected_result="Form shows age/DOB inconsistency error"
            ))
        
        # Phone and country code consistency
        if "phone" in field_types and "country" in field_types:
            scenarios.append(TestScenario(
                scenario_id="phone_country_mismatch",
                title="Phone format and country mismatch",
                description="Test phone number format validation against selected country",
                test_type="negative",
                priority="medium",
                test_data=[
                    "Select country: United States",
                    "Enter phone: +44 20 7946 0958 (UK format)",
                    "Submit form",
                    "Verify format mismatch error"
                ],
                expected_result="Form shows phone format error for selected country"
            ))
        
        # Credit card validation
        if "card-number" in field_types and "cvv" in field_types:
            scenarios.append(TestScenario(
                scenario_id="card_cvv_mismatch",
                title="Credit card number and CVV mismatch",
                description="Test CVV length validation based on card type",
                test_type="negative",
                priority="high",
                test_data=[
                    "Enter Visa card number (starts with 4)",
                    "Enter 4-digit CVV (Amex format)",
                    "Submit form",
                    "Verify CVV length error"
                ],
                expected_result="Form shows incorrect CVV length for card type"
            ))
        
        # Comprehensive positive scenario
        scenarios.append(TestScenario(
            scenario_id="complete_valid_submission",
            title="Complete form with all valid data",
            description="Test successful form submission with all fields properly filled",
            test_type="positive",
            priority="high",
            test_data=[
                "Fill all required fields with valid data",
                "Fill optional fields with valid data",
                "Verify all validations pass",
                "Submit form"
            ],
            expected_result="Form submits successfully without errors"
        ))
        
        # Data persistence scenario
        scenarios.append(TestScenario(
            scenario_id="form_data_persistence",
            title="Form data persistence on validation error",
            description="Test that form retains user input after validation error",
            test_type="positive",
            priority="medium",
            test_data=[
                "Fill form with mostly valid data",
                "Enter invalid data in one field",
                "Submit form",
                "Verify error shown but other data retained"
            ],
            expected_result="Invalid field shows error, valid fields retain entered data"
        ))
        
        # Tab order and accessibility
        scenarios.append(TestScenario(
            scenario_id="keyboard_navigation",
            title="Keyboard-only form navigation",
            description="Test complete form filling using only keyboard navigation",
            test_type="positive",
            priority="medium",
            test_data=[
                "Use Tab key to navigate through all fields",
                "Fill each field using keyboard only",
                "Submit using Enter or Space",
                "Verify logical tab order"
            ],
            expected_result="All fields accessible via keyboard with logical tab order"
        ))
        
        return scenarios

# --- Advanced Screenshot Analysis ---

class AdvancedScreenshotAnalyzer:
    """Enhanced screenshot analysis with AI and computer vision"""
    
    def __init__(self):
        self.ocr_cache = {}
        self.analysis_cache = {}
    
    def analyze_screenshot_with_ai(self, image: Image.Image, context: Dict = None) -> Dict:
        """Analyze screenshot using AI for better field detection"""
        if not AI_AVAILABLE:
            return self.analyze_screenshot_traditional(image)
        
        try:
            # Convert image to base64 for AI analysis
            import io
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Enhanced OCR analysis
            ocr_results = self.enhanced_ocr_analysis(image)
            
            prompt = f"""
            Analyze this form screenshot for comprehensive test data generation:
            
            OCR Results: {json.dumps(ocr_results[:20], indent=2)}  # Limit for token efficiency
            Context: {json.dumps(context or {}, indent=2)}
            
            Identify and categorize all form elements:
            1. Input fields (with labels, placeholders, types)
            2. Dropdown/select elements
            3. Checkboxes and radio buttons
            4. Buttons (submit, reset, cancel)
            5. Form sections and groupings
            6. Validation indicators
            7. Help text and instructions
            
            For each field, provide:
            - Field type (email, password, text, etc.)
            - Label text
            - Placeholder text
            - Required indicator
            - Validation hints
            - Accessibility considerations
            
            Return ONLY valid JSON without any markdown formatting or code blocks:
            {{
                "fields": [
                    {{
                        "type": "field_type",
                        "label": "field_label",
                        "placeholder": "placeholder_text",
                        "required": true,
                        "validation_hints": ["hint1", "hint2"],
                        "accessibility_notes": ["note1", "note2"],
                        "suggested_test_types": ["valid", "invalid", "boundary"]
                    }}
                ],
                "form_structure": {{
                    "sections": ["section1", "section2"],
                    "layout": "single_column",
                    "interaction_patterns": ["pattern1", "pattern2"]
                }},
                "testing_priorities": [
                    "priority1", "priority2", "priority3"
                ]
            }}
            """
            
            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Clean up response to ensure valid JSON
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response as JSON: {e}")
                logger.error(f"Response was: {cleaned_response}")
                # Return fallback structure
                return {
                    "fields": [],
                    "form_structure": {"sections": ["main"], "layout": "unknown", "interaction_patterns": []},
                    "testing_priorities": ["Field validation", "User input handling"]
                }
            
        except Exception as e:
            logger.error(f"Error in AI screenshot analysis: {e}")
            return self.analyze_screenshot_traditional(image)
    
    def enhanced_ocr_analysis(self, image: Image.Image) -> List[Dict]:
        """Enhanced OCR with preprocessing and confidence scoring"""
        try:
            # Multi-stage preprocessing for better OCR
            processed_images = [
                self.preprocess_image_stage1(image),
                self.preprocess_image_stage2(image),
                self.preprocess_image_stage3(image)
            ]
            
            all_results = []
            
            for processed_img in processed_images:
                # Get detailed OCR data
                ocr_data = pytesseract.image_to_data(
                    processed_img, 
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Uniform block of text
                )
                
                # Process OCR results
                for i in range(len(ocr_data['text'])):
                    text = ocr_data['text'][i].strip()
                    confidence = ocr_data['conf'][i]
                    
                    if text and confidence > 30:  # Only high-confidence results
                        all_results.append({
                            'text': text,
                            'confidence': confidence,
                            'left': ocr_data['left'][i],
                            'top': ocr_data['top'][i],
                            'width': ocr_data['width'][i],
                            'height': ocr_data['height'][i],
                            'level': ocr_data['level'][i]
                        })
            
            # Deduplicate and sort by confidence
            unique_results = {}
            for result in all_results:
                key = result['text'].lower()
                if key not in unique_results or result['confidence'] > unique_results[key]['confidence']:
                    unique_results[key] = result
            
            return sorted(unique_results.values(), key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"Enhanced OCR analysis failed: {e}")
            return []
    
    def preprocess_image_stage1(self, image: Image.Image) -> Image.Image:
        """First stage preprocessing - basic cleanup"""
        # Convert to grayscale
        img = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Apply slight blur to reduce noise
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def preprocess_image_stage2(self, image: Image.Image) -> Image.Image:
        """Second stage preprocessing - aggressive cleanup"""
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image.convert('L'))
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return Image.fromarray(processed)
    
    def preprocess_image_stage3(self, image: Image.Image) -> Image.Image:
        """Third stage preprocessing - edge enhancement"""
        img_array = np.array(image.convert('L'))
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Apply sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        return Image.fromarray(sharpened)
    
    def analyze_screenshot_traditional(self, image: Image.Image) -> Dict:
        """Traditional screenshot analysis without AI"""
        # Use existing implementation
        extracted_labels = extract_form_fields(image)
        
        return {
            "fields": [
                {
                    "type": infer_field_type(label),
                    "label": label,
                    "placeholder": "",
                    "required": False,
                    "validation_hints": [],
                    "accessibility_notes": [],
                    "suggested_test_types": ["valid", "invalid", "boundary"]
                }
                for label in extracted_labels
            ],
            "form_structure": {
                "sections": ["main"],
                "layout": "single_column",
                "interaction_patterns": ["standard_form"]
            },
            "testing_priorities": [
                "Field validation",
                "Required field handling",
                "Data format compliance"
            ]
        }

# --- Helper functions for test data generation ---

def infer_field_type(label):
    """
    Enhanced field type detection using patterns and contextual clues
    """
    label = label.lower().strip()

    # Email pattern detection
    if any(term in label for term in ["email", "e-mail", "mail"]):
        return "email"

    # Password pattern detection
    if any(term in label for term in ["password", "pass", "pwd", "secret"]):
        return "password"

    # Phone number pattern detection
    if any(term in label for term in ["phone", "mobile", "cell", "tel", "contact number"]):
        return "phone"

    # Date pattern detection
    if any(term in label for term in ["date", "day", "dob", "birth", "calendar"]):
        if "birth" in label or "dob" in label:
            return "date-of-birth"
        return "date"

    # Domain name detection
    if any(term in label for term in ["domain", "domain name", "hostname", "website name", "site name"]):
        return "domain-name"

    # Numeric field detection
    if any(term in label for term in ["number", "num", "amount", "quantity", "count", "zip", "postal", "age"]):
        if "zip" in label and "code" in label:
            return "zip"
        if "postal" in label and "code" in label:
            return "zip"
        if any(term in label for term in ["zip", "postal"]):
            return "zip"
        if "age" in label:
            return "age"
        if any(term in label for term in ["amount", "price", "cost"]):
            return "currency"
        return "number"

    # Address detection
    if any(term in label for term in ["address", "street", "city", "state", "country"]):
        if "country" in label:
            return "country"
        if "state" in label or "province" in label:
            return "state"
        if "city" in label or "town" in label:
            return "city"
        if "zip" in label or "postal" in label:
            return "zip"
        return "address"

    # Name detection
    if any(term in label for term in ["first name", "last name", "surname", "fullname", "name"]):
        if "first" in label and "name" in label:
            return "first-name"
        if ("last" in label or "surname" in label or "family" in label) and "name" in label:
            return "last-name"
        if "user" in label:
            return "username"
        if "domain" in label:  # Additional check for domain names
            return "domain-name"
        else:
            return "name"

    # Credit card detection
    if any(term in label for term in ["card", "credit", "cc ", "cvv", "cvc", "expiry", "expiration"]):
        if any(term in label for term in ["cvv", "cvc", "security code", "security number"]):
            return "cvv"
        if any(term in label for term in ["expiry", "expiration", "exp "]):
            return "card-expiry"
        if "number" in label:
            return "card-number"
        return "payment"

    # URL/Website detection
    if any(term in label for term in ["url", "website", "site", "web", "link"]):
        return "url"

    # Default to text if no specific pattern is found
    return "text"


def generate_test_data(field_type):
    """
    Generate comprehensive test data based on field type with detailed test cases
    """
    valid_data = []
    invalid_data = []
    boundary_data = []
    edge_cases = []
    exploits = []

    # Common security exploits
    common_exploits = [
        "' OR '1'='1",  # SQL Injection
        "<script>alert(1)</script>",  # XSS
        "<img src=x onerror=alert(1)>",  # XSS image
        "eval(alert(1))",  # JavaScript injection
        "'; DROP TABLE users; --",  # SQL Injection
        "${7*7}",  # OGNL/Expression injection
        "{{7*7}}",  # Template injection
        "><svg/onload=alert(1)>",  # SVG XSS
        "</script><script>alert(1)</script>",  # Script tag breaking
        "\" autofocus onfocus=alert(1) \"",  # Attribute breaking
        "javascript:alert(1)",  # JavaScript protocol
    ]

    # Unicode/special character cases
    unicode_cases = [
        "测试测试",  # Chinese
        "Тестирование",  # Russian
        "😀👍💥",  # Emojis
        "ñáéíóúü",  # Spanish characters
        "∑ ∆ ∞ ♥ ♦ ♣ ♠",  # Symbols
        "الاختبار",  # Arabic
        "🚀🔥⚡️💻"  # More emojis
    ]

    # Add common exploits and unicode cases to the exploits list
    exploits.extend(common_exploits)
    edge_cases.extend(unicode_cases)

    if field_type == "domain-name":
        valid_data = [
            "example.com",
            "sub.example.com",
            "example-domain.co.uk",
            "my-domain123.org",
            "xn--80aswg.xn--p1ai",  # Punycode for IDN domains
            "domain.app",
            "test-1.io"
        ]
        invalid_data = [
            "example",  # Missing TLD
            "example..com",  # Double dot
            "-example.com",  # Starts with hyphen
            "example-.com",  # Ends with hyphen
            "example.com-",  # TLD with hyphen
            "example_domain.com",  # Underscore not allowed in domain names
            "example@domain.com",  # @ not allowed in domain names
            "domain.com/page",  # Path included
            "https://example.com"  # Protocol included
        ]
        boundary_data = [
            "a.io",  # Shortest valid domain (single letter + TLD)
            "a" * 63 + ".com",  # Maximum length for a domain segment
            "a." + "a" * 63 + ".com",  # Subdomain at max length
            "a" * 63 + "." + "b" * 63 + "." + "c" * 63 + ".com"  # Domain approaching 253 char total limit
        ]

    elif field_type == "email":
        valid_data = [
            "test@example.com",
            "user.name+tag@example.com",
            "user-name@example.co.uk",
            "a@b.co",
            "email@domain.com",
            "firstname.lastname@domain.com",
            "firstname+lastname@domain.com",
            "user123@test-domain.org",
            "test.email.123@company.net",
            "valid.email+special@sub.domain.co.uk",
            "user_underscore@domain.io",
            "numbers123@domain456.edu",
            "test.multiple.dots@example.com",
            "plus+symbol+test@gmail.com",
            "hyphens-allowed@test-domain.com",
            "dot.separated@test.com",
            "sub.domain@test.com",
            "user@sub.domain.com",
            "user@subdomain.domain.com",
            "user@domain.com"
        ]
        invalid_data = [
            "plainaddress",
            "@missingusername.com",
            "username@.com",
            "username@domain..com",
            "username@domain@domain.com",
            ".username@domain.com",
            "username@domain.com.",
            "username@-domain.com",
            "username@domain-.com",
            "user name@domain.com",  # Space in username
            "username@domain .com",  # Space in domain
            "username@@domain.com",  # Double @
            "username@domain,com",  # Comma instead of dot
            "",  # Empty string
            "user@",  # Missing domain
            "@domain.com",  # Missing username
            "user@domain",  # Missing TLD
            "user@.domain.com",  # Domain starts with dot
            "user@domain..com",  # Double dots
            "user@domain_underscore.com",  # Underscore in domain
            "user@domain#special.com",  # Special chars in domain
            "user@domain.c",  # TLD too short
            "user@domain.verylongtoleveldomain"  # TLD too long
        ]
        boundary_data = [
            "a@b.c",  # Minimum valid email
            "a" * 64 + "@example.com",  # Maximum local part
            "test@" + "a" * 251 + ".com",  # Close to maximum domain length
            "a" * 63 + "@" + "b" * 63 + ".com",  # Both parts at max
            "user@sub." + "a" * 60 + ".example.com",  # Long subdomain
            "test.email.with+many+symbols@example.com",  # Many special chars
            "1234567890@example.com",  # Numbers only in local
            "test@123456789.com",  # Numbers in domain
            "a@" + "b" * 63 + ".co",  # Max domain part with min TLD
            "test@example." + "a" * 6  # Max TLD length
        ]

    elif field_type == "password":
        valid_data = [
            "P@ssw0rd!",
            "Complex123!",
            "Very$tr0ngP@$$w0rd",
            "Abcd1234!",
            "P@55word",
            "Test-1234",
            "MyStr0ng#Password",
            "Secur3$P@ssw0rd",
            "C0mpl3x!ty2024",
            "P@$$w0rd$ecur3",
            "Str0ng&Secur3!",
            "T3$t1ng#2024",
            "V@l1d$P@ssw0rd",
            "S@f3ty!F1rst",
            "C0d3$3cur1ty!"
        ]
        invalid_data = [
            "password",
            "12345678",
            "abcdefgh",
            "short",
            "no_numbers",
            "NO_LOWERCASE",
            "no_uppercase123",
            "PASSWORD123",  # No lowercase
            "password123",  # No uppercase
            "Password",  # No numbers or symbols
            "Pass123",  # Too short
            "",  # Empty
            " ",  # Space only
            "P@ss",  # Too short
            "aaaaaaaa",  # All same character
            "Password1",  # No special characters
            "P@SSWORD123",  # No lowercase
            "p@ssword123",  # No uppercase
            "Password@",  # No numbers
            "12345678!",  # No letters
            "Pass word123!",  # Contains space
            "PasswordPasswordPassword",  # No numbers/special chars
            "123456789",  # Only numbers
            "!@#$%^&*()",  # Only special chars
            "Pa1!",  # Too short (4 chars)
            "commonpassword",  # Common weak password
            "qwerty123"  # Common keyboard pattern
        ]
        boundary_data = [
            "Aa1!",  # Minimum complexity (4 chars)
            "A" + "a" * 6 + "1!",  # Common minimum length (8)
            "A" + "a" * 62 + "1!",  # Maximum reasonable length (64)
            "Aa1" + "!" * 60,  # Max length with min chars
            "P@ssw0rd" + "1" * 56,  # 64 char password
            "A1!" + "a" * 61,  # Different pattern at boundary
            "Complex1!",  # 9 characters
            "VeryLongPasswordWith123!AndSpecialChars",  # 40+ characters
            "Short1!",  # 7 characters
            "ExactlyTwentyChar1!"  # Exactly 20 characters
        ]

    elif field_type == "phone":
        valid_data = [
            "1234567890",
            "(123) 456-7890",
            "+1 123-456-7890",
            "123.456.7890",
            "+12345678901"
        ]
        invalid_data = [
            "123-456",
            "abcdefghij",
            "(123)456-7890abc",
            "(123)4567890",
            "123 456 789"
        ]
        boundary_data = [
            "+1",  # Too short
            "+" + "1" * 20  # Too long
        ]

    elif field_type in ["number", "age", "zip"]:
        if field_type == "age":
            valid_data = ["18", "25", "65", "100"]
            invalid_data = ["-1", "200", "abc", "12.5"]
            boundary_data = ["0", "1", "120", "121"]
        elif field_type == "zip":
            valid_data = ["12345", "12345-6789"]
            invalid_data = ["1234", "123456", "abcde"]
            boundary_data = ["00000", "99999"]
        else:
            valid_data = ["123", "0", "999999", "-100", "3.14"]
            invalid_data = ["abc", "12a34", ""]
            boundary_data = ["-2147483648", "2147483647"]  # Common 32-bit integer bounds

    elif field_type == "date" or field_type == "date-of-birth":
        valid_data = [
            "2023-05-25",
            "05/25/2023",
            "25/05/2023",
            "May 25, 2023"
        ]
        if field_type == "date-of-birth":
            valid_data = [
                "1990-01-01",
                "01/15/1985",
                "Dec 25, 1970"
            ]
            boundary_data = [
                "1900-01-01",  # Very old
                "2023-05-26",  # Today
                "2005-05-26",  # ~18 years ago
                "1923-05-26"  # 100 years ago
            ]
        else:
            boundary_data = [
                "1970-01-01",  # Unix epoch start
                "2038-01-19",  # 32-bit Unix time overflow
                "2023-05-26",  # Today
                "2023-05-27",  # Tomorrow
                "2023-05-25"  # Yesterday
            ]
        invalid_data = [
            "2023-13-01",  # Invalid month
            "2023-02-30",  # Invalid day
            "abcde",
            "13/13/2023"
        ]

    elif field_type in ["first-name", "last-name", "name"]:
        valid_data = [
            "John",
            "Mary",
            "José",
            "Smith-Jones",
            "O'Connor"
        ]
        invalid_data = [
            "123",
            "User#1",
            ""
        ]
        boundary_data = [
            "A",  # Single character
            "A" * 50  # Very long name
        ]

    elif field_type == "username":
        valid_data = [
            "john_doe",
            "mary.smith",
            "user123",
            "test_user_2023"
        ]
        invalid_data = [
            "user name",  # Space
            "user@name",  # Special character
            "a"  # Too short
        ]
        boundary_data = [
            "ab",  # Minimum length often 2-3
            "a" * 30  # Maximum length often 30
        ]

    elif field_type == "address":
        valid_data = [
            "123 Main St",
            "456 Park Ave, Apt 7B",
            "1000 5th Avenue"
        ]
        invalid_data = [
            ""  # Empty
        ]
        boundary_data = [
            "A",  # Very short
            "A" * 100  # Very long
        ]

    elif field_type in ["city", "state", "country"]:
        if field_type == "city":
            valid_data = ["New York", "Los Angeles", "Chicago", "San Francisco"]
        elif field_type == "state":
            valid_data = ["California", "New York", "Texas", "FL", "CA", "NY"]
        else:  # country
            valid_data = ["United States", "Canada", "United Kingdom", "Australia", "US", "UK"]

        invalid_data = ["123", ""]
        boundary_data = ["A", "A" * 50]

    elif field_type == "currency":
        valid_data = ["100", "99.99", "1,000.00", "0.50", "-100"]
        invalid_data = ["abc", "$100", "100$", "100.999"]
        boundary_data = ["0", "0.01", "9999999.99"]

    elif field_type == "card-number":
        valid_data = [
            "4111111111111111",  # Visa
            "5500000000000004",  # Mastercard
            "340000000000009",  # Amex
            "6011000000000004"  # Discover
        ]
        invalid_data = [
            "411111111111",  # Too short
            "41111111111111111",  # Too long
            "0000000000000000",  # Invalid pattern
            "abcdefghijklmnop"  # Non-numeric
        ]
        boundary_data = [
            "4242424242424242",  # Test card
            "4111111111111111"  # Test card
        ]

    elif field_type == "cvv":
        valid_data = ["123", "1234"]
        invalid_data = ["12", "12345", "abc"]
        boundary_data = ["000", "999"]

    elif field_type == "card-expiry":
        valid_data = ["12/25", "01/30", "05/2025"]

        invalid_data = [
            "00/25",  # Invalid month
            "13/25",  # Invalid month
            "05/20",  # Past date
            "abcd"  # Non-date
        ]
        boundary_data = [
            "05/23",  # Current month
            "05/33"  # Far future
        ]

    elif field_type == "url":
        valid_data = [
            "https://www.example.com",
            "http://example.com/path?query=value",
            "https://subdomain.example.co.uk/path"
        ]
        invalid_data = [
            "example",
            "http://",
            "www.example",
            "https:/example.com"
        ]
        boundary_data = [
            "https://a.b",  # Minimum valid URL
            "https://" + "a" * 250 + ".com"  # Very long URL
        ]

    else:  # Default text field
        valid_data = [
            "Sample text",
            "123456",
            "Text with spaces",
            "Text-with-hyphens"
        ]
        invalid_data = [
            "",  # Empty
            " ",  # Whitespace only
            "   "  # Multiple whitespaces
        ]
        boundary_data = [
            "a",  # Minimum
            "a" * 100  # Maximum
        ]

    # Add some edge cases that apply to all fields
    edge_cases.extend([
        "",  # Empty string
        " ",  # Space
        "\t",  # Tab
        "\n",  # Newline
        "null",  # String "null"
        "undefined",  # String "undefined"
        "None",  # String "None"
        "\u200B"  # Zero-width space
    ])

    return valid_data, invalid_data, boundary_data, edge_cases, exploits


# --- Helper functions for history tracking ---

def get_history_file_path():
    """Get the file path for storing test generation history"""
    # Create in user's home directory to ensure persistence across sessions
    home_dir = os.path.expanduser("~")
    history_dir = os.path.join(home_dir, ".jarvis_test_data_history")

    # Create directory if it doesn't exist
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    return os.path.join(history_dir, "test_data_history.json")

def load_test_history():
    """Load test generation history from file"""
    history_file = get_history_file_path()
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history file: {e}")
    return []

def save_test_history(history_item):
    """Save a test generation to history"""
    history = load_test_history()

    # Add the new item to the history list
    history.append(history_item)

    # Keep only the most recent 50 records
    history = history[-50:]

    # Save the updated history
    history_file = get_history_file_path()
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f)
        return True
    except Exception as e:
        print(f"Error saving history file: {e}")
        return False

def create_history_item(name, field_types, num_fields):
    """Create a history item with metadata"""
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "field_count": num_fields,
        "field_types": field_types,
        "framework": st.session_state.get("selected_framework", "Selenium"),
        "language": st.session_state.get("selected_language", "Python")
    }


# --- Helper functions for image processing ---

def preprocess_image(image):
    """Preprocess the image to improve OCR quality with multiple techniques"""
    # Convert PIL image to OpenCV format
    img = np.array(image)
    
    # Convert to RGB if it's RGBA
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Resize image if it's too small (OCR works better on larger images)
    height, width = gray.shape
    if height < 600 or width < 800:
        scale_factor = max(600 / height, 800 / width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding for better text extraction
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    
    # Apply closing to connect broken text
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Apply opening to remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    
    # Apply noise removal
    denoised = cv2.fastNlMeansDenoising(opened, None, 10, 7, 21)
    
    # Apply sharpening to make text clearer
    kernel_sharpen = np.array([[-1, -1, -1], 
                              [-1,  9, -1], 
                              [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)
    
    return Image.fromarray(sharpened)


def unify_common_fields(labels):
    """
    Unify common fields and handle first/last name vs full name logic.
    "Name", "First name" and "Last name" fields are considered as one common field type.
    Also groups common fields that might have slight variations.
    """
    # Convert labels to lowercase for easier matching
    lowercase_labels = [label.lower() for label in labels]

    # Keep track of processed labels to avoid duplicates
    processed = []
    unified_labels = []

    # Common field groups that should be unified
    field_groups = {
        'name': ['name', 'full name', 'first name', 'firstname', 'first', 'last name', 'lastname', 'surname', 'last', 'given name'],
        'email': ['email', 'e-mail', 'email address', 'e-mail address', 'electronic mail', 'mail'],
        'phone': ['phone', 'telephone', 'mobile', 'cell', 'phone number', 'contact number', 'tel', 'cellular'],
        'address': ['address', 'street address', 'mailing address', 'home address', 'street'],
        'city': ['city', 'town', 'municipality'],
        'state': ['state', 'province', 'region', 'territory'],
        'country': ['country', 'nation', 'nationality'],
        'zip': ['zip', 'postal code', 'zip code', 'postcode', 'postal', 'zipcode'],
        'dob': ['date of birth', 'birth date', 'dob', 'birthdate', 'birthday'],
        'card': ['credit card', 'card number', 'card #', 'card no', 'payment card'],
        'password': ['password', 'passcode', 'pass', 'pwd', 'secret'],
        'username': ['username', 'user name', 'login', 'userid', 'user id'],
        'website': ['website', 'web site', 'url', 'homepage', 'site'],
        'company': ['company', 'organization', 'employer', 'business', 'firm'],
    }

    # Check for name-related fields
    has_first_name = any(
        any(term in label for term in ['first name', 'firstname', 'first']) for label in lowercase_labels)
    has_last_name = any(
        any(term in label for term in ['last name', 'lastname', 'surname', 'last']) for label in lowercase_labels)
    has_full_name = any(label == 'name' or label == 'full name' for label in lowercase_labels)

    # Process each label
    for i, label in enumerate(labels):
        if i in processed:
            continue

        label_lower = label.lower()

        # Special handling for name fields
        if any(name_term in label_lower for name_term in field_groups['name']):
            name_fields = []

            # If we have first and last name, prioritise those over a generic "name" field
            if has_first_name and has_last_name:
                # Find the best first name field
                for j, name_label in enumerate(labels):
                    if any(term in name_label.lower() for term in ['first name', 'firstname', 'first']):
                        name_fields.append(name_label)
                        processed.append(j)
                        break

                # Find the best last name field
                for j, name_label in enumerate(labels):
                    if any(term in name_label.lower() for term in ['last name', 'lastname', 'surname', 'last']):
                        name_fields.append(name_label)
                        processed.append(j)
                        break

                # Add full name only if it's distinct (not just "Name")
                if has_full_name:
                    for j, name_label in enumerate(labels):
                        name_lower = name_label.lower()
                        if (name_lower == 'full name' or
                                ('name' in name_lower and 'first' not in name_lower and 'last' not in name_lower)):
                            if name_lower != 'name':  # Skip generic "Name" if we have first and last
                                name_fields.append(name_label)
                            processed.append(j)
            else:
                # If we don't have both first and last, just add whatever name fields we have
                for j, name_label in enumerate(labels):
                    if any(term in name_label.lower() for term in field_groups['name']):
                        name_fields.append(name_label)
                        processed.append(j)

            # Add unique name fields to unified list
            for name_field in name_fields:
                if not any(unified_label.lower() == name_field.lower() for unified_label in unified_labels):
                    unified_labels.append(name_field)

            # Skip further processing for this label
            processed.append(i)
            continue

        # Check if this label belongs to a field group
        matched_group = None
        for group_name, variations in field_groups.items():
            if group_name == 'name':  # Already handled above
                continue

            if any(variation in label_lower for variation in variations):
                matched_group = group_name
                break

        if matched_group:
            # Find the best label from the field group (prefer the original field names if possible)
            added = False
            for preferred_label in labels:
                preferred_lower = preferred_label.lower()
                # Add the best version of this field (most specific/clearest)
                if any(variation in preferred_lower for variation in field_groups[matched_group]):
                    if not any(unified_label.lower() == preferred_label.lower() for unified_label in unified_labels):
                        unified_labels.append(preferred_label)
                        added = True
                        break

            if not added:
                # If no good version found, standardise based on the group
                standard_names = {
                    'email': 'Email Address',
                    'phone': 'Phone Number',
                    'address': 'Address',
                    'city': 'City',
                    'state': 'State/Province',
                    'country': 'Country',
                    'zip': 'Zip/Postal Code',
                    'dob': 'Date of Birth',
                    'card': 'Card Number'
                }
                unified_labels.append(standard_names.get(matched_group, matched_group.title()))

            # Mark all variations as processed
            for j, other_label in enumerate(labels):
                if any(variation in other_label.lower() for variation in field_groups[matched_group]):
                    processed.append(j)
        else:
            # This label doesn't belong to any field group, add it as is
            if not any(unified_label.lower() == label.lower() for unified_label in unified_labels):
                unified_labels.append(label)
            processed.append(i)

    return unified_labels


def extract_form_fields(image):
    """
    Enhanced form field extraction with improved accuracy and multiple detection methods
    """
    # Try multiple preprocessing approaches
    preprocessed_images = [
        preprocess_image(image),
        image,  # Original image
        ImageEnhance.Contrast(image).enhance(2.0),  # High contrast
        ImageEnhance.Sharpness(image).enhance(2.0)  # Enhanced sharpness
    ]
    
    all_potential_labels = []
    
    # Enhanced form field indicators with more comprehensive patterns
    form_field_indicators = [
        # Common form fields
        'name', 'email', 'phone', 'address', 'username', 'password',
        'date', 'number', 'card', 'zip', 'postal', 'code',
        'city', 'state', 'country', 'first', 'last', 'domain',
        
        # Additional patterns
        'login', 'signin', 'signup', 'register', 'contact',
        'message', 'subject', 'company', 'organization',
        'title', 'description', 'comment', 'feedback',
        'birth', 'age', 'gender', 'occupation', 'website',
        'mobile', 'telephone', 'fax', 'extension',
        
        # Payment fields
        'credit', 'debit', 'payment', 'billing', 'cvv', 'cvc',
        'expiry', 'expiration', 'cardholder', 'security',
        
        # Additional common fields
        'confirm', 'verify', 'repeat', 'retype',
        
        # Extended field patterns for better detection
        'full', 'given', 'family', 'surname', 'middle',
        'street', 'apartment', 'unit', 'building', 'floor',
        'province', 'region', 'territory', 'county',
        'zipcode', 'postcode', 'area', 'pin',
        'mobile', 'landline', 'home', 'work', 'office',
        'personal', 'business', 'corporate',
        'account', 'userid', 'id', 'identifier',
        'pin', 'ssn', 'social', 'national',
        'dob', 'birthday', 'anniversary',
        'website', 'url', 'link', 'homepage',
        'comment', 'note', 'remark', 'additional',
        'prefer', 'choice', 'select', 'option'
    ]
    
    # Process each preprocessed image
    for processed_image in preprocessed_images:
        try:
            # Run OCR with multiple configurations for better detection
            configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :*()[]{}.-_@#$%&',
                r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :*()[]{}.-_@#$%&',
                r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :*()[]{}.-_@#$%&',
                r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 :*()[]{}.-_@#$%&'
            ]
            
            all_ocr_results = []
            
            for config in configs:
                try:
                    ocr_data = pytesseract.image_to_data(
                        processed_image, 
                        output_type=pytesseract.Output.DICT,
                        config=config
                    )
                    all_ocr_results.append(ocr_data)
                except:
                    continue
            
            # Combine results from all OCR configurations
            potential_labels = []
            
            for ocr_data in all_ocr_results:
                # Group text elements by spatial position
                elements = []
                confidence_threshold = 30  # Lower threshold for more inclusive detection
                
                for i, (text, left, top, width, height, conf) in enumerate(
                    zip(ocr_data['text'], ocr_data['left'], ocr_data['top'], 
                        ocr_data['width'], ocr_data['height'], ocr_data['conf'])):
                    
                    if text.strip() and conf >= confidence_threshold:
                        elements.append({
                            'index': i,
                            'text': text.strip(),
                            'left': left,
                            'top': top,
                            'right': left + width,
                            'bottom': top + height,
                            'confidence': conf,
                            'width': width,
                            'height': height
                        })
                
                # Enhanced label detection strategies
                
                # 1. Look for labels with colons (most reliable indicator)
                for element in elements:
                    text = element['text']
                    if ':' in text:
                        label = text.split(':')[0].strip()
                        if len(label) > 1:
                            potential_labels.append(label)
                
                # 2. Look for form field indicator words with better context
                for element in elements:
                    text = element['text'].lower()
                    original_text = element['text']
                    
                    # Check for exact matches and compound terms
                    for indicator in form_field_indicators:
                        if indicator in text:
                            # Check for word boundaries to avoid partial matches
                            if re.search(r'\b' + re.escape(indicator) + r'\b', text):
                                # Extract the complete label (may be multi-word)
                                words = original_text.split()
                                for i, word in enumerate(words):
                                    if indicator in word.lower():
                                        # Include surrounding words for context
                                        start_idx = max(0, i - 1)
                                        end_idx = min(len(words), i + 3)  # Include more context
                                        label = ' '.join(words[start_idx:end_idx])
                                        potential_labels.append(label.strip())
                                        break
                
                # 3. Look for asterisk (*) and "required" indicators
                for i, element in enumerate(elements):
                    if '*' in element['text'] or 'required' in element['text'].lower():
                        # Check nearby elements for the actual label
                        for nearby in elements:
                            distance = abs(nearby['top'] - element['top'])
                            if (distance < 50 and  # Increased tolerance for same line
                                nearby['left'] < element['left'] and  # To the left of the indicator
                                len(nearby['text']) > 1):
                                potential_labels.append(nearby['text'].rstrip(':*').strip())
                
                # 4. Spatial analysis - look for text near form input areas
                for element in elements:
                    text = element['text']
                    if (len(text) > 2 and 
                        not text.isdigit() and 
                        not text.lower() in ['ok', 'cancel', 'submit', 'reset', 'clear', 'button', 'click', 'here'] and
                        any(char.isalpha() for char in text)):
                        
                        # Check if this could be a form label based on position and content
                        if any(indicator in text.lower() for indicator in form_field_indicators):
                            potential_labels.append(text.rstrip(':*').strip())
                        
                        # Also check for common form label patterns
                        if re.search(r'\b(enter|type|input|select|choose)\b', text.lower()):
                            potential_labels.append(text.rstrip(':*').strip())
                
                # 5. Pattern matching for common form structures
                for element in elements:
                    text = element['text']
                    
                    # Match patterns like "Enter your email", "Your name", etc.
                    patterns = [
                        r'enter\s+(?:your\s+)?(\w+(?:\s+\w+)?)',
                        r'your\s+(\w+(?:\s+\w+)?)',
                        r'provide\s+(?:your\s+)?(\w+(?:\s+\w+)?)',
                        r'type\s+(?:your\s+)?(\w+(?:\s+\w+)?)',
                        r'input\s+(?:your\s+)?(\w+(?:\s+\w+)?)',
                        r'(\w+(?:\s+\w+)?)\s*\*',  # Text followed by asterisk
                        r'(\w+(?:\s+\w+)?)\s*\(required\)',  # Text followed by (required)
                    ]
                    
                    for pattern in patterns:
                        matches = re.finditer(pattern, text.lower())
                        for match in matches:
                            extracted = match.group(1) if match.lastindex else match.group(0)
                            if any(indicator in extracted for indicator in form_field_indicators):
                                potential_labels.append(extracted.strip())
                
                # 6. Look for field placeholder text and hints
                for element in elements:
                    text = element['text'].lower()
                    if any(hint in text for hint in ['placeholder', 'hint', 'example', 'e.g.', 'format']):
                        # Look for nearby text that might be the actual label
                        for nearby in elements:
                            if (abs(nearby['top'] - element['top']) < 100 and  # Nearby vertically
                                abs(nearby['left'] - element['left']) < 200 and  # Nearby horizontally
                                len(nearby['text']) > 1 and
                                any(indicator in nearby['text'].lower() for indicator in form_field_indicators)):
                                potential_labels.append(nearby['text'].rstrip(':*').strip())
                
                # 7. Detect labels based on text size and position (likely form labels)
                if elements:
                    # Find text elements that are likely labels based on size and position
                    avg_height = sum(e['height'] for e in elements) / len(elements)
                    avg_width = sum(e['width'] for e in elements) / len(elements)
                    
                    for element in elements:
                        text = element['text']
                        # Labels are often slightly larger than average text
                        if (element['height'] >= avg_height * 0.8 and
                            element['width'] >= avg_width * 0.5 and
                            len(text) > 1 and
                            not text.isdigit() and
                            any(char.isalpha() for char in text) and
                            len(text.split()) <= 4):  # Labels are usually 1-4 words
                            
                            # Additional checks for form label characteristics
                            if (any(indicator in text.lower() for indicator in form_field_indicators[:20]) or  # Check common indicators
                                re.search(r'\b(full|first|last|user|pass|confirm|email|phone|address|city|state|zip|country|date|birth)\b', text.lower())):
                                potential_labels.append(text.rstrip(':*').strip())
            
            all_potential_labels.extend(potential_labels)
            
        except Exception as e:
            logger.warning(f"OCR processing failed for one image variant: {e}")
            continue
    
    # Clean and deduplicate labels
    cleaned_labels = []
    seen_labels = set()
    
    for label in all_potential_labels:
        # Clean up the label
        clean_label = re.sub(r'[^\w\s]', '', label).strip()
        clean_label = ' '.join(clean_label.split())  # Normalize whitespace
        
        if (len(clean_label) > 1 and 
            clean_label.lower() not in seen_labels and
            clean_label.lower() not in ['and', 'the', 'for', 'or', 'of', 'to', 'in', 'on', 'is', 'are', 'was', 'were']):
            
            cleaned_labels.append(clean_label)
            seen_labels.add(clean_label.lower())
    
    # Unify common fields
    unified_labels = unify_common_fields(cleaned_labels)
    
    # Final verification and scoring
    verified_labels = []
    for label in unified_labels:
        label_lower = label.lower()
        score = 0
        
        # Score based on form field indicators
        for indicator in form_field_indicators:
            if indicator in label_lower:
                score += 2
                if label_lower == indicator:  # Exact match gets higher score
                    score += 3
        
        # Score based on common form patterns
        if any(keyword in label_lower for keyword in ['enter', 'your', 'select', 'choose', 'input', 'fill']):
            score += 1
        
        # Score based on label structure
        if re.search(r'^\w+(?:\s+\w+){0,2}$', label):  # 1-3 words
            score += 1
        
        if score >= 2:  # Minimum threshold for inclusion
            verified_labels.append(label)
    
    # If no labels found with high confidence, try a more permissive approach
    if not verified_labels:
        for label in unified_labels:
            if any(indicator in label.lower() for indicator in form_field_indicators[:10]):  # Top 10 most common
                verified_labels.append(label)
    
    return verified_labels


def generate_test_scenarios(field_type, test_data):
    """
    Generate specific test scenarios for each field type
    """
    valid_data, invalid_data, boundary_data, edge_cases, exploits = test_data

    scenarios = []

    # Generate positive test scenarios
    for data in valid_data[:3]:  # Limit to first few to avoid too much data
        scenarios.append({
            "type": "positive",
            "description": f"Submit form with valid {field_type} input: '{data}'",
            "expected_result": "Form submission succeeds",
            "test_data": data
        })

    # Generate negative test scenarios
    for data in invalid_data[:2]:  # Limit to first few to avoid too much data
        scenarios.append({
            "type": "negative",
            "description": f"Submit form with invalid {field_type} input: '{data}'",
            "expected_result": "Form validation prevents submission and shows error message",
            "test_data": data
        })

    # Generate boundary test scenarios
    for data in boundary_data[:2]:
        scenarios.append({
            "type": "boundary",
            "description": f"Submit form with boundary {field_type} value: '{data}'",
            "expected_result": "System handles boundary value appropriately",
            "test_data": data
        })

    # Generate security test scenarios
    for data in exploits[:2]:
        scenarios.append({
            "type": "security",
            "description": f"Submit form with potentially malicious input in {field_type} field",
            "expected_result": "System sanitises or rejects harmful input",
            "test_data": data
        })

    return scenarios


def get_field_constraints(field_type):
    """
    Get expected constraints based on field type
    """
    constraints = {
        "min_length": None,
        "max_length": None,
        "pattern": None,
        "allowed_chars": None,
        "format": None,
        "html_type": "text",
        "validation": []
    }

    if field_type == "email":
        constraints.update({
            "min_length": 3,
            "max_length": 320,  # RFC 5321
            "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "allowed_chars": "Alphanumeric + @._%-+",
            "html_type": "email",
            "validation": ["Format validation", "Domain validation"]
        })

    elif field_type == "domain-name":
        constraints.update({
            "min_length": 3,
            "max_length": 253,  # RFC 1035
            "pattern": r"^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$",
            "allowed_chars": "Letters, numbers, hyphens (not at start/end of segments), period as separator",
            "format": "example.com, sub.example.co.uk",
            "html_type": "text",
            "validation": ["DNS format validation", "TLD validation", "IDN compatibility check"]
        })

    elif field_type == "password":
        constraints.update({
            "min_length": 8,
            "max_length": 64,
            "pattern": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
            "allowed_chars": "Alphanumeric + special characters",
            "html_type": "password",
            "validation": ["Complexity requirements", "Common password check"]
        })

    elif field_type == "phone":
        constraints.update({
            "min_length": 7,
            "max_length": 15,
            "pattern": r"^\+?[0-9\s\-()\.]{7,20}$",
            "allowed_chars": "Digits, spaces, +()- characters",
            "html_type": "tel",
            "validation": ["Format validation"]
        })

    elif field_type == "number":
        constraints.update({
            "pattern": r"^-?\d+(\.\d+)?$",
            "allowed_chars": "Digits, decimal point, minus sign",
            "html_type": "number",
            "validation": ["Numeric validation"]
        })

    elif field_type == "age":
        constraints.update({
            "min_length": 1,
            "max_length": 3,
            "pattern": r"^\d{1,3}$",
            "allowed_chars": "Digits only",
            "html_type": "number",
            "validation": ["Range validation (0-120)"]
        })

    elif field_type == "zip":
        constraints.update({
            "min_length": 5,
            "max_length": 10,
            "pattern": r"^\d{5}(-\d{4})?$",  # US format
            "allowed_chars": "Digits, hyphen",
            "html_type": "text",
            "validation": ["Format validation"]
        })

    elif field_type == "date" or field_type == "date-of-birth":
        constraints.update({
            "pattern": r"^\d{4}-\d{2}-\d{2}$",  # ISO format
            "allowed_chars": "Digits, hyphens, slashes",
            "html_type": "date",
            "format": "YYYY-MM-DD",
            "validation": ["Date validation"]
        })
        if field_type == "date-of-birth":
            constraints["validation"].append("Age verification")

    elif field_type in ["name", "first-name", "last-name"]:
        constraints.update({
            "min_length": 1,
            "max_length": 50,
            "pattern": r"^[A-Za-z\s\-'\.]+$",
            "allowed_chars": "Letters, spaces, hyphens, apostrophes",
            "html_type": "text",
            "validation": ["Character validation"]
        })

    elif field_type == "username":
        constraints.update({
            "min_length": 3,
            "max_length": 30,
            "pattern": r"^[a-zA-Z0-9_\.]+$",
            "allowed_chars": "Alphanumeric, underscore, period",
            "html_type": "text",
            "validation": ["Uniqueness check"]
        })

    elif field_type == "address":
        constraints.update({
            "min_length": 1,
            "max_length": 100,
            "allowed_chars": "Alphanumeric, spaces, common punctuation",
            "html_type": "text",
            "validation": ["Format validation"]
        })

    elif field_type in ["city", "state", "country"]:
        constraints.update({
            "min_length": 1,
            "max_length": 50,
            "allowed_chars": "Letters, spaces, hyphens",
            "html_type": "text",
            "validation": ["Lookup validation"]
        })

    elif field_type == "currency":
        constraints.update({
            "pattern": r"^-?\d+(\.\d{1,2})?$",
            "allowed_chars": "Digits, decimal point, minus sign",
            "html_type": "number",
            "format": "0.00",
            "validation": ["Numeric validation", "Range validation"]
        })

    elif field_type == "card-number":
        constraints.update({
            "min_length": 13,
            "max_length": 19,
            "pattern": r"^\d{13,19}$",
            "allowed_chars": "Digits only",
            "html_type": "text",
            "validation": ["Luhn algorithm", "Card type validation"]
        })

    elif field_type == "cvv":
        constraints.update({
            "min_length": 3,
            "max_length": 4,
            "pattern": r"^\d{3,4}$",
            "allowed_chars": "Digits only",
            "html_type": "text",
            "validation": ["Length validation"]
        })

    elif field_type == "card-expiry":
        constraints.update({
            "pattern": r"^(0[1-9]|1[0-2])\/([0-9]{2}|[0-9]{4})$",
            "allowed_chars": "Digits, slash",
            "html_type": "text",
            "format": "MM/YY or MM/YYYY",
            "validation": ["Date validation", "Future date validation"]
        })

    elif field_type == "url":
        constraints.update({
            "min_length": 3,
            "max_length": 2048,  # Common browser limit
            "pattern": r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$",
            "allowed_chars": "URL valid characters",
            "html_type": "url",
            "validation": ["URL validation"]
        })

    else:  # Default text field
        constraints.update({
            "min_length": 1,
            "max_length": 255,
            "allowed_chars": "Any characters",
            "html_type": "text",
            "validation": []
        })

    return constraints


def get_html_attributes(field_type, constraints):
    """
    Generate recommended HTML attributes for the field
    """
    attributes = []

    # Add type attribute
    attributes.append(f'type="{constraints["html_type"]}"')

    # Add min/max length if applicable
    if constraints["min_length"]:
        attributes.append(f'minlength="{constraints["min_length"]}"')
    if constraints["max_length"]:
        attributes.append(f'maxlength="{constraints["max_length"]}"')

    # Add pattern if available
    if constraints["pattern"]:
        # Remove the ^ and $ from the pattern for HTML pattern attribute
        html_pattern = constraints["pattern"].replace("^", "").replace("$", "")
        attributes.append(f'pattern="{html_pattern}"')

    # Add specific attributes based on field type
    if constraints["html_type"] == "email":
        attributes.append('autocomplete="email"')
    elif constraints["html_type"] == "password":
        attributes.append('autocomplete="new-password"')
    elif constraints["html_type"] == "tel":
        attributes.append('autocomplete="tel"')
    elif field_type == "zip":
        attributes.append('autocomplete="postal-code"')
    elif field_type == "card-number":
        attributes.append('autocomplete="cc-number"')
    elif field_type == "card-expiry":
        attributes.append('autocomplete="cc-exp"')
    elif field_type == "cvv":
        attributes.append('autocomplete="cc-csc"')

    # Common attributes
    if field_type != "password":  # Don't add spellcheck for password fields
        attributes.append('spellcheck="false"')

    # Required attribute placeholder
    attributes.append('required')

    return attributes


# --- Enhanced Main Streamlit UI ---

def show_ui():
    """Enhanced UI with AI-powered features and website traversal"""

    # Sidebar configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Theme selection
        theme = st.selectbox("🎨 Theme", ["Light", "Dark", "Auto"])
        
        # AI Status
        ai_status = "🤖 AI-Enabled" if AI_AVAILABLE else "⚠️ Basic Mode"
        st.info(f"Status: {ai_status}")
        
        # Quick stats
        if 'manual_fields' in st.session_state and st.session_state.manual_fields:
            st.write("**Current Form:**")
            st.write(f"• Fields: {len(st.session_state.manual_fields)}")
            st.write(f"• Required: {sum(1 for f in st.session_state.manual_fields if f.required)}")
        
        # Version info
        st.write("---")
        st.write("**Version:** 2.0 Enhanced")
        st.write("**Features:** All Active")

    # Apply enhanced styling
    theme_css = get_theme_css(theme)
    st.markdown(theme_css, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🧪 AI-Powered Intelligent Test Data Generator</h1>', unsafe_allow_html=True)
    
    # Enhanced status indicator with more details
    status_text = "🤖 AI-Powered" if AI_AVAILABLE else "⚠️ Basic Mode"
    feature_text = "Full AI Analysis | Website Traversal | Advanced Test Generation"
    st.markdown(f'<p class="status-text">{status_text} | {feature_text}</p>', unsafe_allow_html=True)

    # Feature highlights
    with st.expander("✨ New in Version 2.0 - Click to see enhancements!", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔧 Enhanced Test Data:**
            - 15+ email test cases (vs 7 before)
            - 26+ password test cases (vs 7 before)  
            - Comprehensive boundary testing
            - Advanced security test scenarios
            """)
        
        with col2:
            st.markdown("""
            **🎯 Accessibility & Localization:**
            - Screen reader compatibility tests
            - Keyboard navigation scenarios
            - Multi-language text expansion tests
            - Color contrast validation
            """)
        
        with col3:
            st.markdown("""
            **🚀 New Features:**
            - Cross-field scenario generation (6+ scenarios)
            - Performance testing tools
            - Data export (CSV, Excel)
            - Test script generation
            - Advanced analytics dashboard
            """)

    # Main tabs for different input methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "📸 Screenshot Analysis", 
        "🕷️ Website Traversal", 
        "✏️ Manual Entry",
        "📊 Advanced Analytics"
    ])

    with tab1:
        show_screenshot_analysis_tab()
    
    with tab2:
        show_website_traversal_tab()
    
    with tab3:
        show_manual_entry_tab()
    
    with tab4:
        show_analytics_tab()

def get_theme_css(theme):
    """Get CSS for the selected theme"""
    base_css = """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #EC5328;
            margin-bottom: 1rem;
            text-align: center;
            background: linear-gradient(135deg, #EC5328 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .status-text {
            text-align: center;
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .feature-header {
            font-size: 1.5rem;
            color: #EC5328;
            margin-top: 1rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #EC5328 0%, #ff6b6b 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(236, 83, 40, 0.4);
        }
        .info-box {
            background: linear-gradient(135deg, #fef5e7 0%, #fed7aa 100%);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #EC5328;
        }
        .success-box {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid #4caf50;
        }
        .ai-insight {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-left: 5px solid #EC5328;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 12px rgba(236, 83, 40, 0.15);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            border: 1px solid #e2e8f0;
            color: #2d3748;
        }
        .ai-insight h3 {
            color: #EC5328;
            margin-top: 0;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        .ai-insight h4 {
            color: #2d3748;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        .ai-insight ul {
            padding-left: 1.5rem;
            color: #4a5568;
        }
        .ai-insight li {
            margin-bottom: 0.5rem;
            line-height: 1.6;
            color: #4a5568;
        }
        .ai-insight strong {
            color: #2d3748;
            font-weight: 600;
        }
        .ai-insight p {
            color: #4a5568;
            line-height: 1.6;
            margin-bottom: 0.75rem;
        }
        .cross-field-scenario {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        .scenario-header {
            font-weight: 700;
            font-size: 1.1rem;
            color: #1a202c;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #EC5328;
        }
        .scenario-steps {
            background-color: #f7fafc;
            border: 1px solid #cbd5e0;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.75rem;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #2d3748;
        }
        .scenario-description {
            color: #4a5568;
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }
        .scenario-expected {
            color: #2d3748;
            font-weight: 600;
            font-size: 0.9rem;
            background: #e6fffa;
            padding: 0.5rem;
            border-radius: 4px;
            border-left: 3px solid #38b2ac;
            margin-top: 0.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        .field-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .field-card h4 {
            color: #2d3748;
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .test-data-section {
            background: #fef5e7;
            border-radius: 6px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-left: 3px solid #EC5328;
        }
        .error-message {
            background: #fed7d7;
            color: #742a2a;
            padding: 0.75rem;
            border-radius: 6px;
            border-left: 3px solid #e53e3e;
            margin: 0.5rem 0;
        }
        .success-message {
            background: #c6f6d5;
            color: #22543d;
            padding: 0.75rem;
            border-radius: 6px;
            border-left: 3px solid #38a169;
            margin: 0.5rem 0;
        }
        .warning-message {
            background: #fef5e7;
            color: #744210;
            padding: 0.75rem;
            border-radius: 6px;
            border-left: 3px solid #EC5328;
            margin: 0.5rem 0;
        }
        </style>
    """
    
    if theme == "Dark":
        base_css += """
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .cross-field-scenario {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            border: 2px solid #4a5568;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            color: #e2e8f0;
        }
        .scenario-header {
            font-weight: 700;
            font-size: 1.1rem;
            color: #f7fafc;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #EC5328;
        }
        .scenario-steps {
            background-color: #4a5568;
            border: 1px solid #718096;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 0.75rem;
            font-size: 0.95rem;
            line-height: 1.6;
            color: #f7fafc;
        }
        .scenario-description {
            color: #cbd5e0;
            font-size: 1rem;
            line-height: 1.5;
            margin-bottom: 0.5rem;
        }
        .scenario-expected {
            color: #f7fafc;
            font-weight: 600;
            font-size: 0.9rem;
            background: #2c5530;
            padding: 0.5rem;
            border-radius: 4px;
            border-left: 3px solid #68d391;
            margin-top: 0.5rem;
        }
        .ai-insight {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
            border-left: 5px solid #EC5328;
            border: 1px solid #4a5568;
            color: #e2e8f0;
        }
        .ai-insight h3 {
            color: #EC5328;
        }
        .ai-insight h4 {
            color: #e2e8f0;
        }
        .ai-insight ul {
            color: #cbd5e0;
        }
        .ai-insight li {
            color: #cbd5e0;
        }
        .ai-insight strong {
            color: #f7fafc;
        }
        .ai-insight p {
            color: #cbd5e0;
        }
        .status-text {
            color: #a0aec0;
        }
        .feature-header {
            color: #EC5328;
        }
        .field-card {
            background: #2d3748;
            border: 1px solid #4a5568;
            color: #e2e8f0;
        }
        .field-card h4 {
            color: #f7fafc;
        }
        .test-data-section {
            background: #1a202c;
            border-left: 3px solid #EC5328;
            color: #e2e8f0;
        }
        .error-message {
            background: #742a2a;
            color: #fed7d7;
            border-left: 3px solid #fc8181;
        }
        .success-message {
            background: #22543d;
            color: #c6f6d5;
            border-left: 3px solid #68d391;
        }
        .warning-message {
            background: #744210;
            color: #fef5e7;
            border-left: 3px solid #EC5328;
        }
        .metric-card {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            border: 1px solid #4a5568;
            color: #e2e8f0;
        }
        """
    
    return base_css

def show_screenshot_analysis_tab():
    """Enhanced screenshot analysis tab with AI"""
    st.header("📸 Screenshot Analysis")
    
    # Sidebar configuration for this tab
    with st.sidebar:
        st.subheader("🔧 Screenshot Analysis Settings")
        
        # AI Enhancement Options
        if AI_AVAILABLE:
            use_ai_analysis = st.checkbox("🤖 Use AI-Enhanced Analysis", value=True)
            ai_confidence = st.slider("AI Analysis Confidence", 0.1, 1.0, 0.7)
        else:
            use_ai_analysis = False
            st.warning("AI features unavailable - using traditional OCR")
        
        # OCR Settings
        ocr_engine = st.selectbox("OCR Engine", ["Tesseract", "EasyOCR (if available)"])
        preprocessing_level = st.selectbox("Image Preprocessing", ["Basic", "Enhanced", "Aggressive"])
        
        # Analysis Options
        detect_hidden_fields = st.checkbox("Detect Hidden Fields", value=True)
        analyze_field_relationships = st.checkbox("Analyze Field Dependencies", value=True)
        generate_cross_field_tests = st.checkbox("Generate Cross-Field Test Scenarios", value=True)

    st.info("📸 Upload a screenshot of a form to automatically detect fields and generate comprehensive test data.")
    
    uploaded_files = st.file_uploader(
        "Upload screenshot(s) of the webpage", 
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploaded_files:
        analyzer = AdvancedScreenshotAnalyzer()
        ai_generator = AITestDataGenerator()
        
        for i, uploaded_file in enumerate(uploaded_files):
            st.subheader(f"Analysis {i+1}: {uploaded_file.name}")
            
            # Display the image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption=f"Screenshot {i+1}", use_container_width=True)
            
            with col2:
                with st.spinner("🔍 Analyzing image and extracting form fields..."):
                    try:
                        # Extract form fields using traditional OCR
                        extracted_labels = extract_form_fields(image)
                        
                        if not extracted_labels:
                            warning_html = """
                            <div class="warning-message">
                                <strong>⚠️ No form fields detected in the image</strong>
                                <p>The OCR process couldn't identify any form fields. Try these solutions:</p>
                                <ul>
                                    <li><strong>Image Quality:</strong> Ensure the image has clear, readable text</li>
                                    <li><strong>Form Visibility:</strong> Check if form labels are clearly visible</li>
                                    <li><strong>Focus Area:</strong> Try cropping to focus on the form area</li>
                                    <li><strong>Resolution:</strong> Increase image resolution or contrast</li>
                                    <li><strong>Orientation:</strong> Make sure text is horizontal (not rotated)</li>
                                    <li><strong>Alternative:</strong> Use manual field entry instead</li>
                                </ul>
                            </div>
                            """
                            st.markdown(warning_html, unsafe_allow_html=True)
                        else:
                            st.success(f"✅ Successfully detected {len(extracted_labels)} potential form fields!")
                            
                            # Display fields with better styling
                            fields_html = """
                            <div class="success-message">
                                <strong>🏷️ Detected Form Fields:</strong>
                                <ul style="margin-top: 0.5rem;">
                            """
                            for label in extracted_labels:
                                fields_html += f"<li>{label}</li>"
                            fields_html += """
                                </ul>
                                <p style="margin-bottom: 0;"><em>💡 You can edit these fields below before generating test data.</em></p>
                            </div>
                            """
                            st.markdown(fields_html, unsafe_allow_html=True)
                    
                    except Exception as e:
                        error_html = f"""
                        <div class="error-message">
                            <strong>❌ Error processing image</strong>
                            <p><strong>Error Details:</strong> {str(e)}</p>
                            <p><strong>Troubleshooting:</strong></p>
                            <ul>
                                <li>Check if the image format is supported (PNG, JPG, JPEG)</li>
                                <li>Verify the image file is not corrupted</li>
                                <li>Try with a different image</li>
                                <li>Use manual field entry as an alternative</li>
                                <li>Ensure sufficient memory is available for image processing</li>
                            </ul>
                        </div>
                        """
                        st.markdown(error_html, unsafe_allow_html=True)
                        extracted_labels = []
            
            # Generate comprehensive test data if fields were found
            if 'extracted_labels' in locals() and extracted_labels:
                # Convert labels to FormField objects
                form_fields = []
                for label in extracted_labels:
                    field_type = infer_field_type(label)
                    constraints = get_field_constraints(field_type)
                    form_fields.append(FormField(
                        name=label.lower().replace(" ", "_"),
                        field_type=field_type,
                        label=label,
                        placeholder=f"Enter {label.lower()}",
                        required=True,  # Assume required by default
                        validation_rules=constraints.get("validation", []),
                        constraints=constraints
                    ))
                
                # Show field details
                show_field_test_data(form_fields, ai_generator)
                
                # Show cross-field scenarios if enabled
                if generate_cross_field_tests and len(form_fields) > 1:
                    show_cross_field_scenarios(form_fields, ai_generator)
                
                # Show AI insights if available
                if use_ai_analysis and AI_AVAILABLE:
                    analysis_result = analyzer.analyze_screenshot_with_ai(image)
                    show_ai_insights(analysis_result, form_fields)
                else:
                    analysis_result = analyzer.analyze_screenshot_traditional(image)
                    show_ai_insights(analysis_result, form_fields)
                st.success(f"✅ Found {len(form_fields)} form fields")
                
                # Show field details
                if form_fields:
                    for field in form_fields:
                        with st.expander(f"📝 {field.label} ({field.field_type})"):
                            st.json({
                                "name": field.name,
                                "type": field.field_type,
                                "label": field.label,
                                "placeholder": field.placeholder,
                                "required": field.required,
                                "validation_rules": field.validation_rules,
                                "constraints": field.constraints
                            })
            
            # Generate comprehensive test data
            if form_fields:
                st.subheader("🧪 Generated Test Data")
                
                # Generate test data for each field
                test_data_tabs = st.tabs(["🔤 Field Data", "🔗 Cross-Field Scenarios", "🤖 AI Insights"])
                
                with test_data_tabs[0]:
                    show_field_test_data(form_fields, ai_generator)
                
                with test_data_tabs[1]:
                    if generate_cross_field_tests:
                        show_cross_field_scenarios(form_fields, ai_generator)
                
                with test_data_tabs[2]:
                    if AI_AVAILABLE:
                        show_ai_insights(analysis_result, form_fields)

def show_website_traversal_tab():
    """Website traversal and analysis tab"""
    st.header("🕷️ Website Traversal & Analysis")
    
    with st.sidebar:
        st.subheader("🔧 Traversal Settings")
        max_pages = st.slider("Max Pages to Analyze", 5, 100, 25)
        max_depth = st.slider("Crawl Depth", 1, 5, 3)
        rate_limit = st.slider("Rate Limit (seconds)", 1, 10, 2)
        
        # Analysis options
        analyze_security = st.checkbox("Security Analysis", value=True)
        analyze_accessibility = st.checkbox("Accessibility Analysis", value=True)
        analyze_performance = st.checkbox("Performance Analysis", value=True)
        
        if AI_AVAILABLE:
            use_ai_insights = st.checkbox("🤖 AI-Powered Insights", value=True)
        else:
            use_ai_insights = False

    st.info("🕷️ Enter a website URL to automatically discover and analyze all forms across the site.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        website_url = st.text_input("Website URL", placeholder="https://example.com")
    with col2:
        start_traversal = st.button("🚀 Start Analysis", type="primary")
    
    if start_traversal and website_url:
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Initialize traverser
        traverser = WebsiteTraverser()
        traverser.max_depth = max_depth
        traverser.rate_limit_delay = rate_limit
        
        # Perform traversal
        try:
            analyses = traverser.traverse_website(website_url, max_pages)
            
            if analyses:
                st.success(f"🎉 Analysis complete! Found {len(analyses)} pages with forms.")
                
                # Display results
                show_traversal_results(analyses, use_ai_insights)
            else:
                st.warning("No forms found on the analyzed pages.")
                
        except Exception as e:
            st.error(f"Error during website analysis: {str(e)}")

def show_manual_entry_tab():
    """Manual field entry tab"""
    st.header("✏️ Manual Field Entry")
    
    with st.sidebar:
        st.subheader("🔧 Manual Entry Settings")
        field_template = st.selectbox(
            "Field Template",
            ["Blank", "Login Form", "Registration Form", "Contact Form", "Payment Form", "Survey Form"]
        )

    st.info("✏️ Manually define form fields and generate comprehensive test data.")
    
    # Field templates
    if field_template != "Blank":
        if st.button(f"Load {field_template} Template"):
            st.session_state.manual_fields = get_field_template(field_template)
    
    # Manual field entry
    if 'manual_fields' not in st.session_state:
        st.session_state.manual_fields = []
    
    # Add new field
    with st.expander("➕ Add New Field", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            field_name = st.text_input("Field Name")
        with col2:
            field_type = st.selectbox("Field Type", [
                "text", "email", "password", "phone", "number", "date", 
                "url", "domain-name", "age", "currency", "zip", "name",
                "first-name", "last-name", "address", "city", "state", "country"
            ])
        with col3:
            field_required = st.checkbox("Required")
        
        field_label = st.text_input("Field Label")
        
        if st.button("Add Field") and field_name:
            new_field = FormField(
                name=field_name,
                field_type=field_type,
                label=field_label or field_name.title(),
                placeholder="",
                required=field_required,
                validation_rules=[],
                constraints=get_field_constraints(field_type)
            )
            st.session_state.manual_fields.append(new_field)
            st.success(f"Added field: {field_name}")
    
    # Display and edit current fields
    if st.session_state.manual_fields:
        st.subheader("📝 Current Fields")
        
        for i, field in enumerate(st.session_state.manual_fields):
            with st.expander(f"{field.label} ({field.field_type})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json({
                        "name": field.name,
                        "type": field.field_type,
                        "required": field.required,
                        "constraints": field.constraints
                    })
                with col2:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state.manual_fields.pop(i)
                        st.experimental_rerun()
        
        # Generate test data and export options
        if st.session_state.manual_fields:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🧪 Generate Test Data", type="primary"):
                    ai_generator = AITestDataGenerator()
                    show_field_test_data(st.session_state.manual_fields, ai_generator)
            
            with col2:
                if st.button("📊 Generate Cross-Field Tests"):
                    ai_generator = AITestDataGenerator()
                    show_cross_field_scenarios(st.session_state.manual_fields, ai_generator)
            
            with col3:
                if st.button("🎯 Show Testing Insights"):
                    show_ai_insights({}, st.session_state.manual_fields)
            
            # Export section
            st.subheader("📁 Export Test Data")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Export as CSV"):
                    export_test_data_csv(st.session_state.manual_fields)
            
            with export_col2:
                if st.button("📊 Export as Excel"):
                    export_test_data_excel(st.session_state.manual_fields)
            
            with export_col3:
                if st.button("🧪 Generate Test Scripts"):
                    show_test_script_generator(st.session_state.manual_fields)

def export_test_data_csv(fields: List[FormField]):
    """Export test data as CSV"""
    import io
    output = io.StringIO()
    
    # Generate test data for all fields
    ai_generator = AITestDataGenerator()
    
    # Write header
    output.write("Field Name,Field Type,Required,Test Type,Test Data,Expected Result\n")
    
    for field in fields:
        if AI_AVAILABLE:
            test_data = ai_generator.generate_intelligent_test_data(field)
            valid_data = test_data.get('valid_data', [])
            invalid_data = test_data.get('invalid_data', [])
            boundary_data = test_data.get('boundary_data', [])
            security_tests = test_data.get('security_tests', [])
        else:
            test_data_tuple = generate_test_data(field.field_type)
            valid_data, invalid_data, boundary_data, edge_cases, exploits = test_data_tuple
            security_tests = exploits
        
        # Write valid data
        for data in valid_data:
            output.write(f'"{field.name}","{field.field_type}","{field.required}","Valid","{data}","Pass"\n')
        
        # Write invalid data
        for data in invalid_data:
            output.write(f'"{field.name}","{field.field_type}","{field.required}","Invalid","{data}","Validation Error"\n')
        
        # Write boundary data
        for data in boundary_data:
            output.write(f'"{field.name}","{field.field_type}","{field.required}","Boundary","{data}","Check Behavior"\n')
        
        # Write security tests
        for data in security_tests:
            output.write(f'"{field.name}","{field.field_type}","{field.required}","Security","{data}","Block/Sanitize"\n')
    
    csv_data = output.getvalue()
    st.download_button(
        label="💾 Download CSV",
        data=csv_data,
        file_name=f"test_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_test_data_excel(fields: List[FormField]):
    """Export test data as Excel"""
    try:
        import pandas as pd
        import io
        
        # Generate test data for all fields
        ai_generator = AITestDataGenerator()
        all_data = []
        
        for field in fields:
            if AI_AVAILABLE:
                test_data = ai_generator.generate_intelligent_test_data(field)
                valid_data = test_data.get('valid_data', [])
                invalid_data = test_data.get('invalid_data', [])
                boundary_data = test_data.get('boundary_data', [])
                security_tests = test_data.get('security_tests', [])
            else:
                test_data_tuple = generate_test_data(field.field_type)
                valid_data, invalid_data, boundary_data, edge_cases, exploits = test_data_tuple
                security_tests = exploits
            
            # Add all test data types
            for test_type, data_list in [
                ("Valid", valid_data),
                ("Invalid", invalid_data), 
                ("Boundary", boundary_data),
                ("Security", security_tests)
            ]:
                for data in data_list:
                    all_data.append({
                        'Field Name': field.name,
                        'Field Type': field.field_type,
                        'Label': field.label,
                        'Required': field.required,
                        'Test Type': test_type,
                        'Test Data': data,
                        'Expected Result': "Pass" if test_type == "Valid" else "Validation Error" if test_type == "Invalid" else "Check Behavior"
                    })
        
        df = pd.DataFrame(all_data)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Test Data', index=False)
            
            # Add summary sheet
            summary_data = {
                'Metric': ['Total Fields', 'Required Fields', 'Field Types', 'Total Test Cases'],
                'Value': [
                    len(fields),
                    sum(1 for f in fields if f.required),
                    len(set(f.field_type for f in fields)),
                    len(all_data)
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        excel_data = output.getvalue()
        st.download_button(
            label="💾 Download Excel",
            data=excel_data,
            file_name=f"test_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except ImportError:
        st.error("pandas or xlsxwriter not available. Please install: pip install pandas xlsxwriter")

def show_test_script_generator(fields: List[FormField]):
    """Show test script generation options"""
    st.subheader("🧪 Test Script Generator")
    
    col1, col2 = st.columns(2)
    with col1:
        framework = st.selectbox("Test Framework", [
            "Selenium (Python)",
            "Playwright (Python)", 
            "Cypress (JavaScript)",
            "Robot Framework",
            "pytest"
        ])
    
    with col2:
        include_cross_field = st.checkbox("Include Cross-Field Tests", value=True)
    
    if st.button("Generate Test Script"):
        generator = TestScriptGenerator()
        
        if framework == "Selenium (Python)":
            script = generator.generate_selenium_script(fields, "Python")
        elif framework == "Playwright (Python)":
            script = generator.generate_playwright_script(fields, "Python")
        elif framework == "Cypress (JavaScript)":
            script = generator.generate_cypress_script(fields, "JavaScript")
        elif framework == "Robot Framework":
            script = generator.generate_robot_script(fields, "Robot")
        else:
            script = generator.generate_pytest_script(fields, "Python")
        
        st.code(script, language="python" if "Python" in framework else "javascript")
        
        # Download button
        st.download_button(
            label="💾 Download Test Script",
            data=script,
            file_name=f"test_script_{framework.lower().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
            mime="text/plain"
        )

def show_analytics_tab():
    """Test analytics and metrics tab"""
    st.header("📊 Test Analytics & Metrics")
    
    # Load history
    history = load_test_history()
    
    if history:
        # Analytics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Generations", len(history))
        
        with col2:
            total_fields = sum(item.get('field_count', 0) for item in history)
            st.metric("Fields Analyzed", total_fields)
        
        with col3:
            frameworks = [item.get('framework', 'Unknown') for item in history]
            popular_framework = max(set(frameworks), key=frameworks.count) if frameworks else "N/A"
            st.metric("Popular Framework", popular_framework)
        
        with col4:
            languages = [item.get('language', 'Unknown') for item in history]
            popular_language = max(set(languages), key=languages.count) if languages else "N/A"
            st.metric("Popular Language", popular_language)
        
        # Timeline chart
        st.subheader("📈 Usage Timeline")
        
        # Convert history to DataFrame for visualization
        df = pd.DataFrame(history)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            
            daily_usage = df.groupby('date').size().reset_index(name='count')
            
            import plotly.express as px
            fig = px.line(daily_usage, x='date', y='count', title='Daily Test Generation Activity')
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        for item in history[-10:]:  # Show last 10
            st.write(f"**{item['timestamp']}** - {item['name']} ({item['field_count']} fields)")
    else:
        st.info("No test generation history available yet.")

def show_field_test_data(fields: List[FormField], ai_generator: AITestDataGenerator):
    """Display test data for individual fields"""
    for field in fields:
        with st.expander(f"🔤 {field.label} Test Data", expanded=True):
            # Generate test data
            if AI_AVAILABLE:
                test_data = ai_generator.generate_intelligent_test_data(field)
                # AI generator returns a dictionary
                valid_data = test_data.get('valid_data', [])
                invalid_data = test_data.get('invalid_data', [])
                boundary_data = test_data.get('boundary_data', [])
                security_tests = test_data.get('security_tests', [])
                accessibility_tests = test_data.get('accessibility_tests', [])
                localization_tests = test_data.get('localization_tests', [])
            else:
                # generate_test_data returns a tuple
                test_data_tuple = generate_test_data(field.field_type)
                valid_data, invalid_data, boundary_data, edge_cases, exploits = test_data_tuple
                security_tests = exploits
                
                # Generate accessibility tests even in basic mode
                accessibility_tests = [
                    f"Test {field.label} with screen reader",
                    f"Verify {field.label} has proper ARIA labels",
                    f"Check {field.label} keyboard navigation",
                    f"Validate {field.label} high contrast mode",
                    f"Test {field.label} with voice commands",
                    f"Verify {field.label} focus indicators",
                    f"Check {field.label} for color blind users"
                ]
                
                # Generate localization tests even in basic mode
                localization_tests = [
                    f"Test {field.label} with UTF-8 characters",
                    f"Verify {field.label} with RTL languages",
                    f"Test {field.label} with long German text",
                    f"Check {field.label} with Chinese characters",
                    f"Verify {field.label} with Arabic text",
                    f"Test {field.label} with special Unicode symbols",
                    f"Check {field.label} text expansion/contraction"
                ]
            
            # Display in tabs
            data_tabs = st.tabs([
                "✅ Valid Data", "❌ Invalid Data", "⚡ Boundary Cases", 
                "🔒 Security Tests", "♿ Accessibility", "🌍 Localization"
            ])
            
            with data_tabs[0]:
                st.json(valid_data)
            
            with data_tabs[1]:
                st.json(invalid_data)
            
            with data_tabs[2]:
                st.json(boundary_data)
            
            with data_tabs[3]:
                st.json(security_tests)
            
            with data_tabs[4]:
                st.json(accessibility_tests)
            
            with data_tabs[5]:
                st.json(localization_tests)

def show_cross_field_scenarios(fields: List[FormField], ai_generator: AITestDataGenerator):
    """Display cross-field test scenarios with enhanced styling"""
    st.subheader("🔗 Cross-Field Test Scenarios")
    
    with st.spinner("Generating cross-field test scenarios..."):
        scenarios = ai_generator.generate_cross_field_scenarios(fields)
    
    if not scenarios:
        st.warning("No cross-field scenarios generated. Try adding more fields or check AI connectivity.")
        return
    
    st.success(f"Generated {len(scenarios)} cross-field test scenarios")
    
    # Group scenarios by priority
    high_priority = [s for s in scenarios if s.priority == "high"]
    medium_priority = [s for s in scenarios if s.priority == "medium"]
    low_priority = [s for s in scenarios if s.priority == "low"]
    
    # Display by priority groups
    for priority_group, scenarios_list, emoji in [
        ("High Priority", high_priority, "🔴"),
        ("Medium Priority", medium_priority, "🟡"), 
        ("Low Priority", low_priority, "🟢")
    ]:
        if scenarios_list:
            st.markdown(f"### {emoji} {priority_group}")
            
            for scenario in scenarios_list:
                priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                type_icon = {"positive": "✅", "negative": "❌", "boundary": "⚡", "security": "🔒"}
                
                with st.expander(
                    f"{type_icon.get(scenario.test_type, '🔸')} {scenario.title}",
                    expanded=False
                ):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="cross-field-scenario">
                            <div class="scenario-header">📝 Test Description</div>
                            <div class="scenario-description">{scenario.description}</div>
                            
                            <div class="scenario-header">✅ Expected Result</div>
                            <div class="scenario-expected">{scenario.expected_result}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"**Type:** {scenario.test_type.title()}")
                        st.markdown(f"**Priority:** {scenario.priority.title()}")
                        if hasattr(scenario, 'scenario_id') and scenario.scenario_id:
                            st.markdown(f"**ID:** `{scenario.scenario_id}`")
                    
                    if scenario.test_data and isinstance(scenario.test_data, list):
                        st.markdown('<div class="scenario-header">🔧 Test Steps</div>', unsafe_allow_html=True)
                        steps_html = '<div class="scenario-steps"><ol style="margin: 0; padding-left: 1.5rem;">'
                        for i, step in enumerate(scenario.test_data, 1):
                            steps_html += f'<li style="margin: 0.25rem 0; color: #2d3748; font-weight: 500;">{step}</li>'
                        steps_html += '</ol></div>'
                        st.markdown(steps_html, unsafe_allow_html=True)
                    elif scenario.test_data:
                        st.markdown('<div class="scenario-header">💾 Test Data</div>', unsafe_allow_html=True)
                        st.json(scenario.test_data)

def show_ai_insights(analysis_result: Dict, fields: List[FormField]):
    """Display AI-powered insights or fallback analysis"""
    st.subheader("🤖 Testing Insights & Analysis")
    
    # Form structure insights
    structure = analysis_result.get('form_structure', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Form Layout Analysis:**")
        st.write(f"• Layout: {structure.get('layout', 'Standard Form')}")
        st.write(f"• Sections: {len(structure.get('sections', [fields]))}")
        st.write(f"• Total Fields: {len(fields)}")
        st.write(f"• Required Fields: {sum(1 for f in fields if f.required)}")
        
    with col2:
        st.write("**Field Type Distribution:**")
        field_types = {}
        for field in fields:
            field_types[field.field_type] = field_types.get(field.field_type, 0) + 1
        for ftype, count in field_types.items():
            st.write(f"• {ftype.title()}: {count}")
    
    # Generate insights
    if AI_AVAILABLE:
        try:
            field_summary = [{"name": f.name, "type": f.field_type, "required": f.required} for f in fields]
            
            prompt = f"""
            Provide strategic testing insights for this form:
            
            Fields: {json.dumps(field_summary, indent=2)}
            Structure: {json.dumps(structure, indent=2)}
            
            Focus on:
            1. Critical test scenarios most likely to find bugs
            2. User experience considerations
            3. Business logic validation needs
            4. Integration testing requirements
            5. Performance and scalability concerns
            
            Provide actionable, specific recommendations in plain text format using bullet points and clear sections.
            DO NOT use HTML tags, code blocks, or markdown formatting.
            Use simple bullet points (•) and numbered lists (1., 2., etc.).
            """
            
            ai_insights = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Clean up AI response - remove HTML tags and markdown artifacts
            cleaned_insights = ai_insights
            if '```html' in cleaned_insights:
                # Remove HTML code blocks
                cleaned_insights = re.sub(r'```html\s*', '', cleaned_insights)
                cleaned_insights = re.sub(r'```\s*$', '', cleaned_insights)
            
            # Convert common HTML tags to markdown for better display
            cleaned_insights = re.sub(r'<h[1-6]>(.*?)</h[1-6]>', r'**\1**', cleaned_insights)
            cleaned_insights = re.sub(r'<strong>(.*?)</strong>', r'**\1**', cleaned_insights)
            cleaned_insights = re.sub(r'<b>(.*?)</b>', r'**\1**', cleaned_insights)
            cleaned_insights = re.sub(r'<em>(.*?)</em>', r'*\1*', cleaned_insights)
            cleaned_insights = re.sub(r'<i>(.*?)</i>', r'*\1*', cleaned_insights)
            cleaned_insights = re.sub(r'<ul>', '', cleaned_insights)
            cleaned_insights = re.sub(r'</ul>', '', cleaned_insights)
            cleaned_insights = re.sub(r'<li>(.*?)</li>', r'• \1', cleaned_insights)
            cleaned_insights = re.sub(r'<br\s*/?>', '\n', cleaned_insights)
            cleaned_insights = re.sub(r'<p>(.*?)</p>', r'\1\n', cleaned_insights)
            
            # Remove any remaining HTML tags
            cleaned_insights = re.sub(r'<[^>]+>', '', cleaned_insights)
            
            st.markdown(f'<div class="ai-insight"><h3>🧠 AI-Generated Testing Strategy</h3><p>{cleaned_insights}</p></div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating AI insights: {e}")
            show_fallback_insights(fields)
    else:
        show_fallback_insights(fields)

def show_fallback_insights(fields: List[FormField]):
    """Show expert testing insights when AI is not available"""
    insights_html = """
    <div class="ai-insight">
        <h3>🧠 Expert Testing Strategy</h3>
        <h4>Priority Testing Areas:</h4>
        <ul>
    """
    
    # Generate insights based on field types
    field_types = [f.field_type for f in fields]
    required_fields = [f for f in fields if f.required]
    
    if "email" in field_types:
        insights_html += "<li><strong>Email Validation:</strong> Test with various email formats, international domains, and edge cases like very long emails.</li>"
    
    if "password" in field_types:
        insights_html += "<li><strong>Password Security:</strong> Verify complexity requirements, common password rejection, and secure transmission.</li>"
        if len([f for f in fields if "confirm" in f.name.lower()]) > 0:
            insights_html += "<li><strong>Password Confirmation:</strong> Test mismatch scenarios and real-time validation feedback.</li>"
    
    if "phone" in field_types:
        insights_html += "<li><strong>Phone Validation:</strong> Test international formats, different separators, and country-specific validation.</li>"
    
    if len(required_fields) > 0:
        insights_html += f"<li><strong>Required Field Validation:</strong> Test all {len(required_fields)} required fields individually and in combination.</li>"
    
    if "card-number" in field_types or "cvv" in field_types:
        insights_html += "<li><strong>Payment Security:</strong> Test card validation, CVV verification, and secure data handling.</li>"
    
    insights_html += """
        </ul>
        <h4>Cross-Field Testing:</h4>
        <ul>
            <li>Test field dependencies and conditional validation</li>
            <li>Verify data consistency across related fields</li>
            <li>Test form completion workflows and error recovery</li>
        </ul>
        <h4>User Experience Testing:</h4>
        <ul>
            <li>Test keyboard navigation and accessibility</li>
            <li>Verify responsive behavior on different screen sizes</li>
            <li>Test error message clarity and positioning</li>
            <li>Verify form persistence during validation errors</li>
        </ul>
        <h4>Security Considerations:</h4>
        <ul>
            <li>Test input sanitization and XSS prevention</li>
            <li>Verify CSRF protection mechanisms</li>
            <li>Test SQL injection prevention in form processing</li>
            <li>Verify secure transmission of sensitive data</li>
        </ul>
    </div>
    """
    
    st.markdown(insights_html, unsafe_allow_html=True)

def show_traversal_results(analyses: List[WebsiteAnalysis], use_ai_insights: bool):
    """Display website traversal results"""
    # Summary metrics
    total_forms = sum(len(analysis.forms) for analysis in analyses)
    total_fields = sum(len(analysis.fields) for analysis in analyses)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Pages Analyzed", len(analyses))
    with col2:
        st.metric("Forms Found", total_forms)
    with col3:
        st.metric("Total Fields", total_fields)
    
    # Detailed results
    for i, analysis in enumerate(analyses):
        with st.expander(f"📄 Page {i+1}: {analysis.url}", expanded=False):
            
            # Forms summary
            if analysis.forms:
                st.subheader("📝 Forms Found")
                for j, form in enumerate(analysis.forms):
                    st.write(f"**Form {j+1}:** {len(form['fields'])} fields")
                    
                    # Show field details
                    for field in form['fields']:
                        st.write(f"  • {field.get('label', field.get('name', 'Unnamed'))} ({field.get('type', 'unknown')})")
            
            # Issues summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if analysis.security_considerations:
                    st.subheader("🔒 Security Issues")
                    for issue in analysis.security_considerations:
                        st.warning(issue)
            
            with col2:
                if analysis.accessibility_issues:
                    st.subheader("♿ Accessibility Issues")
                    for issue in analysis.accessibility_issues:
                        st.info(issue)
            
            with col3:
                if analysis.performance_notes:
                    st.subheader("⚡ Performance Notes")
                    for note in analysis.performance_notes:
                        st.info(note)
            
            # AI insights
            if use_ai_insights and analysis.ai_insights:
                st.subheader("🤖 AI Analysis")
                st.markdown(f'<div class="ai-insight">{analysis.ai_insights}</div>', unsafe_allow_html=True)

def get_field_template(template_name: str) -> List[FormField]:
    """Get predefined field templates"""
    templates = {
        "Login Form": [
            FormField("email", "email", "Email Address", "", True, [], get_field_constraints("email")),
            FormField("password", "password", "Password", "", True, [], get_field_constraints("password"))
        ],
        "Registration Form": [
            FormField("first_name", "first-name", "First Name", "", True, [], get_field_constraints("first-name")),
            FormField("last_name", "last-name", "Last Name", "", True, [], get_field_constraints("last-name")),
            FormField("email", "email", "Email Address", "", True, [], get_field_constraints("email")),
            FormField("password", "password", "Password", "", True, [], get_field_constraints("password")),
            FormField("confirm_password", "password", "Confirm Password", "", True, [], get_field_constraints("password")),
            FormField("phone", "phone", "Phone Number", "", False, [], get_field_constraints("phone")),
            FormField("date_of_birth", "date-of-birth", "Date of Birth", "", False, [], get_field_constraints("date-of-birth"))
        ],
        "Contact Form": [
            FormField("name", "name", "Full Name", "", True, [], get_field_constraints("name")),
            FormField("email", "email", "Email Address", "", True, [], get_field_constraints("email")),
            FormField("subject", "text", "Subject", "", True, [], get_field_constraints("text")),
            FormField("message", "text", "Message", "", True, [], get_field_constraints("text"))
        ],
        "Payment Form": [
            FormField("card_number", "card-number", "Card Number", "", True, [], get_field_constraints("card-number")),
            FormField("expiry", "card-expiry", "Expiry Date", "", True, [], get_field_constraints("card-expiry")),
            FormField("cvv", "cvv", "CVV", "", True, [], get_field_constraints("cvv")),
            FormField("cardholder_name", "name", "Cardholder Name", "", True, [], get_field_constraints("name")),
            FormField("billing_address", "address", "Billing Address", "", True, [], get_field_constraints("address"))
        ],
        "Survey Form": [
            FormField("age", "age", "Age", "", False, [], get_field_constraints("age")),
            FormField("occupation", "text", "Occupation", "", False, [], get_field_constraints("text")),
            FormField("experience", "number", "Years of Experience", "", False, [], get_field_constraints("number")),
            FormField("rating", "number", "Rating (1-10)", "", True, [], get_field_constraints("number"))
        ]
    }
    
    return templates.get(template_name, [])

# --- Advanced Test Script Generation ---

class TestScriptGenerator:
    """Generate comprehensive test scripts for various frameworks"""
    
    def __init__(self):
        self.frameworks = {
            "Selenium": self.generate_selenium_script,
            "Robot Framework": self.generate_robot_script,
            "Cypress": self.generate_cypress_script,
            "Playwright": self.generate_playwright_script,
            "pytest": self.generate_pytest_script
        }
    
    def generate_test_script(self, fields: List[FormField], framework: str, language: str = "Python") -> str:
        """Generate test script for specified framework"""
        generator = self.frameworks.get(framework, self.generate_selenium_script)
        return generator(fields, language)
    
    def generate_selenium_script(self, fields: List[FormField], language: str) -> str:
        """Generate Selenium test script"""
        if language.lower() == "python":
            return self._generate_selenium_python(fields)
        elif language.lower() == "java":
            return self._generate_selenium_java(fields)
        else:
            return self._generate_selenium_python(fields)
    
    def _generate_selenium_python(self, fields: List[FormField]) -> str:
        """Generate Python Selenium script"""
        script = '''import pytest
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

class TestFormValidation:
    """AI-Generated comprehensive form validation tests"""
    
    @classmethod
    def setup_class(cls):
        """Setup test class with WebDriver"""
        options = Options()
        options.add_argument("--headless")  # Remove for visual testing
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        cls.driver = webdriver.Chrome(options=options)
        cls.wait = WebDriverWait(cls.driver, 10)
        
    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        cls.driver.quit()
        
    def setup_method(self):
        """Setup for each test method"""
        self.driver.get("YOUR_FORM_URL_HERE")  # Replace with actual URL
        
'''
        
        # Generate test methods for each field
        for field in fields:
            script += self._generate_field_test_methods_python(field)
        
        # Add cross-field validation tests
        script += self._generate_cross_field_tests_python(fields)
        
        # Add utility methods
        script += '''
    def fill_form_field(self, field_locator, value):
        """Utility method to fill form field"""
        try:
            element = self.wait.until(EC.element_to_be_clickable((By.NAME, field_locator)))
            element.clear()
            element.send_keys(value)
            return True
        except TimeoutException:
            try:
                element = self.driver.find_element(By.ID, field_locator)
                element.clear()
                element.send_keys(value)
                return True
            except NoSuchElementException:
                return False
    
    def submit_form(self):
        """Submit the form"""
        try:
            submit_button = self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='submit'], button[type='submit']")))
            submit_button.click()
            return True
        except TimeoutException:
            return False
    
    def get_validation_message(self, field_name):
        """Get validation message for a field"""
        try:
            # Try HTML5 validation message first
            field = self.driver.find_element(By.NAME, field_name)
            return self.driver.execute_script("return arguments[0].validationMessage;", field)
        except:
            # Try custom validation message
            try:
                error_element = self.driver.find_element(By.CSS_SELECTOR, f"[data-error-for='{field_name}'], .error-{field_name}")
                return error_element.text
            except:
                return None

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
'''
        
        return script
    
    def _generate_field_test_methods_python(self, field: FormField) -> str:
        """Generate test methods for a specific field"""
        field_name = field.name
        field_type = field.field_type
        
        methods = f'''
    def test_{field_name}_valid_input(self):
        """Test {field.label} with valid inputs"""
        valid_values = {self._get_sample_valid_data(field)}
        
        for value in valid_values:
            self.setup_method()  # Reset form
            assert self.fill_form_field("{field_name}", value), f"Failed to fill {field_name}"
            
            if self.submit_form():
                # Check for success indicators
                time.sleep(1)  # Allow for form processing
                validation_msg = self.get_validation_message("{field_name}")
                assert not validation_msg or validation_msg == "", f"Unexpected validation error: {{validation_msg}}"
    
    def test_{field_name}_invalid_input(self):
        """Test {field.label} with invalid inputs"""
        invalid_values = {self._get_sample_invalid_data(field)}
        
        for value in invalid_values:
            self.setup_method()  # Reset form
            assert self.fill_form_field("{field_name}", value), f"Failed to fill {field_name}"
            
            # Try to submit and expect validation error
            self.submit_form()
            time.sleep(0.5)  # Allow for validation
            validation_msg = self.get_validation_message("{field_name}")
            assert validation_msg, f"Expected validation error for invalid input: {{value}}"
    
    def test_{field_name}_boundary_cases(self):
        """Test {field.label} with boundary values"""
        boundary_values = {self._get_sample_boundary_data(field)}
        
        for value in boundary_values:
            self.setup_method()  # Reset form
            assert self.fill_form_field("{field_name}", value), f"Failed to fill {field_name}"
            
            # Submit and check behavior
            self.submit_form()
            time.sleep(0.5)
            # Boundary cases might be valid or invalid - check context
            validation_msg = self.get_validation_message("{field_name}")
            # Log the result for manual verification
            print(f"Boundary test for {{value}}: validation_msg={{validation_msg}}")
'''
        
        if field.required:
            methods += f'''
    def test_{field_name}_required_validation(self):
        """Test that {field.label} is required"""
        self.setup_method()
        # Leave field empty and try to submit
        assert self.submit_form() == True, "Form submission should be attempted"
        time.sleep(0.5)
        validation_msg = self.get_validation_message("{field_name}")
        assert validation_msg, f"Expected required field validation for {field_name}"
'''
        
        return methods
    
    def _generate_cross_field_tests_python(self, fields: List[FormField]) -> str:
        """Generate cross-field validation tests"""
        # Build field data strings separately to avoid f-string nesting issues
        field_data_lines = []
        for field in fields:
            sample_data = self._get_sample_valid_data(field)
            value = sample_data[0] if sample_data else '""'
            field_data_lines.append(f'            "{field.name}": {value},\n')

        required_field_lines = []
        for field in fields:
            if field.required:
                sample_data = self._get_sample_valid_data(field)
                value = sample_data[0] if sample_data else '""'
                required_field_lines.append(f'            "{field.name}": {value},\n')

        return '''
    def test_form_complete_valid_submission(self):
        """Test complete form submission with all valid data"""
        # Fill all fields with valid data
        test_data = {
''' + ''.join(field_data_lines) + '''        }
        
        for field_name, value in test_data.items():
            assert self.fill_form_field(field_name, value), f"Failed to fill {field_name}"
        
        # Submit form
        assert self.submit_form(), "Form submission failed"
        time.sleep(1)
        
        # Check for success (this depends on your application's behavior)
        # You might check for redirect, success message, etc.
        current_url = self.driver.current_url
        # Add your success validation logic here
    
    def test_form_partial_submission(self):
        """Test form submission with only required fields"""
        required_fields = {
''' + ''.join(required_field_lines) + '''        }
        
        for field_name, value in required_fields.items():
            assert self.fill_form_field(field_name, value), f"Failed to fill required field {field_name}"
        
        assert self.submit_form(), "Form submission with required fields failed"
'''
    
    def generate_robot_script(self, fields: List[FormField], language: str) -> str:
        """Generate Robot Framework test script"""
        return f'''*** Settings ***
Documentation    AI-Generated Form Validation Tests
Library          SeleniumLibrary
Library          Collections
Test Setup       Setup Test
Test Teardown    Teardown Test

*** Variables ***
${{FORM_URL}}    YOUR_FORM_URL_HERE
${{BROWSER}}     chrome
${{HEADLESS}}    True

*** Test Cases ***
''' + ''.join([self._generate_robot_field_tests(field) for field in fields]) + '''

Test Complete Form Submission
    [Documentation]    Test complete form with valid data
    [Tags]    positive    integration
    Fill Form With Valid Data
    Submit Form
    Verify Successful Submission

Test Required Fields Validation
    [Documentation]    Test that required fields are validated
    [Tags]    negative    validation
    Submit Form
    Verify Required Field Errors

*** Keywords ***
Setup Test
    Open Browser    ${{FORM_URL}}    ${{BROWSER}}    options=add_argument("--headless")
    Maximize Browser Window
    Set Selenium Timeout    10

Teardown Test
    Close Browser

Fill Form Field
    [Arguments]    ${{field_name}}    ${{value}}
    Wait Until Element Is Visible    name:${{field_name}}
    Clear Element Text    name:${{field_name}}
    Input Text    name:${{field_name}}    ${{value}}

Submit Form
    Click Button    css:input[type='submit'], button[type='submit']

Verify Successful Submission
    # Add your success verification logic here
    Wait Until Location Contains    success
    # OR check for success message
    # Wait Until Element Is Visible    css:.success-message

Verify Required Field Errors
    # Check for HTML5 validation or custom error messages
    Execute Javascript    return document.querySelector('input:invalid') !== null

Fill Form With Valid Data
''' + ''.join([f'    Fill Form Field    {field.name}    {self._get_sample_valid_data(field)[0] if self._get_sample_valid_data(field) else ""}\n' for field in fields])
    
    def _generate_robot_field_tests(self, field: FormField) -> str:
        """Generate Robot Framework tests for a field"""
        return f'''
Test {field.label.replace(' ', '_')} Valid Input
    [Documentation]    Test {field.label} with valid inputs
    [Tags]    positive    {field.field_type}
    @{{valid_values}}=    Create List    {self._format_robot_list(self._get_sample_valid_data(field))}
    FOR    ${{value}}    IN    @{{valid_values}}
        Fill Form Field    {field.name}    ${{value}}
        Submit Form
        # Add validation logic here
    END

Test {field.label.replace(' ', '_')} Invalid Input
    [Documentation]    Test {field.label} with invalid inputs
    [Tags]    negative    {field.field_type}
    @{{invalid_values}}=    Create List    {self._format_robot_list(self._get_sample_invalid_data(field))}
    FOR    ${{value}}    IN    @{{invalid_values}}
        Fill Form Field    {field.name}    ${{value}}
        Submit Form
        # Verify validation error appears
        Execute Javascript    return document.querySelector('input[name="{field.name}"]:invalid') !== null
    END
'''
    
    def generate_cypress_script(self, fields: List[FormField], language: str) -> str:
        """Generate Cypress test script"""
        return f'''// AI-Generated Cypress Form Validation Tests
describe('Form Validation Tests', () => {{
  beforeEach(() => {{
    cy.visit('YOUR_FORM_URL_HERE'); // Replace with actual URL
  }});
''' + ''.join([self._generate_cypress_field_tests(field) for field in fields]) + '''
  
  it('should submit form with all valid data', () => {
    // Fill all fields with valid data
''' + ''.join([f"    cy.get('[name=\"{field.name}\"]').type('{self._get_sample_valid_data(field)[0] if self._get_sample_valid_data(field) else ''}');\n" for field in fields]) + '''    
    cy.get('input[type="submit"], button[type="submit"]').click();
    
    // Verify successful submission
    cy.url().should('include', 'success'); // Adjust based on your app
  });

  it('should validate required fields', () => {
    cy.get('input[type="submit"], button[type="submit"]').click();
    
    // Check for HTML5 validation
''' + ''.join([f"    cy.get('[name=\"{field.name}\"]').should('have.attr', 'required');\n" for field in fields if field.required]) + '''  });
}});'''
    
    def _generate_cypress_field_tests(self, field: FormField) -> str:
        """Generate Cypress tests for a field"""
        valid_data = self._get_sample_valid_data(field)
        invalid_data = self._get_sample_invalid_data(field)
        
        return f'''
  context('{field.label} Tests', () => {{
    it('should accept valid {field.field_type} input', () => {{
      const validValues = {json.dumps(valid_data[:3])};
      
      validValues.forEach(value => {{
        cy.get('[name="{field.name}"]').clear().type(value);
        cy.get('input[type="submit"], button[type="submit"]').click();
        // Add specific validation for success
      }});
    }});
    
    it('should reject invalid {field.field_type} input', () => {{
      const invalidValues = {json.dumps(invalid_data[:3])};
      
      invalidValues.forEach(value => {{
        cy.get('[name="{field.name}"]').clear().type(value);
        cy.get('input[type="submit"], button[type="submit"]').click();
        
        // Check for validation error
        cy.get('[name="{field.name}"]').should('have.class', 'invalid')
          .or('satisfy', ($el) => {{
            return $el[0].validationMessage !== '';
          }});
      }});
    }});
  }});'''

    def generate_playwright_script(self, fields: List[FormField], language: str) -> str:
        """Generate Playwright test script"""
        return f'''// AI-Generated Playwright Form Validation Tests
import {{ test, expect }} from '@playwright/test';

test.describe('Form Validation Tests', () => {{
  test.beforeEach(async ({{ page }}) => {{
    await page.goto('YOUR_FORM_URL_HERE'); // Replace with actual URL
  }});
''' + ''.join([self._generate_playwright_field_tests(field) for field in fields]) + '''
  
  test('complete form submission with valid data', async ({{ page }}) => {{
    // Fill all fields with valid data
''' + ''.join([f"    await page.fill('[name=\"{field.name}\"]', '{self._get_sample_valid_data(field)[0] if self._get_sample_valid_data(field) else ''}');\n" for field in fields]) + '''    
    await page.click('input[type="submit"], button[type="submit"]');
    
    // Verify successful submission
    await expect(page).toHaveURL(/success/); // Adjust based on your app
  }});

  test('required fields validation', async ({{ page }}) => {{
    await page.click('input[type="submit"], button[type="submit"]');
    
    // Check for validation messages
''' + ''.join([f"    await expect(page.locator('[name=\"{field.name}\"]')).toHaveAttribute('required');\n" for field in fields if field.required]) + '''  });
}});'''
    
    def _generate_playwright_field_tests(self, field: FormField) -> str:
        """Generate Playwright tests for a field"""
        valid_data = self._get_sample_valid_data(field)
        invalid_data = self._get_sample_invalid_data(field)
        
        return f'''
  test.describe('{field.label} validation', () => {{
    test('accepts valid {field.field_type} input', async ({{ page }}) => {{
      const validValues = {json.dumps(valid_data[:3])};
      
      for (const value of validValues) {{
        await page.fill('[name="{field.name}"]', value);
        await page.click('input[type="submit"], button[type="submit"]');
        // Add validation logic here
      }}
    }});
    
    test('rejects invalid {field.field_type} input', async ({{ page }}) => {{
      const invalidValues = {json.dumps(invalid_data[:3])};
      
      for (const value of invalidValues) {{
        await page.fill('[name="{field.name}"]', value);
        await page.click('input[type="submit"], button[type="submit"]');
        
        // Check for validation error
        const validationMessage = await page.locator('[name="{field.name}"]').evaluate(el => el.validationMessage);
        expect(validationMessage).toBeTruthy();
      }}
    }});
  }});'''

    def generate_pytest_script(self, fields: List[FormField], language: str) -> str:
        """Generate pytest-specific test script"""
        return self._generate_selenium_python(fields)  # Use Selenium Python as base
    
    # Helper methods for test data
    def _get_sample_valid_data(self, field: FormField) -> List[str]:
        """Get sample valid data for field"""
        valid_data, _, _, _, _ = generate_test_data(field.field_type)
        return valid_data[:3]  # Return first 3 for brevity
    
    def _get_sample_invalid_data(self, field: FormField) -> List[str]:
        """Get sample invalid data for field"""
        _, invalid_data, _, _, _ = generate_test_data(field.field_type)
        return invalid_data[:3]  # Return first 3 for brevity
    
    def _get_sample_boundary_data(self, field: FormField) -> List[str]:
        """Get sample boundary data for field"""
        _, _, boundary_data, _, _ = generate_test_data(field.field_type)
        return boundary_data[:3]  # Return first 3 for brevity
    
    def _format_robot_list(self, data_list: List[str]) -> str:
        """Format list for Robot Framework"""
        return '    '.join([f'"{item}"' for item in data_list])

# --- Performance and Load Testing Integration ---

class PerformanceTestGenerator:
    """Generate performance and load tests"""
    
    def generate_load_test_script(self, fields: List[FormField], base_url: str) -> str:
        """Generate load testing script using locust"""
        return f'''# AI-Generated Load Testing Script for Form Submission
from locust import HttpUser, task, between
import random
import json

class FormSubmissionUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup for each user"""
        self.test_data = {{
''' + ''.join([f'            "{field.name}": {json.dumps(self._get_load_test_data(field))},\n' for field in fields]) + '''        }}
    
    @task(3)
    def submit_valid_form(self):
        """Submit form with valid data"""
        form_data = {{
''' + ''.join([f'            "{field.name}": random.choice(self.test_data["{field.name}"]),\n' for field in fields]) + '''        }}
        
        response = self.client.post("/submit-form", data=form_data)
        
        if response.status_code != 200:
            print(f"Form submission failed: {{response.status_code}}")
    
    @task(1)
    def submit_invalid_form(self):
        """Submit form with invalid data to test error handling"""
        invalid_data = {{
            "{fields[0].name if fields else 'email'}": "invalid_data_test",
        }}
        
        response = self.client.post("/submit-form", data=invalid_data)
        # Invalid submissions might return 400, which is expected
    
    @task(2)
    def load_form_page(self):
        """Load the form page"""
        response = self.client.get("/")
        
        if response.status_code != 200:
            print(f"Page load failed: {{response.status_code}}")

# Run with: locust -f this_file.py --host={base_url}
'''
    
    def _get_load_test_data(self, field: FormField) -> List[str]:
        """Get diverse data for load testing"""
        valid_data, _, _, _, _ = generate_test_data(field.field_type)
        return valid_data[:5]  # Return multiple options for variety

# --- Export and Reporting Features ---

class TestDataExporter:
    """Export test data in various formats"""
    
    def export_to_csv(self, fields: List[FormField], test_data: Dict) -> str:
        """Export test data to CSV format"""
        import io
        output = io.StringIO()
        
        # Write header
        output.write("Field Name,Field Type,Test Type,Test Data,Expected Result\n")
        
        for field in fields:
            field_test_data = test_data.get(field.name, {})
            
            # Valid data
            for data in field_test_data.get('valid_data', []):
                output.write(f'"{field.name}","{field.field_type}","Valid","{data}","Pass"\n')
            
            # Invalid data
            for data in field_test_data.get('invalid_data', []):
                output.write(f'"{field.name}","{field.field_type}","Invalid","{data}","Validation Error"\n')
        
        return output.getvalue()
    
    def export_to_json(self, fields: List[FormField], test_data: Dict) -> str:
        """Export test data to JSON format"""
        export_data = {
            "metadata": {
                "generated_at": datetime.datetime.now().isoformat(),
                "field_count": len(fields),
                "generator_version": "2.0.0"
            },
            "fields": []
        }
        
        for field in fields:
            field_data = {
                "name": field.name,
                "type": field.field_type,
                "label": field.label,
                "required": field.required,
                "constraints": field.constraints,
                "test_data": test_data.get(field.name, {})
            }
            export_data["fields"].append(field_data)
        
        return json.dumps(export_data, indent=2)
    
    def export_to_excel(self, fields: List[FormField], test_data: Dict) -> bytes:
        """Export test data to Excel format"""
        try:
            import pandas as pd
            
            # Create DataFrames for different test types
            all_data = []
            
            for field in fields:
                field_test_data = test_data.get(field.name, {})
                
                for test_type, values in field_test_data.items():
                    if isinstance(values, list):
                        for value in values:
                            all_data.append({
                                'Field Name': field.name,
                                'Field Type': field.field_type,
                                'Label': field.label,
                                'Required': field.required,
                                'Test Type': test_type,
                                'Test Data': value,
                                'Expected Result': 'Pass' if 'valid' in test_type else 'Validation Error'
                            })
            
            df = pd.DataFrame(all_data)
            
            # Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Test Data', index=False)
                
                # Add summary sheet
                summary_df = pd.DataFrame({
                    'Metric': ['Total Fields', 'Required Fields', 'Test Cases Generated'],
                    'Value': [
                        len(fields),
                        sum(1 for f in fields if f.required),
                        len(all_data)
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            return output.getvalue()
            
        except ImportError:
            raise ImportError("pandas and openpyxl required for Excel export")

# --- AI-Powered Analysis and Insights ---

def analyze_form_complexity(fields: List[FormField]) -> Dict:
    """Analyze form complexity and provide insights"""
    analysis = {
        "complexity_score": 0,
        "risk_factors": [],
        "recommendations": [],
        "testing_priority": "medium"
    }
    
    # Calculate complexity score
    base_score = len(fields) * 2
    
    # Add complexity for different field types
    complex_types = ["password", "email", "phone", "card-number", "date-of-birth"]
    for field in fields:
        if field.field_type in complex_types:
            base_score += 3
        if field.required:
            base_score += 2
    
    analysis["complexity_score"] = base_score
    
    # Identify risk factors
    if len(fields) > 10:
        analysis["risk_factors"].append("Large number of fields may impact user experience")
    
    required_fields = [f for f in fields if f.required]
    if len(required_fields) / len(fields) > 0.7:
        analysis["risk_factors"].append("High percentage of required fields")
    
    # Security-sensitive fields
    sensitive_types = ["password", "card-number", "cvv"]
    sensitive_fields = [f for f in fields if f.field_type in sensitive_types]
    if sensitive_fields:
        analysis["risk_factors"].append("Contains security-sensitive fields requiring extra validation")
    
    # Generate recommendations
    if base_score > 50:
        analysis["testing_priority"] = "high"
        analysis["recommendations"].append("Implement comprehensive validation testing")
        analysis["recommendations"].append("Consider progressive form disclosure")
    elif base_score > 25:
        analysis["testing_priority"] = "medium"
        analysis["recommendations"].append("Focus on cross-field validation")
    else:
        analysis["testing_priority"] = "low"
        analysis["recommendations"].append("Standard validation testing sufficient")
    
    return analysis

def get_ai_testing_strategy(fields: List[FormField]) -> str:
    """Get AI-generated testing strategy"""
    if not AI_AVAILABLE:
        return "AI testing strategy unavailable - using default approach"
    
    try:
        field_summary = []
        for field in fields:
            field_summary.append({
                "name": field.name,
                "type": field.field_type,
                "required": field.required,
                "constraints": field.constraints
            })
        
        complexity_analysis = analyze_form_complexity(fields)
        
        prompt = f"""
        Create a comprehensive testing strategy for this form:
        
        Fields: {json.dumps(field_summary, indent=2)}
        Complexity Analysis: {json.dumps(complexity_analysis, indent=2)}
        
        Provide a strategic testing approach covering:
        1. Test prioritization based on risk assessment
        2. Specific test scenarios to focus on
        3. Automation recommendations
        4. Performance testing considerations
        5. Security testing priorities
        6. User experience testing aspects
        7. Cross-browser/device testing needs
        8. Maintenance and regression testing strategy
        
        Format as a clear, actionable testing plan.
        """
        
        return azure_openai_client.generate_response(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
    except Exception as e:
        logger.error(f"Error generating AI testing strategy: {e}")
        return f"Error generating AI strategy: {str(e)}"

#
# # --- Main UI Functions ---
#
#         # Extract form fields with spinner to show this is processing
#         with st.spinner("🔍 Analysing image and extracting form fields..."):
#             try:
#                 extracted_labels = extract_form_fields(image)
#
#                 if not extracted_labels:
#                     warning_html = """
#                     <div class="warning-message">
#                         <strong>⚠️ No form fields detected in the image</strong>
#                         <p>The OCR process couldn't identify any form fields. Try these solutions:</p>
#                         <ul>
#                             <li><strong>Image Quality:</strong> Ensure the image has clear, readable text</li>
#                             <li><strong>Form Visibility:</strong> Check if form labels are clearly visible</li>
#                             <li><strong>Focus Area:</strong> Try cropping to focus on the form area</li>
#                             <li><strong>Resolution:</strong> Increase image resolution or contrast</li>
#                             <li><strong>Orientation:</strong> Make sure text is horizontal (not rotated)</li>
#                             <li><strong>Alternative:</strong> Use manual field entry below</li>
#                         </ul>
#                     </div>
#                     """
#                     st.markdown(warning_html, unsafe_allow_html=True)
#                     extracted_labels = []
#                     # Add notification for empty field detection
#                     if NOTIFICATIONS_AVAILABLE:
#                         notifications.add_notification(
#                             module_name="intelligent_test_data_generation",
#                             status="warning",
#                             message="No form fields detected in image",
#                             details="The OCR process couldn't identify any form fields in the uploaded image",
#                             action_steps=["Try uploading a clearer screenshot", "Use manual field entry instead",
#                                          "Make sure form labels are clearly visible in the image"]
#                         )
#                 else:
#                     with col2:
#                         st.success(f"✅ Successfully detected {len(extracted_labels)} potential form fields!")
#
#                         # Display fields with better styling
#                         fields_html = """
#                         <div class="success-message">
#                             <strong>🏷️ Detected Form Fields:</strong>
#                             <ul style="margin-top: 0.5rem;">
#                         """
#                         for label in extracted_labels:
#                             fields_html += f"<li>{label}</li>"
#                         fields_html += """
#                             </ul>
#                             <p style="margin-bottom: 0;"><em>💡 You can edit these fields below before generating test data.</em></p>
#                         </div>
#                         """
#                         st.markdown(fields_html, unsafe_allow_html=True)
#
#                     # Add notification for successful field detection
#                     if NOTIFICATIONS_AVAILABLE:
#                         notifications.add_notification(
#                             module_name="intelligent_test_data_generation",
#                             status="success",
#                             message="Form fields detected successfully",
#                             details=f"Detected {len(extracted_labels)} form fields from uploaded image",
#                             action_steps=["Review and edit detected fields if needed", "Generate test data for these fields"]
#                         )
#
#                         # Track execution in metrics
#                         if hasattr(notifications, 'handle_execution_result'):
#                             notifications.handle_execution_result(
#                                 module_name="intelligent_test_data_generation",
#                                 success=True,
#                                 execution_details=f"Detected {len(extracted_labels)} form fields from image"
#                             )
#             except Exception as e:
#                 error_html = f"""
#                 <div class="error-message">
#                     <strong>❌ Error processing image</strong>
#                     <p><strong>Error Details:</strong> {str(e)}</p>
#                     <p><strong>Troubleshooting:</strong></p>
#                     <ul>
#                         <li>Check if the image format is supported (PNG, JPG, JPEG)</li>
#                         <li>Verify the image file is not corrupted</li>
#                         <li>Try with a different image</li>
#                         <li>Use manual field entry as an alternative</li>
#                         <li>Ensure sufficient memory is available for image processing</li>
#                     </ul>
#                 </div>
#                 """
#                 st.markdown(error_html, unsafe_allow_html=True)
#                 extracted_labels = []
#                 # Add notification for image processing error
#                 if NOTIFICATIONS_AVAILABLE:
#                     notifications.add_notification(
#                         module_name="intelligent_test_data_generation",
#                         status="error",
#                         message="Failed to process image",
#                         details=f"Error: {str(e)}",
#                         action_steps=["Check if the image format is supported (PNG, JPG, JPEG)",
#                                      "Try with a different image", "Use manual field entry instead"]
#                     )
#
#                     # Track execution failure in metrics
#                     if hasattr(notifications, 'handle_execution_result'):
#                         notifications.handle_execution_result(
#                             module_name="intelligent_test_data_generation",
#                             success=False,
#                             execution_details=f"Image processing error: {str(e)}"
#                         )
#
#         # Let user edit the detected fields
#         if extracted_labels:
#             with st.expander("Edit Detected Fields", expanded=True):
#                 edited_labels = []
#                 for i, label in enumerate(extracted_labels):
#                     col1, col2 = st.columns([3, 1])
#                     with col1:
#                         edited_label = st.text_input(f"Field {i + 1}", value=label, key=f"field_{i}")
#                     with col2:
#                         field_type = st.selectbox(
#                             "Field Type",
#                             ["auto-detect", "text", "email", "password", "phone", "number",
#                              "date", "date-of-birth", "address", "first-name", "last-name",
#                              "city", "state", "country", "zip", "currency", "card-number",
#                              "card-expiry", "cvv", "url", "domain-name"],
#                             index=0,
#                             key=f"type_{i}"
#                         )
#                     if edited_label.strip():  # Only add non-empty fields
#                         edited_labels.append((edited_label, field_type))
#
#                 # Option to add more fields manually
#                 st.divider()
#                 st.subheader("Add additional fields")
#                 col1, col2 = st.columns([3, 1])
#                 with col1:
#                     new_field = st.text_input("New field label")
#                 with col2:
#                     new_field_type = st.selectbox(
#                         "Field Type",
#                         ["auto-detect", "text", "email", "password", "phone", "number",
#                          "date", "date-of-birth", "address", "first-name", "last-name",
#                          "city", "state", "country", "zip", "currency", "card-number",
#                          "card-expiry", "cvv", "url", "domain-name"],
#                         index=0,
#                         key="new_field_type"
#                     )
#                 if st.button("Add Field") and new_field.strip():
#                     edited_labels.append((new_field, new_field_type))
#
#                     # Add notification for successful field addition
#                     if NOTIFICATIONS_AVAILABLE:
#                         notifications.add_notification(
#                             module_name="intelligent_test_data_generation",
#                             status="success",
#                             message="New field added successfully",
#                             details=f"Added field '{new_field}' with type '{new_field_type if new_field_type != 'auto-detect' else infer_field_type(new_field)}'",
#                             action_steps=["Add more fields if needed", "Generate test data when ready"]
#                         )
#
#                     st.rerun()
#
#             # Generate test data for the edited fields
#             if st.button("Generate Test Data", type="primary", use_container_width=True):
#                 if not edited_labels:
#                     st.warning("⚠️ No fields to generate test data for.")
#                     # Add notification for empty fields error
#                     if NOTIFICATIONS_AVAILABLE:
#                         notifications.add_notification(
#                             module_name="intelligent_test_data_generation",
#                             status="warning",
#                             message="No fields to generate test data for",
#                             details="Please add or detect form fields before generating test data",
#                             action_steps=["Upload a form screenshot or manually add form fields"]
#                         )
#                 else:
#                     with st.spinner("🔄 Generating comprehensive test data..."):
#                         data = []
#                         test_scenarios = []
#
#                         try:
#                             for label, field_type_option in edited_labels:
#                                 # If auto-detect, use our field detection logic
#                                 if field_type_option == "auto-detect":
#                                     field_type = infer_field_type(label)
#                                 else:
#                                     field_type = field_type_option
#
#                                 # Get constraints for this field type
#                                 constraints = get_field_constraints(field_type)
#
#                                 # Generate HTML attributes
#                                 html_attributes = get_html_attributes(field_type, constraints)
#
#                                 # Generate comprehensive test data
#                                 test_data = generate_test_data(field_type)
#                                 valid_data, invalid_data = test_data[0], test_data[1]
#                                 exploits = test_data[2] if len(test_data) > 2 else []
#
#                                 # Generate test scenarios if requested
#                                 if include_scenarios:
#                                     field_scenarios = generate_test_scenarios(field_type, test_data)
#                                     for scenario in field_scenarios:
#                                         scenario["field"] = label
#                                     test_scenarios.extend(field_scenarios)
#
#                                 # Prepare the data for the dataframe
#                                 data_item = {
#                                     "Field Label": label,
#                                     "Field Type": field_type,
#                                     "Min Length": str(constraints["min_length"]) if constraints["min_length"] is not None else "-",
#                                     "Max Length": str(constraints["max_length"]) if constraints["max_length"] is not None else "-",
#                                     "Allowed Characters": constraints["allowed_chars"] or "-",
#                                     "Format": constraints["format"] or "-",
#                                     "Frontend Attribute Suggestions": ", ".join(html_attributes),
#                                     "Valid Test Data": ", ".join(valid_data[:5]),  # Limit to first 5 examples
#                                     "Invalid Test Data": ", ".join(invalid_data[:5]),
#                                 }
#
#                                 # Add boundary and security data if requested
#                                 if include_boundary_values and len(test_data) > 2:
#                                     data_item["Boundary Test Data"] = ", ".join(test_data[2][:3])
#
#                                 if include_security_tests and len(test_data) > 4:
#                                     data_item["Security Test Data"] = ", ".join(test_data[4][:3])
#
#                                 data_item["Recommended Validation"] = ", ".join(constraints["validation"])
#
#                                 data.append(data_item)
#
#                             # Create and display the dataframe
#                             df = pd.DataFrame(data)
#                             st.subheader("�� Field Analysis and Test Data")
#                             st.dataframe(df, use_container_width=True)
#
#                             # Send success notification
#                             if NOTIFICATIONS_AVAILABLE:
#                                 notifications.add_notification(
#                                     module_name="intelligent_test_data_generation",
#                                     status="success",
#                                     message="Test data generated successfully",
#                                     details=f"Generated test data for {len(data)} fields with {len(test_scenarios)} test scenarios",
#                                     action_steps=["Review the generated test data", "Export data in desired format", "Generate test code"]
#                                 )
#
#                                 # Track execution in metrics
#                                 if hasattr(notifications, 'handle_execution_result'):
#                                     notifications.handle_execution_result(
#                                         module_name="intelligent_test_data_generation",
#                                         success=True,
#                                         execution_details=f"Generated test data for {len(data)} fields"
#                                     )
#
#                         except Exception as e:
#                             st.error(f"❌ Error generating test data: {str(e)}")
#                             # Send error notification
#                             if NOTIFICATIONS_AVAILABLE:
#                                 notifications.add_notification(
#                                     module_name="intelligent_test_data_generation",
#                                     status="error",
#                                     message="Failed to generate test data",
#                                     details=f"Error: {str(e)}",
#                                     action_steps=["Check field definitions", "Try again with different field types"]
#                                 )
#
#                                 # Track execution failure in metrics
#                                 if hasattr(notifications, 'handle_execution_result'):
#                                     notifications.handle_execution_result(
#                                         module_name="intelligent_test_data_generation",
#                                         success=False,
#                                         execution_details=f"Error: {str(e)}"
#                                     )
#                             return
#
#                         # Display test scenarios if generated
#                         if include_scenarios and test_scenarios:
#                             st.subheader("📋 Test Scenarios")
#                             scenarios_df = pd.DataFrame(test_scenarios)
#                             st.dataframe(scenarios_df, use_container_width=True)
#
#                         # Save to history if requested
#                         if save_to_history:
#                             field_types = [item["Field Type"] for item in data]
#                             history_item = create_history_item("Test Generation", field_types, len(data))
#                             if save_test_history(history_item):
#                                 st.success("Test generation saved to history.")
#
#                                 # Add notification for successful history save
#                                 if NOTIFICATIONS_AVAILABLE:
#                                     notifications.add_notification(
#                                         module_name="intelligent_test_data_generation",
#                                         status="success",
#                                         message="Test generation saved to history",
#                                         details=f"Saved data for {len(data)} fields with the following types: {', '.join(set(field_types))}",
#                                         action_steps=["View history in the Team Usage tab", "Generate more test data sets for comparison"]
#                                     )
#                             else:
#                                 st.error("Failed to save test generation to history.")
#
#                                 # Add notification for history save error
#                                 if NOTIFICATIONS_AVAILABLE:
#                                     notifications.add_notification(
#                                         module_name="intelligent_test_data_generation",
#                                         status="error",
#                                         message="Failed to save history",
#                                         details="Could not save test generation to history file",
#                                         action_steps=["Check file permissions", "Ensure history directory is writable", "Try again later"]
#                                     )
#
#                         # Generate code samples
#                         st.subheader("💻 Implementation Examples")
#                         code_tab1, code_tab2, code_tab3 = st.tabs(["HTML Form", "JavaScript Validation", "Test Script"])
#
#                         with code_tab1:
#                             html_code = "<form id='validationForm' method='post'>\n"
#                             for label, field_type_option in edited_labels:
#                                 field_id = label.lower().replace(" ", "_")
#                                 if field_type_option == "auto-detect":
#                                     field_type = infer_field_type(label)
#                                 else:
#                                     field_type = field_type_option
#                                 constraints = get_field_constraints(field_type)
#                                 attrs = get_html_attributes(field_type, constraints)
#
#                                 html_code += f"  <div class='form-group'>\n"
#                                 html_code += f"    <label for='{field_id}'>{label}</label>\n"
#                                 html_code += f"    <input id='{field_id}' name='{field_id}' {' '.join(attrs)} />\n"
#                                 html_code += f"  </div>\n"
#                             html_code += "  <button type='submit'>Submit</button>\n</form>"
#                             st.code(html_code, language="html")
#
#                     with code_tab2:
#                         js_code = "// Basic JavaScript form validation\ndocument.addEventListener('DOMContentLoaded', function() {\n"
#                         js_code += "  const form = document.getElementById('validationForm');\n"
#                         js_code += "  form.addEventListener('submit', function(e) {\n"
#                         js_code += "    let valid = true;\n"
#                         for label, field_type_option in edited_labels:
#                             field_id = label.lower().replace(" ", "_")
#                             if field_type_option == "auto-detect":
#                                 field_type = infer_field_type(label)
#                             else:
#                                 field_type = field_type_option
#                             constraints = get_field_constraints(field_type)
#                             if constraints["min_length"]:
#                                 js_code += f"    if (form['{field_id}'].value.length < {constraints['min_length']}) {{ valid = false; alert('{label}: Too short'); }}\n"
#                             if constraints["max_length"]:
#                                 js_code += f"    if (form['{field_id}'].value.length > {constraints['max_length']}) {{ valid = false; alert('{label}: Too long'); }}\n"
#                             if constraints["pattern"]:
#                                 js_code += f"    if (!form['{field_id}'].value.match(/{constraints['pattern'].replace('^', '').replace('$', '')}/)) {{ valid = false; alert('{label}: Invalid format'); }}\n"
#                         js_code += "    if (!valid) e.preventDefault();\n"
#                         js_code += "  });\n"
#                         js_code += "});"
#                         st.code(js_code, language="javascript")
#
#                     with code_tab3:
#                         # Add tabs for different frameworks
#                         framework_tabs = st.tabs([
#                             f"{test_framework}",
#                             "API Tests" if api_testing else "Data Sets",
#                             "Accessibility Tests" if include_accessibility_tests else "Batch Tests"
#                         ])
#
#                         with framework_tabs[0]:
#                             # Determine which test code to generate based on selected framework
#                             if test_framework == "Robot Framework":
#                                 # Robot Framework example
#                                 test_script = "*** Settings ***\n"
#                                 test_script += "Library    SeleniumLibrary\n\n"
#                                 test_script += "*** Variables ***\n"
#                                 for label, field_type_option in edited_labels:
#                                     field_id = label.lower().replace(" ", "_")
#                                     if field_type_option == "auto-detect":
#                                         field_type = infer_field_type(label)
#                                     else:
#                                         field_type = field_type_option
#                                     test_data = generate_test_data(field_type)[0]
#                                     if test_data:
#                                         test_script += f"${{{field_id}}}    {test_data[0]}\n"
#
#                                 test_script += "\n*** Test Cases ***\n"
#                                 test_script += "Verify Form Submission\n"
#                                 test_script += "    Open Browser    http://localhost:8000    chrome\n"
#                                 for label, field_type_option in edited_labels:
#                                     field_id = label.lower().replace(" ", "_")
#                                     test_script += f"    Input Text    id:{field_id}    ${{{field_id}}}\n"
#                                 test_script += "    Click Button    css:button[type=submit]\n"
#                                 test_script += "    Page Should Contain    Success\n"
#                                 test_script += "    [Teardown]    Close Browser\n"
#
#                             elif test_framework == "Cypress":
#                                 # Cypress example
#                                 test_script = "describe('Form submission', () => {\n"
#                                 test_script += "  it('should submit the form with valid data', () => {\n"
#                                 test_script += "    cy.visit('http://localhost:8000');\n\n"
#                                 for label, field_type_option in edited_labels:
#                                     field_id = label.lower().replace(" ", "_")
#                                     if field_type_option == "auto-detect":
#                                         field_type = infer_field_type(label)
#                                     else:
#                                         field_type = field_type_option
#                                     test_data = generate_test_data(field_type)[0]
#                                     if test_data:
#                                         test_script += f"    cy.get('#{field_id}').type('{test_data[0]}');\n"
#                                 test_script += "\n    cy.get('button[type=submit]').click();\n"
#                                 test_script += "    cy.contains('Success').should('be.visible');\n"
#                                 test_script += "  });\n});"
#
#                             elif test_framework == "pytest":
#                                 # pytest example
#                                 test_script = "import pytest\n"
#                                 test_script += "from selenium import webdriver\n"
#                                 test_script += "from selenium.webdriver.common.by import By\n\n"
#                                 test_script += "def test_form_submission():\n"
#                                 test_script += "    driver = webdriver.Chrome()\n"
#                                 test_script += "    driver.get('http://localhost:8000')\n\n"
#                                 for label, field_type_option in edited_labels:
#                                     field_id = label.lower().replace(" ", "_")
#                                     if field_type_option == "auto-detect":
#                                         field_type = infer_field_type(label)
#                                     else:
#                                         field_type = field_type_option
#                                     test_data = generate_test_data(field_type)[0]
#                                     if test_data:
#                                         test_script += f"    driver.find_element(By.ID, '{field_id}').send_keys('{test_data[0]}')\n"
#                                 test_script += "\n    driver.find_element(By.CSS_SELECTOR, 'button[type=submit]').click()\n"
#                                 test_script += "    assert 'Success' in driver.page_source\n"
#                                 test_script += "    driver.quit()"
#
#                             elif test_framework == "Playwright":
#                                 # Playwright example
#                                 if language == "Python":
#                                     test_script = "from playwright.sync_api import sync_playwright\n\n"
#                                     test_script += "def test_form_submission():\n"
#                                     test_script += "    with sync_playwright() as p:\n"
#                                     test_script += "        browser = p.chromium.launch()\n"
#                                     test_script += "        page = browser.new_page()\n"
#                                     test_script += "        page.goto('http://localhost:8000')\n\n"
#                                     for label, field_type_option in edited_labels:
#                                         field_id = label.lower().replace(" ", "_")
#                                         if field_type_option == "auto-detect":
#                                             field_type = infer_field_type(label)
#                                         else:
#                                             field_type = field_type_option
#                                         test_data = generate_test_data(field_type)[0]
#                                         if test_data:
#                                             test_script += f"        page.fill('#{field_id}', '{test_data[0]}')\n"
#                                     test_script += "\n        page.click('button[type=submit]')\n"
#                                     test_script += "        assert page.inner_text('body').find('Success') > -1\n"
#                                     test_script += "        browser.close()"
#                                 else:  # JavaScript
#                                     test_script = "const { test, expect } = require('@playwright/test');\n\n"
#                                     test_script += "test('form submission', async ({ page }) => {\n"
#                                     test_script += "  await page.goto('http://localhost:8000');\n\n"
#                                     for label, field_type_option in edited_labels:
#                                         field_id = label.lower().replace(" ", "_")
#                                         if field_type_option == "auto-detect":
#                                             field_type = infer_field_type(label)
#                                         else:
#                                             field_type = field_type_option
#                                         test_data = generate_test_data(field_type)[0]
#                                         if test_data:
#                                             test_script += f"  await page.fill('#{field_id}', '{test_data[0]}');\n"
#                                     test_script += "\n  await page.click('button[type=submit]');\n"
#                                     test_script += "  await expect(page.locator('body')).toContainText('Success');\n"
#                                     test_script += "});"
#
#                             else:  # Default to Selenium
#                                 # Selenium example
#                                 if language == "Python":
#                                     test_script = "# Python Selenium test example\n"
#                                     test_script += "from selenium import webdriver\nfrom selenium.webdriver.common.by import By\n\n"
#                                     test_script += "driver = webdriver.Chrome()\n"
#                                     test_script += "driver.get('http://localhost:8000')  # Change to your form URL\n\n"
#                                     for label, field_type_option in edited_labels:
#                                         field_id = label.lower().replace(" ", "_")
#                                         if field_type_option == "auto-detect":
#                                             field_type = infer_field_type(label)
#                                         else:
#                                             field_type = field_type_option
#                                         test_data = generate_test_data(field_type)[0]
#                                         if test_data:
#                                             test_script += f"driver.find_element(By.ID, '{field_id}').send_keys('{test_data[0]}')\n"
#                                     test_script += "driver.find_element(By.CSS_SELECTOR, 'button[type=submit]').click()\n"
#                                     test_script += "# Add assertions/checks as needed\n"
#                                     test_script += "assert 'Success' in driver.page_source\n"
#                                     test_script += "driver.quit()"
#                                 elif language == "Java":
#                                     test_script = "// Java Selenium example\n"
#                                     test_script += "import org.openqa.selenium.By;\n"
#                                     test_script += "import org.openqa.selenium.WebDriver;\n"
#                                     test_script += "import org.openqa.selenium.chrome.ChromeDriver;\n\n"
#                                     test_script += "public class FormTest {\n"
#                                     test_script += "    public static void main(String[] args) {\n"
#                                     test_script += "        WebDriver driver = new ChromeDriver();\n"
#                                     test_script += "        driver.get(\"http://localhost:8000\");\n\n"
#                                     for label, field_type_option in edited_labels:
#                                         field_id = label.lower().replace(" ", "_")
#                                         if field_type_option == "auto-detect":
#                                             field_type = infer_field_type(label)
#                                         else:
#                                             field_type = field_type_option
#                                         test_data = generate_test_data(field_type)[0]
#                                         if test_data:
#                                             test_script += f"        driver.findElement(By.id(\"{field_id}\")).sendKeys(\"{test_data[0]}\");\n"
#                                     test_script += "\n        driver.findElement(By.cssSelector(\"button[type=submit]\")).click();\n"
#                                     test_script += "        // Add assertions as needed\n"
#                                     test_script += "        assert driver.getPageSource().contains(\"Success\");\n"
#                                     test_script += "        driver.quit();\n"
#                                     test_script += "    }\n}"
#                                 else:  # JavaScript
#                                     test_script = "// JavaScript Selenium example\n"
#                                     test_script += "const {Builder, By, Key} = require('selenium-webdriver');\n\n"
#                                     test_script += "async function runTest() {\n"
#                                     test_script += "  let driver = await new Builder().forBrowser('chrome').build();\n"
#                                     test_script += "  try {\n"
#                                     test_script += "    await driver.get('http://localhost:8000');\n\n"
#                                     for label, field_type_option in edited_labels:
#                                         field_id = label.lower().replace(" ", "_")
#                                         if field_type_option == "auto-detect":
#                                             field_type = infer_field_type(label)
#                                         else:
#                                             field_type = field_type_option
#                                         test_data = generate_test_data(field_type)[0]
#                                         if test_data:
#                                             test_script += f"    await driver.findElement(By.id('{field_id}')).sendKeys('{test_data[0]}');\n"
#                                     test_script += "\n    await driver.findElement(By.css('button[type=submit]')).click();\n"
#                                     test_script += "    // Add assertions as needed\n"
#                                     test_script += "  } finally {\n"
#                                     test_script += "    await driver.quit();\n"
#                                     test_script += "  }\n"
#                                     test_script += "}\n\n"
#                                     test_script += "runTest();"
#
#                             st.code(test_script,
#                                     language=language.lower() if language != "Robot Framework" else "robotframework")
#
#                         with framework_tabs[1]:
#                             if api_testing:
#                                 # API test code
#                                 api_test_code = """# API Test Example using pytest
#                 import pytest
#                 import requests
#                 import json
#
#                 BASE_URL = "http://localhost:8000/api"  # Change to your API URL
#
#                 def test_form_submission_api():
#                     # Prepare payload with test data
#                     payload = {"""
#                                 for label, field_type_option in edited_labels:
#                                     field_id = label.lower().replace(" ", "_")
#                                     if field_type_option == "auto-detect":
#                                         field_type = infer_field_type(label)
#                                     else:
#                                         field_type = field_type_option
#                                     test_data = generate_test_data(field_type)[0]
#                                     if test_data:
#                                         api_test_code += f"        '{field_id}': '{test_data[0]}',\n"
#
#                                 api_test_code += """    }
#
#                     response = requests.post(f"{BASE_URL}/submit", json=payload)
#
#                     # Check status code
#                     assert response.status_code == 200
#
#                     # Validate response structure
#                     response_data = response.json()
#                     assert "status" in response_data
#                     assert response_data["status"] == "success"
#                 """
#                                 st.code(api_test_code, language="python")
#
#                                 # Add REST API contract sample
#                                 st.subheader("API Contract Example (OpenAPI)")
#                                 openapi_spec = """{
#                   "openapi": "3.0.0",
#                   "info": {
#                     "title": "Form Submission API",
#                     "version": "1.0.0",
#                     "description": "API for form data submission"
#                   },
#                   "paths": {
#                     "/api/submit": {
#                       "post": {
#                         "summary": "Submit form data",
#                         "requestBody": {
#                           "required": true,
#                           "content": {
#                             "application/json": {
#                               "schema": {
#                                 "type": "object",
#                                 "properties": {
#                 """
#                                 for i, (label, field_type_option) in enumerate(edited_labels):
#                                     field_id = label.lower().replace(" ", "_")
#                                     if field_type_option == "auto-detect":
#                                         field_type = infer_field_type(label)
#                                     else:
#                                         field_type = field_type_option
#
#                                     # Determine schema type based on field type
#                                     schema_type = "string"
#                                     if field_type in ["number", "age", "currency"]:
#                                         schema_type = "number"
#
#                                     constraints = get_field_constraints(field_type)
#                                     openapi_spec += f"                  \"{field_id}\": {{\n"
#                                     openapi_spec += f"                    \"type\": \"{schema_type}\""
#
#                                     if constraints["min_length"]:
#                                         openapi_spec += f",\n                    \"minLength\": {constraints['min_length']}"
#                                     if constraints["max_length"]:
#                                         openapi_spec += f",\n                    \"maxLength\": {constraints['max_length']}"
#                                     if constraints["pattern"]:
#                                         openapi_spec += f",\n                    \"pattern\": \"{constraints['pattern']}\""
#
#                                     openapi_spec += f"\n                  }}"
#                                     if i < len(edited_labels) - 1:
#                                         openapi_spec += ","
#                                     openapi_spec += "\n"
#
#                                 openapi_spec += """                }
#                               }
#                             }
#                           }
#                         },
#                         "responses": {
#                           "200": {
#                             "description": "Form data submitted successfully",
#                             "content": {
#                               "application/json": {
#                                 "schema": {
#                                   "type": "object",
#                                   "properties": {
#                                     "status": {
#                                       "type": "string",
#                                       "enum": ["success"]
#                                     },
#                                     "message": {
#                                       "type": "string"
#                                     },
#                                     "id": {
#                                       "type": "string",
#                                       "format": "uuid"
#                                     }
#                                   }
#                                 }
#                               }
#                             }
#                           },
#                           "400": {
#                             "description": "Invalid input data",
#                             "content": {
#                               "application/json": {
#                                 "schema": {
#                                   "type": "object",
#                                   "properties": {
#                                     "status": {
#                                       "type": "string",
#                                       "enum": ["error"]
#                                     },
#                                     "message": {
#                                       "type": "string"
#                                     },
#                                     "errors": {
#                                       "type": "object"
#                                     }
#                                   }
#                                 }
#                               }
#                             }
#                           }
#                         }
#                       }
#                     }
#                   }
#                 }"""
#                                 st.code(openapi_spec, language="json")
#                             else:
#                                 # Generate multiple data sets for data-driven testing
#                                 st.subheader("Data-Driven Test Data Sets")
#                                 st.write("Generate multiple sets of test data for data-driven testing")
#
#                                 num_sets = st.slider("Number of data sets to generate", 5, 100, 10)
#
#                                 if st.button("Generate Test Data Sets"):
#                                     with st.spinner("Generating data sets..."):
#                                         # Generate the data
#                                         data_sets = []
#                                         for i in range(num_sets):
#                                             data_set = {}
#                                             for label, field_type_option in edited_labels:
#                                                 field_id = label.lower().replace(" ", "_")
#                                                 if field_type_option == "auto-detect":
#                                                     field_type = infer_field_type(label)
#                                                 else:
#                                                     field_type = field_type_option
#
#                                                 # Get random valid value
#                                                 valid_data = generate_test_data(field_type)[0]
#                                                 if valid_data:
#                                                     import random
#                                                     data_set[field_id] = random.choice(valid_data)
#                                                 else:
#                                                     data_set[field_id] = ""
#
#                                             data_sets.append(data_set)
#
#                                         # Display the data sets
#                                         sets_df = pd.DataFrame(data_sets)
#                                         st.dataframe(sets_df, use_container_width=True)
#
#                                         # Add download buttons
#                                         col1, col2, col3 = st.columns(3)
#
#                                         with col1:
#                                             csv = sets_df.to_csv(index=False)
#                                             b64_csv = base64.b64encode(csv.encode()).decode()
#                                             st.download_button(
#                                                 label="Download CSV",
#                                                 data=csv,
#                                                 file_name="test_data_sets.csv",
#                                                 mime="text/csv"
#                                             )
#
#                                         with col2:
#                                             json_str = sets_df.to_json(orient="records")
#                                             st.download_button(
#                                                 label="Download JSON",
#                                                 data=json_str,
#                                                 file_name="test_data_sets.json",
#                                                 mime="application/json"
#                                             )
#
#                                         with col3:
#                                             # Generate Python test code that uses the data
#                                             py_test_code = "import pytest\nimport pandas as pd\n\n"
#                                             py_test_code += "# Load test data from CSV\n"
#                                             py_test_code += "test_data = pd.read_csv('test_data_sets.csv').to_dict('records')\n\n"
#                                             py_test_code += "@pytest.mark.parameterise('data', test_data)\n"
#                                             py_test_code += "def test_form_with_test_data(data):\n"
#                                             py_test_code += "    # Use the data from the parameterised test\n"
#                                             py_test_code += "    print(f\"Testing with data set: {data}\")\n"
#                                             py_test_code += "    # Here you would run your test with the data\n\n"
#
#                                             st.download_button(
#                                                 label="Download Test Script",
#                                                 data=py_test_code,
#                                                 file_name="data_driven_test.py",
#                                                 mime="text/plain"
#                                             )
#
#                         with framework_tabs[2]:
#                             if include_accessibility_tests:
#                                 # Accessibility test code
#                                 a11y_code = """// Accessibility Tests with Cypress and axe-core
#                 describe('Form Accessibility Tests', () => {
#                   beforeEach(() => {
#                     cy.visit('/');
#                     cy.injectAxe();
#                   });
#
#                   it('should have no accessibility violations', () => {
#                     // Run the accessibility audit
#                     cy.checkA11y();
#                   });
#
#                   it('should have properly labeled form fields', () => {
#                 """
#                                 for label, field_type_option in edited_labels:
#                                     field_id = label.lower().replace(" ", "_")
#                                     a11y_code += f"    cy.get('label[for=\"{field_id}\"]').should('be.visible');\n"
#                                     a11y_code += f"    cy.get('#{field_id}').should('have.attr', 'aria-required', 'true');\n"
#
#                                 a11y_code += """  });
#
#                   it('should have keyboard navigation', () => {
#                     // Test keyboard navigation through the form
#                     cy.get('body').focus().tab(); // First tab should focus first form field
#                 """
#
#                                 # Add tab navigation checks
#                                 for i, (label, field_type_option) in enumerate(edited_labels):
#                                     field_id = label.lower().replace(" ", "_")
#                                     if i > 0:
#                                         a11y_code += "    cy.focused().tab();\n"
#                                     a11y_code += f"    cy.focused().should('have.attr', 'id', '{field_id}');\n"
#
#                                 a11y_code += """    cy.focused().tab();
#                     cy.focused().should('have.attr', 'type', 'submit'); // Submit button should be focused last
#                   });
#                 });"""
#                                 st.code(a11y_code, language="javascript")
#
#                                 # Add accessibility report template
#                                 st.subheader("Accessibility Compliance Checklist")
#
#                                 col1, col2 = st.columns(2)
#                                 with col1:
#                                     st.info("WCAG 2.1 Level A")
#                                     st.write("- ✓ All form fields have associated labels")
#                                     st.write("- ✓ All interactive elements are keyboard accessible")
#                                     st.write("- ✓ Color is not used as the only visual means of conveying information")
#                                     st.write("- ✓ Form has clear instructions")
#
#                                 with col2:
#                                     st.info("WCAG 2.1 Level AA")
#                                     st.write("- ✓ Text has sufficient color contrast (4.5:1)")
#                                     st.write("- ✓ Form is resizable up to 200% without loss of content")
#                                     st.write("- ✓ Form has visible focus indicators")
#                                     st.write("- ✓ Error messages are identified by assistive technology")
#                             else:
#                                 # Batch test generator
#                                 st.subheader("Batch Test Generator")
#
#                                 st.write("Generate a batch of tests covering multiple scenarios")
#
#                                 batch_options = st.multiselect(
#                                     "Test Types to Include",
#                                     ["Valid Input", "Invalid Input", "Boundary Values", "Security Tests",
#                                      "Performance Tests"],
#                                     ["Valid Input", "Invalid Input"]
#                                 )
#                                 if st.button("Generate Batch Tests"):
#                                     with st.spinner("Generating batch tests..."):
#                                         batch_tests = []
#                                         for option in batch_options:
#                                             if option == "Valid Input":
#                                                 batch_tests.append(
#                                                     "Test valid form submission with all fields filled correctly.")
#                                             elif option == "Invalid Input":
#                                                 batch_tests.append(
#                                                     "Test form submission with invalid data (e.g., wrong email format).")
#                                             elif option == "Boundary Values":
#                                                 batch_tests.append(
#                                                     "Test boundary values for numeric fields (e.g., min/max age).")
#                                             elif option == "Security Tests":
#                                                 batch_tests.append(
#                                                     "Test for SQL injection vulnerabilities in text fields.")
#                                             elif option == "Performance Tests":
#                                                 batch_tests.append("Test form submission performance under load.")
#
#                                         st.write("Generated Batch Tests:")
#                                         for test in batch_tests:
#                                             st.write(f"- {test}")


if __name__ == "__main__":
    show_ui()
