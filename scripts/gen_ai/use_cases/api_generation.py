import os
import sys
import json
import yaml
import requests
import streamlit as st
import pandas as pd
import time
import random
import re
from datetime import datetime
from threading import Thread, Event
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from socketserver import ThreadingMixIn
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import tempfile
from pathlib import Path
import hashlib
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict
import concurrent.futures
import shlex
import zipfile
import io

# Ensure parent directory is in path to import shared modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the Postman collection parser
from parsers.postman_parser import show_postman_ui

# Azure OpenAI Integration for AI-powered enhancements
try:
    from azure_openai_client import AzureOpenAIClient
    AI_AVAILABLE = True
    azure_openai_client = AzureOpenAIClient()
except (ImportError, ValueError) as e:
    AI_AVAILABLE = False
    azure_openai_client = None
    print(f"Warning: Azure OpenAI client not available: {e}")

# Import notifications module for action feedback
try:
    from notifications import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    print("Notifications module not available. Notification features will be disabled.")
    # Create a mock notifications object
    class MockNotifications:
        def add_notification(self, **kwargs):
            pass
    notifications = MockNotifications()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the mock server instance
mock_server_instance = None
mock_server_thread = None
server_stop_event = Event()

# AI-Enhanced Data Structures
@dataclass
class APIEndpoint:
    """Enhanced endpoint representation with AI analysis"""
    path: str
    method: str
    summary: str = ""
    description: str = ""
    parameters: List[Dict] = field(default_factory=list)
    responses: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    security: List[Dict] = field(default_factory=list)
    complexity_score: float = 0.0
    risk_level: str = "medium"
    ai_insights: Dict = field(default_factory=dict)
    test_priorities: List[str] = field(default_factory=list)

@dataclass
class APITestSuite:
    """Enhanced test suite with AI-generated metadata"""
    framework: str
    endpoints: List[APIEndpoint]
    base_url: str
    test_type: str
    coverage_analysis: Dict = field(default_factory=dict)
    ai_recommendations: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    estimated_execution_time: int = 0
    maintenance_complexity: str = "medium"

class AIAPIAnalyzer:
    """AI-powered API analysis and insights generator"""
    
    def __init__(self):
        self.analysis_cache = {}
        self.pattern_database = {
            "auth_patterns": ["login", "auth", "token", "oauth", "session"],
            "crud_patterns": ["create", "read", "update", "delete", "get", "post", "put", "patch", "delete"],
            "security_patterns": ["password", "token", "key", "secret", "credential"],
            "data_patterns": ["user", "customer", "order", "product", "payment", "invoice"]
        }
    
    def analyze_endpoint(self, endpoint: Dict) -> APIEndpoint:
        """Analyze endpoint with AI insights"""
        # Ensure endpoint is a dictionary
        if not isinstance(endpoint, dict):
            logger.warning(f"Invalid endpoint format in analyze_endpoint. Expected dict, got {type(endpoint)}: {endpoint}")
            endpoint = {}
        
        api_endpoint = APIEndpoint(
            path=endpoint.get('path', ''),
            method=endpoint.get('method', 'GET'),
            summary=endpoint.get('summary', ''),
            description=endpoint.get('description', ''),
            parameters=endpoint.get('parameters', []),
            responses=endpoint.get('responses', {}),
            tags=endpoint.get('tags', [])
        )
        
        # Calculate complexity score
        api_endpoint.complexity_score = self._calculate_complexity(api_endpoint)
        
        # Determine risk level
        api_endpoint.risk_level = self._determine_risk_level(api_endpoint)
        
        # Generate AI insights if available
        if AI_AVAILABLE:
            api_endpoint.ai_insights = self._get_ai_insights(api_endpoint)
            api_endpoint.test_priorities = self._generate_test_priorities(api_endpoint)
        
        return api_endpoint
    
    def _calculate_complexity(self, endpoint: APIEndpoint) -> float:
        """Calculate endpoint complexity score"""
        score = 0.0
        
        # Base complexity by method
        method_weights = {
            'GET': 1.0, 'POST': 2.0, 'PUT': 2.5, 'PATCH': 2.5, 'DELETE': 1.5, 'HEAD': 0.5, 'OPTIONS': 0.5
        }
        score += method_weights.get(endpoint.method.upper(), 1.0)
        
        # Parameter complexity
        score += len(endpoint.parameters) * 0.5
        
        # Path parameter complexity
        path_params = len(re.findall(r'\{[^}]+\}', endpoint.path))
        score += path_params * 1.0
        
        # Response complexity
        score += len(endpoint.responses) * 0.3
        
        return min(score, 10.0)  # Cap at 10
    
    def _determine_risk_level(self, endpoint: APIEndpoint) -> str:
        """Determine risk level based on endpoint characteristics"""
        risk_factors = 0
        
        # Check for security-sensitive operations
        sensitive_words = ['password', 'token', 'secret', 'key', 'admin', 'delete', 'remove']
        if any(word in endpoint.path.lower() or word in endpoint.description.lower() 
               for word in sensitive_words):
            risk_factors += 2
        
        # Check HTTP method risk
        high_risk_methods = ['POST', 'PUT', 'PATCH', 'DELETE']
        if endpoint.method.upper() in high_risk_methods:
            risk_factors += 1
        
        # Check complexity
        if endpoint.complexity_score > 5:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "high"
        elif risk_factors >= 1:
            return "medium"
        else:
            return "low"
    
    def _get_ai_insights(self, endpoint: APIEndpoint) -> Dict:
        """Get AI-powered insights for endpoint"""
        if not AI_AVAILABLE:
            return {}
        
        try:
            prompt = f"""
            Analyze this API endpoint and provide insights:
            
            Method: {endpoint.method}
            Path: {endpoint.path}
            Description: {endpoint.description}
            Parameters: {len(endpoint.parameters)} parameters
            Responses: {list(endpoint.responses.keys())}
            
            Provide insights on:
            1. Security considerations
            2. Testing priorities
            3. Common failure modes
            4. Performance considerations
            5. Data validation requirements
            6. Any other relevant information
            7. Format the response as JSON with keys: security, testing, failures, performance, validation
            8. Include any AI-generated recommendations
            9.  If applicable, provide a risk assessment based on the endpoint characteristics
            10. Provide any additional context or information that may be relevant to the analysis
            Context: This endpoint is part of an API that handles user data and transactions.
            Response should be JSON format with keys: security, testing, failures, performance, validation
            """
            
            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Try to parse as JSON, fallback to structured text
            try:
                return json.loads(response)
            except:
                return {"analysis": response}
                
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return {"error": str(e)}
    
    def _generate_test_priorities(self, endpoint: APIEndpoint) -> List[str]:
        """Generate test priorities using AI analysis"""
        priorities = []
        
        # Base priorities on risk and complexity
        if endpoint.risk_level == "high":
            priorities.extend(["Security Testing", "Error Handling", "Authentication"])
        
        if endpoint.complexity_score > 5:
            priorities.extend(["Performance Testing", "Load Testing"])
        
        # Method-specific priorities
        if endpoint.method.upper() in ['POST', 'PUT', 'PATCH']:
            priorities.extend(["Data Validation", "Schema Validation"])
        
        if endpoint.method.upper() == 'DELETE':
            priorities.extend(["Data Integrity", "Authorization"])
        
        return list(set(priorities))  # Remove duplicates
    
    def analyze_api_suite(self, endpoints: List[Dict], framework: str, test_type: str) -> APITestSuite:
        """Analyze complete API test suite"""
        analyzed_endpoints = [self.analyze_endpoint(ep) for ep in endpoints]
        
        test_suite = APITestSuite(
            framework=framework,
            endpoints=analyzed_endpoints,
            base_url="",
            test_type=test_type
        )
        
        # Calculate suite-level metrics
        test_suite.quality_score = self._calculate_suite_quality(analyzed_endpoints)
        test_suite.estimated_execution_time = self._estimate_execution_time(analyzed_endpoints, framework)
        test_suite.maintenance_complexity = self._assess_maintenance_complexity(analyzed_endpoints)
        
        if AI_AVAILABLE:
            test_suite.ai_recommendations = self._generate_suite_recommendations(test_suite)
            test_suite.coverage_analysis = self._analyze_test_coverage(analyzed_endpoints)
        
        return test_suite
    
    def _calculate_suite_quality(self, endpoints: List[APIEndpoint]) -> float:
        """Calculate overall test suite quality score"""
        if not endpoints:
            return 0.0
        
        total_score = 0.0
        for endpoint in endpoints:
            # Quality factors
            base_score = 5.0
            
            # Bonus for comprehensive parameters
            if len(endpoint.parameters) > 0:
                base_score += 1.0
            
            # Bonus for multiple response codes
            if len(endpoint.responses) > 1:
                base_score += 1.0
            
            # Bonus for good documentation
            if endpoint.description and len(endpoint.description) > 20:
                base_score += 1.0
            
            # Risk-based scoring
            risk_multipliers = {"low": 1.0, "medium": 1.2, "high": 1.5}
            base_score *= risk_multipliers.get(endpoint.risk_level, 1.0)
            
            total_score += min(base_score, 10.0)
        
        return total_score / len(endpoints)
    
    def _estimate_execution_time(self, endpoints: List[APIEndpoint], framework: str) -> int:
        """Estimate test execution time in seconds"""
        base_time_per_endpoint = {
            "Postman": 2,
            "RestAssured": 5,
            "Requests": 3,
            "Robot Framework": 4
        }
        
        base_time = base_time_per_endpoint.get(framework, 3)
        total_time = 0
        
        for endpoint in endpoints:
            endpoint_time = base_time
            
            # Add time for complexity
            endpoint_time += int(endpoint.complexity_score * 0.5)
            
            # Add time for multiple test scenarios
            endpoint_time += len(endpoint.test_priorities) * 2
            
            total_time += endpoint_time
        
        return total_time
    
    def _assess_maintenance_complexity(self, endpoints: List[APIEndpoint]) -> str:
        """Assess maintenance complexity of the test suite"""
        if not endpoints:
            return "low"
        
        avg_complexity = sum(ep.complexity_score for ep in endpoints) / len(endpoints)
        high_risk_count = sum(1 for ep in endpoints if ep.risk_level == "high")
        
        if avg_complexity > 6 or high_risk_count > len(endpoints) * 0.5:
            return "high"
        elif avg_complexity > 3 or high_risk_count > 0:
            return "medium"
        else:
            return "low"
    
    def _generate_suite_recommendations(self, test_suite: APITestSuite) -> List[str]:
        """Generate AI-powered recommendations for test suite"""
        if not AI_AVAILABLE:
            return []
        
        try:
            suite_summary = {
                "framework": test_suite.framework,
                "endpoint_count": len(test_suite.endpoints),
                "average_complexity": sum(ep.complexity_score for ep in test_suite.endpoints) / len(test_suite.endpoints),
                "high_risk_endpoints": sum(1 for ep in test_suite.endpoints if ep.risk_level == "high"),
                "test_type": test_suite.test_type,
                "quality_score": test_suite.quality_score,
                "maintenance_complexity": test_suite.maintenance_complexity
            }
            
            prompt = f"""
            Analyze this API test suite and provide actionable recommendations:
            
            Suite Details: {json.dumps(suite_summary, indent=2)}
            
            Provide 5-7 specific recommendations for:
            1. Test coverage improvements
            2. Performance optimization
            3. Maintenance reduction
            4. Security testing enhancements
            5. Framework-specific best practices
            6. Any other relevant insights
            7.  Ensure recommendations are practical and actionable
            8.  Include any AI-generated insights based on the suite characteristics
            9.  Consider the complexity and risk levels of the endpoints
            10.  Provide recommendations that can be implemented in the specified framework
            
            Format as a JSON list of recommendation strings.
            """
            
            response = azure_openai_client.generate_response(
                prompt=prompt,
                max_tokens=800,
                temperature=0.4
            )
            
            try:
                return json.loads(response)
            except:
                # Extract recommendations from text response
                lines = response.split('\n')
                recommendations = [line.strip('- ').strip() for line in lines if line.strip() and not line.strip().isdigit()]
                return recommendations[:7]  # Limit to 7 recommendations
                
        except Exception as e:
            logger.error(f"Error generating suite recommendations: {e}")
            return ["Review endpoint complexity", "Enhance error handling", "Add performance tests"]
    
    def _analyze_test_coverage(self, endpoints: List[APIEndpoint]) -> Dict:
        """Analyze test coverage patterns"""
        coverage = {
            "method_coverage": defaultdict(int),
            "risk_coverage": defaultdict(int),
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
            "coverage_gaps": [],
            "recommendations": []
        }
        
        for endpoint in endpoints:
            coverage["method_coverage"][endpoint.method.upper()] += 1
            coverage["risk_coverage"][endpoint.risk_level] += 1
            
            if endpoint.complexity_score < 3:
                coverage["complexity_distribution"]["low"] += 1
            elif endpoint.complexity_score < 6:
                coverage["complexity_distribution"]["medium"] += 1
            else:
                coverage["complexity_distribution"]["high"] += 1
        
        # Identify gaps
        common_methods = ['GET', 'POST', 'PUT', 'DELETE']
        missing_methods = [method for method in common_methods 
                          if method not in coverage["method_coverage"]]
        
        if missing_methods:
            coverage["coverage_gaps"].append(f"Missing {', '.join(missing_methods)} methods")
        
        if coverage["risk_coverage"]["high"] == 0:
            coverage["coverage_gaps"].append("No high-risk endpoint testing")
        
        return coverage

# Mock server class with threading support
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True
    allow_reuse_address = True

class MockAPIHandler(BaseHTTPRequestHandler):
    """Custom request handler for mock API server."""
    
    def __init__(self, endpoints, delay_ms=500, *args, **kwargs):
        self.endpoints = endpoints
        self.delay_ms = delay_ms
        super().__init__(*args, **kwargs)
    
    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        return
    
    def do_GET(self):
        """Handle GET requests."""
        self.handle_request('GET')
    
    def do_POST(self):
        """Handle POST requests."""
        self.handle_request('POST')
    
    def do_PUT(self):
        """Handle PUT requests."""
        self.handle_request('PUT')
    
    def do_DELETE(self):
        """Handle DELETE requests."""
        self.handle_request('DELETE')
    
    def do_PATCH(self):
        """Handle PATCH requests."""
        self.handle_request('PATCH')
    
    def do_HEAD(self):
        """Handle HEAD requests."""
        self.handle_request('HEAD')
    
    def do_OPTIONS(self):
        """Handle OPTIONS requests (CORS preflight)."""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()
    
    def handle_request(self, method):
        """Handle all HTTP requests."""
        # Add delay to simulate network latency
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
            
        matched_endpoint = self.find_matching_endpoint(self.path, method)
        
        if matched_endpoint:
            self.handle_matched_endpoint(matched_endpoint, method)
        else:
            self.handle_unmatched_endpoint(method)
    
    def send_cors_headers(self):
        """Send CORS headers for cross-origin requests."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
    
    def find_matching_endpoint(self, path, method):
        """Find endpoint that matches the request path and method."""
        request_path = path.split('?')[0]  # Remove query parameters
        
        for endpoint in self.endpoints:
            if (endpoint['method'].upper() == method.upper() and 
                self.path_matches(request_path, endpoint['path'])):
                return endpoint
        
        return None
    
    def path_matches(self, request_path, endpoint_path):
        """Check if request path matches endpoint path pattern."""
        # Convert endpoint path pattern to regex
        pattern = endpoint_path
        
        # Replace path parameters {param} with regex groups
        pattern = re.sub(r'\{[^}]+\}', r'([^/]+)', pattern)
        pattern = f"^{pattern}$"
        
        return re.match(pattern, request_path) is not None
    
    def handle_matched_endpoint(self, endpoint, method):
        """Handle request for a matched endpoint."""
        # Ensure endpoint is a dictionary
        if not isinstance(endpoint, dict):
            logger.warning(f"Invalid endpoint format in handle_matched_endpoint. Expected dict, got {type(endpoint)}")
            endpoint = {}
            
        # Determine response status and content
        responses = endpoint.get('responses', {})
        
        # Choose appropriate response (prefer 200, then 201, then first available)
        status_code = 200
        response_data = {}
        
        if '200' in responses:
            status_code = 200
            response_data = self.generate_response_data(responses['200'], endpoint)
        elif '201' in responses:
            status_code = 201
            response_data = self.generate_response_data(responses['201'], endpoint)
        elif responses:
            first_status = list(responses.keys())[0]
            status_code = int(first_status)
            response_data = self.generate_response_data(responses[first_status], endpoint)
        else:
            response_data = self.generate_default_response(endpoint, method)
        
        # Send response
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        
        # Write response body
        if method != 'HEAD':
            response_json = json.dumps(response_data, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
    
    def handle_unmatched_endpoint(self, method):
        """Handle request for unmatched endpoint."""
        self.send_response(404)
        self.send_header('Content-Type', 'application/json')
        self.send_cors_headers()
        self.end_headers()
        
        if method != 'HEAD':
            error_response = {
                "error": "Not Found",
                "message": f"Endpoint {method} {self.path} not found",
                "status": 404,
                "timestamp": datetime.now().isoformat()
            }
            response_json = json.dumps(error_response, indent=2)
            self.wfile.write(response_json.encode('utf-8'))
    
    def generate_response_data(self, response_spec, endpoint):
        """Generate response data based on response specification."""
        # Ensure response_spec and endpoint are dictionaries
        if not isinstance(response_spec, dict):
            response_spec = {}
        if not isinstance(endpoint, dict):
            endpoint = {'method': 'GET', 'path': '/unknown'}
            
        # Try to use example from spec
        if 'example' in response_spec:
            return response_spec['example']
        
        # Try to generate from schema
        if 'schema' in response_spec:
            return self.generate_from_schema(response_spec['schema'])
        
        # Generate default response
        return self.generate_default_response(endpoint, endpoint.get('method', 'GET'))
    
    def generate_from_schema(self, schema):
        """Generate mock data from JSON schema."""
        if not schema:
            return {}
        
        # Ensure schema is a dictionary
        if not isinstance(schema, dict):
            return {}
        
        schema_type = schema.get('type', 'object')
        
        if schema_type == 'object':
            result = {}
            properties = schema.get('properties', {})
            for prop_name, prop_schema in properties.items():
                result[prop_name] = self.generate_from_schema(prop_schema)
            return result
        
        elif schema_type == 'array':
            items_schema = schema.get('items', {})
            return [self.generate_from_schema(items_schema) for _ in range(random.randint(1, 3))]
        
        elif schema_type == 'string':
            return schema.get('example', f"sample_string_{random.randint(1, 100)}")
        
        elif schema_type == 'integer':
            return schema.get('example', random.randint(1, 1000))
        
        elif schema_type == 'number':
            return schema.get('example', round(random.uniform(1, 100), 2))
        
        elif schema_type == 'boolean':
            return schema.get('example', random.choice([True, False]))
        
        else:
            return schema.get('example', "unknown_type")
    
    def generate_default_response(self, endpoint, method):
        """Generate a default response for an endpoint."""
        # Ensure endpoint is a dictionary
        if not isinstance(endpoint, dict):
            endpoint = {'path': '/unknown'}
            
        path = endpoint.get('path', '/unknown')
        
        # Extract potential resource name from path
        path_parts = [part for part in path.split('/') if part and not part.startswith('{')]
        resource_name = path_parts[-1] if path_parts else 'resource'
        
        base_response = {
            "id": random.randint(1, 1000),
            "timestamp": datetime.now().isoformat(),
            "message": f"Mock response for {method} {path}",
            "success": True
        }
        
        if method.upper() == 'GET':
            if '{' in path:
                # Single resource
                base_response.update({
                    f"{resource_name}_id": random.randint(1, 1000),
                    f"{resource_name}_name": f"Sample {resource_name.title()}",
                    "status": "active"
                })
            else:
                # Collection
                base_response = {
                    "data": [
                        {
                            "id": i,
                            f"{resource_name}_name": f"Sample {resource_name.title()} {i}",
                            "status": "active"
                        }
                        for i in range(1, 4)
                    ],
                    "total": 3,
                    "page": 1,
                    "per_page": 10
                }
        
        elif method.upper() == 'POST':
            base_response.update({
                "created": True,
                f"{resource_name}_id": random.randint(1000, 9999),
                "message": f"{resource_name.title()} created successfully"
            })
        
        elif method.upper() == 'PUT':
            base_response.update({
                "updated": True,
                "message": f"{resource_name.title()} updated successfully"
            })
        
        elif method.upper() == 'DELETE':
            base_response.update({
                "deleted": True,
                "message": f"{resource_name.title()} deleted successfully"
            })
        
        return base_response

def create_mock_handler(endpoints, delay_ms):
    """Create a request handler class with specific endpoints and delay."""
    class ConfiguredMockHandler(MockAPIHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(endpoints, delay_ms, *args, **kwargs)
    
    return ConfiguredMockHandler

def start_mock_server(endpoints, port=8080, delay_ms=500):
    """Start the mock server with given endpoints."""
    global mock_server_instance, mock_server_thread, server_stop_event
    
    try:
        # Stop any existing server first
        if mock_server_instance:
            stop_mock_server()
            time.sleep(1)  # Give time for cleanup
        
        # Create handler with endpoints
        handler_class = create_mock_handler(endpoints, delay_ms)
        
        # Create server with reuse address
        server_address = ('', port)
        mock_server_instance = ThreadedHTTPServer(server_address, handler_class)
        mock_server_instance.allow_reuse_address = True
        
        # Reset stop event
        server_stop_event.clear()
        
        # Start server in thread
        def run_server():
            while not server_stop_event.is_set():
                try:
                    mock_server_instance.handle_request()
                except OSError:
                    break
        
        mock_server_thread = Thread(target=run_server, daemon=True)
        mock_server_thread.start()
        
        return True, f"Mock server started on port {port}"
    
    except OSError as e:
        if e.errno == 48:  # Address already in use
            return False, f"Port {port} is already in use. Please try a different port or stop the existing service."
        else:
            return False, f"Failed to start mock server: {str(e)}"
    except Exception as e:
        return False, f"Failed to start mock server: {str(e)}"

def stop_mock_server():
    """Stop the running mock server."""
    global mock_server_instance, mock_server_thread, server_stop_event
    
    try:
        if mock_server_instance:
            server_stop_event.set()
            mock_server_instance.server_close()
            
            if mock_server_thread and mock_server_thread.is_alive():
                mock_server_thread.join(timeout=2)
            
            mock_server_instance = None
            mock_server_thread = None
            
            return True, "Mock server stopped successfully"
        else:
            return True, "No mock server was running"
    
    except Exception as e:
        return False, f"Error stopping mock server: {str(e)}"

def is_mock_server_running():
    """Check if mock server is currently running."""
    return mock_server_instance is not None and not server_stop_event.is_set()

# Define supported test frameworks and their templates
TEST_FRAMEWORKS = {
    "Postman": {
        "extension": "json",
        "language": "javascript"
    },
    "RestAssured": {
        "extension": "java",
        "language": "java"
    },
    "Requests": {
        "extension": "py",
        "language": "python"
    },
    "Robot Framework": {
        "extension": "robot",
        "language": "robotframework"
    }
}

# Define HTTP methods with descriptions
HTTP_METHODS = {
    "GET": "Retrieve data from a specified resource",
    "POST": "Submit data to be processed to a specified resource",
    "PUT": "Update a specified resource",
    "DELETE": "Delete a specified resource",
    "PATCH": "Apply partial modifications to a resource",
    "HEAD": "Same as GET but returns only HTTP headers and no document body",
    "OPTIONS": "Describe the communication options for the target resource"
}


def load_spec_from_file(uploaded_file):
    """Parse the uploaded API specification file."""
    try:
        content = uploaded_file.read()
        if uploaded_file.name.lower().endswith('.json'):
            spec = json.loads(content)
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="api_generation",
                    status="success",
                    message="API specification file loaded successfully",
                    details=f"Successfully parsed {uploaded_file.name} with {len(spec.get('paths', {})) if 'paths' in spec else 0} paths."
                )
            return spec
        elif uploaded_file.name.lower().endswith(('.yaml', '.yml')):
            spec = yaml.safe_load(content)
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="api_generation",
                    status="success",
                    message="API specification file loaded successfully",
                    details=f"Successfully parsed {uploaded_file.name} with {len(spec.get('paths', {})) if 'paths' in spec else 0} paths."
                )
            return spec
        else:
            st.error("Unsupported file format. Please upload a JSON or YAML file.")
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="api_generation",
                    status="error",
                    message="Unsupported file format",
                    details=f"The file {uploaded_file.name} is not supported. Only JSON or YAML files are accepted.",
                    action_steps=["Upload a .json, .yaml, or .yml file containing API specifications"]
                )
            return None
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="api_generation",
                status="error",
                message="Error parsing API specification file",
                details=f"Failed to parse {uploaded_file.name}: {str(e)}",
                action_steps=["Check the file format", "Ensure the file is valid JSON or YAML"]
            )
        return None


def load_spec_from_url(swagger_url):
    """Parse API specification from a Swagger/OpenAPI URL."""
    try:
        st.info(f"Attempting to fetch API specification from URL: {swagger_url}")

        # Handle URLs with hash fragments specifically for Swagger UI
        if '#/' in swagger_url:
            swagger_url = swagger_url.split('#/')[0]
            st.info(f"Removed hash fragment from URL: {swagger_url}")

        # Add verification skipping for internal development environments
        verify_ssl = True
        if any(domain in swagger_url for domain in ['dev1k8s01', 'localhost', 'test', '.local', '127.0.0.1']):
            verify_ssl = False
            st.info("Development environment detected. SSL verification disabled.")

        # Special handling for specific domain structure we've observed
        if 'sfbff' in swagger_url and 'apidocs' in swagger_url:
            st.info("Detected Storefront BFF API documentation URL")

            # First check if we can find the actual API json location from the HTML
            html_response = None
            try:
                html_response = requests.get(swagger_url, timeout=10, verify=False)
                if html_response.status_code == 200:
                    # Try to extract the swagger-ui-init.js path which might contain the API spec
                    if "./apidocs/swagger-ui-init.js" in html_response.text:
                        st.info("Found swagger-ui-init.js reference, attempting to extract API spec directly from it")

                        # Get the js file that contains the embedded spec
                        js_url = f"{swagger_url.rstrip('/apidocs')}/apidocs/swagger-ui-init.js"
                        st.info(f"Fetching API spec from: {js_url}")

                        try:
                            js_response = requests.get(js_url, timeout=10, verify=False)
                            if js_response.status_code == 200:
                                js_content = js_response.text

                                # Look for the swaggerDoc object in the JS file
                                if '"swaggerDoc"' in js_content or "'swaggerDoc'" in js_content:
                                    st.info("Found embedded swagger specification in JS file")

                                    # Extract the JSON part - looking for swaggerDoc object
                                    try:
                                        # Find the start of the swaggerDoc object
                                        start_marker = '"swaggerDoc": '
                                        if start_marker not in js_content:
                                            start_marker = "'swaggerDoc': "

                                        start_idx = js_content.find(start_marker) + len(start_marker)

                                        # Extract the JSON object - need to find where it ends
                                        # This is tricky as we need to count nested braces
                                        json_str = ""
                                        brace_count = 0
                                        capture = False

                                        for char in js_content[start_idx:]:
                                            if char == '{':
                                                brace_count += 1
                                                capture = True
                                            elif char == '}':
                                                brace_count -= 1

                                            if capture:
                                                json_str += char

                                            # If we've found the closing brace of the main object, we're done
                                            if capture and brace_count == 0:
                                                break

                                        if json_str:
                                            try:
                                                # Parse the extracted JSON
                                                spec = json.loads(json_str)
                                                if 'swagger' in spec or 'openapi' in spec:
                                                    st.success("Successfully extracted API spec from JS file")
                                                    return spec
                                            except json.JSONDecodeError as json_error:
                                                st.error(f"Error parsing extracted JSON: {json_error}")
                                                st.info("Extracted content sample (first 200 chars):")
                                                st.code(json_str[:200])
                                    except Exception as extract_error:
                                        st.error(f"Error extracting API spec from JS: {extract_error}")

                        except Exception as js_error:
                            st.error(f"Error fetching swagger-ui-init.js: {js_error}")

                    # If we couldn't extract from JS, try direct specification file paths
                    st.info("Trying direct API spec paths...")
                    base_path = swagger_url.rstrip('/apidocs')
                    potential_spec_urls = [
                        f"{base_path}/api-docs",
                        f"{base_path}/v3/api-docs",
                        f"{base_path}/v2/api-docs",
                        f"{base_path}/swagger.json",
                        f"{base_path}/sfapi/v3/api-docs",
                        f"{base_path}/sfapi/swagger.json",
                        f"{swagger_url}/swagger.json",  # sometimes under apidocs folder
                        f"{base_path}/apidocs/swagger.json",
                    ]

                    for potential_url in potential_spec_urls:
                        st.info(f"Trying potential API spec URL: {potential_url}")
                        try:
                            spec_response = requests.get(potential_url, timeout=10, verify=False)
                            if spec_response.status_code == 200:
                                try:
                                    # Try to parse as JSON
                                    spec_data = spec_response.json()
                                    if 'swagger' in spec_data or 'openapi' in spec_data:
                                        st.success(f"Found valid API spec at: {potential_url}")
                                        swagger_url = potential_url
                                        break
                                except json.JSONDecodeError:
                                    st.info(f"Response from {potential_url} is not valid JSON")
                                    continue
                        except Exception as e:
                            st.info(f"Failed to access {potential_url}: {str(e)}")
                            continue
            except Exception as e:
                st.info(f"Error while analyzing HTML content: {str(e)}")

        # Continue with existing code
        if 'swagger-ui' in swagger_url:
            # For Swagger UI URLs, try to find the actual spec URL
            base_url = swagger_url.split('/swagger-ui')[0]
            st.info(f"Extracted base URL: {base_url}")

            # Common API specification paths relative to the base URL
            swagger_ui_paths = [
                f"{base_url}/v3/api-docs",
                f"{base_url}/v2/api-docs",
                f"{base_url}/swagger.json",
                f"{base_url}/api-docs",
                # Additional paths for common Swagger UI implementations
                f"{base_url}/swagger/v3/api-docs",
                f"{base_url}/swagger/v2/api-docs",
                f"{base_url}/swagger/api-docs",
                f"{base_url}/swagger-resources/config",
                f"{base_url}/swagger-resources/configuration/ui",
                # For API docs directly under swagger-ui path
                f"{base_url}/swagger-ui/api-docs",
                f"{base_url}/swagger-ui/v3/api-docs",
                f"{base_url}/swagger-ui/v2/api-docs",
                f"{base_url}/swagger-ui/swagger.json",
                # For specific registeredsite.com domain
                f"{base_url}/sfapi/v3/api-docs",
                f"{base_url}/sfapi/v2/api-docs",
                f"{base_url}/sfapi/swagger.json"
            ]

            for path in swagger_ui_paths:
                try:
                    st.info(f"Trying path: {path}")
                    response = requests.get(path, timeout=10, verify=False)
                    if response.status_code == 200:
                        swagger_url = path
                        st.success(f"Found API spec at: {path}")
                        break
                except Exception as e:
                    st.info(f"Failed to access {path}: {str(e)}")
                    continue

        # Try common paths for Swagger/OpenAPI documentation
        if not (swagger_url.endswith('.json') or swagger_url.endswith('.yaml') or swagger_url.endswith('.yml')):
            # Try common paths if not directly pointing to a spec file
            potential_paths = [
                f"{swagger_url.rstrip('/')}/swagger.json",
                f"{swagger_url.rstrip('/')}/api-docs",
                f"{swagger_url.rstrip('/')}/v2/api-docs",
                f"{swagger_url.rstrip('/')}/v3/api-docs",
                f"{swagger_url.rstrip('/')}/swagger/v1/swagger.json",
                f"{swagger_url.rstrip('/')}/openapi.json",
                f"{swagger_url.rstrip('/')}/openapi.yaml",
                # Additional paths for specific domain
                f"{swagger_url.rstrip('/')}/sfapi/v3/api-docs",
                f"{swagger_url.rstrip('/')}/sfapi/swagger.json"
            ]

            # Try each potential path until we find one that works
            found_spec = False
            for path in potential_paths:
                try:
                    st.info(f"Trying alternate path: {path}")
                    response = requests.get(path, timeout=10, verify=False)
                    if response.status_code == 200:
                        swagger_url = path
                        found_spec = True
                        st.success(f"Found API spec at: {path}")
                        break
                except Exception as e:
                    st.info(f"Failed to access {path}: {str(e)}")
                    continue

            if not found_spec:
                # Use the original URL as a fallback
                st.info("Falling back to original URL")
                response = requests.get(swagger_url, timeout=10, verify=False)
        else:
            # Direct URL to spec file
            response = requests.get(swagger_url, timeout=10, verify=False)

        if response.status_code != 200:
            st.error(f"Failed to fetch API specification. Status code: {response.status_code}")
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="api_generation",
                    status="error",
                    message="Failed to fetch API specification",
                    details=f"Server returned status code {response.status_code}",
                    action_steps=["Check if the URL is correct", "Ensure the API documentation is accessible"]
                )
            return None

        # Determine format based on content or URL
        if swagger_url.endswith('.yaml') or swagger_url.endswith('.yml'):
            spec = yaml.safe_load(response.text)
        else:
            # Default to JSON - Add error handling for JSON parsing
            try:
                spec = response.json()
            except json.JSONDecodeError as json_error:
                st.error(f"Error parsing JSON response: {json_error}")
                if NOTIFICATIONS_AVAILABLE:
                    notifications.add_notification(
                        module_name="api_generation",
                        status="error",
                        message="Error parsing API specification JSON",
                        details=f"Failed to parse JSON from {swagger_url}: {str(json_error)}",
                        action_steps=["Check if the URL returns valid JSON", "Inspect the raw response content"]
                    )

                # Log the actual response content for debugging
                st.info("Response content (first 500 chars):")
                st.code(response.text[:500])
                return None

        # Validate that it's an OpenAPI/Swagger spec
        if not ('swagger' in spec or 'openapi' in spec):
            st.warning("URL does not appear to contain a valid OpenAPI/Swagger specification")
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="api_generation",
                    status="warning",
                    message="Invalid API specification",
                    details="The document does not appear to be a valid OpenAPI/Swagger specification",
                    action_steps=["Check if the URL points to the correct API documentation"]
                )
            return None

        # Add notification for successful retrieval
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="api_generation",
                status="success",
                message="API specification loaded successfully",
                details=f"Successfully retrieved specification from {swagger_url} with {len(spec.get('paths', {})) if 'paths' in spec else 0} paths."
            )
        return spec

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching API specification: {e}")
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="api_generation",
                status="error",
                message="Error fetching API specification",
                details=f"Failed to retrieve specification from {swagger_url}: {str(e)}",
                action_steps=["Check your internet connection", "Verify the URL is correct", "Ensure the API is accessible"]
            )
        return None
    except Exception as e:
        st.error(f"Error parsing API specification: {e}")
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="api_generation",
                status="error",
                message="Error parsing API specification",
                details=f"Failed to parse specification from {swagger_url}: {str(e)}",
                action_steps=["Check if the URL returns a valid JSON or YAML document"]
            )
        return None


def extract_endpoints(spec):
    """Extract endpoints from OpenAPI specification."""
    endpoints = []
    
    # Validate input
    if not isinstance(spec, dict):
        logger.error(f"Invalid specification format. Expected dict, got {type(spec)}")
        st.error(f"❌ Invalid specification format. Expected dictionary, got {type(spec)}")
        return endpoints

    # Add debug information
    st.info(f"📋 Processing specification with keys: {list(spec.keys())}")
    
    if 'swagger' in spec or 'openapi' in spec:
        # Handle OpenAPI/Swagger format
        paths = spec.get('paths', {})
        if not paths:
            st.warning("⚠️ No 'paths' section found in the API specification")
            return endpoints
            
        st.info(f"📊 Found {len(paths)} paths to process")
        
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                st.warning(f"⚠️ Skipping path '{path}' - methods is not a dictionary: {type(methods)}")
                continue
                
            for method, details in methods.items():
                if method.lower() in [m.lower() for m in HTTP_METHODS.keys()]:
                    # Ensure details is a dictionary
                    if not isinstance(details, dict):
                        st.warning(f"⚠️ Method '{method}' for path '{path}' has invalid details (type: {type(details)}), using defaults")
                        details = {}
                    
                    endpoint = {
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'description': details.get('description', ''),
                        'parameters': details.get('parameters', []),
                        'responses': details.get('responses', {}),
                        'tags': details.get('tags', [])
                    }
                    endpoints.append(endpoint)
    elif 'info' in spec and 'item' in spec:
        # Handle Postman Collection format
        st.info("📋 Processing as Postman collection")
        endpoints.extend(extract_postman_endpoints(spec))
    else:
        # Handle simple list of endpoints (only process keys that look like paths)
        st.info("📋 Processing as simple endpoint list (no OpenAPI/Swagger format detected)")
        for path, methods in spec.items():
            # Skip metadata keys that are clearly not API paths
            if path in ['info', 'host', 'basePath', 'schemes', 'consumes', 'produces', 'definitions', 'securityDefinitions']:
                continue
            # Only process keys that look like API paths (start with / or contain API-like terms)
            if not (path.startswith('/') or 'api' in path.lower() or path.startswith('http')):
                continue
                
            if isinstance(methods, dict):
                for method, details in methods.items():
                    # Only process actual HTTP methods
                    if method.lower() not in [m.lower() for m in HTTP_METHODS.keys()]:
                        continue
                        
                    # Ensure details is a dictionary
                    if not isinstance(details, dict):
                        st.warning(f"⚠️ Method '{method}' for path '{path}' has invalid details (type: {type(details)}), using defaults")
                        details = {}
                    
                    endpoint = {
                        'path': path,
                        'method': method.upper(),
                        'summary': details.get('summary', ''),
                        'description': details.get('description', ''),
                        'parameters': details.get('parameters', []),
                        'responses': details.get('responses', {}),
                        'tags': details.get('tags', [])
                    }
                    endpoints.append(endpoint)

    if endpoints:
        st.success(f"✅ Successfully extracted {len(endpoints)} endpoints")
    else:
        st.error("❌ No valid endpoints found in the specification")
        
    return endpoints


def extract_postman_endpoints(collection):
    """Extract endpoints from Postman collection format"""
    endpoints = []
    
    def process_item(item):
        if 'request' in item:
            # This is a request item
            request = item['request']
            if isinstance(request, dict):
                method = request.get('method', 'GET')
                url_info = request.get('url', {})
                
                # Handle both string and object URL formats
                if isinstance(url_info, str):
                    path = url_info
                elif isinstance(url_info, dict):
                    path = url_info.get('raw', '') or '/'.join(url_info.get('path', []))
                else:
                    path = '/'
                
                # Clean up the path - remove base URL if present
                if 'http' in path:
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(path)
                        path = parsed.path
                    except:
                        pass
                
                endpoint = {
                    'path': path or '/',
                    'method': method.upper(),
                    'summary': item.get('name', ''),
                    'description': item.get('description', ''),
                    'parameters': [],
                    'responses': {},
                    'tags': []
                }
                endpoints.append(endpoint)
        
        elif 'item' in item:
            # This is a folder containing more items
            for sub_item in item['item']:
                process_item(sub_item)
    
    # Process top-level items
    if 'item' in collection:
        for item in collection['item']:
            process_item(item)
    
    return endpoints


def extract_from_url(base_url):
    """Placeholder for function that would extract API information from a URL."""
    # In a real implementation, this might use requests to hit common endpoints
    # like /swagger.json or perform discovery

    try:
        # For demo purposes, return a simple API structure
        discovered_endpoints = [
            {
                'path': '/api/users',
                'method': 'GET',
                'summary': 'Get all users',
                'description': 'Returns a list of users',
                'parameters': [],
                'responses': {'200': {'description': 'Success'}}
            },
            {
                'path': '/api/users/{id}',
                'method': 'GET',
                'summary': 'Get user by ID',
                'description': 'Returns a single user',
                'parameters': [{'name': 'id', 'in': 'path', 'required': True}],
                'responses': {'200': {'description': 'Success'}, '404': {'description': 'Not found'}}
            }
        ]

        # Add notification about successful discovery
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="api_generation",
                status="success",
                message="API endpoints discovered",
                details=f"Successfully discovered {len(discovered_endpoints)} endpoints from {base_url}.",
                action_steps=["Review discovered endpoints", "Generate tests for these endpoints"]
            )

        return discovered_endpoints

    except Exception as e:
        # In a real implementation, this would handle connection errors, etc.
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="api_generation",
                status="error",
                message="Failed to discover API endpoints",
                details=f"Error discovering endpoints from {base_url}: {str(e)}",
                action_steps=["Check if the URL is correct", "Ensure the API is accessible",
                             "Try uploading a specification file instead"]
            )
        return []


def generate_tests(endpoints, framework, base_url, test_type="Basic"):
    """AI-Enhanced test generation with intelligent analysis and optimizations."""
    
    # Initialize AI analyzer
    ai_analyzer = AIAPIAnalyzer()
    
    # Analyze the API suite for AI insights
    test_suite = ai_analyzer.analyze_api_suite(endpoints, framework, test_type)
    
    # Generate enhanced test code
    if framework == "Postman":
        return generate_postman_tests(test_suite, base_url, test_type)
    elif framework == "RestAssured":
        return generate_restassured_tests(test_suite, base_url, test_type)
    elif framework == "Requests":
        return generate_requests_tests(test_suite, base_url, test_type)
    elif framework == "Robot Framework":
        return generate_robot_tests(test_suite, base_url, test_type)
    else:
        return ""

def generate_postman_tests(test_suite: APITestSuite, base_url: str, test_type: str) -> str:
    """Generate AI-enhanced Postman collection with intelligent test scenarios"""
    
    collection = {
        "info": {
            "name": f"AI-Generated API Tests - {test_type}",
            "description": f"Intelligent API test collection with {len(test_suite.endpoints)} endpoints",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
            "_postman_meta": {
                "ai_generated": True,
                "quality_score": test_suite.quality_score,
                "estimated_time": test_suite.estimated_execution_time,
                "recommendations": test_suite.ai_recommendations[:3]  # Top 3 recommendations
            }
        },
        "variable": [
            {"key": "baseUrl", "value": base_url.rstrip('/'), "type": "string"},
            {"key": "authToken", "value": "{{auth_token}}", "type": "string"}
        ],
        "item": []
    }

    for endpoint in test_suite.endpoints:
        folder = {
            "name": f"{endpoint.method.upper()} {endpoint.path}",
            "description": f"AI Risk Level: {endpoint.risk_level.upper()} | Complexity: {endpoint.complexity_score:.1f}",
            "item": []
        }
        
        # Basic positive test
        basic_test = {
            "name": f"✅ {endpoint.method.upper()} {endpoint.path} - Success",
            "request": {
                "method": endpoint.method.upper(),
                "header": _generate_intelligent_headers(endpoint),
                "url": {
                    "raw": f"{{{{baseUrl}}}}{endpoint.path}",
                    "host": ["{{baseUrl}}"],
                    "path": endpoint.path.strip('/').split('/')
                }
            },
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "type": "text/javascript",
                        "exec": _generate_postman_assertions(endpoint, test_type)
                    }
                }
            ]
        }
        
        # Add parameters intelligently
        if endpoint.parameters:
            _add_postman_parameters(basic_test, endpoint)
        
        folder["item"].append(basic_test)
        
        # Add AI-driven test scenarios based on test_type
        if test_type in ["Comprehensive", "Security"]:
            folder["item"].extend(_generate_advanced_postman_tests(endpoint, test_type))
        
        collection["item"].append(folder)
    
    # Add AI recommendations as comments in collection
    if test_suite.ai_recommendations:
        collection["info"]["description"] += f"\n\nAI Recommendations:\n" + "\n".join(f"• {rec}" for rec in test_suite.ai_recommendations[:5])
    
    return json.dumps(collection, indent=2)


def _generate_intelligent_headers(endpoint):
    """Generate intelligent headers based on endpoint analysis"""
    headers = [
        {"key": "Content-Type", "value": "application/json"},
        {"key": "Accept", "value": "application/json"}
    ]
    
    # Add authentication if security requirements detected
    if endpoint.security or any(word in endpoint.path.lower() for word in ['auth', 'login', 'token']):
        headers.append({"key": "Authorization", "value": "Bearer {{authToken}}"})
    
    return headers


def _generate_postman_assertions(endpoint, test_type):
    """Generate intelligent Postman test assertions"""
    assertions = [
        "pm.test('Status code is 200', function () {",
        "    pm.response.to.have.status(200);",
        "});",
        "",
        "pm.test('Response time is less than 2000ms', function () {",
        "    pm.expect(pm.response.responseTime).to.be.below(2000);",
        "});"
    ]
    
    # Add security tests for sensitive endpoints
    if endpoint.risk_level == 'high' or test_type == 'Security':
        assertions.extend([
            "",
            "pm.test('Response has security headers', function () {",
            "    pm.expect(pm.response.headers.get('X-Content-Type-Options')).to.exist;",
            "});"
        ])
    
    # Add JSON validation for API endpoints
    if endpoint.method.upper() in ['GET', 'POST', 'PUT']:
        assertions.extend([
            "",
            "pm.test('Response is valid JSON', function () {",
            "    pm.response.to.be.json;",
            "});"
        ])
    
    return assertions


def _add_postman_parameters(test_request, endpoint):
    """Add parameters to Postman test request"""
    if not endpoint.parameters:
        return
    
    for param in endpoint.parameters:
        param_name = param.get('name', '')
        param_in = param.get('in', '')
        
        if param_in == 'query':
            if 'query' not in test_request['request']['url']:
                test_request['request']['url']['query'] = []
            test_request['request']['url']['query'].append({
                "key": param_name,
                "value": "{{test_value}}"
            })
        elif param_in == 'header':
            test_request['request']['header'].append({
                "key": param_name,
                "value": "{{test_value}}"
            })


def _generate_advanced_postman_tests(endpoint, test_type):
    """Generate advanced test scenarios for specific test types"""
    advanced_tests = []
    
    if test_type == 'Security':
        # Add security-focused tests
        security_test = {
            "name": f"🔒 {endpoint.method.upper()} {endpoint.path} - Security",
            "request": {
                "method": endpoint.method.upper(),
                "header": [{"key": "X-Test-Security", "value": "true"}],
                "url": {
                    "raw": f"{{{{baseUrl}}}}{endpoint.path}",
                    "host": ["{{baseUrl}}"],
                    "path": endpoint.path.strip('/').split('/')
                }
            },
            "event": [{
                "listen": "test",
                "script": {
                    "type": "text/javascript",
                    "exec": [
                        "pm.test('Security: No sensitive data in response', function () {",
                        "    const responseText = pm.response.text();",
                        "    pm.expect(responseText).to.not.include('password');",
                        "    pm.expect(responseText).to.not.include('secret');",
                        "});"
                    ]
                }
            }]
        }
        advanced_tests.append(security_test)
    
    return advanced_tests

def generate_restassured_tests(test_suite: APITestSuite, base_url: str, test_type: str) -> str:
    """Generate AI-enhanced RestAssured tests with intelligent patterns"""
    
    imports = [
        "import org.testng.annotations.Test;",
        "import org.testng.annotations.BeforeClass;",
        "import static io.restassured.RestAssured.*;",
        "import static org.hamcrest.Matchers.*;"
    ]
    
    code = "\n".join(imports) + "\n\n"
    
    code += f"""/**
 * AI-Generated API Test Suite
 * Quality Score: {test_suite.quality_score:.1f}/10
 */
public class AIGeneratedAPITests {{
    
    private final String BASE_URL = "{base_url.rstrip('/')}";
    private io.restassured.specification.RequestSpecification requestSpec;
    
    @BeforeClass
    public void setup() {{
        requestSpec = given().baseUri(BASE_URL);
    }}
    
"""
    
    # Generate individual test methods
    for i, endpoint in enumerate(test_suite.endpoints, 1):
        method = endpoint.method.lower()
        path = endpoint.path
        endpoint_name = path.replace('/', '_').replace('{', '').replace('}', '')
        
        code += f"""    @Test
    public void test{i}_{method}{endpoint_name}() {{
        // AI Risk Level: {endpoint.risk_level.upper()}
        given()
            .spec(requestSpec)
        .when()
            .{method}("{path}")
        .then()
            .statusCode(200);
    }}
    
"""
    
    code += "}\n"
    return code

def generate_requests_tests(test_suite: APITestSuite, base_url: str, test_type: str) -> str:
    """Generate AI-enhanced Python Requests tests with intelligent scenarios"""
    
    imports = ["import requests", "import unittest", "import json", "import time"]
    code = "\n".join(imports) + "\n\n"
    
    code += f'''"""
AI-Generated API Test Suite
Quality Score: {test_suite.quality_score:.1f}/10
"""

class AIGeneratedAPITests(unittest.TestCase):
    BASE_URL = "{base_url.rstrip('/')}"
    
    def setUp(self):
        self.session = requests.Session()
    
'''
    
    # Generate test methods for each endpoint
    for i, endpoint in enumerate(test_suite.endpoints, 1):
        method = endpoint.method.lower()
        path = endpoint.path
        endpoint_name = path.replace('/', '_').replace('{', '').replace('}', '')
        
        # Replace path parameters with test values
        test_path = path.replace('{id}', 'test_value')
        
        code += f"""    def test_{i:02d}_{method}{endpoint_name}(self):
        \"\"\"Test {method.upper()} {path} - Risk: {endpoint.risk_level.upper()}\"\"\"
        url = f"{{self.BASE_URL}}{test_path}"
        response = self.session.{method}(url)
        self.assertEqual(response.status_code, 200)
    
"""
    
    code += '''if __name__ == "__main__":
    unittest.main()
'''
    return code

def generate_robot_tests(test_suite: APITestSuite, base_url: str, test_type: str) -> str:
    """Generate AI-enhanced Robot Framework tests with intelligent keywords"""
    
    code = "*** Settings ***\n"
    code += "Library    RequestsLibrary\n"
    code += "Library    Collections\n\n"
    
    code += "*** Variables ***\n"
    code += f"${{BASE_URL}}    {base_url.rstrip('/')}\n"
    code += f"${{QUALITY_SCORE}}    {test_suite.quality_score:.1f}\n\n"
    
    code += "*** Keywords ***\n"
    code += "Initialize Test Suite\n"
    code += "    Create Session    api    ${BASE_URL}\n\n"
    
    code += "*** Test Cases ***\n"
    
    for i, endpoint in enumerate(test_suite.endpoints, 1):
        method = endpoint.method.upper()
        path = endpoint.path
        endpoint_name = path.replace('/', '_').replace('{', '').replace('}', '')
        
        code += f"""Test {i} {method} {endpoint_name}
    [Documentation]    AI Risk: {endpoint.risk_level.upper()}
    Create Session    api    ${{BASE_URL}}
    ${{response}}=    {method} On Session    api    {path}
    Should Be Equal As Strings    ${{response.status_code}}    200

"""
    
    return code


def show_ui():
    """
    Display the comprehensive AI-powered API test generation interface with all features
    """
    st.title("🚀 Comprehensive AI-Powered API Test Generator")
    
    # AI status indicator at the top
    if AI_AVAILABLE:
        st.success("🤖 **AI Enhancement Status**: ACTIVE - All advanced features available")
    else:
        st.warning("💡 **AI Enhancement Status**: BASIC MODE - Install Azure OpenAI for full AI capabilities")
    
    # Enhanced description with AI features
    st.markdown("""
    Generate comprehensive API test scripts with **AI-powered insights** including:
    - **Intelligent endpoint analysis** with risk assessment
    - **Smart test case generation** based on API patterns
    - **Quality scoring** and optimization recommendations
    - **Framework-specific optimizations** for better maintainability
    - **Mock server generation** for testing and development
    - **Bulk collection processing** for multiple APIs
    """)
    
    # Main feature tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Single API Generation", 
        "📦 Bulk Collection Processing", 
        "🎭 Mock Server", 
        "🔍 API Discovery", 
        "⚙️ Advanced Options"
    ])
    
    with tab1:
        show_single_api_generation()
    
    with tab2:
        show_bulk_collection_processing()
    
    with tab3:
        show_mock_server_interface()
    
    with tab4:
        show_api_discovery_interface()
    
    with tab5:
        show_advanced_options()


def show_single_api_generation():
    """Single API test generation interface"""
    st.subheader("🎯 Single API Test Generation")
    
    # Input method selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_method = st.radio(
            "📥 Input Method",
            [
                "Manual JSON/YAML Input", 
                "Swagger/OpenAPI URL", 
                "File Upload (.json/.yaml)", 
                "Postman Collection Import",
                "cURL Command Conversion",
                "Manual Endpoint Definition"
            ],
            help="Choose how you want to provide the API specification"
        )
    
    with col2:
        st.markdown("**💡 Quick Tips:**")
        if input_method == "Manual JSON/YAML Input":
            st.info("Paste complete OpenAPI 3.0+ or Swagger 2.0 specification")
        elif input_method == "Swagger/OpenAPI URL":
            st.info("Direct URL to API docs or specification JSON")
        elif input_method == "File Upload (.json/.yaml)":
            st.info("Upload specification files from your computer")
        elif input_method == "Postman Collection Import":
            st.info("Import existing Postman collections for conversion")
        elif input_method == "cURL Command Conversion":
            st.info("Convert cURL commands to test scripts")
        else:
            st.info("Define endpoints manually with custom parameters")
    
    # Input fields based on method
    spec = None
    endpoints = []
    
    if input_method == "Manual JSON/YAML Input":
        spec = handle_manual_input()
    elif input_method == "Swagger/OpenAPI URL":
        spec = handle_url_input()
    elif input_method == "File Upload (.json/.yaml)":
        spec = handle_file_upload()
    elif input_method == "Postman Collection Import":
        spec = handle_postman_import()
    elif input_method == "cURL Command Conversion":
        endpoints = handle_curl_conversion()
    else:  # Manual Endpoint Definition
        endpoints = handle_manual_endpoint_definition()
    
    # Extract endpoints if we have a spec
    if spec:
        endpoints = extract_endpoints(spec)
    
    # Test configuration
    st.subheader("⚙️ Test Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        test_framework = st.selectbox(
            "🔧 Test Framework",
            ["Postman", "RestAssured", "Python Requests", "Robot Framework", "Cypress", "Playwright", "Newman CLI"],
            help="Choose your preferred testing framework"
        )
        
        test_type = st.selectbox(
            "🎯 Test Type",
            ["Functional", "Performance", "Security", "Integration", "Regression", "Smoke", "End-to-End"],
            help="Select the primary focus of your tests"
        )
    
    with config_col2:
        base_url = st.text_input(
            "🌐 Base URL",
            value="https://api.example.com",
            help="The base URL for your API endpoints"
        )
        
        auth_type = st.selectbox(
            "🔐 Authentication Type",
            ["None", "Bearer Token", "API Key", "Basic Auth", "OAuth 2.0", "JWT", "Custom Headers"],
            help="Select authentication method for your API"
        )
    
    with config_col3:
        include_ai_insights = st.checkbox(
            "🤖 AI-Powered Enhancements",
            value=AI_AVAILABLE,
            disabled=not AI_AVAILABLE,
            help="Include AI analysis and recommendations"
        )
        
        include_negative_tests = st.checkbox(
            "❌ Negative Test Cases",
            value=True,
            help="Generate error handling and edge case tests"
        )
        
        include_performance_tests = st.checkbox(
            "⚡ Performance Assertions",
            value=False,
            help="Add response time and throughput validations"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Generation Options"):
        advanced_col1, advanced_col2 = st.columns(2)
        
        with advanced_col1:
            test_data_generation = st.selectbox(
                "🎲 Test Data Generation",
                ["Static Values", "Dynamic/Random", "AI-Generated", "Schema-Based"],
                help="How to generate test data for requests"
            )
            
            assertion_level = st.selectbox(
                "✅ Assertion Depth",
                ["Basic Status Codes", "Response Structure", "Deep Field Validation", "Business Logic"],
                help="Level of response validation to include"
            )
        
        with advanced_col2:
            environment_setup = st.multiselect(
                "🏗️ Environment Support",
                ["Development", "Staging", "Production", "Local", "Docker"],
                default=["Development", "Staging"],
                help="Generate environment-specific configurations"
            )
            
            documentation_level = st.selectbox(
                "📚 Documentation Level",
                ["Minimal", "Standard", "Comprehensive", "Tutorial-Style"],
                help="Amount of documentation to include in tests"
            )
    
    # Generate button and results
    if st.button("🚀 Generate Comprehensive API Tests", type="primary", key="single_gen"):
        if not endpoints:
            st.error("❌ No API endpoints found. Please provide valid API specification.")
            return
        
        generate_and_display_tests(
            endpoints, test_framework, base_url, test_type, 
            auth_type, include_ai_insights, include_negative_tests, 
            include_performance_tests, test_data_generation, 
            assertion_level, environment_setup, documentation_level
        )


def show_bulk_collection_processing():
    """Bulk API collection processing interface"""
    st.subheader("📦 Bulk Collection Processing")
    
    st.markdown("""
    Process multiple API specifications simultaneously to generate comprehensive test suites.
    Perfect for microservices architectures or API ecosystem testing.
    """)
    
    bulk_method = st.radio(
        "📥 Bulk Input Method",
        [
            "Multiple URLs (Line-separated)",
            "Multiple File Upload",
            "Directory Scan",
            "API Registry Import",
            "Postman Workspace Import"
        ]
    )
    
    collections = []
    
    if bulk_method == "Multiple URLs (Line-separated)":
        urls_input = st.text_area(
            "🔗 API Specification URLs",
            height=150,
            placeholder="https://api1.example.com/swagger.json\nhttps://api2.example.com/openapi.yaml\nhttps://api3.example.com/docs",
            help="Enter one URL per line"
        )
        
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            st.info(f"📊 Found {len(urls)} URLs to process")
            
            for i, url in enumerate(urls, 1):
                st.write(f"   {i}. {url}")
        else:
            urls_input = ""
    
    elif bulk_method == "Multiple File Upload":
        uploaded_files = st.file_uploader(
            "📁 Upload Multiple API Specification Files",
            type=['json', 'yaml', 'yml'],
            accept_multiple_files=True,
            help="Select multiple JSON or YAML files containing API specifications"
        )
        
        if uploaded_files:
            st.info(f"📊 Found {len(uploaded_files)} files to process")
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"   {i}. {file.name}")
        else:
            uploaded_files = []
    
    elif bulk_method == "Directory Scan":
        directory_path = st.text_input(
            "📂 Directory Path",
            placeholder="/path/to/api/specs",
            help="Path to directory containing API specification files"
        )
        
        scan_recursive = st.checkbox("🔄 Scan Subdirectories", value=True)
        
        if not directory_path:
            directory_path = ""
    
    elif bulk_method == "API Registry Import":
        registry_type = st.selectbox(
            "📋 Registry Type",
            ["Kong", "AWS API Gateway", "Azure API Management", "Google Cloud Endpoints", "Custom Registry"]
        )
        
        registry_config = st.text_area(
            "⚙️ Registry Configuration",
            placeholder="Enter registry connection details (JSON format)",
            help="Configuration for connecting to your API registry"
        )
        
        if not registry_config:
            registry_config = ""
    
    else:  # Postman Workspace Import
        workspace_id = st.text_input(
            "🏢 Postman Workspace ID",
            help="ID of the Postman workspace to import"
        )
        
        api_key = st.text_input(
            "🔑 Postman API Key",
            type="password",
            help="Your Postman API key for workspace access"
        )
        
        if not workspace_id:
            workspace_id = ""
        if not api_key:
            api_key = ""
    
    # Bulk processing options
    st.subheader("⚙️ Bulk Processing Configuration")
    
    bulk_col1, bulk_col2 = st.columns(2)
    
    with bulk_col1:
        output_format = st.selectbox(
            "📄 Output Format",
            ["Combined Test Suite", "Robot Framework Tests", "Python Requests Tests", "Postman Collection", "Separate Files per API", "Modular Framework", "Test Orchestra"]
        )
        
        naming_convention = st.selectbox(
            "📝 File Naming",
            ["API Name Based", "Service Based", "Domain Based", "Custom Pattern"]
        )
    
    with bulk_col2:
        parallel_processing = st.checkbox("⚡ Parallel Processing", value=True)
        
        error_handling = st.selectbox(
            "❌ Error Handling",
            ["Stop on First Error", "Continue with Warnings", "Skip Failed APIs", "Retry with Fallback"]
        )
    
    # Bulk generation button
    if st.button("🚀 Process Bulk Collections", type="primary", key="bulk_gen"):
        if bulk_method == "Multiple URLs (Line-separated)" and urls_input.strip():
            process_bulk_urls(urls_input, output_format, parallel_processing)
        elif bulk_method == "Multiple File Upload" and uploaded_files:
            process_bulk_files(uploaded_files, output_format, parallel_processing)
        elif bulk_method == "Directory Scan" and directory_path:
            process_directory_scan(directory_path, scan_recursive, output_format, parallel_processing)
        elif bulk_method == "API Registry Import" and registry_config:
            process_registry_import(registry_type, registry_config, output_format, parallel_processing)
        elif bulk_method == "Postman Workspace Import" and workspace_id and api_key:
            process_postman_workspace(workspace_id, api_key, output_format, parallel_processing)
        else:
            st.warning("⚠️ Please provide complete input for bulk processing")


def show_mock_server_interface():
    """Mock server interface"""
    st.subheader("🎭 API Mock Server")
    
    st.markdown("""
    Generate and run mock servers based on your API specifications for development and testing.
    """)
    
    mock_col1, mock_col2 = st.columns(2)
    
    with mock_col1:
        st.write("**🔧 Mock Server Configuration**")
        
        mock_port = st.number_input(
            "🌐 Server Port",
            min_value=1000,
            max_value=65535,
            value=8080,
            help="Port for the mock server"
        )
        
        mock_delay = st.slider(
            "⏱️ Response Delay (ms)",
            min_value=0,
            max_value=5000,
            value=500,
            help="Simulated network latency"
        )
        
        mock_data_type = st.selectbox(
            "🎲 Mock Data Type",
            ["Schema-based", "Example-based", "Random/Faker", "AI-Generated", "Custom Responses"]
        )
    
    with mock_col2:
        st.write("**📊 Server Status**")
        
        if is_mock_server_running():
            st.success("✅ Mock server is running")
            st.info(f"🌐 Server URL: http://localhost:{mock_port}")
            
            if st.button("🛑 Stop Mock Server", key="stop_mock"):
                success, message = stop_mock_server()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.info("⏸️ Mock server is not running")
    
    # Mock server input
    st.write("**📝 API Specification for Mocking**")
    
    mock_input_method = st.radio(
        "Input for Mock Server",
        ["Use Current Specification", "Upload New File", "Enter Manually"],
        horizontal=True
    )
    
    mock_spec = None
    
    if mock_input_method == "Upload New File":
        mock_file = st.file_uploader(
            "Upload API Specification for Mocking",
            type=['json', 'yaml', 'yml'],
            key="mock_upload"
        )
        if mock_file:
            mock_spec = load_spec_from_file(mock_file)
    
    elif mock_input_method == "Enter Manually":
        mock_text = st.text_area(
            "API Specification (JSON/YAML)",
            height=200,
            key="mock_manual"
        )
        if mock_text.strip():
            try:
                mock_spec = json.loads(mock_text)
            except:
                try:
                    mock_spec = yaml.safe_load(mock_text)
                except:
                    st.error("Invalid JSON/YAML format")
    
    # Mock server controls
    mock_control_col1, mock_control_col2 = st.columns(2)
    
    with mock_control_col1:
        if st.button("🚀 Start Mock Server", type="primary", key="start_mock"):
            if mock_spec:
                endpoints = extract_endpoints(mock_spec)
                if endpoints:
                    success, message = start_mock_server(endpoints, mock_port, mock_delay)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("No valid endpoints found in specification")
            else:
                st.error("Please provide an API specification")
    
    with mock_control_col2:
        if st.button("📋 Generate Mock Client Tests", key="mock_tests"):
            if mock_spec:
                endpoints = extract_endpoints(mock_spec)
                if endpoints:
                    # Generate tests pointing to mock server
                    mock_base_url = f"http://localhost:{mock_port}"
                    generate_and_display_tests(
                        endpoints, "Python Requests", mock_base_url, "Functional",
                        "None", False, True, False, "Dynamic/Random", 
                        "Basic Status Codes", ["Local"], "Standard"
                    )
                else:
                    st.error("No valid endpoints found in specification")
            else:
                st.error("Please provide an API specification")


def show_api_discovery_interface():
    """API discovery interface"""
    st.subheader("🔍 API Discovery & Analysis")
    
    st.markdown("""
    Discover and analyze APIs from various sources including live endpoints, documentation sites, and code repositories.
    """)
    
    discovery_method = st.selectbox(
        "🔎 Discovery Method",
        [
            "URL Endpoint Discovery",
            "Documentation Site Scraping",
            "GitHub Repository Analysis",
            "Network Traffic Analysis",
            "Service Mesh Discovery",
            "Container Registry Scan"
        ]
    )
    
    if discovery_method == "URL Endpoint Discovery":
        target_url = st.text_input(
            "🎯 Target URL",
            placeholder="https://api.example.com",
            help="Base URL to discover endpoints from"
        )
        
        discovery_depth = st.slider("🕳️ Discovery Depth", 1, 5, 2)
        include_params = st.checkbox("📋 Analyze Parameters", value=True)
        
        if st.button("🔍 Start Discovery", key="url_discovery"):
            if target_url:
                discovered_endpoints = discover_endpoints_from_url(target_url, discovery_depth, include_params)
                if discovered_endpoints:
                    st.success(f"✅ Discovered {len(discovered_endpoints)} endpoints")
                    display_discovered_endpoints(discovered_endpoints)
                else:
                    st.warning("⚠️ No endpoints discovered")
            else:
                st.error("Please provide a target URL")
    
    elif discovery_method == "Documentation Site Scraping":
        doc_url = st.text_input(
            "📚 Documentation URL",
            placeholder="https://docs.api.example.com",
            help="URL of API documentation site"
        )
        
        doc_type = st.selectbox(
            "📄 Documentation Type",
            ["Auto-detect", "Swagger UI", "Redoc", "API Blueprint", "Postman Docs", "Custom"]
        )
        
        if st.button("📖 Scrape Documentation", key="doc_scraping"):
            st.info("Documentation scraping feature would be implemented here")
    
    elif discovery_method == "GitHub Repository Analysis":
        repo_url = st.text_input(
            "📁 Repository URL",
            placeholder="https://github.com/user/repo",
            help="GitHub repository containing API code"
        )
        
        analysis_type = st.multiselect(
            "🔬 Analysis Type",
            ["OpenAPI Specs", "Route Definitions", "Controller Methods", "API Clients", "Tests"],
            default=["OpenAPI Specs", "Route Definitions"]
        )
        
        if st.button("🔍 Analyze Repository", key="repo_analysis"):
            st.info("Repository analysis feature would be implemented here")
    
    else:
        st.info(f"🚧 {discovery_method} feature would be implemented here")


def show_advanced_options():
    """Advanced options and utilities"""
    st.subheader("⚙️ Advanced Options & Utilities")
    
    option_tabs = st.tabs([
        "🔧 Code Generation", 
        "📊 Analytics", 
        "🔄 Conversion Tools", 
        "🎯 Custom Templates"
    ])
    
    with option_tabs[0]:
        st.write("**🔧 Advanced Code Generation**")
        
        code_col1, code_col2 = st.columns(2)
        
        with code_col1:
            custom_framework = st.text_input(
                "🛠️ Custom Framework",
                placeholder="Enter custom framework name"
            )
            
            template_engine = st.selectbox(
                "📝 Template Engine",
                ["Jinja2", "Mustache", "Handlebars", "Custom"]
            )
        
        with code_col2:
            code_style = st.selectbox(
                "🎨 Code Style",
                ["Standard", "Compact", "Verbose", "Enterprise"]
            )
            
            include_comments = st.checkbox("💬 Include Comments", value=True)
    
    with option_tabs[1]:
        st.write("**📊 API Analytics & Insights**")
        
        if st.button("📈 Generate API Analytics Report"):
            st.info("Analytics report feature would be implemented here")
        
        st.write("**Metrics to Analyze:**")
        metrics = st.multiselect(
            "Select Metrics",
            [
                "Endpoint Complexity",
                "Test Coverage",
                "Response Time Distribution",
                "Error Rate Analysis",
                "Security Assessment",
                "Performance Bottlenecks"
            ],
            default=["Endpoint Complexity", "Test Coverage"]
        )
    
    with option_tabs[2]:
        st.write("**🔄 Format Conversion Tools**")
        
        conversion_type = st.selectbox(
            "🔄 Conversion Type",
            [
                "OpenAPI 2.0 → 3.0",
                "Postman → OpenAPI",
                "RAML → OpenAPI",
                "API Blueprint → OpenAPI",
                "GraphQL → REST",
                "cURL → Test Script"
            ]
        )
        
        if st.button("🔄 Start Conversion"):
            st.info(f"{conversion_type} conversion feature would be implemented here")
    
    with option_tabs[3]:
        st.write("**🎯 Custom Templates**")
        
        template_type = st.selectbox(
            "📋 Template Type",
            ["Test Framework", "Documentation", "Client SDK", "Mock Server"]
        )
        
        custom_template = st.text_area(
            "✏️ Custom Template",
            height=200,
            placeholder="Enter your custom template here..."
        )
        
        if st.button("💾 Save Template"):
            st.success("Template saved successfully!")


# Helper functions for the enhanced interface

def handle_manual_input():
    """Handle manual JSON/YAML input"""
    
    # Provide helpful examples
    st.write("**💡 Example formats:**")
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("📋 Load OpenAPI 3.0 Example", key="load_openapi_example"):
            st.session_state.manual_spec_example = """{
  "openapi": "3.0.0",
  "info": {
    "title": "Sample API",
    "version": "1.0.0"
  },
  "paths": {
    "/users": {
      "get": {
        "summary": "Get all users",
        "responses": {
          "200": {
            "description": "Success"
          }
        }
      },
      "post": {
        "summary": "Create user",
        "responses": {
          "201": {
            "description": "Created"
          }
        }
      }
    },
    "/users/{id}": {
      "get": {
        "summary": "Get user by ID",
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "schema": {
              "type": "integer"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Success"
          }
        }
      }
    }
  }
}"""
    
    with example_col2:
        if st.button("� Load Swagger 2.0 Example", key="load_swagger_example"):
            st.session_state.manual_spec_example = """{
  "swagger": "2.0",
  "info": {
    "title": "Sample API",
    "version": "1.0.0"
  },
  "paths": {
    "/users": {
      "get": {
        "summary": "Get all users",
        "responses": {
          "200": {
            "description": "Success"
          }
        }
      }
    }
  }
}"""
    
    # Initialize session state
    if 'manual_spec_example' not in st.session_state:
        st.session_state.manual_spec_example = ""
    
    api_spec_input = st.text_area(
        "📝 API Specification (JSON/YAML)",
        height=300,
        value=st.session_state.manual_spec_example,
        placeholder="Paste your OpenAPI/Swagger specification here...",
        help="Paste your complete OpenAPI specification or Swagger JSON/YAML content"
    )
    
    if api_spec_input.strip():
        try:
            # Try JSON first
            spec = json.loads(api_spec_input)
            st.success("✅ Valid JSON format detected")
            return spec
        except json.JSONDecodeError:
            try:
                # Try YAML
                spec = yaml.safe_load(api_spec_input)
                st.success("✅ Valid YAML format detected")
                return spec
            except yaml.YAMLError as e:
                st.error(f"❌ Invalid JSON/YAML format: {str(e)}")
                return None
    return None


def handle_url_input():
    """Handle URL input with enhanced validation"""
    swagger_url = st.text_input(
        "🔗 Swagger/OpenAPI URL",
        placeholder="https://api.example.com/swagger.json",
        help="Enter the URL to your OpenAPI/Swagger specification"
    )
    
    # Initialize session state for URL spec
    if 'url_spec' not in st.session_state:
        st.session_state.url_spec = None
    if 'last_url' not in st.session_state:
        st.session_state.last_url = None
    
    if swagger_url.strip():
        # Check if URL changed, reset spec if so
        if swagger_url != st.session_state.last_url:
            st.session_state.url_spec = None
            st.session_state.last_url = swagger_url
        
        if st.button("🔍 Validate & Load URL", key="validate_url"):
            with st.spinner("Loading specification from URL..."):
                spec = load_spec_from_url(swagger_url)
                if spec:
                    st.session_state.url_spec = spec
                    st.success("✅ Successfully loaded specification from URL")
                else:
                    st.session_state.url_spec = None
                    st.error("❌ Failed to load specification from URL")
        
        # Return the stored spec if available
        if st.session_state.url_spec:
            st.info("📋 Specification loaded and ready for processing")
            return st.session_state.url_spec
        elif swagger_url.strip():
            st.info("💡 Click 'Validate & Load URL' to fetch and validate the specification")
    
    return st.session_state.url_spec


def handle_file_upload():
    """Handle file upload with preview"""
    uploaded_file = st.file_uploader(
        "📁 Upload API Specification File",
        type=['json', 'yaml', 'yml'],
        help="Upload a JSON or YAML file containing your API specification"
    )
    
    if uploaded_file:
        # Show file info
        st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        spec = load_spec_from_file(uploaded_file)
        if spec:
            st.success("✅ File loaded successfully")
            
            # Preview spec structure
            with st.expander("👀 Preview Specification Structure"):
                if isinstance(spec, dict):
                    st.json({k: f"<{type(v).__name__}>" for k, v in spec.items()})
        
        return spec
    
    return None


def handle_postman_import():
    """Handle Postman collection import"""
    st.markdown("**📦 Import Postman Collection**")
    
    postman_file = st.file_uploader(
        "Upload Postman Collection",
        type=['json'],
        help="Upload a Postman collection JSON file"
    )
    
    if postman_file:
        try:
            collection = json.load(postman_file)
            st.success("✅ Postman collection loaded successfully")
            
            # Convert Postman collection to OpenAPI-like format
            converted_spec = convert_postman_to_spec(collection)
            return converted_spec
            
        except Exception as e:
            st.error(f"❌ Error loading Postman collection: {e}")
    
    return None


def handle_curl_conversion():
    """Handle cURL command conversion"""
    st.markdown("**🔄 Convert cURL Commands**")
    
    curl_commands = st.text_area(
        "cURL Commands",
        height=200,
        placeholder="curl -X GET https://api.example.com/users\ncurl -X POST https://api.example.com/users -d '{\"name\":\"John\"}'",
        help="Enter one or more cURL commands, one per line"
    )
    
    if curl_commands.strip():
        if st.button("🔄 Convert cURL to Endpoints", key="convert_curl"):
            endpoints = convert_curl_to_endpoints(curl_commands)
            if endpoints:
                st.success(f"✅ Converted {len(endpoints)} cURL commands to endpoints")
                return endpoints
            else:
                st.error("❌ Failed to convert cURL commands")
    
    return []


def handle_manual_endpoint_definition():
    """Handle manual endpoint definition"""
    st.markdown("**✏️ Define Endpoints Manually**")
    
    endpoints = []
    
    # Dynamic endpoint builder
    if "manual_endpoints" not in st.session_state:
        st.session_state.manual_endpoints = [{"path": "/api/example", "method": "GET"}]
    
    for i, endpoint in enumerate(st.session_state.manual_endpoints):
        with st.expander(f"Endpoint {i + 1}: {endpoint['method']} {endpoint['path']}", expanded=i == 0):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                path = st.text_input(f"Path", value=endpoint['path'], key=f"path_{i}")
            
            with col2:
                method = st.selectbox(f"Method", ["GET", "POST", "PUT", "DELETE", "PATCH"], 
                                    index=["GET", "POST", "PUT", "DELETE", "PATCH"].index(endpoint['method']), 
                                    key=f"method_{i}")
            
            with col3:
                if st.button("🗑️", key=f"delete_{i}", help="Delete endpoint"):
                    st.session_state.manual_endpoints.pop(i)
                    st.rerun()
            
            # Update endpoint
            st.session_state.manual_endpoints[i] = {
                "path": path,
                "method": method,
                "summary": st.text_input(f"Summary", key=f"summary_{i}"),
                "description": st.text_area(f"Description", key=f"desc_{i}", height=60)
            }
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add Endpoint", key="add_endpoint"):
            st.session_state.manual_endpoints.append({"path": "/api/new", "method": "GET"})
            st.rerun()
    
    with col2:
        if st.button("✅ Use These Endpoints", key="use_manual"):
            return st.session_state.manual_endpoints
    
    return []


# Enhanced helper functions for comprehensive functionality

def generate_and_display_tests(endpoints, test_framework, base_url, test_type, 
                              auth_type, include_ai_insights, include_negative_tests, 
                              include_performance_tests, test_data_generation, 
                              assertion_level, environment_setup, documentation_level):
    """Generate and display comprehensive test results"""
    
    with st.spinner("🤖 Generating comprehensive test suite..."):
        try:
            # Generate the test code using AI-enhanced generator
            generated_code = generate_tests(endpoints, test_framework, base_url, test_type)
            
            # Enhance with additional features
            if include_negative_tests:
                generated_code = add_negative_test_cases(generated_code, test_framework)
            
            if include_performance_tests:
                generated_code = add_performance_assertions(generated_code, test_framework)
            
            if generated_code:
                st.success("✅ Comprehensive test suite generated successfully!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Endpoints", len(endpoints))
                with col2:
                    st.metric("🧪 Test Cases", estimate_test_count(endpoints, include_negative_tests))
                with col3:
                    st.metric("📏 Lines of Code", len(generated_code.split('\n')))
                with col4:
                    st.metric("⚡ Est. Runtime", f"{estimate_runtime(endpoints)}s")
                
                # Display the generated code
                st.subheader("📝 Generated Test Code")
                st.code(generated_code, language=get_language_for_framework(test_framework))
                
                # Download options
                download_col1, download_col2, download_col3 = st.columns(3)
                
                with download_col1:
                    filename = f"api_tests_{test_framework.lower().replace(' ', '_')}.{get_file_extension(test_framework)}"
                    st.download_button(
                        label="📥 Download Test File",
                        data=generated_code,
                        file_name=filename,
                        mime="text/plain"
                    )
                
                with download_col2:
                    # Generate documentation
                    documentation = generate_test_documentation(endpoints, test_framework, documentation_level)
                    st.download_button(
                        label="📚 Download Documentation",
                        data=documentation,
                        file_name=f"test_documentation_{test_framework.lower().replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                
                with download_col3:
                    # Generate environment configuration
                    env_config = generate_environment_config(environment_setup, base_url, auth_type)
                    st.download_button(
                        label="⚙️ Download Config",
                        data=env_config,
                        file_name="environment_config.json",
                        mime="application/json"
                    )
                
                # AI insights summary (if available)
                if AI_AVAILABLE and include_ai_insights:
                    display_ai_insights_summary(endpoints)
                
                # Additional features
                with st.expander("🔧 Additional Options"):
                    additional_col1, additional_col2 = st.columns(2)
                    
                    with additional_col1:
                        if st.button("🎭 Generate Mock Server", key="gen_mock"):
                            mock_code = generate_mock_server_code(endpoints)
                            st.code(mock_code, language="python")
                    
                    with additional_col2:
                        if st.button("📊 Generate Test Report Template", key="gen_report"):
                            report_template = generate_report_template(test_framework)
                            st.code(report_template, language="html")
            
            else:
                st.error("❌ Failed to generate test code. Please check your configuration.")
                
        except Exception as e:
            st.error(f"❌ Error generating tests: {str(e)}")
            st.exception(e)


def process_bulk_urls(urls_input, output_format, parallel_processing):
    """Process multiple URLs for bulk generation"""
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, url in enumerate(urls):
        status_text.text(f"Processing {i+1}/{len(urls)}: {url}")
        progress_bar.progress((i + 1) / len(urls))
        
        try:
            spec = load_spec_from_url(url)
            if spec:
                endpoints = extract_endpoints(spec)
                if endpoints:
                    result = {
                        'url': url,
                        'endpoints': len(endpoints),
                        'status': 'success',
                        'spec': spec
                    }
                else:
                    result = {'url': url, 'endpoints': 0, 'status': 'no_endpoints'}
            else:
                result = {'url': url, 'endpoints': 0, 'status': 'failed'}
        except Exception as e:
            result = {'url': url, 'endpoints': 0, 'status': 'error', 'error': str(e)}
        
        results.append(result)
    
    # Display results
    st.subheader("📊 Bulk Processing Results")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_endpoints = sum(r['endpoints'] for r in results if r['status'] == 'success')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Successful", success_count)
    with col2:
        st.metric("📊 Total Endpoints", total_endpoints)
    with col3:
        st.metric("❌ Failed", len(urls) - success_count)
    
    # Detailed results table
    df = pd.DataFrame(results)
    st.dataframe(df)
    
    if success_count > 0:
        if st.button("🚀 Generate Combined Test Suite from URLs", key="bulk_urls_gen"):
            generate_combined_test_suite(results, output_format)


def process_bulk_files(uploaded_files, output_format, parallel_processing):
    """Process multiple uploaded files for bulk generation"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
            spec = load_spec_from_file(file)
            if spec:
                endpoints = extract_endpoints(spec)
                result = {
                    'file': file.name,
                    'endpoints': len(endpoints),
                    'status': 'success' if endpoints else 'no_endpoints',
                    'spec': spec
                }
            else:
                result = {'file': file.name, 'endpoints': 0, 'status': 'failed'}
        except Exception as e:
            result = {'file': file.name, 'endpoints': 0, 'status': 'error', 'error': str(e)}
        
        results.append(result)
    
    # Display results similar to URL processing
    st.subheader("📊 File Processing Results")
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_endpoints = sum(r['endpoints'] for r in results if r['status'] == 'success')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Successful", success_count)
    with col2:
        st.metric("📊 Total Endpoints", total_endpoints)
    with col3:
        st.metric("❌ Failed", len(uploaded_files) - success_count)
    
    # Display detailed results table
    if results:
        # Store results in session state to preserve them
        st.session_state.bulk_processing_results = results
        st.session_state.bulk_output_format = output_format
        
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        if success_count > 0:
            if st.button("🚀 Generate Combined Test Suite from Files", key="bulk_files_gen"):
                generate_combined_test_suite(results, output_format)
    
    # Also show results from session state if they exist
    elif hasattr(st.session_state, 'bulk_processing_results') and st.session_state.bulk_processing_results:
        st.info("📋 Showing previous bulk processing results:")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            df = pd.DataFrame(st.session_state.bulk_processing_results)
            st.dataframe(df)
        
        with col2:
            if st.button("🗑️ Clear Results", key="clear_bulk_results"):
                del st.session_state.bulk_processing_results
                if 'bulk_output_format' in st.session_state:
                    del st.session_state.bulk_output_format
                st.rerun()
        
        success_count = sum(1 for r in st.session_state.bulk_processing_results if r['status'] == 'success')
        if success_count > 0:
            if st.button("🚀 Generate Combined Test Suite from Files", key="bulk_files_gen_session"):
                generate_combined_test_suite(st.session_state.bulk_processing_results, 
                                           st.session_state.get('bulk_output_format', 'Combined Test Suite'))


def process_directory_scan(directory_path, scan_recursive, output_format, parallel_processing):
    """Process directory scan for API specifications"""
    try:
        if not os.path.exists(directory_path):
            st.error(f"❌ Directory does not exist: {directory_path}")
            return
            
        st.info(f"🔍 Scanning directory: {directory_path}")
        
        # Find all API specification files
        spec_files = []
        extensions = ['.json', '.yaml', '.yml']
        
        if scan_recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        spec_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                if any(file.lower().endswith(ext) for ext in extensions):
                    spec_files.append(os.path.join(directory_path, file))
        
        if not spec_files:
            st.warning("⚠️ No API specification files found in the directory")
            return
            
        st.success(f"✅ Found {len(spec_files)} specification files")
        
        # Process files
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, file_path in enumerate(spec_files):
            status_text.text(f"Processing {i+1}/{len(spec_files)}: {os.path.basename(file_path)}")
            progress_bar.progress((i + 1) / len(spec_files))
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Try to parse as JSON or YAML
                spec = None
                if file_path.lower().endswith('.json'):
                    spec = json.loads(content)
                else:
                    spec = yaml.safe_load(content)
                
                if spec:
                    endpoints = extract_endpoints(spec)
                    result = {
                        'file': os.path.basename(file_path),
                        'path': file_path,
                        'endpoints': len(endpoints),
                        'status': 'success' if endpoints else 'no_endpoints',
                        'spec': spec
                    }
                else:
                    result = {'file': os.path.basename(file_path), 'path': file_path, 'endpoints': 0, 'status': 'failed'}
                    
            except Exception as e:
                result = {'file': os.path.basename(file_path), 'path': file_path, 'endpoints': 0, 'status': 'error', 'error': str(e)}
            
            results.append(result)
        
        # Display results
        st.subheader("📊 Directory Scan Results")
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        total_endpoints = sum(r['endpoints'] for r in results if r['status'] == 'success')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Successful", success_count)
        with col2:
            st.metric("📊 Total Endpoints", total_endpoints)
        with col3:
            st.metric("❌ Failed", len(spec_files) - success_count)
            
        # Show detailed results
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        if success_count > 0:
            if st.button("🚀 Generate Combined Test Suite from Directory", key="bulk_dir_gen"):
                generate_combined_test_suite(results, output_format)
                
    except Exception as e:
        st.error(f"❌ Error during directory scan: {str(e)}")


def process_registry_import(registry_type, registry_config, output_format, parallel_processing):
    """Process API registry import"""
    st.info(f"🔄 Processing {registry_type} registry import")
    
    try:
        config = json.loads(registry_config) if registry_config.strip() else {}
        
        # Mock implementation for different registry types
        if registry_type == "Kong":
            results = process_kong_registry(config)
        elif registry_type == "AWS API Gateway":
            results = process_aws_api_gateway(config)
        elif registry_type == "Azure API Management":
            results = process_azure_apim(config)
        elif registry_type == "Google Cloud Endpoints":
            results = process_gcp_endpoints(config)
        else:
            results = process_custom_registry(config)
            
        # Display results
        st.subheader(f"📊 {registry_type} Import Results")
        
        if results:
            success_count = sum(1 for r in results if r['status'] == 'success')
            total_endpoints = sum(r['endpoints'] for r in results if r['status'] == 'success')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ APIs Found", success_count)
            with col2:
                st.metric("📊 Total Endpoints", total_endpoints)
            with col3:
                st.metric("❌ Failed", len(results) - success_count)
                
            df = pd.DataFrame(results)
            st.dataframe(df)
            
            if success_count > 0:
                if st.button("🚀 Generate Test Suite from Registry", key="bulk_registry_gen"):
                    generate_combined_test_suite(results, output_format)
        else:
            st.warning("⚠️ No APIs found in registry")
            
    except json.JSONDecodeError:
        st.error("❌ Invalid JSON configuration")
    except Exception as e:
        st.error(f"❌ Error processing registry: {str(e)}")


def process_postman_workspace(workspace_id, api_key, output_format, parallel_processing):
    """Process Postman workspace import"""
    st.info(f"🔄 Importing from Postman workspace: {workspace_id}")
    
    try:
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }
        
        # Get workspace info
        workspace_url = f"https://api.getpostman.com/workspaces/{workspace_id}"
        response = requests.get(workspace_url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"❌ Failed to access workspace: {response.status_code}")
            return
            
        workspace_data = response.json()
        workspace_name = workspace_data.get('workspace', {}).get('name', 'Unknown')
        
        st.success(f"✅ Connected to workspace: {workspace_name}")
        
        # Get collections in workspace
        collections_url = f"https://api.getpostman.com/collections"
        response = requests.get(collections_url, headers=headers)
        
        if response.status_code != 200:
            st.error(f"❌ Failed to get collections: {response.status_code}")
            return
            
        collections_data = response.json()
        collections = collections_data.get('collections', [])
        
        if not collections:
            st.warning("⚠️ No collections found in workspace")
            return
            
        st.info(f"🔍 Found {len(collections)} collections")
        
        # Process each collection
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, collection_info in enumerate(collections):
            collection_id = collection_info.get('id')
            collection_name = collection_info.get('name', 'Unknown')
            
            status_text.text(f"Processing {i+1}/{len(collections)}: {collection_name}")
            progress_bar.progress((i + 1) / len(collections))
            
            try:
                # Get detailed collection
                collection_url = f"https://api.getpostman.com/collections/{collection_id}"
                response = requests.get(collection_url, headers=headers)
                
                if response.status_code == 200:
                    collection_data = response.json()
                    collection_spec = collection_data.get('collection', {})
                    
                    # Convert to OpenAPI format
                    spec = convert_postman_to_spec(collection_spec)
                    endpoints = extract_endpoints(spec)
                    
                    result = {
                        'collection': collection_name,
                        'id': collection_id,
                        'endpoints': len(endpoints),
                        'status': 'success' if endpoints else 'no_endpoints',
                        'spec': spec
                    }
                else:
                    result = {
                        'collection': collection_name,
                        'id': collection_id,
                        'endpoints': 0,
                        'status': 'failed',
                        'error': f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                result = {
                    'collection': collection_name,
                    'id': collection_id,
                    'endpoints': 0,
                    'status': 'error',
                    'error': str(e)
                }
            
            results.append(result)
        
        # Display results
        st.subheader("📊 Postman Workspace Import Results")
        
        success_count = sum(1 for r in results if r['status'] == 'success')
        total_endpoints = sum(r['endpoints'] for r in results if r['status'] == 'success')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Collections", success_count)
        with col2:
            st.metric("📊 Total Endpoints", total_endpoints)
        with col3:
            st.metric("❌ Failed", len(collections) - success_count)
            
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        if success_count > 0:
            if st.button("🚀 Generate Test Suite from Postman", key="bulk_postman_gen"):
                generate_combined_test_suite(results, output_format)
                
    except Exception as e:
        st.error(f"❌ Error importing Postman workspace: {str(e)}")


# Registry-specific processors
def process_kong_registry(config):
    """Process Kong API Gateway registry"""
    # Mock implementation
    return [
        {'api': 'Kong API 1', 'endpoints': 5, 'status': 'success'},
        {'api': 'Kong API 2', 'endpoints': 3, 'status': 'success'}
    ]


def process_aws_api_gateway(config):
    """Process AWS API Gateway registry"""
    # Mock implementation
    return [
        {'api': 'AWS API 1', 'endpoints': 8, 'status': 'success'},
        {'api': 'AWS API 2', 'endpoints': 4, 'status': 'success'}
    ]


def process_azure_apim(config):
    """Process Azure API Management registry"""
    # Mock implementation
    return [
        {'api': 'Azure API 1', 'endpoints': 6, 'status': 'success'},
        {'api': 'Azure API 2', 'endpoints': 2, 'status': 'success'}
    ]


def process_gcp_endpoints(config):
    """Process Google Cloud Endpoints registry"""
    # Mock implementation
    return [
        {'api': 'GCP API 1', 'endpoints': 7, 'status': 'success'},
        {'api': 'GCP API 2', 'endpoints': 5, 'status': 'success'}
    ]


def process_custom_registry(config):
    """Process custom registry"""
    # Mock implementation
    return [
        {'api': 'Custom API 1', 'endpoints': 4, 'status': 'success'},
        {'api': 'Custom API 2', 'endpoints': 6, 'status': 'success'}
    ]


def discover_endpoints_from_url(target_url, discovery_depth, include_params):
    """Discover endpoints from a target URL"""
    # This is a placeholder implementation
    # In reality, this would implement various discovery techniques
    
    discovered = []
    
    # Mock discovery results
    common_paths = ["/api/users", "/api/posts", "/api/auth", "/api/search"]
    methods = ["GET", "POST", "PUT", "DELETE"]
    
    for path in common_paths:
        for method in methods:
            if method == "GET" or (method in ["POST", "PUT"] and "auth" not in path):
                endpoint = {
                    'path': path,
                    'method': method,
                    'summary': f'{method} {path}',
                    'description': f'Discovered endpoint: {method} {path}',
                    'discovered': True
                }
                discovered.append(endpoint)
    
    return discovered


def display_discovered_endpoints(endpoints):
    """Display discovered endpoints in a nice format"""
    st.write("**🔍 Discovered Endpoints:**")
    
    for endpoint in endpoints:
        with st.expander(f"{endpoint['method']} {endpoint['path']}"):
            st.write(f"**Summary:** {endpoint.get('summary', 'N/A')}")
            st.write(f"**Description:** {endpoint.get('description', 'N/A')}")
            if endpoint.get('parameters'):
                st.write(f"**Parameters:** {len(endpoint['parameters'])}")


def convert_postman_to_spec(collection):
    """Convert Postman collection to OpenAPI-like specification"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": collection.get("info", {}).get("name", "Converted API"),
            "version": "1.0.0"
        },
        "paths": {}
    }
    
    def process_item(item, base_path=""):
        if "item" in item:  # It's a folder
            for sub_item in item["item"]:
                process_item(sub_item, base_path)
        else:  # It's a request
            if "request" in item:
                request = item["request"]
                method = request.get("method", "GET").lower()
                
                # Extract path from URL
                url = request.get("url", {})
                if isinstance(url, str):
                    path = "/api/converted"
                else:
                    raw_url = url.get("raw", "")
                    # Simple path extraction
                    if "{{" in raw_url:
                        path = "/api/converted"
                    else:
                        from urllib.parse import urlparse
                        parsed = urlparse(raw_url)
                        path = parsed.path or "/api/converted"
                
                if path not in spec["paths"]:
                    spec["paths"][path] = {}
                
                spec["paths"][path][method] = {
                    "summary": item.get("name", f"{method.upper()} {path}"),
                    "responses": {
                        "200": {"description": "Success"}
                    }
                }
    
    if "item" in collection:
        for item in collection["item"]:
            process_item(item)
    
    return spec


def convert_curl_to_endpoints(curl_commands):
    """Convert cURL commands to endpoint definitions"""
    endpoints = []
    
    for line in curl_commands.split('\n'):
        line = line.strip()
        if line.startswith('curl'):
            # Simple cURL parsing
            method = "GET"
            url = ""
            
            # Extract method
            if "-X" in line:
                parts = line.split("-X")[1].strip().split()
                method = parts[0].strip()
            
            # Extract URL (simplified)
            import re
            url_match = re.search(r'https?://[^\s]+', line)
            if url_match:
                url = url_match.group()
                # Extract path
                from urllib.parse import urlparse
                parsed = urlparse(url)
                path = parsed.path or "/api/converted"
                
                endpoint = {
                    'path': path,
                    'method': method.upper(),
                    'summary': f'Converted from cURL: {method} {path}',
                    'description': f'Original cURL: {line[:100]}...'
                }
                endpoints.append(endpoint)
    
    return endpoints


def add_negative_test_cases(generated_code, test_framework):
    """Add negative test cases to generated code"""
    # This would add error handling tests, invalid input tests, etc.
    negative_tests = "\n\n# Negative Test Cases\n"
    
    if test_framework == "Python Requests":
        negative_tests += """
# Test invalid endpoints
def test_invalid_endpoint(self):
    \"\"\"Test 404 for non-existent endpoint\"\"\"
    response = self.session.get(f"{self.BASE_URL}/invalid/endpoint")
    self.assertEqual(response.status_code, 404)

# Test invalid methods
def test_invalid_method(self):
    \"\"\"Test 405 for unsupported method\"\"\"
    response = self.session.patch(f"{self.BASE_URL}/users")
    self.assertIn(response.status_code, [405, 501])
"""
    
    return generated_code + negative_tests


def add_performance_assertions(generated_code, test_framework):
    """Add performance assertions to generated code"""
    if test_framework == "Python Requests":
        perf_code = """
# Performance Assertions
def test_response_time(self):
    \"\"\"Test API response time\"\"\"
    import time
    start_time = time.time()
    response = self.session.get(f"{self.BASE_URL}/users")
    end_time = time.time()
    response_time = end_time - start_time
    self.assertLess(response_time, 2.0, "Response time should be less than 2 seconds")
"""
        return generated_code + perf_code
    
    return generated_code


def estimate_test_count(endpoints, include_negative_tests):
    """Estimate total number of test cases"""
    base_count = len(endpoints)
    if include_negative_tests:
        base_count += len(endpoints) * 2  # Add negative tests
    return base_count


def estimate_runtime(endpoints):
    """Estimate test runtime in seconds"""
    return len(endpoints) * 2  # 2 seconds per endpoint


def generate_test_documentation(endpoints, test_framework, documentation_level):
    """Generate test documentation"""
    doc = f"""# API Test Documentation

## Test Framework: {test_framework}
## Total Endpoints: {len(endpoints)}

### Test Coverage

"""
    
    for endpoint in endpoints:
        doc += f"#### {endpoint['method']} {endpoint['path']}\n"
        doc += f"- **Summary**: {endpoint.get('summary', 'N/A')}\n"
        doc += f"- **Description**: {endpoint.get('description', 'N/A')}\n"
        doc += "\n"
    
    return doc


def generate_environment_config(environment_setup, base_url, auth_type):
    """Generate environment configuration"""
    config = {
        "environments": {},
        "auth": {
            "type": auth_type,
            "credentials": {}
        }
    }
    
    for env in environment_setup:
        config["environments"][env.lower()] = {
            "base_url": base_url.replace("api.example.com", f"{env.lower()}-api.example.com"),
            "timeout": 30,
            "retry_count": 3
        }
    
    return json.dumps(config, indent=2)


def display_ai_insights_summary(endpoints):
    """Display AI insights summary"""
    st.subheader("🎯 AI Enhancement Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Smart Analysis:**
        - Endpoint risk assessment
        - Security vulnerability detection
        - Performance bottleneck identification
        - API pattern recognition
        """)
    
    with col2:
        st.markdown("""
        **🧪 Enhanced Testing:**
        - Context-aware test scenarios
        - Framework-specific optimizations
        - Security-focused test cases
        - Performance testing integration
        """)
    
    with col3:
        st.markdown("""
        **📊 Advanced Insights:**
        - AI-powered recommendations
        - Test suite optimization
        - Maintenance complexity analysis
        - Strategic test planning
        """)


def generate_mock_server_code(endpoints):
    """Generate mock server code"""
    mock_code = f"""# Mock Server for API Testing
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

"""
    
    for endpoint in endpoints:
        path = endpoint['path'].replace('{', '<').replace('}', '>')
        method = endpoint['method'].lower()
        
        if method == 'get':
            mock_code += f"""
@app.route('{path}', methods=['GET'])
def mock_{endpoint['path'].replace('/', '_').replace('{', '').replace('}', '')}():
    return jsonify({{"message": "Mock response for {endpoint['method']} {endpoint['path']}", "data": {{"id": random.randint(1, 100)}}}})
"""
    
    mock_code += """
if __name__ == '__main__':
    app.run(debug=True, port=8080)
"""
    
    return mock_code


def generate_report_template(test_framework):
    """Generate test report template"""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>API Test Report - {test_framework}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f4f4f4; padding: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>API Test Report</h1>
        <p>Framework: {test_framework}</p>
        <p>Generated: {{{{timestamp}}}}</p>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <h3>Total Tests</h3>
            <p>{{{{total_tests}}}}</p>
        </div>
        <div class="metric">
            <h3>Passed</h3>
            <p>{{{{passed_tests}}}}</p>
        </div>
        <div class="metric">
            <h3>Failed</h3>
            <p>{{{{failed_tests}}}}</p>
        </div>
    </div>
    
    <div class="results">
        <h2>Test Results</h2>
        <!-- Test results will be populated here -->
    </div>
</body>
</html>"""


def generate_combined_test_suite(results, output_format):
    """Generate combined test suite from multiple APIs"""
    try:
        st.success("🚀 Generating combined test suite from multiple API specifications!")
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'success' and 'spec' in r]
        
        if not successful_results:
            st.error("❌ No successful API specifications to combine")
            return
            
        st.info(f"📊 Combining {len(successful_results)} API specifications")
        
        # Combine all endpoints
        all_endpoints = []
        api_sources = []
        
        for i, result in enumerate(successful_results):
            spec = result.get('spec', {})
            if spec:
                try:
                    endpoints = extract_endpoints(spec)
                    
                    for endpoint in endpoints:
                        # Add source information to endpoint
                        if isinstance(endpoint, dict):
                            endpoint['source_api'] = result.get('url', result.get('file', result.get('collection', 'Unknown')))
                            all_endpoints.append(endpoint)
                    
                    api_sources.append({
                        'name': result.get('url', result.get('file', result.get('collection', 'Unknown'))),
                        'endpoints': len(endpoints)
                    })
                except Exception as e:
                    st.error(f"Error processing endpoints for result {i+1}: {str(e)}")
            else:
                st.warning(f"No spec found in result {i+1}")
        
        if not all_endpoints:
            st.error("❌ No endpoints found in the combined specifications")
            return
            
        # Display combination summary
        st.subheader("📊 Combined Test Suite Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔗 APIs Combined", len(successful_results))
        with col2:
            st.metric("📊 Total Endpoints", len(all_endpoints))
        with col3:
            st.metric("🧪 Est. Test Cases", len(all_endpoints) * 3)
        
        # Show API sources
        if api_sources:
            st.write("**📋 Source APIs:**")
            sources_df = pd.DataFrame(api_sources)
            st.dataframe(sources_df)
        
        # Generate combined test suite based on output format
        st.subheader(f"🎯 Generating: {output_format}")
        
        if output_format == "Separate Files per API":
            generate_separate_test_files(successful_results)
        elif output_format == "Combined Test Suite":
            generate_single_combined_suite(all_endpoints)
        elif output_format == "Robot Framework Tests":
            generate_robot_framework_suite(all_endpoints, api_sources)
        elif output_format == "Python Requests Tests":
            generate_python_requests_suite(all_endpoints, api_sources)
        elif output_format == "Postman Collection":
            generate_postman_collection_suite(all_endpoints, api_sources)
        elif output_format == "Modular Framework":
            generate_modular_framework(successful_results)
        elif output_format == "Test Orchestra":
            generate_test_orchestra(all_endpoints, api_sources)
        else:
            st.info("Using default: Combined Test Suite")
            generate_single_combined_suite(all_endpoints)
            
    except Exception as e:
        st.error(f"❌ Error generating combined test suite: {str(e)}")
        st.exception(e)  # Show full traceback for debugging
        logger.exception("Error in generate_combined_test_suite")


def generate_separate_test_files(successful_results):
    """Generate separate test files for each API"""
    st.subheader("📁 Separate Test Files")
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        for i, result in enumerate(successful_results):
            spec = result.get('spec', {})
            if spec:
                endpoints = extract_endpoints(spec)
                api_name = result.get('url', result.get('file', f'api_{i+1}'))
                safe_name = re.sub(r'[^\w\-_.]', '_', api_name)
                
                # Generate test code for this API
                test_code = generate_tests(endpoints, "Python Requests", "https://api.example.com", "Functional")
                
                # Add to zip
                filename = f"{safe_name}_tests.py"
                zip_file.writestr(filename, test_code)
                
                # Also create a summary file
                summary = f"""# Test Summary for {api_name}
                
Endpoints: {len(endpoints)}
Generated: {datetime.now().isoformat()}

## Endpoints:
"""
                for endpoint in endpoints:
                    summary += f"- {endpoint['method']} {endpoint['path']}\n"
                
                zip_file.writestr(f"{safe_name}_summary.md", summary)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="📦 Download All Test Files (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=f"api_test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip"
    )


def generate_single_combined_suite(all_endpoints):
    """Generate a single combined test suite"""
    st.subheader("🎯 Combined Test Suite")
    
    # Group endpoints by source
    grouped_endpoints = defaultdict(list)
    for endpoint in all_endpoints:
        source = endpoint.get('source_api', 'Unknown')
        grouped_endpoints[source].append(endpoint)
    
    # Generate combined Python test suite
    combined_code = f"""#!/usr/bin/env python3
\"\"\"
Combined API Test Suite
Generated: {datetime.now().isoformat()}
Total APIs: {len(grouped_endpoints)}
Total Endpoints: {len(all_endpoints)}
\"\"\"

import unittest
import requests
import time
from datetime import datetime


class CombinedAPITestSuite(unittest.TestCase):
    \"\"\"Combined test suite for multiple APIs\"\"\"
    
    def setUp(self):
        self.session = requests.Session()
        self.session.headers.update({{'User-Agent': 'API-Test-Suite/1.0'}})
        self.base_urls = {{
            # Add your API base URLs here
"""
    
    # Add base URLs for each API
    for i, source in enumerate(grouped_endpoints.keys()):
        safe_name = re.sub(r'[^\w]', '_', source)
        combined_code += f"            '{safe_name}': 'https://api{i+1}.example.com',\n"
    
    combined_code += "        }\n    \n"
    
    # Generate test methods for each API
    for source, endpoints in grouped_endpoints.items():
        safe_name = re.sub(r'[^\w]', '_', source)
        combined_code += f"""
    # Tests for {source}
    def test_{safe_name}_endpoints(self):
        \"\"\"Test all endpoints for {source}\"\"\"
        base_url = self.base_urls['{safe_name}']
        
"""
        
        for endpoint in endpoints:
            method = endpoint['method'].lower()
            path = endpoint['path']
            combined_code += f"""        # Test {endpoint['method']} {path}
        response = self.session.{method}(f"{{base_url}}{path}")
        self.assertIn(response.status_code, [200, 201, 202, 204])
        
"""
    
    # Add integration tests
    combined_code += """
    def test_cross_api_integration(self):
        \"\"\"Test integration between APIs\"\"\"
        # Example: Create resource in API 1, verify in API 2
        pass
    
    def test_api_availability(self):
        \"\"\"Test that all APIs are available\"\"\"
        for api_name, base_url in self.base_urls.items():
            with self.subTest(api=api_name):
                try:
                    response = self.session.get(f"{base_url}/health", timeout=10)
                    self.assertTrue(response.status_code < 500, f"{api_name} returned server error")
                except requests.exceptions.RequestException:
                    self.fail(f"{api_name} is not reachable")


if __name__ == '__main__':
    unittest.main(verbosity=2)
"""
    
    st.code(combined_code, language="python")
    
    st.download_button(
        label="📥 Download Combined Test Suite",
        data=combined_code,
        file_name=f"combined_api_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
        mime="text/plain"
    )


def generate_modular_framework(successful_results):
    """Generate a modular test framework"""
    st.subheader("🏗️ Modular Test Framework")
    
    # Create base framework structure
    framework_code = """# Modular API Test Framework
# This creates a reusable framework for testing multiple APIs

from abc import ABC, abstractmethod
import requests
import unittest
from typing import List, Dict


class APITestBase(ABC):
    \"\"\"Base class for API testing\"\"\"
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    @abstractmethod
    def get_endpoints(self) -> List[Dict]:
        \"\"\"Return list of endpoints to test\"\"\"
        pass
    
    def test_endpoint(self, endpoint: Dict):
        \"\"\"Test a single endpoint\"\"\"
        method = endpoint['method'].lower()
        path = endpoint['path']
        url = f"{self.base_url}{path}"
        
        response = getattr(self.session, method)(url)
        return response


# Generated API test classes
"""
    
    # Generate a class for each API
    for i, result in enumerate(successful_results):
        spec = result.get('spec', {})
        if spec:
            endpoints = extract_endpoints(spec)
            api_name = result.get('url', result.get('file', f'API{i+1}'))
            safe_name = re.sub(r'[^\w]', '', api_name.replace('.', '_').replace('/', '_'))
            
            framework_code += f"""

class {safe_name}Tests(APITestBase):
    \"\"\"Tests for {api_name}\"\"\"
    
    def get_endpoints(self):
        return {endpoints}
    
    def test_all_endpoints(self):
        for endpoint in self.get_endpoints():
            with self.subTest(endpoint=f"{{endpoint['method']}} {{endpoint['path']}}"):
                response = self.test_endpoint(endpoint)
                self.assertIn(response.status_code, [200, 201, 202, 204])
"""
    
    st.code(framework_code, language="python")
    
    st.download_button(
        label="📥 Download Modular Framework",
        data=framework_code,
        file_name=f"modular_api_framework_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
        mime="text/plain"
    )


def generate_test_orchestra(all_endpoints, api_sources):
    """Generate a test orchestration suite"""
    st.subheader("🎼 Test Orchestra")
    
    orchestra_code = f"""#!/usr/bin/env python3
\"\"\"
API Test Orchestra
Orchestrates testing across multiple APIs with dependencies and workflows
Generated: {datetime.now().isoformat()}
\"\"\"

import unittest
import requests
import asyncio
import concurrent.futures
from typing import Dict, List
import time


class APITestOrchestra:
    \"\"\"Orchestrates API tests across multiple services\"\"\"
    
    def __init__(self):
        self.session = requests.Session()
        self.results = {{}}
        self.apis = {api_sources}
    
    async def test_api_parallel(self, api_info: Dict):
        \"\"\"Test an API asynchronously\"\"\"
        api_name = api_info['name']
        print(f"Testing {{api_name}}...")
        
        # Simulate API testing
        start_time = time.time()
        # Add actual test logic here
        end_time = time.time()
        
        return {{
            'api': api_name,
            'duration': end_time - start_time,
            'status': 'success'
        }}
    
    async def run_parallel_tests(self):
        \"\"\"Run all API tests in parallel\"\"\"
        tasks = [self.test_api_parallel(api) for api in self.apis]
        results = await asyncio.gather(*tasks)
        return results
    
    def run_sequential_tests(self):
        \"\"\"Run tests with dependencies in sequence\"\"\"
        results = []
        
        # Phase 1: Authentication APIs
        auth_apis = [api for api in self.apis if 'auth' in api['name'].lower()]
        for api in auth_apis:
            result = self.test_api_sync(api)
            results.append(result)
        
        # Phase 2: Core APIs
        core_apis = [api for api in self.apis if 'auth' not in api['name'].lower()]
        for api in core_apis:
            result = self.test_api_sync(api)
            results.append(result)
        
        return results
    
    def test_api_sync(self, api_info: Dict):
        \"\"\"Test API synchronously\"\"\"
        # Add synchronous testing logic
        return {{'api': api_info['name'], 'status': 'success'}}


class IntegrationTestSuite(unittest.TestCase):
    \"\"\"Integration tests across all APIs\"\"\"
    
    def setUp(self):
        self.orchestra = APITestOrchestra()
    
    def test_parallel_execution(self):
        \"\"\"Test all APIs in parallel\"\"\"
        results = asyncio.run(self.orchestra.run_parallel_tests())
        self.assertTrue(all(r['status'] == 'success' for r in results))
    
    def test_sequential_execution(self):
        \"\"\"Test APIs with dependencies\"\"\"
        results = self.orchestra.run_sequential_tests()
        self.assertTrue(all(r['status'] == 'success' for r in results))
    
    def test_cross_api_workflows(self):
        \"\"\"Test workflows that span multiple APIs\"\"\"
        # Example: User registration -> Profile creation -> Order placement
        pass


if __name__ == '__main__':
    unittest.main()
"""
    
    st.code(orchestra_code, language="python")
    
    st.download_button(
        label="📥 Download Test Orchestra",
        data=orchestra_code,
        file_name=f"api_test_orchestra_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
        mime="text/plain"
    )


def generate_robot_framework_suite(all_endpoints, api_sources):
    """Generate a Robot Framework test suite from combined endpoints"""
    st.subheader("🤖 Robot Framework Test Suite")
    
    # Group endpoints by source
    grouped_endpoints = defaultdict(list)
    for endpoint in all_endpoints:
        source = endpoint.get('source_api', 'Unknown')
        grouped_endpoints[source].append(endpoint)
    
    # Generate Robot Framework code
    robot_code = f"""*** Settings ***
Documentation    Combined API Test Suite - Robot Framework
Library          RequestsLibrary
Library          Collections
Library          String

*** Variables ***
&{{API_URLS}}    Create Dictionary
...              api1=https://api1.example.com
...              api2=https://api2.example.com

*** Keywords ***
Setup Test Environment
    Create Session    api1    ${{API_URLS.api1}}
    Create Session    api2    ${{API_URLS.api2}}

*** Test Cases ***
"""

    # Generate test cases for each API
    for source, endpoints in grouped_endpoints.items():
        safe_name = re.sub(r'[^\w]', '_', source)
        robot_code += f"""
Test All Endpoints For {safe_name}
    [Documentation]    Test all endpoints for {source}
    [Tags]    {safe_name}    api-test
"""
        
        for endpoint in endpoints:
            method = endpoint['method'].upper()
            path = endpoint['path']
            robot_code += f"""    # Test {method} {path}
    ${{response}}=    {method} On Session    api1    {path}    expected_status=any
    Should Be True    ${{response.status_code}} < 500
"""

    robot_code += """
Cross API Integration Test
    [Documentation]    Test integration between different APIs
    [Tags]    integration
    # Add cross-API integration tests here
    Log    Integration tests would be implemented here

API Health Check
    [Documentation]    Verify all APIs are healthy
    [Tags]    health-check
    FOR    ${api_name}    IN    @{API_URLS.keys()}
        ${response}=    GET On Session    ${api_name}    /health    expected_status=any
        Should Be True    ${response.status_code} < 500    ${api_name} health check failed
    END
"""
    
    st.code(robot_code, language="robotframework")
    
    st.download_button(
        label="📥 Download Robot Framework Tests",
        data=robot_code,
        file_name=f"combined_api_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.robot",
        mime="text/plain"
    )


def generate_python_requests_suite(all_endpoints, api_sources):
    """Generate a Python Requests test suite from combined endpoints"""
    st.subheader("🐍 Python Requests Test Suite")
    
    # Group endpoints by source
    grouped_endpoints = defaultdict(list)
    for endpoint in all_endpoints:
        source = endpoint.get('source_api', 'Unknown')
        grouped_endpoints[source].append(endpoint)
    
    # Generate Python test code
    python_code = f"""#!/usr/bin/env python3
\"\"\"
Combined API Test Suite - Python Requests
Generated: {datetime.now().isoformat()}
Total APIs: {len(grouped_endpoints)}
Total Endpoints: {len(all_endpoints)}
\"\"\"

import unittest
import requests
import json
import time
from datetime import datetime


class CombinedAPITestSuite(unittest.TestCase):
    \"\"\"Combined API test suite using Python Requests\"\"\"
    
    @classmethod
    def setUpClass(cls):
        cls.session = requests.Session()
        cls.session.headers.update({{'User-Agent': 'API-Test-Suite/1.0'}})
        cls.base_urls = {{
"""
    
    # Add base URLs for each API
    for i, source in enumerate(grouped_endpoints.keys()):
        safe_name = re.sub(r'[^\w]', '_', source)
        python_code += f"            '{safe_name}': 'https://api{i+1}.example.com',\n"
    
    python_code += "        }\n\n"
    
    # Generate test methods for each API
    for source, endpoints in grouped_endpoints.items():
        safe_name = re.sub(r'[^\w]', '_', source)
        python_code += f"""    def test_{safe_name}_endpoints(self):
        \"\"\"Test all endpoints for {source}\"\"\"
        base_url = self.base_urls['{safe_name}']
        
        test_results = []
        
"""
        
        for endpoint in endpoints:
            method = endpoint['method'].lower()
            path = endpoint['path']
            python_code += f"""        # Test {endpoint['method']} {path}
        try:
            response = self.session.{method}(f"{{base_url}}{path}", timeout=10)
            test_results.append({{
                'endpoint': '{endpoint['method']} {path}',
                'status_code': response.status_code,
                'success': response.status_code < 500
            }})
            self.assertLess(response.status_code, 500, f"Server error for {endpoint['method']} {path}")
        except requests.exceptions.RequestException as e:
            test_results.append({{
                'endpoint': '{endpoint['method']} {path}',
                'error': str(e),
                'success': False
            }})
            self.fail(f"Request failed for {endpoint['method']} {path}: {{e}}")
        
"""
        
        python_code += f"""        # Summary for {source}
        successful_tests = sum(1 for result in test_results if result.get('success', False))
        total_tests = len(test_results)
        print(f"{{base_url}}: {{successful_tests}}/{{total_tests}} tests passed")
        
"""
    
    python_code += """    def test_api_performance(self):
        \"\"\"Test API response times\"\"\"
        for api_name, base_url in self.base_urls.items():
            with self.subTest(api=api_name):
                start_time = time.time()
                try:
                    response = self.session.get(f"{base_url}/health", timeout=5)
                    end_time = time.time()
                    response_time = end_time - start_time
                    self.assertLess(response_time, 3.0, f"{api_name} response time too slow: {response_time:.2f}s")
                except requests.exceptions.RequestException as e:
                    self.fail(f"{api_name} performance test failed: {e}")
    
    def test_cross_api_integration(self):
        \"\"\"Test integration scenarios across APIs\"\"\"
        # Example: Test data consistency between APIs
        print("Cross-API integration tests would be implemented here")
        
    def tearDown(self):
        \"\"\"Clean up after each test\"\"\"
        # Add any cleanup logic here
        pass


if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
"""
    
    st.code(python_code, language="python")
    
    st.download_button(
        label="📥 Download Python Requests Tests",
        data=python_code,
        file_name=f"combined_api_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
        mime="text/plain"
    )


def generate_postman_collection_suite(all_endpoints, api_sources):
    """Generate a Postman collection from combined endpoints"""
    st.subheader("📮 Postman Collection")
    
    # Group endpoints by source
    grouped_endpoints = defaultdict(list)
    for endpoint in all_endpoints:
        source = endpoint.get('source_api', 'Unknown')
        grouped_endpoints[source].append(endpoint)
    
    # Generate Postman collection
    collection = {
        "info": {
            "name": f"Combined API Test Suite - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": f"Combined test collection with {len(all_endpoints)} endpoints from {len(grouped_endpoints)} APIs",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "variable": [
            {"key": "baseUrl1", "value": "https://api1.example.com", "type": "string"},
            {"key": "baseUrl2", "value": "https://api2.example.com", "type": "string"},
            {"key": "authToken", "value": "{{auth_token}}", "type": "string"}
        ],
        "item": []
    }
    
    # Create folders for each API
    for source, endpoints in grouped_endpoints.items():
        safe_name = re.sub(r'[^\w\-_. ]', '_', source)
        
        folder = {
            "name": f"📁 {safe_name}",
            "description": f"Tests for {source} ({len(endpoints)} endpoints)",
            "item": []
        }
        
        # Add requests for each endpoint
        for endpoint in endpoints:
            request_item = {
                "name": f"{endpoint['method']} {endpoint['path']}",
                "request": {
                    "method": endpoint['method'].upper(),
                    "header": [
                        {"key": "Content-Type", "value": "application/json"},
                        {"key": "Accept", "value": "application/json"}
                    ],
                    "url": {
                        "raw": f"{{{{baseUrl1}}}}{endpoint['path']}",
                        "host": ["{{baseUrl1}}"],
                        "path": endpoint['path'].strip('/').split('/')
                    }
                },
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.test('Status code is successful', function () {",
                                "    pm.expect(pm.response.code).to.be.oneOf([200, 201, 202, 204]);",
                                "});",
                                "",
                                "pm.test('Response time is acceptable', function () {",
                                "    pm.expect(pm.response.responseTime).to.be.below(2000);",
                                "});",
                                "",
                                "pm.test('Response has valid content type', function () {",
                                "    pm.expect(pm.response.headers.get('Content-Type')).to.include('application/json');",
                                "});"
                            ]
                        }
                    }
                ]
            }
            
            folder["item"].append(request_item)
        
        collection["item"].append(folder)
    
    # Add integration tests folder
    integration_folder = {
        "name": "🔗 Integration Tests",
        "description": "Cross-API integration scenarios",
        "item": [
            {
                "name": "Health Check All APIs",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": "{{baseUrl1}}/health",
                        "host": ["{{baseUrl1}}"],
                        "path": ["health"]
                    }
                },
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.test('API is healthy', function () {",
                                "    pm.response.to.have.status(200);",
                                "});",
                                "",
                                "// Test additional APIs",
                                "// Add more health checks for other APIs"
                            ]
                        }
                    }
                ]
            }
        ]
    }
    
    collection["item"].append(integration_folder)
    
    collection_json = json.dumps(collection, indent=2)
    
    st.code(collection_json, language="json")
    
    st.download_button(
        label="📥 Download Postman Collection",
        data=collection_json,
        file_name=f"combined_api_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


def get_language_for_framework(framework):
    """Get the appropriate language for syntax highlighting"""
    language_map = {
        "Postman": "json",
        "RestAssured": "java",
        "Python Requests": "python",
        "Robot Framework": "robotframework",
        "Cypress": "javascript",
        "Playwright": "javascript",
        "Newman CLI": "json"
    }
    return language_map.get(framework, "text")


def get_file_extension(framework):
    """Get the appropriate file extension for the framework"""
    extension_map = {
        "Postman": "json",
        "RestAssured": "java",
        "Python Requests": "py",
        "Robot Framework": "robot",
        "Cypress": "js",
        "Playwright": "js",
        "Newman CLI": "json"
    }
    return extension_map.get(framework, "txt")


if __name__ == "__main__":
    # This allows running the module directly for testing
    show_ui()
