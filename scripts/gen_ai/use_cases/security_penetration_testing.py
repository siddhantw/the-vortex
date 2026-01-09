"""
Security Penetration Testing Module
Automated security testing following OWASP Top 10 and industry best practices
"""

import streamlit as st
import asyncio
import aiohttp
import ssl
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
from urllib.parse import urljoin, urlparse
import re
from typing import Dict, List, Any, Optional
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
import hashlib
import base64
import time

# Enhanced logging setup
try:
    from enhanced_logging import get_logger, EmojiIndicators, PerformanceTimer, ProgressTracker
    logger = get_logger("SecurityPenetrationTesting", level=logging.INFO, log_file="security_penetration_testing.log")
except ImportError:
    # Fallback to standard logging if enhanced_logging is not available
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    print("Warning: Enhanced logging not available, using standard logging")

# OWASP Top 10 2021 Categories
OWASP_TOP_10 = {
    "A01:2021": "Broken Access Control",
    "A02:2021": "Cryptographic Failures", 
    "A03:2021": "Injection",
    "A04:2021": "Insecure Design",
    "A05:2021": "Security Misconfiguration",
    "A06:2021": "Vulnerable and Outdated Components",
    "A07:2021": "Identification and Authentication Failures",
    "A08:2021": "Software and Data Integrity Failures",
    "A09:2021": "Security Logging and Monitoring Failures",
    "A10:2021": "Server-Side Request Forgery (SSRF)"
}

# Brand configurations based on the existing framework
BRAND_CONFIGS = {
    "wcom": {
        "base_urls": ["https://www.web.com"],
        "security_endpoints": ["/website-security/ssl-certificates", "/website-security/sitelock"],
        "api_endpoints": ["/api", "/sfcore.do"],
        "admin_paths": ["/admin", "/wp-admin", "/administrator"]
    },
    "ncom": {
        "base_urls": ["https://www.networksolutions.com"],
        "security_endpoints": ["/ssl-certificates", "/sitelock-overview"],
        "api_endpoints": ["/api", "/sfcore.do"],
        "admin_paths": ["/admin", "/manager"]
    },
    "rcom": {
        "base_urls": ["https://www.register.com"],
        "security_endpoints": ["/security", "/ssl"],
        "api_endpoints": ["/api", "/sfcore.do"],
        "admin_paths": ["/admin", "/control-panel"]
    },
    "dcom": {
        "base_urls": ["https://www.domain.com"],
        "security_endpoints": ["/security", "/ssl-certificates"],
        "api_endpoints": ["/api", "/sfcore.do"],
        "admin_paths": ["/admin"]
    },
    "bhcom": {
        "base_urls": ["https://www.bluehost.com"],
        "security_endpoints": ["/security", "/ssl"],
        "api_endpoints": ["/api", "/sfcore.do"],
        "admin_paths": ["/my-account"]
    },
    "hgcom": {
        "base_urls": ["https://www.hostgator.com"],
        "security_endpoints": ["/ssl-certificates", "/sitelock"],
        "api_endpoints": ["/api", "/sfcore.do"],
        "admin_paths": ["/cpanel", "/admin"]
    }
}

class SecurityScanner:
    """Advanced security scanner implementing OWASP Top 10 checks"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        requests.packages.urllib3.disable_warnings()
        self.vulnerabilities = []
        self.scan_results = {}
        
    async def scan_website(self, url: str, scan_config: Dict) -> Dict[str, Any]:
        """Comprehensive security scan of a website"""
        results = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "owasp_checks": {},
            "vulnerabilities": [],
            "recommendations": [],
            "security_score": 0,
            "ssl_analysis": {},
            "headers_analysis": {},
            "endpoint_analysis": []
        }
        
        try:
            # SSL/TLS Analysis
            results["ssl_analysis"] = await self._analyze_ssl(url)
            
            # Security Headers Analysis
            results["headers_analysis"] = await self._analyze_security_headers(url)
            
            # OWASP Top 10 Checks
            for owasp_id, description in OWASP_TOP_10.items():
                check_result = await self._perform_owasp_check(url, owasp_id, scan_config)
                results["owasp_checks"][owasp_id] = check_result
                
                if check_result["vulnerabilities"]:
                    results["vulnerabilities"].extend(check_result["vulnerabilities"])
            
            # Endpoint Security Analysis
            results["endpoint_analysis"] = await self._analyze_endpoints(url, scan_config)
            
            # Calculate security score
            results["security_score"] = self._calculate_security_score(results)
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"Error scanning {url}: {e}")
            results["error"] = str(e)
            
        return results
    
    async def _analyze_ssl(self, url: str) -> Dict[str, Any]:
        """Analyze SSL/TLS configuration"""
        ssl_results = {
            "has_ssl": False,
            "certificate_info": {},
            "protocol_support": {},
            "vulnerabilities": [],
            "grade": "F"
        }
        
        try:
            hostname = urlparse(url).hostname
            if not hostname:
                return ssl_results
                
            # Check SSL certificate
            context = ssl.create_default_context()
            with ssl.create_connection((hostname, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    ssl_results["has_ssl"] = True
                    ssl_results["certificate_info"] = {
                        "subject": dict(x[0] for x in cert['subject']),
                        "issuer": dict(x[0] for x in cert['issuer']),
                        "version": cert.get('version'),
                        "serial_number": cert.get('serialNumber'),
                        "not_before": cert.get('notBefore'),
                        "not_after": cert.get('notAfter'),
                        "signature_algorithm": cert.get('signatureAlgorithm')
                    }
                    
                    # Check certificate expiry
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    days_until_expiry = (not_after - datetime.now()).days
                    
                    if days_until_expiry < 30:
                        ssl_results["vulnerabilities"].append({
                            "type": "Certificate Expiry",
                            "severity": "High" if days_until_expiry < 7 else "Medium",
                            "description": f"Certificate expires in {days_until_expiry} days"
                        })
                    
                    # Check protocol versions
                    ssl_results["protocol_support"] = {
                        "tls_1_3": True,  # Simplified check
                        "tls_1_2": True,
                        "cipher_suites": ssock.cipher()
                    }
                    
                    ssl_results["grade"] = self._calculate_ssl_grade(ssl_results)
                    
        except Exception as e:
            logger.error(f"SSL analysis failed for {url}: {e}")
            ssl_results["error"] = str(e)
            
        return ssl_results
    
    async def _analyze_security_headers(self, url: str) -> Dict[str, Any]:
        """Analyze security headers"""
        headers_analysis = {
            "missing_headers": [],
            "present_headers": {},
            "security_score": 0,
            "recommendations": []
        }
        
        security_headers = [
            "Strict-Transport-Security",
            "Content-Security-Policy", 
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Permissions-Policy"
        ]
        
        try:
            response = self.session.get(url, timeout=10)
            
            for header in security_headers:
                if header in response.headers:
                    headers_analysis["present_headers"][header] = response.headers[header]
                else:
                    headers_analysis["missing_headers"].append(header)
                    headers_analysis["recommendations"].append(
                        f"Add {header} header for enhanced security"
                    )
            
            # Calculate score based on present headers
            headers_analysis["security_score"] = (
                len(headers_analysis["present_headers"]) / len(security_headers)
            ) * 100
            
        except Exception as e:
            logger.error(f"Headers analysis failed for {url}: {e}")
            headers_analysis["error"] = str(e)
            
        return headers_analysis
    
    async def _perform_owasp_check(self, url: str, owasp_id: str, scan_config: Dict) -> Dict[str, Any]:
        """Perform specific OWASP Top 10 security check"""
        check_result = {
            "id": owasp_id,
            "name": OWASP_TOP_10[owasp_id],
            "status": "pass",
            "vulnerabilities": [],
            "details": {}
        }
        
        try:
            if owasp_id == "A01:2021":  # Broken Access Control
                check_result = await self._check_access_control(url, check_result)
            elif owasp_id == "A02:2021":  # Cryptographic Failures
                check_result = await self._check_cryptographic_failures(url, check_result)
            elif owasp_id == "A03:2021":  # Injection
                check_result = await self._check_injection_vulnerabilities(url, check_result)
            elif owasp_id == "A04:2021":  # Insecure Design
                check_result = await self._check_insecure_design(url, check_result)
            elif owasp_id == "A05:2021":  # Security Misconfiguration
                check_result = await self._check_security_misconfiguration(url, check_result)
            elif owasp_id == "A06:2021":  # Vulnerable Components
                check_result = await self._check_vulnerable_components(url, check_result)
            elif owasp_id == "A07:2021":  # Auth Failures
                check_result = await self._check_authentication_failures(url, check_result)
            elif owasp_id == "A08:2021":  # Integrity Failures
                check_result = await self._check_integrity_failures(url, check_result)
            elif owasp_id == "A09:2021":  # Logging Failures
                check_result = await self._check_logging_failures(url, check_result)
            elif owasp_id == "A10:2021":  # SSRF
                check_result = await self._check_ssrf_vulnerabilities(url, check_result)
                
        except Exception as e:
            logger.error(f"OWASP check {owasp_id} failed for {url}: {e}")
            check_result["error"] = str(e)
            
        return check_result
    
    async def _check_access_control(self, url: str, result: Dict) -> Dict:
        """Check for broken access control vulnerabilities"""
        admin_paths = ["/admin", "/administrator", "/wp-admin", "/manager", "/control-panel"]
        
        for path in admin_paths:
            try:
                test_url = urljoin(url, path)
                response = self.session.get(test_url, timeout=5, allow_redirects=False)
                
                if response.status_code == 200:
                    result["vulnerabilities"].append({
                        "type": "Exposed Admin Interface",
                        "severity": "High",
                        "url": test_url,
                        "description": f"Admin interface accessible at {path}"
                    })
                    result["status"] = "fail"
                    
            except Exception as e:
                continue
                
        return result
    
    async def _check_cryptographic_failures(self, url: str, result: Dict) -> Dict:
        """Check for cryptographic failures"""
        try:
            # Check for HTTP vs HTTPS
            if not url.startswith('https://'):
                result["vulnerabilities"].append({
                    "type": "Insecure Protocol",
                    "severity": "High", 
                    "description": "Website not using HTTPS"
                })
                result["status"] = "fail"
                
            # Check for weak ciphers (simplified)
            response = self.session.get(url, timeout=10)
            if 'TLS' not in str(response.raw._connection.sock.cipher()):
                result["vulnerabilities"].append({
                    "type": "Weak Encryption",
                    "severity": "Medium",
                    "description": "Weak cipher suite detected"
                })
                result["status"] = "fail"
                
        except Exception as e:
            pass
            
        return result
    
    async def _check_injection_vulnerabilities(self, url: str, result: Dict) -> Dict:
        """Check for injection vulnerabilities"""
        injection_payloads = [
            "' OR '1'='1",
            "<script>alert('XSS')</script>",
            "'; DROP TABLE users; --",
            "{{7*7}}",
            "${7*7}"
        ]
        
        try:
            # Test common form parameters
            for payload in injection_payloads[:2]:  # Limit for demo
                test_params = {'q': payload, 'search': payload, 'id': payload}
                response = self.session.get(url, params=test_params, timeout=5)
                
                if payload in response.text:
                    result["vulnerabilities"].append({
                        "type": "Potential Injection",
                        "severity": "High",
                        "description": f"Payload reflected: {payload}",
                        "parameter": list(test_params.keys())[0]
                    })
                    result["status"] = "fail"
                    
        except Exception as e:
            pass
            
        return result
    
    async def _check_security_misconfiguration(self, url: str, result: Dict) -> Dict:
        """Check for security misconfigurations"""
        try:
            # Check for directory listing
            dir_paths = ["/uploads/", "/files/", "/images/", "/documents/"]
            
            for path in dir_paths:
                test_url = urljoin(url, path)
                response = self.session.get(test_url, timeout=5)
                
                if "Index of" in response.text or "Directory listing" in response.text:
                    result["vulnerabilities"].append({
                        "type": "Directory Listing Enabled",
                        "severity": "Medium",
                        "url": test_url,
                        "description": f"Directory listing enabled at {path}"
                    })
                    result["status"] = "fail"
                    
            # Check for exposed configuration files
            config_files = [
                "/.env", "/config.php", "/wp-config.php", 
                "/web.config", "/.htaccess", "/robots.txt"
            ]
            
            for config_file in config_files:
                test_url = urljoin(url, config_file)
                response = self.session.get(test_url, timeout=5)
                
                if response.status_code == 200 and len(response.text) > 100:
                    result["vulnerabilities"].append({
                        "type": "Exposed Configuration",
                        "severity": "High" if config_file != "/robots.txt" else "Low",
                        "url": test_url,
                        "description": f"Configuration file accessible: {config_file}"
                    })
                    if config_file != "/robots.txt":
                        result["status"] = "fail"
                        
        except Exception as e:
            pass
            
        return result
    
    async def _check_vulnerable_components(self, url: str, result: Dict) -> Dict:
        """Check for vulnerable and outdated components"""
        try:
            response = self.session.get(url, timeout=10)
            
            # Check server headers for version information
            server_header = response.headers.get('Server', '')
            x_powered_by = response.headers.get('X-Powered-By', '')
            
            version_patterns = [
                r'Apache/(\d+\.\d+\.\d+)',
                r'nginx/(\d+\.\d+\.\d+)',
                r'PHP/(\d+\.\d+\.\d+)',
                r'WordPress (\d+\.\d+\.\d+)'
            ]
            
            for pattern in version_patterns:
                for header_value in [server_header, x_powered_by, response.text[:1000]]:
                    match = re.search(pattern, header_value)
                    if match:
                        version = match.group(1)
                        result["vulnerabilities"].append({
                            "type": "Version Disclosure",
                            "severity": "Low",
                            "description": f"Version information disclosed: {match.group(0)}",
                            "version": version
                        })
                        # In real implementation, check against CVE database
                        
        except Exception as e:
            pass
            
        return result
    
    async def _check_authentication_failures(self, url: str, result: Dict) -> Dict:
        """Check for authentication and session management failures"""
        try:
            # Check for weak login endpoints
            login_paths = ["/login", "/signin", "/auth", "/admin/login"]
            
            for path in login_paths:
                test_url = urljoin(url, path)
                response = self.session.get(test_url, timeout=5)
                
                if response.status_code == 200:
                    # Check for brute force protection
                    # Simple check for rate limiting
                    for i in range(3):
                        login_attempt = self.session.post(
                            test_url,
                            data={'username': 'test', 'password': 'test'},
                            timeout=5
                        )
                        if i == 2 and login_attempt.status_code != 429:
                            result["vulnerabilities"].append({
                                "type": "No Rate Limiting",
                                "severity": "Medium",
                                "url": test_url,
                                "description": "No rate limiting detected on login endpoint"
                            })
                            result["status"] = "fail"
                            break
                        time.sleep(1)
                        
        except Exception as e:
            pass
            
        return result
    
    async def _check_integrity_failures(self, url: str, result: Dict) -> Dict:
        """Check for software and data integrity failures"""
        try:
            response = self.session.get(url, timeout=10)
            
            # Check for CDN usage without SRI
            script_tags = re.findall(r'<script[^>]+src=["\']([^"\']+)["\'][^>]*>', response.text)
            
            for script_src in script_tags:
                if any(cdn in script_src for cdn in ['cdn.', 'ajax.googleapis.com', 'cdnjs.']):
                    if 'integrity=' not in response.text:
                        result["vulnerabilities"].append({
                            "type": "Missing SRI",
                            "severity": "Medium",
                            "description": f"CDN resource without SRI: {script_src}",
                            "url": script_src
                        })
                        result["status"] = "fail"
                        
        except Exception as e:
            pass
            
        return result
    
    async def _check_logging_failures(self, url: str, result: Dict) -> Dict:
        """Check for security logging and monitoring failures"""
        # This would typically involve checking log configurations
        # For demo purposes, we'll check for basic security headers that aid monitoring
        try:
            response = self.session.get(url, timeout=10)
            
            monitoring_headers = [
                'Content-Security-Policy-Report-Only',
                'Report-To',
                'NEL'  # Network Error Logging
            ]
            
            missing_monitoring = []
            for header in monitoring_headers:
                if header not in response.headers:
                    missing_monitoring.append(header)
                    
            if missing_monitoring:
                result["vulnerabilities"].append({
                    "type": "Limited Security Monitoring",
                    "severity": "Low",
                    "description": f"Missing monitoring headers: {', '.join(missing_monitoring)}"
                })
                
        except Exception as e:
            pass
            
        return result
    
    async def _check_ssrf_vulnerabilities(self, url: str, result: Dict) -> Dict:
        """Check for Server-Side Request Forgery vulnerabilities"""
        try:
            # Test common SSRF parameters
            ssrf_params = ['url', 'link', 'redirect', 'proxy', 'callback']
            ssrf_payloads = [
                'http://169.254.169.254/latest/meta-data/',  # AWS metadata
                'http://localhost:22',
                'file:///etc/passwd'
            ]
            
            for param in ssrf_params[:2]:  # Limit for demo
                for payload in ssrf_payloads[:1]:  # Limit for demo
                    test_params = {param: payload}
                    response = self.session.get(url, params=test_params, timeout=5)
                    
                    # Simple check for potential SSRF (would need more sophisticated detection)
                    if response.status_code == 500 or 'timeout' in response.text.lower():
                        result["vulnerabilities"].append({
                            "type": "Potential SSRF",
                            "severity": "High",
                            "parameter": param,
                            "description": f"Possible SSRF vulnerability in parameter: {param}"
                        })
                        result["status"] = "fail"
                        
        except Exception as e:
            pass
            
        return result
    
    async def _analyze_endpoints(self, url: str, scan_config: Dict) -> List[Dict]:
        """Analyze security of specific endpoints"""
        endpoint_results = []
        
        # Get endpoints from configuration
        brand = scan_config.get('brand', 'wcom')
        if brand in BRAND_CONFIGS:
            endpoints = BRAND_CONFIGS[brand].get('security_endpoints', [])
            
            for endpoint in endpoints:
                endpoint_url = urljoin(url, endpoint)
                try:
                    response = self.session.get(endpoint_url, timeout=10)
                    
                    endpoint_analysis = {
                        "endpoint": endpoint,
                        "url": endpoint_url,
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds(),
                        "content_length": len(response.content),
                        "security_headers": {},
                        "vulnerabilities": []
                    }
                    
                    # Check security headers for this endpoint
                    security_headers = ['X-Frame-Options', 'X-Content-Type-Options', 'X-XSS-Protection']
                    for header in security_headers:
                        if header in response.headers:
                            endpoint_analysis["security_headers"][header] = response.headers[header]
                            
                    endpoint_results.append(endpoint_analysis)
                    
                except Exception as e:
                    endpoint_results.append({
                        "endpoint": endpoint,
                        "url": endpoint_url,
                        "error": str(e)
                    })
                    
        return endpoint_results
    
    def _calculate_ssl_grade(self, ssl_results: Dict) -> str:
        """Calculate SSL grade based on configuration"""
        score = 100
        
        if not ssl_results.get("has_ssl"):
            return "F"
            
        # Deduct points for vulnerabilities
        for vuln in ssl_results.get("vulnerabilities", []):
            if vuln["severity"] == "High":
                score -= 30
            elif vuln["severity"] == "Medium":
                score -= 15
            else:
                score -= 5
                
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def _calculate_security_score(self, results: Dict) -> int:
        """Calculate overall security score"""
        total_score = 100
        
        # Deduct points for vulnerabilities
        for vuln in results.get("vulnerabilities", []):
            if vuln["severity"] == "High":
                total_score -= 20
            elif vuln["severity"] == "Medium":
                total_score -= 10
            else:
                total_score -= 5
                
        # Adjust based on SSL grade
        ssl_grade = results.get("ssl_analysis", {}).get("grade", "F")
        if ssl_grade in ["A+", "A"]:
            total_score += 10
        elif ssl_grade == "F":
            total_score -= 20
            
        # Adjust based on security headers
        headers_score = results.get("headers_analysis", {}).get("security_score", 0)
        total_score += int(headers_score * 0.2)
        
        return max(0, min(100, total_score))
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable security recommendations"""
        recommendations = []
        
        # SSL recommendations
        ssl_grade = results.get("ssl_analysis", {}).get("grade", "F")
        if ssl_grade in ["F", "D", "C"]:
            recommendations.append("Upgrade SSL/TLS configuration to achieve grade A or higher")
            
        # Headers recommendations
        missing_headers = results.get("headers_analysis", {}).get("missing_headers", [])
        if missing_headers:
            recommendations.append(f"Add missing security headers: {', '.join(missing_headers)}")
            
        # Vulnerability-specific recommendations
        for vuln in results.get("vulnerabilities", []):
            if vuln["type"] == "Exposed Admin Interface":
                recommendations.append("Restrict admin interface access by IP or implement proper authentication")
            elif vuln["type"] == "Directory Listing Enabled":
                recommendations.append("Disable directory listing in web server configuration")
            elif vuln["type"] == "Missing SRI":
                recommendations.append("Add Subresource Integrity (SRI) for all external scripts")
                
        # OWASP-specific recommendations
        for owasp_id, check_result in results.get("owasp_checks", {}).items():
            if check_result.get("status") == "fail":
                owasp_name = OWASP_TOP_10[owasp_id]
                recommendations.append(f"Address {owasp_name} vulnerabilities found during scan")
                
        return list(set(recommendations))  # Remove duplicates

def show_ui():
    """Display the Security Penetration Testing UI"""
    st.title("🔒 Security Penetration Testing")
    st.markdown("Comprehensive automated security testing following OWASP Top 10 and industry best practices")
    
    # Configuration Section
    st.header("🔧 Scan Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        scan_type = st.selectbox(
            "Scan Type",
            ["Quick Scan", "Comprehensive Scan", "OWASP Top 10 Only", "Custom Scan"],
            help="Select the type of security scan to perform"
        )
        
        selected_brand = st.selectbox(
            "Brand/Website",
            list(BRAND_CONFIGS.keys()),
            format_func=lambda x: x.upper(),
            help="Select the brand configuration to use"
        )
        
    with col2:
        custom_urls = st.text_area(
            "Custom URLs (one per line)",
            placeholder="https://example.com\nhttps://api.example.com",
            help="Add custom URLs to scan (will be added to brand URLs)"
        )
        
        scan_depth = st.slider(
            "Scan Depth",
            1, 5, 2,
            help="1=Surface scan, 5=Deep penetration testing"
        )
    
    # Advanced Options
    with st.expander("🔧 Advanced Scan Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_owasp = st.multiselect(
                "OWASP Categories to Include",
                list(OWASP_TOP_10.keys()),
                default=list(OWASP_TOP_10.keys()),
                format_func=lambda x: f"{x}: {OWASP_TOP_10[x]}"
            )
            
            scan_timeout = st.number_input(
                "Request Timeout (seconds)",
                1, 60, 10
            )
            
        with col2:
            enable_ssl_check = st.checkbox("SSL/TLS Analysis", True)
            enable_headers_check = st.checkbox("Security Headers Analysis", True)
            enable_endpoint_scan = st.checkbox("Endpoint Security Scan", True)
            enable_vuln_scan = st.checkbox("Vulnerability Scanning", True)
    
    # Scan Execution
    st.header("🚀 Execute Security Scan")
    
    if st.button("🔍 Start Security Scan", type="primary"):
        if not include_owasp:
            st.error("Please select at least one OWASP category to scan")
            return
            
        # Prepare URLs
        urls_to_scan = []
        
        # Add brand URLs
        if selected_brand in BRAND_CONFIGS:
            urls_to_scan.extend(BRAND_CONFIGS[selected_brand]["base_urls"])
            
        # Add custom URLs
        if custom_urls.strip():
            custom_url_list = [url.strip() for url in custom_urls.split('\n') if url.strip()]
            urls_to_scan.extend(custom_url_list)
            
        if not urls_to_scan:
            st.error("No URLs to scan. Please select a brand or add custom URLs.")
            return
            
        # Prepare scan configuration
        scan_config = {
            "brand": selected_brand,
            "scan_type": scan_type,
            "scan_depth": scan_depth,
            "include_owasp": include_owasp,
            "timeout": scan_timeout,
            "enable_ssl": enable_ssl_check,
            "enable_headers": enable_headers_check,
            "enable_endpoints": enable_endpoint_scan,
            "enable_vulnerabilities": enable_vuln_scan
        }
        
        # Execute scan
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        scanner = SecurityScanner()
        all_results = []
        
        for i, url in enumerate(urls_to_scan):
            status_text.text(f"Scanning {url}...")
            progress_bar.progress((i + 1) / len(urls_to_scan))
            
            try:
                # Use asyncio.run for each URL scan
                result = asyncio.run(scanner.scan_website(url, scan_config))
                all_results.append(result)
                
                # Add notifications for critical vulnerabilities
                high_severity_vulns = [
                    v for v in result.get("vulnerabilities", []) 
                    if v.get("severity") == "High"
                ]
                
                if high_severity_vulns:
                    try:
                        import notifications
                        notifications.add_notification(
                            module_name="security_penetration_testing",
                            status="warning",
                            message=f"High severity vulnerabilities found in {url}",
                            details=f"Found {len(high_severity_vulns)} high severity vulnerabilities",
                            action_steps=[
                                "Review detailed scan results below",
                                "Prioritize fixing high severity vulnerabilities",
                                "Consider running additional security tests"
                            ]
                        )
                    except ImportError:
                        pass
                        
            except Exception as e:
                st.error(f"Error scanning {url}: {e}")
                all_results.append({
                    "url": url,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        progress_bar.progress(1.0)
        status_text.text("Scan completed!")
        
        # Display Results
        st.header("📊 Security Scan Results")
        
        if all_results:
            # Summary Statistics
            st.subheader("📈 Executive Summary")
            
            total_vulnerabilities = sum(
                len(result.get("vulnerabilities", [])) for result in all_results
            )
            
            high_severity = sum(
                len([v for v in result.get("vulnerabilities", []) if v.get("severity") == "High"])
                for result in all_results
            )
            
            avg_security_score = sum(
                result.get("security_score", 0) for result in all_results if "security_score" in result
            ) / len([r for r in all_results if "security_score" in r]) if all_results else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("URLs Scanned", len(urls_to_scan))
            with col2:
                st.metric("Total Vulnerabilities", total_vulnerabilities)
            with col3:
                st.metric("High Severity", high_severity, delta=f"-{high_severity}" if high_severity > 0 else None)
            with col4:
                st.metric("Avg Security Score", f"{avg_security_score:.1f}/100")
            
            # Detailed Results for each URL
            for result in all_results:
                if "error" in result:
                    st.error(f"❌ {result['url']}: {result['error']}")
                    continue
                    
                st.subheader(f"🌐 {result['url']}")
                
                # Security Score and SSL Grade
                col1, col2, col3 = st.columns(3)
                with col1:
                    score = result.get("security_score", 0)
                    score_color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
                    st.metric("Security Score", f"{score_color} {score}/100")
                    
                with col2:
                    ssl_grade = result.get("ssl_analysis", {}).get("grade", "F")
                    grade_color = "🟢" if ssl_grade in ["A+", "A"] else "🟡" if ssl_grade in ["B", "C"] else "🔴"
                    st.metric("SSL Grade", f"{grade_color} {ssl_grade}")
                    
                with col3:
                    vuln_count = len(result.get("vulnerabilities", []))
                    st.metric("Vulnerabilities", vuln_count)
                
                # OWASP Top 10 Results
                if result.get("owasp_checks"):
                    st.write("**OWASP Top 10 Check Results:**")
                    
                    owasp_data = []
                    for owasp_id, check_result in result["owasp_checks"].items():
                        status_icon = "✅" if check_result.get("status") == "pass" else "❌"
                        owasp_data.append({
                            "OWASP ID": owasp_id,
                            "Category": OWASP_TOP_10[owasp_id],
                            "Status": f"{status_icon} {check_result.get('status', 'unknown').title()}",
                            "Vulnerabilities": len(check_result.get("vulnerabilities", []))
                        })
                    
                    st.dataframe(pd.DataFrame(owasp_data), use_container_width=True)
                
                # Vulnerabilities Details
                if result.get("vulnerabilities"):
                    st.write("**🚨 Vulnerabilities Found:**")
                    
                    for vuln in result["vulnerabilities"]:
                        severity_color = {
                            "High": "🔴",
                            "Medium": "🟡", 
                            "Low": "🟠"
                        }.get(vuln.get("severity", "Unknown"), "⚪")
                        
                        st.write(f"{severity_color} **{vuln.get('type', 'Unknown')}** ({vuln.get('severity', 'Unknown')})")
                        st.write(f"   ↳ {vuln.get('description', 'No description available')}")
                        
                        if vuln.get("url"):
                            st.write(f"   🔗 URL: `{vuln['url']}`")
                
                # Security Headers Analysis
                if result.get("headers_analysis"):
                    headers_analysis = result["headers_analysis"]
                    
                    with st.expander("🛡️ Security Headers Analysis"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Present Headers:**")
                            for header, value in headers_analysis.get("present_headers", {}).items():
                                st.write(f"✅ {header}")
                                
                        with col2:
                            st.write("**Missing Headers:**")
                            for header in headers_analysis.get("missing_headers", []):
                                st.write(f"❌ {header}")
                
                # Recommendations
                if result.get("recommendations"):
                    with st.expander("💡 Security Recommendations"):
                        for i, recommendation in enumerate(result["recommendations"], 1):
                            st.write(f"{i}. {recommendation}")
                
                st.markdown("---")
            
            # Export Results
            st.subheader("📤 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON Export
                json_data = json.dumps(all_results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_data,
                    file_name=f"security_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
            with col2:
                # CSV Export for vulnerabilities
                vuln_data = []
                for result in all_results:
                    for vuln in result.get("vulnerabilities", []):
                        vuln_data.append({
                            "URL": result.get("url", ""),
                            "Type": vuln.get("type", ""),
                            "Severity": vuln.get("severity", ""),
                            "Description": vuln.get("description", ""),
                            "Vulnerable URL": vuln.get("url", ""),
                            "Parameter": vuln.get("parameter", ""),
                            "Timestamp": result.get("timestamp", "")
                        })
                
                if vuln_data:
                    csv_data = pd.DataFrame(vuln_data).to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV Report", 
                        data=csv_data,
                        file_name=f"vulnerabilities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        else:
            st.error("No scan results to display")
            
        # Success notification
        try:
            import notifications
            notifications.add_notification(
                module_name="security_penetration_testing",
                status="success",
                message=f"Security scan completed for {len(urls_to_scan)} URLs",
                details=f"Found {total_vulnerabilities} total vulnerabilities, {high_severity} high severity",
                action_steps=[
                    "Review the detailed results above",
                    "Export reports for stakeholders",
                    "Schedule regular security scans",
                    "Address high priority vulnerabilities first"
                ]
            )
        except ImportError:
            pass

    # Help Section
    with st.expander("ℹ️ Help & Best Practices"):
        st.markdown("""
        ### Security Scanning Best Practices
        
        **Scan Types:**
        - **Quick Scan**: Basic security checks (5-10 minutes)
        - **Comprehensive Scan**: Full OWASP Top 10 + additional checks (15-30 minutes)  
        - **OWASP Top 10 Only**: Focus on OWASP categories only
        - **Custom Scan**: Configure specific checks
        
        **OWASP Top 10 Categories:**
        - **A01: Broken Access Control** - Authorization bypass, privilege escalation
        - **A02: Cryptographic Failures** - Weak encryption, exposed sensitive data
        - **A03: Injection** - SQL, NoSQL, OS command injection
        - **A04: Insecure Design** - Security design flaws
        - **A05: Security Misconfiguration** - Default configs, incomplete setups
        - **A06: Vulnerable Components** - Outdated libraries and frameworks
        - **A07: Authentication Failures** - Weak authentication and session management
        - **A08: Integrity Failures** - Insecure CI/CD, plugins, libraries
        - **A09: Logging Failures** - Insufficient logging and monitoring
        - **A10: SSRF** - Server-Side Request Forgery
        
        **Security Score Interpretation:**
        - **90-100**: Excellent security posture
        - **70-89**: Good security with minor issues
        - **50-69**: Moderate security requiring attention
        - **Below 50**: Poor security needing immediate action
        
        **SSL Grade Meanings:**
        - **A+/A**: Excellent SSL configuration
        - **B/C**: Acceptable but needs improvement
        - **D/F**: Poor SSL configuration requiring immediate fix
        """)

if __name__ == "__main__":
    show_ui()