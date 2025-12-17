#!/usr/bin/env python3
"""
GTMetrix-Style Performance Checker
A comprehensive website performance testing tool that accepts a list of URLs and provides detailed performance analysis
similar to GTMetrix, including Core Web Vitals, performance grades, waterfall charts, and detailed insights.
"""

import argparse
import base64
import copy
import json
import logging
import os
import random
import subprocess
import tempfile
import time
import requests
import datetime
import csv
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import glob
from pathlib import Path
from urllib.parse import urlparse, urljoin
import hashlib

# Configure matplotlib to use non-interactive backend to prevent hanging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import seaborn as sns
from bs4 import BeautifulSoup
from jinja2 import Template
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Import the historical data manager
from historical_data_manager import HistoricalDataManager
from waterfall_utils import WaterfallUtils

# Paths to Lighthouse and Node.js
LIGHTHOUSE_PATH = os.path.expanduser('~/.nvm/versions/node/v23.1.0/bin/lighthouse')
NODE_PATH = os.path.expanduser('~/.nvm/versions/node/v23.1.0/bin/node')

# Google PageSpeed Insights API key (encoded for security)
ENCODED_API_KEY = "QUl6YVN5Q3c4Q3c1TmF5RWdIYzJ4bnRWRXdqN2J6dzdqd1l6TUU4"

# Chrome User Experience Report API key
CRUX_API_KEY = "AIzaSyBnvJmRw0kLK7NhtrA4X1VG-IuobCBx4_I"  # Users should provide their own CrUX API key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


class GTMetrixStyleChecker:
    """Main class for comprehensive website performance testing"""
    
    def __init__(self, output_dir=None):
        """Initialize the performance checker"""
        self.output_dir = output_dir or os.path.join(os.getcwd(), "performance_reports")
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.report_dir = os.path.join(self.output_dir, f"performance_report_{self.timestamp}")
        self.data_manager = HistoricalDataManager(base_dir=self.output_dir)
        
        # Create output directories
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "screenshots"), exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "waterfalls"), exist_ok=True)
        os.makedirs(os.path.join(self.report_dir, "charts"), exist_ok=True)
        
        # Performance thresholds (similar to GTMetrix grading)
        self.performance_thresholds = {
            'lcp': {'good': 2.5, 'needs_improvement': 4.0},
            'fcp': {'good': 1.8, 'needs_improvement': 3.0},
            'cls': {'good': 0.1, 'needs_improvement': 0.25},
            'inp': {'good': 200, 'needs_improvement': 500},
            'ttfb': {'good': 0.8, 'needs_improvement': 1.8},
            'tbt': {'good': 200, 'needs_improvement': 600},
            'speed_index': {'good': 3.4, 'needs_improvement': 5.8}
        }
        
        logging.info(f"Performance checker initialized. Reports will be saved to: {self.report_dir}")

    def get_api_key(self):
        """Get the Google PageSpeed Insights API key"""
        env_api_key = os.environ.get('PAGESPEED_API_KEY')
        if env_api_key:
            return env_api_key
        return base64.b64decode(ENCODED_API_KEY).decode('utf-8')

    def load_urls_from_file(self, file_path):
        """Load URLs from a file (supports CSV, TXT, or JSON)"""
        urls = []
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        if 'url' in row:
                            urls.append({'url': row['url'], 'name': row.get('name', row['url'])})
                        else:
                            # Assume first column is URL
                            urls.append({'url': list(row.values())[0], 'name': row.get('name', list(row.values())[0])})
            
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as jsonfile:
                    data = json.load(jsonfile)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str):
                                urls.append({'url': item, 'name': item})
                            elif isinstance(item, dict):
                                urls.append({'url': item['url'], 'name': item.get('name', item['url'])})
            
            else:  # TXT or other
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    for line in txtfile:
                        url = line.strip()
                        if url and not url.startswith('#'):
                            urls.append({'url': url, 'name': url})
            
            logging.info(f"Loaded {len(urls)} URLs from {file_path}")
            return urls
            
        except Exception as e:
            logging.error(f"Error loading URLs from file {file_path}: {str(e)}")
            return []

    def run_lighthouse_analysis(self, url, device_mode='desktop', runs=3):
        """Run comprehensive Lighthouse analysis with multiple runs for accuracy"""
        logging.info(f"Running Lighthouse analysis for {url} in {device_mode} mode ({runs} runs)")
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--enable-blink-features=InteractionToNextPaint")
        chrome_options.add_argument("--enable-features=InteractionToNextPaint")
        
        # Configure viewport based on device mode
        if device_mode == 'desktop':
            chrome_options.add_argument("--window-size=1920,1080")
        else:
            chrome_options.add_argument("--window-size=360,640")
        
        reports = []
        
        for run in range(runs):
            try:
                with tempfile.TemporaryDirectory(prefix=f"lighthouse_run_{run}_") as temp_dir:
                    output_file = os.path.join(temp_dir, "report.json")
                    
                    # Configure Lighthouse command based on device mode
                    base_chrome_flags = "--headless=new --no-sandbox --disable-dev-shm-usage"
                    
                    if device_mode == 'desktop':
                        lighthouse_cmd = [
                            NODE_PATH, LIGHTHOUSE_PATH,
                            url,
                            "--output=json",
                            f"--output-path={output_file}",
                            "--quiet",
                            f"--chrome-flags={base_chrome_flags} --window-size=1920,1080",
                            "--form-factor=desktop",
                            "--preset=desktop",
                            "--throttling-method=simulate",
                            "--disable-storage-reset"
                        ]
                    else:  # mobile
                        lighthouse_cmd = [
                            NODE_PATH, LIGHTHOUSE_PATH,
                            url,
                            "--output=json",
                            f"--output-path={output_file}",
                            "--quiet",
                            f"--chrome-flags={base_chrome_flags} --window-size=360,640",
                            "--form-factor=mobile",
                            "--throttling-method=simulate",
                            "--disable-storage-reset"
                        ]
                    
                    logging.info(f"Running Lighthouse (attempt {run + 1}/{runs})")
                    result = subprocess.run(lighthouse_cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0 and os.path.exists(output_file):
                        with open(output_file, 'r') as f:
                            report = json.load(f)
                            reports.append(report)
                            logging.info(f"Lighthouse run {run + 1} completed successfully")
                    else:
                        logging.warning(f"Lighthouse run {run + 1} failed: {result.stderr}")
            
            except Exception as e:
                logging.error(f"Error in Lighthouse run {run + 1}: {str(e)}")
            
            # Add delay between runs
            if run < runs - 1:
                time.sleep(2)
        
        if not reports:
            logging.error(f"All Lighthouse runs failed for {url}")
            return None
        
        # Average the reports for more accurate results
        return self.average_lighthouse_reports(reports)

    def average_lighthouse_reports(self, reports):
        """Average multiple Lighthouse reports for more accurate results"""
        if not reports:
            return None
        
        if len(reports) == 1:
            return reports[0]
        
        avg_report = copy.deepcopy(reports[0])
        
        # Metrics to average
        metrics_to_avg = [
            "first-contentful-paint",
            "largest-contentful-paint",
            "interaction-to-next-paint",
            "experimental-interaction-to-next-paint",
            "max-potential-fid",
            "speed-index",
            "total-blocking-time",
            "cumulative-layout-shift",
            "interactive",
            "server-response-time"
        ]
        
        # Average audit values
        for metric in metrics_to_avg:
            if metric in avg_report.get("audits", {}):
                values = []
                for report in reports:
                    if metric in report.get("audits", {}) and "numericValue" in report["audits"][metric]:
                        values.append(report["audits"][metric]["numericValue"])
                
                if values:
                    avg_value = sum(values) / len(values)
                    avg_report["audits"][metric]["numericValue"] = avg_value
                    # Update display value as well
                    if metric in ["cumulative-layout-shift"]:
                        avg_report["audits"][metric]["displayValue"] = f"{avg_value:.3f}"
                    elif "ms" in avg_report["audits"][metric].get("displayValue", ""):
                        avg_report["audits"][metric]["displayValue"] = f"{int(avg_value)} ms"
                    elif "s" in avg_report["audits"][metric].get("displayValue", ""):
                        avg_report["audits"][metric]["displayValue"] = f"{avg_value:.1f} s"
        
        # Average category scores
        for category in avg_report.get("categories", {}):
            scores = []
            for report in reports:
                if category in report.get("categories", {}) and "score" in report["categories"][category]:
                    scores.append(report["categories"][category]["score"])
            
            if scores:
                avg_score = sum(scores) / len(scores)
                avg_report["categories"][category]["score"] = avg_score
        
        logging.info(f"Averaged {len(reports)} Lighthouse reports")
        return avg_report

    def fetch_crux_data(self, url):
        """Fetch Chrome User Experience Report data with improved error handling"""
        if not CRUX_API_KEY:
            logging.info("CrUX API key not provided, skipping CrUX data fetch")
            return None

        def normalize_url(url):
            """Normalize URL for CrUX API (remove fragments, some query params)"""
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(url)
            # Remove fragment and normalize
            normalized = urlunparse((
                parsed.scheme,
                parsed.netloc.lower(),
                parsed.path.rstrip('/') if parsed.path != '/' else '/',
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            return normalized

        def try_origin_fallback(url):
            """Try fetching data for just the origin if specific URL fails"""
            from urllib.parse import urlparse
            parsed = urlparse(url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            return origin

        try:
            api_url = f"https://chromeuxreport.googleapis.com/v1/records:queryRecord?key={CRUX_API_KEY}"
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }

            # First try: Original URL (normalized)
            normalized_url = normalize_url(url)
            payload = {"url": normalized_url}

            logging.info(f"Sending CrUX API request for URL: {normalized_url}")
            response = requests.post(api_url, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                logging.info(f"Successfully fetched CrUX data for {normalized_url}")
                return response.json()

            elif response.status_code == 404:
                # Second try: Origin only (many sites only have origin-level data)
                origin_url = try_origin_fallback(url)
                if origin_url != normalized_url:
                    logging.info(f"URL-specific CrUX data not found, trying origin: {origin_url}")
                    payload = {"url": origin_url}
                    response = requests.post(api_url, json=payload, headers=headers, timeout=10)

                    if response.status_code == 200:
                        logging.info(f"Successfully fetched CrUX data for origin: {origin_url}")
                        return response.json()
                    else:
                        logging.info(f"No CrUX data available for {origin_url} (insufficient traffic data)")
                        return None
                else:
                    logging.info(f"No CrUX data available for {normalized_url} (insufficient traffic data)")
                    return None

            else:
                # Other HTTP errors
                try:
                    error_details = response.json()
                    logging.warning(f"CrUX API error {response.status_code}: {error_details}")
                except:
                    logging.warning(f"CrUX API error {response.status_code}: {response.text[:200]}")
                return None

        except requests.exceptions.Timeout:
            logging.warning("CrUX API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"CrUX API request failed: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching CrUX data: {str(e)}")
            return None

    def capture_screenshot(self, url, device_mode='desktop'):
        """Capture screenshot of the webpage with enhanced error handling"""
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--ignore-ssl-errors")
            chrome_options.add_argument("--ignore-certificate-errors-spki-list")
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

            if device_mode == 'desktop':
                chrome_options.add_argument("--window-size=1920,1080")
            else:
                chrome_options.add_argument("--window-size=375,667")

            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)

            # Wait for page to load
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Additional wait for dynamic content
            time.sleep(3)

            # Scroll to top to ensure consistent screenshots
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"screenshot_{url_hash}_{device_mode}.png"
            filepath = os.path.join(self.report_dir, "screenshots", filename)

            # Ensure screenshot directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Take screenshot with error handling
            try:
                driver.save_screenshot(filepath)

                # Verify screenshot was created and has content
                if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:  # At least 1KB
                    logging.info(f"Screenshot saved successfully: {filename}")
                    return filename
                else:
                    logging.warning(f"Screenshot file is too small or doesn't exist: {filepath}")
                    return None

            except Exception as screenshot_error:
                logging.error(f"Failed to save screenshot: {str(screenshot_error)}")
                return None

            except Exception as driver_error:
                logging.error(f"WebDriver error for {url}: {str(driver_error)}")
                return None

            finally:
                if driver:
                    try:
                        driver.quit()
                    except Exception as quit_error:
                        logging.warning(f"Error closing driver: {str(quit_error)}")

        except Exception as e:
            logging.error(f"Error capturing screenshot for {url}: {str(e)}")
            return None

    def generate_waterfall_chart(self, lighthouse_report, url):
        """Generate accurate waterfall chart from Lighthouse network data with enhanced fallbacks"""
        try:
            # Extract network requests from Lighthouse report
            network_requests = []

            # Try multiple sources for network data
            requests_data = []

            # Primary source: network-requests audit
            if "audits" in lighthouse_report and "network-requests" in lighthouse_report["audits"]:
                audit_data = lighthouse_report["audits"]["network-requests"].get("details", {}).get("items", [])
                if audit_data:
                    requests_data.extend(audit_data)
                    logging.info(f"Found {len(audit_data)} requests in network-requests audit for {url}")

            # Fallback 1: Try resource-summary audit
            if not requests_data and "audits" in lighthouse_report and "resource-summary" in lighthouse_report["audits"]:
                resource_data = lighthouse_report["audits"]["resource-summary"].get("details", {}).get("items", [])
                if resource_data:
                    # Convert resource summary to request-like format with more accurate timing
                    base_start_time = 0.1  # Start after navigation
                    for i, resource in enumerate(resource_data):
                        # Estimate timing based on resource type and size
                        resource_type = resource.get('resourceType', 'other')
                        transfer_size = resource.get('size', 1000)

                        # More realistic timing based on resource type
                        if resource_type == 'Document':
                            start_time = 0
                            duration = 0.2 + (transfer_size / 50000)  # Base time + size factor
                        elif resource_type == 'Stylesheet':
                            start_time = base_start_time + (i * 0.05)
                            duration = 0.1 + (transfer_size / 100000)
                        elif resource_type == 'Script':
                            start_time = base_start_time + (i * 0.08)
                            duration = 0.15 + (transfer_size / 80000)
                        elif resource_type == 'Image':
                            start_time = base_start_time + (i * 0.03)
                            duration = 0.05 + (transfer_size / 200000)
                        else:
                            start_time = base_start_time + (i * 0.04)
                            duration = 0.08 + (transfer_size / 150000)

                        requests_data.append({
                            'url': f"{url}/{resource.get('label', f'resource-{i}')}",
                            'startTime': start_time,
                            'endTime': start_time + duration,
                            'transferSize': transfer_size,
                            'resourceType': resource_type,
                            'statusCode': 200,
                            'mimeType': resource.get('mimeType', ''),
                            'priority': 'Medium'
                        })
                    logging.info(f"Using resource-summary fallback with {len(resource_data)} items for {url}")

            # Fallback 2: Try critical-request-chains audit with better timing
            if not requests_data and "audits" in lighthouse_report and "critical-request-chains" in lighthouse_report["audits"]:
                chains = lighthouse_report["audits"]["critical-request-chains"].get("details", {}).get("chains", {})
                if chains:
                    def extract_from_chains(chain_data, start_time=0, depth=0):
                        chain_requests = []
                        for url_key, chain_info in chain_data.items():
                            # More realistic duration calculation
                            transfer_size = chain_info.get('transferSize', 1000)
                            duration = 0.1 + (transfer_size / 100000) + (depth * 0.05)

                            chain_requests.append({
                                'url': url_key,
                                'startTime': start_time,
                                'endTime': start_time + duration,
                                'transferSize': transfer_size,
                                'resourceType': 'document' if depth == 0 else 'other',
                                'statusCode': 200,
                                'priority': 'High' if depth == 0 else 'Medium'
                            })

                            # Process children with cascading timing
                            if 'children' in chain_info:
                                child_start = start_time + duration * 0.7  # Start children before parent finishes
                                chain_requests.extend(extract_from_chains(chain_info['children'], child_start, depth + 1))
                        return chain_requests

                    requests_data = extract_from_chains(chains)
                    logging.info(f"Using critical-request-chains fallback with {len(requests_data)} items for {url}")

            # Fallback 3: Create more realistic synthetic data
            if not requests_data:
                logging.warning(f"No network data found, creating realistic synthetic waterfall for {url}")
                synthetic_requests = [
                    {
                        'url': url,
                        'startTime': 0,
                        'endTime': 0.8,  # More realistic document load time
                        'transferSize': 50000,
                        'resourceType': 'Document',
                        'statusCode': 200,
                        'synthetic': True,
                        'priority': 'VeryHigh'
                    }
                ]

                # More realistic resource patterns with proper timing
                realistic_resources = [
                    {'type': 'Stylesheet', 'size': 25000, 'start': 0.2, 'duration': 0.3},
                    {'type': 'Stylesheet', 'size': 15000, 'start': 0.25, 'duration': 0.2},
                    {'type': 'Script', 'size': 75000, 'start': 0.4, 'duration': 0.5},
                    {'type': 'Script', 'size': 45000, 'start': 0.6, 'duration': 0.3},
                    {'type': 'Image', 'size': 120000, 'start': 0.8, 'duration': 0.4},
                    {'type': 'Image', 'size': 80000, 'start': 0.9, 'duration': 0.3},
                    {'type': 'Font', 'size': 35000, 'start': 0.7, 'duration': 0.2},
                    {'type': 'XHR', 'size': 5000, 'start': 1.1, 'duration': 0.1},
                ]

                for i, resource in enumerate(realistic_resources):
                    synthetic_requests.append({
                        'url': f"{url}/assets/{resource['type'].lower()}-{i+1}",
                        'startTime': resource['start'],
                        'endTime': resource['start'] + resource['duration'],
                        'transferSize': resource['size'],
                        'resourceType': resource['type'],
                        'statusCode': 200,
                        'synthetic': True,
                        'priority': 'High' if resource['type'] in ['Stylesheet', 'Script'] else 'Medium'
                    })

                requests_data = synthetic_requests

            # Get more accurate page load timing baseline
            page_start_time = 0
            navigation_start = 0

            if "audits" in lighthouse_report and "metrics" in lighthouse_report["audits"]:
                metrics_details = lighthouse_report["audits"]["metrics"].get("details", {}).get("items", [])
                if metrics_details and len(metrics_details) > 0:
                    metrics_data = metrics_details[0]
                    # Use more accurate timing references
                    navigation_start = metrics_data.get("observedNavigationStart", 0)
                    page_start_time = navigation_start

            # Process requests with more accurate timing calculations
            for req in requests_data:
                url_req = req.get('url', '')
                start_time = req.get('startTime', 0)
                end_time = req.get('endTime', 0)

                # Skip invalid entries
                if not url_req:
                    continue

                # Handle timing edge cases more accurately
                if start_time >= end_time:
                    # Set minimum realistic duration based on resource type
                    resource_type = req.get('resourceType', 'other').lower()
                    if 'document' in resource_type:
                        min_duration = 0.1  # 100ms minimum for documents
                    elif 'script' in resource_type or 'stylesheet' in resource_type:
                        min_duration = 0.05  # 50ms minimum for CSS/JS
                    else:
                        min_duration = 0.02  # 20ms minimum for others

                    end_time = start_time + min_duration

                # More accurate relative timing calculation
                if page_start_time and not req.get('synthetic', False):
                    # For real Lighthouse data, convert from seconds to milliseconds relative to navigation start
                    relative_start = max(0, (start_time - page_start_time) * 1000)
                    duration = (end_time - start_time) * 1000
                else:
                    # For synthetic or fallback data, assume it's already in the right units
                    relative_start = start_time * 1000
                    duration = (end_time - start_time) * 1000

                # Enhanced resource type detection
                resource_type = self.detect_resource_type(url_req, req.get('resourceType', ''), req.get('mimeType', ''))

                # Extract detailed timing information if available
                timing_details = self.extract_timing_details(req)

                network_requests.append({
                    'url': url_req,
                    'startTime': relative_start,
                    'endTime': relative_start + duration,
                    'duration': max(duration, 10),  # Minimum 10ms duration for visibility
                    'transferSize': req.get('transferSize', 0),
                    'resourceSize': req.get('resourceSize', 0),
                    'resourceType': resource_type,
                    'statusCode': req.get('statusCode', 200),
                    'mimeType': req.get('mimeType', ''),
                    'priority': req.get('priority', 'Medium'),
                    'timing': timing_details,
                    'fromCache': req.get('fromDiskCache', False) or req.get('fromMemoryCache', False),
                    'synthetic': req.get('synthetic', False)
                })

            if not network_requests:
                logging.error(f"Failed to generate any network data for waterfall chart: {url}")
                return None

            # Sort requests by start time for proper waterfall display
            network_requests.sort(key=lambda x: x['startTime'])

            # Filter out extremely short requests but keep a minimum set
            filtered_requests = [req for req in network_requests if req['duration'] > 5]  # > 5ms

            # Ensure we have at least some requests to display
            if not filtered_requests and network_requests:
                filtered_requests = network_requests[:10]  # Show top 10 if all are very short

            if not filtered_requests:
                logging.warning(f"No requests to display in waterfall chart for {url}")
                return None

            # Clear any existing plots to prevent memory issues
            plt.clf()
            plt.close('all')

            # Create waterfall chart with proper sizing
            num_requests = len(filtered_requests)
            chart_height = max(8, min(20, num_requests * 0.5))  # Dynamic height with limits

            try:
                fig, ax = plt.subplots(figsize=(16, chart_height))
            except Exception as fig_error:
                logging.error(f"Failed to create matplotlib figure for {url}: {str(fig_error)}")
                # Try with smaller figure
                fig, ax = plt.subplots(figsize=(12, 8))

            # Enhanced color mapping for different resource types
            color_map = {
                'Document': '#1f77b4',     # Blue
                'Stylesheet': '#2ca02c',   # Green
                'Script': '#ff7f0e',       # Orange
                'Image': '#d62728',        # Red
                'Font': '#9467bd',         # Purple
                'Media': '#8c564b',        # Brown
                'XHR': '#e377c2',          # Pink
                'Fetch': '#e377c2',        # Pink
                'WebSocket': '#17becf',    # Cyan
                'Manifest': '#bcbd22',     # Olive
                'Other': '#7f7f7f'         # Gray
            }

            # Priority-based alpha values for better visual hierarchy
            priority_alpha = {
                'VeryHigh': 1.0,
                'High': 0.9,
                'Medium': 0.8,
                'Low': 0.7,
                'VeryLow': 0.6
            }

            y_pos = 0
            max_time = 0
            timing_bars = []

            for req in filtered_requests:
                try:
                    start_time = req['startTime']
                    duration = req['duration']
                    end_time = start_time + duration
                    max_time = max(max_time, end_time)

                    resource_type = req['resourceType']
                    color = color_map.get(resource_type, color_map['Other'])
                    alpha = priority_alpha.get(req['priority'], 0.8)

                    # Apply different styling for cached resources
                    if req['fromCache']:
                        alpha *= 0.6
                        edge_style = '--'
                        edge_color = 'gray'
                    else:
                        edge_style = '-'
                        edge_color = 'white'

                    # Draw the main request bar with better styling
                    rect = patches.Rectangle((start_time, y_pos), duration, 0.8,
                                           linewidth=1.5, edgecolor=edge_color,
                                           facecolor=color, alpha=alpha,
                                           linestyle=edge_style)
                    ax.add_patch(rect)

                    # Add timing phase details if available
                    if req['timing'] and any(req['timing'].values()):
                        timing_bars.append((y_pos, start_time, req['timing'], req['duration']))

                    # Create more informative labels
                    url_parts = req['url'].split('/')
                    filename = url_parts[-1] if url_parts else req['url']

                    # Better filename handling
                    if len(filename) > 35:
                        filename = filename[:32] + '...'
                    elif not filename or filename == '':
                        filename = f"{resource_type.lower()}"

                    # Add comprehensive size information
                    size_info = ""
                    transfer_size = req['transferSize']
                    if transfer_size > 0:
                        if transfer_size >= 1024 * 1024:
                            size_info = f" ({transfer_size / (1024 * 1024):.1f}MB)"
                        elif transfer_size >= 1024:
                            size_info = f" ({transfer_size / 1024:.1f}KB)"
                        else:
                            size_info = f" ({transfer_size}B)"

                    # Add status and cache indicators
                    status_info = ""
                    if req['statusCode'] >= 400:
                        status_info = f" [{req['statusCode']}]"

                    cache_indicator = " [cached]" if req['fromCache'] else ""
                    synthetic_indicator = " [synthetic]" if req.get('synthetic') else ""

                    label_text = f"{filename}{size_info}{status_info}{cache_indicator}{synthetic_indicator}"

                    # Position label with better spacing
                    label_x = max_time * 0.01  # Dynamic positioning
                    ax.text(label_x, y_pos + 0.4, label_text, fontsize=8, va='center',
                           fontweight='bold' if not req['fromCache'] else 'normal',
                           color='black' if not req['fromCache'] else 'gray')

                    # Add duration text on the right
                    duration_text = f"{duration:.0f}ms"
                    ax.text(start_time + duration + (max_time * 0.01), y_pos + 0.4, duration_text,
                           fontsize=7, va='center', color='navy', fontweight='bold')

                    y_pos += 1

                except Exception as bar_error:
                    logging.warning(f"Error drawing bar for request {req.get('url', 'unknown')}: {str(bar_error)}")
                    y_pos += 1
                    continue

            # Draw timing phase details with better accuracy
            for y, start, timing, total_duration in timing_bars:
                try:
                    current_time = start
                    phase_height = 0.15

                    # DNS lookup
                    if timing.get('dns', 0) > 0:
                        dns_duration = min(timing['dns'], total_duration * 0.3)
                        rect = patches.Rectangle((current_time, y + 0.85), dns_duration, phase_height,
                                               facecolor='#ff9999', alpha=0.8, edgecolor='none')
                        ax.add_patch(rect)
                        current_time += dns_duration

                    # TCP connect
                    if timing.get('connect', 0) > 0:
                        connect_duration = min(timing['connect'], total_duration * 0.3)
                        rect = patches.Rectangle((current_time, y + 0.85), connect_duration, phase_height,
                                               facecolor='#ffcc99', alpha=0.8, edgecolor='none')
                        ax.add_patch(rect)
                        current_time += connect_duration

                    # SSL handshake
                    if timing.get('ssl', 0) > 0:
                        ssl_duration = min(timing['ssl'], total_duration * 0.2)
                        rect = patches.Rectangle((current_time, y + 0.85), ssl_duration, phase_height,
                                               facecolor='#99ccff', alpha=0.8, edgecolor='none')
                        ax.add_patch(rect)
                except Exception as timing_error:
                    logging.warning(f"Error drawing timing details: {str(timing_error)}")
                    continue

            # Set proper axis limits with better scaling
            if max_time > 0:
                ax.set_xlim(0, max_time * 1.05)
            else:
                ax.set_xlim(0, 1000)

            ax.set_ylim(-0.5, len(filtered_requests) + 0.5)
            ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
            ax.set_title(f'Network Waterfall Chart - {urlparse(url).netloc}',
                        fontsize=14, fontweight='bold', pad=20)

            # Improve grid and styling
            ax.set_yticks([])
            ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Enhanced legend with better information
            resource_counts = {}
            total_size = 0
            for req in filtered_requests:
                resource_type = req['resourceType']
                resource_counts[resource_type] = resource_counts.get(resource_type, 0) + 1
                total_size += req.get('transferSize', 0)

            legend_elements = []
            for resource_type, color in color_map.items():
                count = resource_counts.get(resource_type, 0)
                if count > 0:
                    legend_elements.append(
                        patches.Patch(facecolor=color, label=f'{resource_type} ({count})')
                    )

            if legend_elements:
                try:
                    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1),
                             frameon=True, fancybox=True, shadow=True, fontsize=9)
                except Exception as legend_error:
                    logging.warning(f"Error creating legend: {str(legend_error)}")

            # Add comprehensive performance summary
            total_requests = len(filtered_requests)
            cached_requests = sum(1 for req in filtered_requests if req['fromCache'])
            failed_requests = sum(1 for req in filtered_requests if req['statusCode'] >= 400)

            summary_text = f"Requests: {total_requests} | Cached: {cached_requests} | Failed: {failed_requests} | Total Size: {self.format_bytes(total_size)} | Load Time: {max_time:.0f}ms"

            try:
                ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                       facecolor='lightblue', alpha=0.8), fontweight='bold')
            except Exception as text_error:
                logging.warning(f"Error adding summary text: {str(text_error)}")

            # Adjust layout with proper margins
            try:
                plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.12)
            except Exception as layout_error:
                logging.warning(f"Error adjusting layout: {str(layout_error)}")

            # Generate filename and save with better error handling
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"waterfall_{url_hash}.png"
            filepath = os.path.join(self.report_dir, "waterfalls", filename)

            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
            except Exception as dir_error:
                logging.error(f"Failed to create waterfall directory: {str(dir_error)}")
                return None

            # Save with high quality
            try:
                plt.savefig(filepath, format='png', dpi=300, bbox_inches='tight',
                           pad_inches=0.3, facecolor='white', edgecolor='none')
                plt.close(fig)

                # Verify file creation and size
                if os.path.exists(filepath) and os.path.getsize(filepath) > 2000:  # At least 2KB
                    file_size = os.path.getsize(filepath)
                    logging.info(f"Accurate waterfall chart saved: {filename} ({file_size} bytes) with {total_requests} requests")
                    return filename
                else:
                    logging.error(f"Waterfall chart file is too small or doesn't exist: {filepath}")
                    return None

            except Exception as save_error:
                logging.error(f"Failed to save waterfall chart: {str(save_error)}")
                try:
                    plt.close(fig)
                except:
                    pass
                return None

        except Exception as e:
            logging.error(f"Error generating waterfall chart for {url}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

            # Clean up matplotlib resources
            try:
                plt.close('all')
            except:
                pass

            return None

    def calculate_performance_grade(self, metrics):
        """Calculate performance grade similar to GTMetrix"""
        scores = []

        # Core Web Vitals scoring
        lcp_score = self.get_metric_score(metrics.get('lcp'), self.performance_thresholds['lcp'])
        fcp_score = self.get_metric_score(metrics.get('fcp'), self.performance_thresholds['fcp'])
        cls_score = self.get_metric_score(metrics.get('cls'), self.performance_thresholds['cls'])
        inp_score = self.get_metric_score(metrics.get('inp'), self.performance_thresholds['inp'])

        # Additional metrics
        ttfb_score = self.get_metric_score(metrics.get('ttfb'), self.performance_thresholds['ttfb'])
        tbt_score = self.get_metric_score(metrics.get('tbt'), self.performance_thresholds['tbt'])

        scores.extend([lcp_score, fcp_score, cls_score, inp_score, ttfb_score, tbt_score])
        scores = [s for s in scores if s is not None]

        if not scores:
            return 'F', 0

        avg_score = sum(scores) / len(scores)

        # Convert to letter grade
        if avg_score >= 90:
            return 'A', avg_score
        elif avg_score >= 80:
            return 'B', avg_score
        elif avg_score >= 70:
            return 'C', avg_score
        elif avg_score >= 60:
            return 'D', avg_score
        elif avg_score >= 50:
            return 'E', avg_score
        else:
            return 'F', avg_score

    def get_metric_score(self, value, thresholds):
        """Convert metric value to score (0-100)"""
        if value is None or value == "N/A":
            return None

        try:
            numeric_value = float(str(value).replace('s', '').replace('ms', '').replace(',', ''))

            if numeric_value <= thresholds['good']:
                return 100
            elif numeric_value <= thresholds['needs_improvement']:
                # Linear interpolation between good and needs improvement
                range_size = thresholds['needs_improvement'] - thresholds['good']
                score_range = 50  # From 100 to 50
                score = 100 - ((numeric_value - thresholds['good']) / range_size) * score_range
                return max(50, score)
            else:
                # Poor performance, score between 0-50
                return max(0, 50 - (numeric_value - thresholds['needs_improvement']) / thresholds['needs_improvement'] * 50)

        except (ValueError, TypeError):
            return None

    def extract_metrics_from_lighthouse(self, report):
        """Extract performance metrics from Lighthouse report"""
        if not report or "audits" not in report:
            return {}

        audits = report["audits"]

        def safe_extract(audit_key, default="N/A"):
            if audit_key in audits and "numericValue" in audits[audit_key]:
                return audits[audit_key]["numericValue"]
            return default

        def safe_extract_display(audit_key, default="N/A"):
            if audit_key in audits and "displayValue" in audits[audit_key]:
                return audits[audit_key]["displayValue"]
            return default

        # Enhanced INP extraction with proper unit handling and comprehensive fallback
        def extract_inp():
            """Extract INP with comprehensive fallback and proper unit conversion"""

            # Try standard INP audit first
            inp_audit = audits.get("interaction-to-next-paint")
            if inp_audit and "numericValue" in inp_audit:
                numeric_value = inp_audit["numericValue"]
                logging.debug(f"Found standard INP audit: {numeric_value}")

                # Handle unit conversion - Lighthouse may return values in different units
                if numeric_value > 10000:  # Likely in microseconds, convert to milliseconds
                    return numeric_value / 1000
                else:
                    return numeric_value

            # Try experimental INP audit
            exp_inp_audit = audits.get("experimental-interaction-to-next-paint")
            if exp_inp_audit and "numericValue" in exp_inp_audit:
                numeric_value = exp_inp_audit["numericValue"]
                logging.debug(f"Found experimental INP audit: {numeric_value}")

                # Handle unit conversion
                if numeric_value > 10000:  # Likely in microseconds
                    return numeric_value / 1000
                else:
                    return numeric_value

            # Try max-potential-fid as fallback (legacy INP-like metric)
            max_fid_audit = audits.get("max-potential-fid")
            if max_fid_audit and "numericValue" in max_fid_audit:
                numeric_value = max_fid_audit["numericValue"]
                logging.debug(f"Using max-potential-fid as INP fallback: {numeric_value}")
                return numeric_value

            # Check metrics section with multiple possible field names
            metrics_audit = audits.get("metrics", {}).get("details", {}).get("items", [])
            if metrics_audit and len(metrics_audit) > 0:
                metrics_data = metrics_audit[0]

                # Try various field names that Lighthouse might use for INP
                inp_field_names = [
                    "interactionToNextPaint",
                    "experimental-interaction-to-next-paint",
                    "observedInteractionToNextPaint",
                    "maxPotentialFID",
                    "observedMaxPotentialFID"
                ]

                for field_name in inp_field_names:
                    if field_name in metrics_data:
                        value = metrics_data[field_name]
                        if value is not None and value > 0:
                            logging.debug(f"Found INP in metrics section under {field_name}: {value}")

                            # Handle unit conversion for metrics section values
                            if value > 10000:  # Likely in microseconds
                                return value / 1000
                            else:
                                return value

            # Try CrUX data as last resort if available in the report
            if "environment" in report and "networkUserAgent" in report["environment"]:
                # This indicates we might have real user data available
                for audit_name in audits:
                    audit = audits[audit_name]
                    if ("inp" in audit_name.lower() or "interaction" in audit_name.lower()) and "numericValue" in audit:
                        value = audit["numericValue"]
                        if value is not None and value > 0:
                            logging.debug(f"Found INP in audit {audit_name}: {value}")
                            return value / 1000 if value > 10000 else value

            logging.debug("No INP data found in any location")
            return "N/A"

        # Get INP value with enhanced extraction and validation
        inp_value = extract_inp()

        # Validate INP value for reasonableness
        if inp_value != "N/A":
            try:
                inp_numeric = float(inp_value)
                # INP should typically be between 0 and 30000ms (30 seconds) for websites
                # Values above 30s are likely errors, but up to 30s can be legitimate poor performance
                if inp_numeric < 0:
                    logging.warning(f"INP value {inp_numeric} is negative, using N/A")
                    inp_value = "N/A"
                elif inp_numeric > 30000:
                    logging.warning(f"INP value {inp_numeric}ms is extremely high (>30s), likely an error. Using capped value of 30000ms")
                    inp_value = 30000  # Cap at 30 seconds instead of rejecting
                else:
                    inp_value = inp_numeric
                    if inp_numeric > 10000:
                        logging.info(f"INP value {inp_numeric}ms is very high but within acceptable range for poor-performing sites")
            except (ValueError, TypeError):
                logging.warning(f"Could not convert INP value {inp_value} to numeric, using N/A")
                inp_value = "N/A"

        metrics = {
            'fcp': safe_extract("first-contentful-paint") / 1000 if safe_extract("first-contentful-paint") != "N/A" else "N/A",
            'lcp': safe_extract("largest-contentful-paint") / 1000 if safe_extract("largest-contentful-paint") != "N/A" else "N/A",
            'cls': safe_extract("cumulative-layout-shift"),
            'inp': inp_value,  # Raw value preserved for calculations
            'ttfb': safe_extract("server-response-time") / 1000 if safe_extract("server-response-time") != "N/A" else "N/A",
            'tbt': safe_extract("total-blocking-time"),
            'tti': safe_extract("interactive") / 1000 if safe_extract("interactive") != "N/A" else "N/A",
            'speed_index': safe_extract("speed-index") / 1000 if safe_extract("speed-index") != "N/A" else "N/A",
            'performance_score': report.get("categories", {}).get("performance", {}).get("score", 0) * 100,
            'accessibility_score': report.get("categories", {}).get("accessibility", {}).get("score", 0) * 100,
            'best_practices_score': report.get("categories", {}).get("best-practices", {}).get("score", 0) * 100,
            'seo_score': report.get("categories", {}).get("seo", {}).get("score", 0) * 100,
        }

        # Calculate Core Web Vitals score
        cwv_scores = []
        if metrics['lcp'] != "N/A":
            cwv_scores.append(self.get_metric_score(metrics['lcp'], self.performance_thresholds['lcp']) or 0)
        if metrics['fcp'] != "N/A":
            cwv_scores.append(self.get_metric_score(metrics['fcp'], self.performance_thresholds['fcp']) or 0)
        if metrics['cls'] != "N/A":
            cwv_scores.append(self.get_metric_score(metrics['cls'], self.performance_thresholds['cls']) or 0)
        if metrics['inp'] != "N/A":
            cwv_scores.append(self.get_metric_score(metrics['inp'], self.performance_thresholds['inp']) or 0)

        metrics['core_web_vitals_score'] = sum(cwv_scores) / len(cwv_scores) if cwv_scores else 0

        return metrics

    def generate_performance_recommendations(self, metrics, lighthouse_report):
        """Generate detailed performance recommendations"""
        recommendations = []

        # FCP Recommendations
        if metrics.get("fcp") != "N/A" and float(metrics["fcp"]) > 1.8:
            recommendations.append({
                'type': 'FCP',
                'severity': 'high' if float(metrics["fcp"]) > 3.0 else 'medium',
                'title': 'Improve First Contentful Paint',
                'description': f'Your FCP is {metrics["fcp"]:.2f}s. Good FCP is under 1.8s.',
                'suggestions': [
                    'Optimize server response times',
                    'Remove render-blocking resources',
                    'Optimize CSS delivery',
                    'Use a Content Delivery Network (CDN)'
                ]
            })

        # LCP Recommendations
        if metrics.get("lcp") != "N/A" and float(metrics["lcp"]) > 2.5:
            recommendations.append({
                'type': 'LCP',
                'severity': 'high' if float(metrics["lcp"]) > 4.0 else 'medium',
                'title': 'Improve Largest Contentful Paint',
                'description': f'Your LCP is {metrics["lcp"]:.2f}s. Good LCP is under 2.5s.',
                'suggestions': [
                    'Optimize images and use next-gen formats',
                    'Preload key resources',
                    'Optimize server response times',
                    'Remove unused CSS and JavaScript'
                ]
            })

        # CLS Recommendations
        if metrics.get("cls") != "N/A" and float(metrics["cls"]) > 0.1:
            recommendations.append({
                'type': 'CLS',
                'severity': 'high' if float(metrics["cls"]) > 0.25 else 'medium',
                'title': 'Improve Cumulative Layout Shift',
                'description': f'Your CLS is {metrics["cls"]:.3f}. Good CLS is under 0.1.',
                'suggestions': [
                    'Set size attributes on images and video elements',
                    'Reserve space for ad slots',
                    'Avoid inserting content above existing content',
                    'Use CSS aspect ratio for responsive images'
                ]
            })

        # INP Recommendations
        if metrics.get("inp") != "N/A" and float(metrics["inp"]) > 200:
            recommendations.append({
                'type': 'INP',
                'severity': 'high' if float(metrics["inp"]) > 500 else 'medium',
                'title': 'Improve Interaction to Next Paint',
                'description': f'Your INP is {metrics["inp"]:.0f}ms. Good INP is under 200ms.',
                'suggestions': [
                    'Optimize JavaScript execution',
                    'Break up long-running tasks',
                    'Minimize main thread work',
                    'Use web workers for heavy computations'
                ]
            })

        # TBT Recommendations
        if metrics.get("tbt") != "N/A" and float(metrics["tbt"]) > 200:
            recommendations.append({
                'type': 'TBT',
                'severity': 'medium',
                'title': 'Reduce Total Blocking Time',
                'description': f'Your TBT is {metrics["tbt"]:.0f}ms. Good TBT is under 200ms.',
                'suggestions': [
                    'Split large JavaScript bundles',
                    'Remove unused JavaScript',
                    'Minimize third-party scripts',
                    'Use code splitting and lazy loading'
                ]
            })

        return recommendations

    def analyze_url(self, url_data, device_modes=['desktop', 'mobile'], runs=3):
        """Perform comprehensive analysis of a single URL"""
        url = url_data['url']
        name = url_data['name']

        logging.info(f"Starting comprehensive analysis for: {name} ({url})")

        results = []

        for device_mode in device_modes:
            try:
                logging.info(f"Analyzing {name} in {device_mode} mode")

                # Run Lighthouse analysis
                lighthouse_report = self.run_lighthouse_analysis(url, device_mode, runs)
                if not lighthouse_report:
                    logging.error(f"Failed to get Lighthouse report for {url} in {device_mode} mode")
                    continue

                # Extract metrics
                metrics = self.extract_metrics_from_lighthouse(lighthouse_report)

                # Calculate performance grade
                grade, score = self.calculate_performance_grade(metrics)

                # Generate recommendations
                recommendations = self.generate_performance_recommendations(metrics, lighthouse_report)

                # Capture screenshot
                screenshot = self.capture_screenshot(url, device_mode)

                # Generate waterfall chart
                waterfall = self.generate_waterfall_chart(lighthouse_report, url)

                # Fetch CrUX data
                crux_data = self.fetch_crux_data(url)

                # Compile results
                result = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'name': name,
                    'url': url,
                    'device_mode': device_mode,
                    'grade': grade,
                    'score': score,
                    'metrics': metrics,
                    'recommendations': recommendations,
                    'screenshot': screenshot,
                    'waterfall': waterfall,
                    'crux_data': crux_data,
                    'lighthouse_report': lighthouse_report
                }

                results.append(result)
                logging.info(f"Analysis completed for {name} ({device_mode}): Grade {grade} ({score:.1f})")

            except Exception as e:
                logging.error(f"Error analyzing {url} in {device_mode} mode: {str(e)}")

        return results

    def process_urls(self, urls, device_modes=['desktop', 'mobile'], max_workers=5, runs=3):
        """Process multiple URLs with parallel execution"""
        all_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.analyze_url, url_data, device_modes, runs): url_data for url_data in urls}

            for future in as_completed(future_to_url):
                url_data = future_to_url[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logging.error(f"Error processing {url_data['url']}: {str(e)}")

        return all_results

    def generate_comparison_charts(self, results):
        """Generate comparison charts for all analyzed URLs"""
        if not results:
            return

        # Performance score comparison
        plt.figure(figsize=(14, 8))  # Increased width for better spacing

        # Group results by URL and device mode
        desktop_results = [r for r in results if r['device_mode'] == 'desktop']
        mobile_results = [r for r in results if r['device_mode'] == 'mobile']

        if desktop_results:
            names = [r['name'][:25] + ('...' if len(r['name']) > 25 else '') for r in desktop_results]  # Shortened names
            scores = [r['score'] for r in desktop_results]

            x = np.arange(len(names))
            width = 0.35

            plt.bar(x - width/2, scores, width, label='Desktop', alpha=0.8, color='#4285F4')

            if mobile_results:
                mobile_scores = [r['score'] for r in mobile_results]
                plt.bar(x + width/2, mobile_scores, width, label='Mobile', alpha=0.8, color='#EA4335')

            plt.xlabel('Websites')
            plt.ylabel('Performance Score')
            plt.title('Performance Score Comparison')
            plt.xticks(x, names, rotation=45, ha='right')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)

            # Use subplots_adjust instead of tight_layout for better control
            plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.9)

            plt.savefig(os.path.join(self.report_dir, "charts", "performance_comparison.png"),
                       dpi=300, bbox_inches='tight', pad_inches=0.2)
            plt.close()

        # Core Web Vitals comparison
        self.generate_cwv_comparison_chart(results)

    def generate_cwv_comparison_chart(self, results):
        """Generate Core Web Vitals comparison chart"""
        if not results:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        desktop_results = [r for r in results if r['device_mode'] == 'desktop']

        if desktop_results:
            names = [r['name'][:20] + ('...' if len(r['name']) > 20 else '') for r in desktop_results]

            # LCP
            lcp_values = [float(r['metrics']['lcp']) if r['metrics']['lcp'] != "N/A" else 0 for r in desktop_results]
            bars1 = ax1.bar(names, lcp_values, color=['#4CAF50' if v <= 2.5 else '#FF9800' if v <= 4.0 else '#F44336' for v in lcp_values])
            ax1.set_title('Largest Contentful Paint (LCP)')
            ax1.set_ylabel('Seconds')
            ax1.axhline(y=2.5, color='green', linestyle='--', alpha=0.7)
            ax1.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7)
            ax1.tick_params(axis='x', rotation=45)

            # FCP
            fcp_values = [float(r['metrics']['fcp']) if r['metrics']['fcp'] != "N/A" else 0 for r in desktop_results]
            bars2 = ax2.bar(names, fcp_values, color=['#4CAF50' if v <= 1.8 else '#FF9800' if v <= 3.0 else '#F44336' for v in fcp_values])
            ax2.set_title('First Contentful Paint (FCP)')
            ax2.set_ylabel('Seconds')
            ax2.axhline(y=1.8, color='green', linestyle='--', alpha=0.7)
            ax2.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7)
            ax2.tick_params(axis='x', rotation=45)

            # CLS
            cls_values = [float(r['metrics']['cls']) if r['metrics']['cls'] != "N/A" else 0 for r in desktop_results]
            bars3 = ax3.bar(names, cls_values, color=['#4CAF50' if v <= 0.1 else '#FF9800' if v <= 0.25 else '#F44336' for v in cls_values])
            ax3.set_title('Cumulative Layout Shift (CLS)')
            ax3.set_ylabel('Score')
            ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.7)
            ax3.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7)
            ax3.tick_params(axis='x', rotation=45)

            # INP
            inp_values = [float(r['metrics']['inp']) if r['metrics']['inp'] != "N/A" else 0 for r in desktop_results]
            bars4 = ax4.bar(names, inp_values, color=['#4CAF50' if v <= 200 else '#FF9800' if v <= 500 else '#F44336' for v in inp_values])
            ax4.set_title('Interaction to Next Paint (INP)')
            ax4.set_ylabel('Milliseconds')
            ax4.axhline(y=200, color='green', linestyle='--', alpha=0.7)
            ax4.axhline(y=500, color='orange', linestyle='--', alpha=0.7)
            ax4.tick_params(axis='x', rotation=45)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.15, hspace=0.3, wspace=0.3)
        plt.savefig(os.path.join(self.report_dir, "charts", "core_web_vitals_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_html_report(self, results):
        """Generate comprehensive HTML report with modern UI/UX"""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --info-color: #3b82f6;
            --light-bg: #f8fafc;
            --white: #ffffff;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            line-height: 1.6;
            color: var(--gray-800);
            background: linear-gradient(135deg, var(--light-bg) 0%, #e2e8f0 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            box-shadow: var(--shadow-xl);
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.1;
        }

        .header-content {
            position: relative;
            z-index: 1;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 12px;
            background: linear-gradient(45deg, #fff, #e2e8f0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .header p {
            font-size: 1.25rem;
            opacity: 0.9;
            font-weight: 300;
        }

        .timestamp {
            font-size: 0.875rem;
            opacity: 0.8;
            margin-top: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Cards */
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }

        .card {
            background: var(--white);
            border-radius: 16px;
            padding: 28px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--gray-100);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        }

        .card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }

        .card h3 {
            color: var(--gray-700);
            margin-bottom: 20px;
            font-size: 1.125rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-icon {
            width: 20px;
            height: 20px;
            opacity: 0.7;
        }

        /* Grade styling */
        .grade {
            font-size: 4rem;
            font-weight: 800;
            text-align: center;
            margin: 16px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .grade.A { 
            background: linear-gradient(135deg, var(--success-color), #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .grade.B { 
            background: linear-gradient(135deg, #65a30d, #84cc16);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .grade.C { 
            background: linear-gradient(135deg, var(--warning-color), #d97706);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .grade.D { 
            background: linear-gradient(135deg, #dc2626, #b91c1c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .grade.F { 
            background: linear-gradient(135deg, var(--danger-color), #dc2626);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Metrics */
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 16px 0;
            padding: 16px 20px;
            background: var(--gray-50);
            border-radius: 12px;
            transition: all 0.2s ease;
            border-left: 4px solid transparent;
        }

        .metric:hover {
            background: var(--gray-100);
            transform: translateX(4px);
        }

        .metric-good { 
            border-left-color: var(--success-color);
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        }
        .metric-needs-improvement { 
            border-left-color: var(--warning-color);
            background: linear-gradient(135deg, #fffbeb, #fef3c7);
        }
        .metric-poor { 
            border-left-color: var(--danger-color);
            background: linear-gradient(135deg, #fef2f2, #fecaca);
        }

        .metric-label {
            font-weight: 500;
            color: var(--gray-700);
        }

        .metric-value {
            font-weight: 600;
            font-size: 1.125rem;
        }

        /* Tabs */
        .device-tabs {
            display: flex;
            margin-bottom: 24px;
            background: var(--gray-100);
            border-radius: 12px;
            padding: 4px;
        }

        .device-tab {
            flex: 1;
            padding: 12px 24px;
            background: transparent;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
            color: var(--gray-600);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .device-tab.active {
            background: var(--white);
            color: var(--primary-color);
            box-shadow: var(--shadow-sm);
        }

        .device-tab:hover:not(.active) {
            background: var(--gray-200);
            color: var(--gray-700);
        }

        /* Content */
        .device-content {
            display: none;
            animation: fadeIn 0.3s ease;
        }

        .device-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Performance Overview */
        .performance-overview {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 32px;
            margin: 32px 0;
        }

        .performance-grade-card {
            background: linear-gradient(135deg, var(--white) 0%, var(--gray-50) 100%);
            border-radius: 20px;
            padding: 32px;
            text-align: center;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--gray-200);
        }

        .performance-score {
            font-size: 1.5rem;
            color: var(--gray-600);
            margin-top: 8px;
        }

        /* Charts */
        .chart-container {
            text-align: center;
            margin: 32px 0;
            background: var(--white);
            border-radius: 16px;
            padding: 24px;
            box-shadow: var(--shadow-md);
        }

        .chart-container h4 {
            margin-bottom: 20px;
            color: var(--gray-700);
            font-size: 1.25rem;
            font-weight: 600;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
        }

        /* Recommendations */
        .recommendations {
            margin-top: 32px;
        }

        .recommendation {
            background: var(--white);
            border: 1px solid var(--gray-200);
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: var(--shadow-sm);
            border-left: 4px solid transparent;
        }

        .recommendation.high {
            border-left-color: var(--danger-color);
            background: linear-gradient(135deg, #fef2f2, var(--white));
        }

        .recommendation.medium {
            border-left-color: var(--warning-color);
            background: linear-gradient(135deg, #fffbeb, var(--white));
        }

        .recommendation h4 {
            color: var(--gray-800);
            margin-bottom: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .severity-badge {
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
        }

        .severity-badge.high {
            background: var(--danger-color);
            color: white;
        }

        .severity-badge.medium {
            background: var(--warning-color);
            color: white;
        }

        .suggestion-list {
            list-style-type: none;
            padding-left: 0;
        }

        .suggestion-list li {
            padding: 8px 0;
            padding-left: 24px;
            position: relative;
            color: var(--gray-700);
        }

        .suggestion-list li:before {
            content: "→";
            position: absolute;
            left: 0;
            color: var(--primary-color);
            font-weight: 600;
        }

        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 24px 0;
            background: var(--white);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }

        th, td {
            padding: 16px;
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
        }

        th {
            background: var(--gray-50);
            font-weight: 600;
            color: var(--gray-700);
        }

        tr:hover {
            background: var(--gray-50);
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 64px;
            padding: 32px;
            color: var(--gray-600);
            background: var(--white);
            border-radius: 16px;
            box-shadow: var(--shadow-sm);
        }

        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: 16px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .performance-overview {
                grid-template-columns: 1fr;
                gap: 24px;
            }

            .summary-cards {
                grid-template-columns: 1fr;
            }
        }

        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid var(--gray-300);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        /* Progress bars */
        .progress-bar {
            width: 100%;
            height: 8px;
            background: var(--gray-200);
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }

        .progress-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }

        .progress-fill.good { background: linear-gradient(90deg, var(--success-color), #059669); }
        .progress-fill.needs-improvement { background: linear-gradient(90deg, var(--warning-color), #d97706); }
        .progress-fill.poor { background: linear-gradient(90deg, var(--danger-color), #dc2626); }

        /* Tooltips */
        .tooltip {
            position: relative;
            cursor: help;
        }

        .tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--gray-900);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.875rem;
            white-space: nowrap;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1><i class="fas fa-tachometer-alt"></i> Performance Analysis Report</h1>
                <p>Comprehensive website performance insights powered by Lighthouse</p>
                <div class="timestamp">
                    <i class="fas fa-clock"></i>
                    Generated on {{ timestamp }}
                </div>
            </div>
        </div>

        <!-- Summary Statistics -->
        <div class="summary-cards">
            <div class="card">
                <h3><i class="fas fa-globe card-icon"></i>URLs Analyzed</h3>
                <div class="grade">{{ total_urls }}</div>
                <p style="text-align: center; color: var(--gray-600);">Websites tested</p>
            </div>
            <div class="card">
                <h3><i class="fas fa-chart-line card-icon"></i>Average Performance</h3>
                <div class="grade {{ avg_grade }}">{{ avg_score }}%</div>
                <div class="progress-bar">
                    <div class="progress-fill {{ 'good' if avg_score >= 90 else 'needs-improvement' if avg_score >= 50 else 'poor' }}" 
                         style="width: {{ avg_score }}%"></div>
                </div>
            </div>
            <div class="card">
                <h3><i class="fas fa-trophy card-icon"></i>Best Performing</h3>
                {% if best_site %}
                <div style="font-weight: 600; margin-bottom: 8px;">{{ best_site.name[:100] }}{{ '...' if best_site.name|length > 30 else '' }}</div>
                <div class="grade {{ best_site.grade }}">{{ best_site.grade }}</div>
                {% else %}
                <div class="grade">-</div>
                {% endif %}
            </div>
            <div class="card">
                <h3><i class="fas fa-exclamation-triangle card-icon"></i>Needs Attention</h3>
                {% if worst_site %}
                <div style="font-weight: 600; margin-bottom: 8px;">{{ worst_site.name[:100] }}{{ '...' if worst_site.name|length > 30 else '' }}</div>
                <div class="grade {{ worst_site.grade }}">{{ worst_site.grade }}</div>
                {% else %}
                <div class="grade">-</div>
                {% endif %}
            </div>
        </div>

        <!-- Comparison Charts -->
        {% if total_urls >= 1 %}
        <div class="card">
            <h3><i class="fas fa-chart-bar card-icon"></i>Performance Comparison</h3>
            <div class="chart-container">
                <h4>Overall Performance Scores</h4>
                <img src="charts/performance_comparison.png" alt="Performance Comparison Chart" onerror="this.style.display='none'">
            </div>
            <div class="chart-container">
                <h4>Core Web Vitals Analysis</h4>
                <img src="charts/core_web_vitals_comparison.png" alt="Core Web Vitals Comparison Chart" onerror="this.style.display='none'">
            </div>
        </div>
        {% endif %}

        <!-- Individual URL Results -->
        {% for url_group in url_results %}
        <div class="card" style="margin-bottom: 40px;">
            <h2 style="color: var(--gray-800); margin-bottom: 8px; display: flex; align-items: center; gap: 12px;">
                <i class="fas fa-link" style="color: var(--primary-color);"></i>
                {{ url_group.name }}
            </h2>
            <p style="margin-bottom: 24px;">
                <strong>URL:</strong> 
                <a href="{{ url_group.url }}" target="_blank" style="color: var(--primary-color); text-decoration: none;">
                    {{ url_group.url }}
                    <i class="fas fa-external-link-alt" style="font-size: 0.8em; margin-left: 4px;"></i>
                </a>
            </p>

            <div class="device-tabs">
                {% for result in url_group.results %}
                <button class="device-tab {{ 'active' if loop.first }}" 
                        onclick="showDevice('{{ url_group.url_hash }}', '{{ result.device_mode }}')">
                    <i class="fas fa-{{ 'desktop' if result.device_mode == 'desktop' else 'mobile-alt' }}"></i>
                    {{ result.device_mode.title() }}
                </button>
                {% endfor %}
            </div>

            {% for result in url_group.results %}
            <div id="{{ url_group.url_hash }}_{{ result.device_mode }}" class="device-content {{ 'active' if loop.first }}">
                <div class="performance-overview">
                    <div class="performance-grade-card">
                        <h3 style="margin-bottom: 16px; color: var(--gray-700);">Performance Grade</h3>
                        <div class="grade {{ result.grade }}">{{ result.grade }}</div>
                        <div class="performance-score">{{ "%.1f"|format(result.score) }}%</div>
                        <div class="progress-bar" style="margin-top: 16px;">
                            <div class="progress-fill {{ 'good' if result.score >= 90 else 'needs-improvement' if result.score >= 50 else 'poor' }}" 
                                 style="width: {{ result.score }}%"></div>
                        </div>
                    </div>
                    <div>
                        <h3 style="margin-bottom: 20px; color: var(--gray-700);">
                            <i class="fas fa-heartbeat" style="color: var(--danger-color); margin-right: 8px;"></i>
                            Core Web Vitals
                        </h3>
                        <div class="metric {{ 'metric-good' if result.metrics.lcp != 'N/A' and result.metrics.lcp|float <= 2.5 else 'metric-needs-improvement' if result.metrics.lcp != 'N/A' and result.metrics.lcp|float <= 4.0 else 'metric-poor' }}">
                            <span class="metric-label">
                                <span class="tooltip" data-tooltip="Time for largest content element to render">
                                    Largest Contentful Paint (LCP)
                                </span>
                            </span>
                            <span class="metric-value">{% if result.metrics.lcp != 'N/A' %}{{ "%.2f"|format(result.metrics.lcp) }}s{% else %}N/A{% endif %}</span>
                        </div>
                        <div class="metric {{ 'metric-good' if result.metrics.fcp != 'N/A' and result.metrics.fcp|float <= 1.8 else 'metric-needs-improvement' if result.metrics.fcp != 'N/A' and result.metrics.fcp|float <= 3.0 else 'metric-poor' }}">
                            <span class="metric-label">
                                <span class="tooltip" data-tooltip="Time for first content to appear">
                                    First Contentful Paint (FCP)
                                </span>
                            </span>
                            <span class="metric-value">{% if result.metrics.fcp != 'N/A' %}{{ "%.2f"|format(result.metrics.fcp) }}s{% else %}N/A{% endif %}</span>
                        </div>
                        <div class="metric {{ 'metric-good' if result.metrics.cls != 'N/A' and result.metrics.cls|float <= 0.1 else 'metric-needs-improvement' if result.metrics.cls != 'N/A' and result.metrics.cls|float <= 0.25 else 'metric-poor' }}">
                            <span class="metric-label">
                                <span class="tooltip" data-tooltip="Measure of visual stability">
                                    Cumulative Layout Shift (CLS)
                                </span>
                            </span>
                            <span class="metric-value">{% if result.metrics.cls != 'N/A' %}{{ "%.3f"|format(result.metrics.cls) }}{% else %}N/A{% endif %}</span>
                        </div>
                        <div class="metric {{ 'metric-good' if result.metrics.inp != 'N/A' and result.metrics.inp|float <= 200 else 'metric-needs-improvement' if result.metrics.inp != 'N/A' and result.metrics.inp|float <= 500 else 'metric-poor' }}">
                            <span class="metric-label">
                                <span class="tooltip" data-tooltip="Responsiveness to user interactions">
                                    Interaction to Next Paint (INP)
                                </span>
                            </span>
                            <span class="metric-value">{% if result.metrics.inp != 'N/A' %}{{ "%.0f"|format(result.metrics.inp) }}ms{% else %}N/A{% endif %}</span>
                        </div>
                    </div>
                </div>

                <!-- Waterfall Chart -->
                {% if result.waterfall %}
                <div class="chart-container">
                    <h4><i class="fas fa-water" style="margin-right: 8px; color: var(--info-color);"></i>Network Waterfall Chart</h4>
                    <img src="waterfalls/{{ result.waterfall }}" alt="Network Waterfall Chart">
                </div>
                {% endif %}

                <!-- Screenshot -->
                {% if result.screenshot %}
                <div class="chart-container">
                    <h4><i class="fas fa-camera" style="margin-right: 8px; color: var(--success-color);"></i>Page Screenshot</h4>
                    <img src="screenshots/{{ result.screenshot }}" alt="Page Screenshot" style="max-height: 400px; object-fit: contain;">
                </div>
                {% endif %}

                <!-- Performance Recommendations -->
                {% if result.recommendations %}
                <div class="recommendations">
                    <h3 style="margin-bottom: 20px; color: var(--gray-700);">
                        <i class="fas fa-lightbulb" style="color: var(--warning-color); margin-right: 8px;"></i>
                        Performance Recommendations
                    </h3>
                    {% for rec in result.recommendations %}
                    <div class="recommendation {{ rec.severity }}">
                        <h4>
                            {{ rec.title }}
                            <span class="severity-badge {{ rec.severity }}">{{ rec.severity }}</span>
                        </h4>
                        <p style="margin-bottom: 16px; color: var(--gray-700);">{{ rec.description }}</p>
                        <ul class="suggestion-list">
                            {% for suggestion in rec.suggestions %}
                            <li>{{ suggestion }}</li>
                            {% endfor %}
                        </ul>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}

                <!-- Detailed Metrics Table -->
                <div style="margin-top: 32px;">
                    <h3 style="margin-bottom: 20px; color: var(--gray-700);">
                        <i class="fas fa-table" style="color: var(--info-color); margin-right: 8px;"></i>
                        Detailed Metrics
                    </h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Performance Score</td>
                                <td>{{ "%.1f"|format(result.metrics.performance_score) }}%</td>
                                <td><span class="severity-badge {{ 'good' if result.metrics.performance_score >= 90 else 'medium' if result.metrics.performance_score >= 50 else 'high' }}">{{ 'Good' if result.metrics.performance_score >= 90 else 'Needs Improvement' if result.metrics.performance_score >= 50 else 'Poor' }}</span></td>
                            </tr>
                            <tr>
                                <td>Accessibility Score</td>
                                <td>{{ "%.1f"|format(result.metrics.accessibility_score) }}%</td>
                                <td><span class="severity-badge {{ 'good' if result.metrics.accessibility_score >= 90 else 'medium' if result.metrics.accessibility_score >= 50 else 'high' }}">{{ 'Good' if result.metrics.accessibility_score >= 90 else 'Needs Improvement' if result.metrics.accessibility_score >= 50 else 'Poor' }}</span></td>
                            </tr>
                            <tr>
                                <td>Best Practices Score</td>
                                <td>{{ "%.1f"|format(result.metrics.best_practices_score) }}%</td>
                                <td><span class="severity-badge {{ 'good' if result.metrics.best_practices_score >= 90 else 'medium' if result.metrics.best_practices_score >= 50 else 'high' }}">{{ 'Good' if result.metrics.best_practices_score >= 90 else 'Needs Improvement' if result.metrics.best_practices_score >= 50 else 'Poor' }}</span></td>
                            </tr>
                            <tr>
                                <td>SEO Score</td>
                                <td>{{ "%.1f"|format(result.metrics.seo_score) }}%</td>
                                <td><span class="severity-badge {{ 'good' if result.metrics.seo_score >= 90 else 'medium' if result.metrics.seo_score >= 50 else 'high' }}">{{ 'Good' if result.metrics.seo_score >= 90 else 'Needs Improvement' if result.metrics.seo_score >= 50 else 'Poor' }}</span></td>
                            </tr>
                            <tr>
                                <td>Total Blocking Time</td>
                                <td>{% if result.metrics.tbt != 'N/A' %}{{ "%.0f"|format(result.metrics.tbt) }}ms{% else %}N/A{% endif %}</td>
                                <td>{% if result.metrics.tbt != 'N/A' %}<span class="severity-badge {{ 'good' if result.metrics.tbt|float <= 200 else 'medium' if result.metrics.tbt|float <= 600 else 'high' }}">{{ 'Good' if result.metrics.tbt|float <= 200 else 'Needs Improvement' if result.metrics.tbt|float <= 600 else 'Poor' }}</span>{% else %}N/A{% endif %}</td>
                            </tr>
                            <tr>
                                <td>Time to Interactive</td>
                                <td>{% if result.metrics.tti != 'N/A' %}{{ "%.2f"|format(result.metrics.tti) }}s{% else %}N/A{% endif %}</td>
                                <td>{% if result.metrics.tti != 'N/A' %}<span class="severity-badge {{ 'good' if result.metrics.tti|float <= 3.8 else 'medium' if result.metrics.tti|float <= 7.3 else 'high' }}">{{ 'Good' if result.metrics.tti|float <= 3.8 else 'Needs Improvement' if result.metrics.tti|float <= 7.3 else 'Poor' }}</span>{% else %}N/A{% endif %}</td>
                            </tr>
                            <tr>
                                <td>Speed Index</td>
                                <td>{% if result.metrics.speed_index != 'N/A' %}{{ "%.2f"|format(result.metrics.speed_index) }}s{% else %}N/A{% endif %}</td>
                                <td>{% if result.metrics.speed_index != 'N/A' %}<span class="severity-badge {{ 'good' if result.metrics.speed_index|float <= 3.4 else 'medium' if result.metrics.speed_index|float <= 5.8 else 'high' }}">{{ 'Good' if result.metrics.speed_index|float <= 3.4 else 'Needs Improvement' if result.metrics.speed_index|float <= 5.8 else 'Poor' }}</span>{% else %}N/A{% endif %}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            {% endfor %}
        </div>
        {% endfor %}

        <div class="footer">
            <p><i class="fas fa-code" style="margin-right: 8px;"></i>Report generated by GTMetrix-Style Performance Checker</p>
            <p style="margin-top: 8px; font-size: 0.875rem; opacity: 0.8;">{{ timestamp }}</p>
        </div>
    </div>

    <script>
        function showDevice(urlHash, deviceMode) {
            // Hide all device content for this URL
            const allContent = document.querySelectorAll(`[id^="${urlHash}_"]`);
            allContent.forEach(content => content.classList.remove('active'));

            // Remove active class from all tabs for this URL
            const urlTabs = document.querySelectorAll(`[onclick*="${urlHash}"]`);
            urlTabs.forEach(tab => tab.classList.remove('active'));

            // Show selected device content
            const targetContent = document.getElementById(`${urlHash}_${deviceMode}`);
            if (targetContent) {
                targetContent.classList.add('active');
            }

            // Add active class to clicked tab
            event.target.classList.add('active');
        }

        // Add smooth scrolling
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });

        // Add loading states for images with improved handling
        document.querySelectorAll('img').forEach(img => {
            // Check if image is already loaded (cached)
            if (img.complete && img.naturalHeight !== 0) {
                img.style.opacity = '1';
            } else {
                // Set initial state
                img.style.opacity = '0';
                img.style.transition = 'opacity 0.3s ease';
                
                // Handle successful load
                img.addEventListener('load', function() {
                    this.style.opacity = '1';
                });
                
                // Handle load errors
                img.addEventListener('error', function() {
                    this.style.opacity = '1';
                    this.style.border = '2px dashed #ccc';
                    this.style.padding = '20px';
                    this.alt = 'Image failed to load: ' + this.src;
                });
                
                // Fallback timeout to ensure images become visible
                setTimeout(() => {
                    if (img.style.opacity === '0') {
                        img.style.opacity = '1';
                    }
                }, 2000);
            }
        });
    </script>
</body>
</html>
        """

        # Prepare data for template
        template_data = self.prepare_template_data(results)

        # Render template
        template = Template(template_str)
        html_content = template.render(**template_data)

        # Save HTML report
        report_path = os.path.join(self.report_dir, "performance_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info(f"HTML report generated: {report_path}")
        return report_path

    def prepare_template_data(self, results):
        """Prepare data for HTML template rendering"""
        if not results:
            return {}

        # Group results by URL
        url_groups = {}
        for result in results:
            url = result['url']
            if url not in url_groups:
                url_groups[url] = {
                    'name': result['name'],
                    'url': url,
                    'url_hash': hashlib.md5(url.encode()).hexdigest()[:8],
                    'results': []
                }
            url_groups[url]['results'].append(result)

        # Calculate summary statistics
        desktop_results = [r for r in results if r['device_mode'] == 'desktop']

        avg_score = sum(r['score'] for r in desktop_results) / len(desktop_results) if desktop_results else 0
        avg_grade = 'A' if avg_score >= 90 else 'B' if avg_score >= 80 else 'C' if avg_score >= 70 else 'D' if avg_score >= 60 else 'E' if avg_score >= 50  else 'F'

        best_site = max(desktop_results, key=lambda x: x['score']) if desktop_results else None
        worst_site = min(desktop_results, key=lambda x: x['score']) if desktop_results else None

        return {
            'timestamp': datetime.datetime.now().strftime('%B %d, %Y at %H:%M'),
            'total_urls': len(url_groups),
            'avg_score': round(avg_score, 1),
            'avg_grade': avg_grade,
            'best_site': best_site,
            'worst_site': worst_site,
            'url_results': list(url_groups.values())
        }

    def save_to_historical_data(self, results):
        """Save results to historical data for dashboard tracking"""
        if not results:
            return

        # Convert results to DataFrame format
        records = []
        for result in results:
            record = {
                'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'timestamp': result['timestamp'],
                'name': result['name'],
                'url': result['url'],
                'device_mode': result['device_mode'],
                'grade': result['grade'],
                'score': result['score'],
                **result['metrics']
            }
            records.append(record)

        df = pd.DataFrame(records)

        # Save to historical data
        report_dir = self.data_manager.save_current_run_data(df)

        # Copy the HTML report and assets to the archived directory
        import shutil

        try:
            # Copy HTML report
            src_html = os.path.join(self.report_dir, "performance_report.html")
            if os.path.exists(src_html):
                shutil.copy2(src_html, os.path.join(report_dir, "performance_report.html"))

            # Copy charts, screenshots, waterfalls directories
            for asset_dir in ['charts', 'screenshots', 'waterfalls']:
                src_dir = os.path.join(self.report_dir, asset_dir)
                dst_dir = os.path.join(report_dir, asset_dir)
                if os.path.exists(src_dir):
                    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

            logging.info(f"Report assets archived to: {report_dir}")

        except Exception as e:
            logging.error(f"Error archiving report assets: {str(e)}")

    def run_analysis(self, urls, device_modes=['desktop', 'mobile'], max_workers=5, runs=3):
        """Run complete performance analysis"""
        logging.info(f"Starting performance analysis for {len(urls)} URLs")

        # Process all URLs
        results = self.process_urls(urls, device_modes, max_workers, runs)

        if not results:
            logging.error("No results generated")
            return None

        # Generate comparison charts
        self.generate_comparison_charts(results)

        # Generate HTML report
        report_path = self.generate_html_report(results)

        # Save to historical data
        self.save_to_historical_data(results)

        logging.info(f"Performance analysis completed. Report saved to: {report_path}")
        return {
            'report_path': report_path,
            'report_dir': self.report_dir,
            'results': results
        }


    def detect_resource_type(self, url, resource_type, mime_type):
        """Enhanced resource type detection"""
        # Normalize the resource type from Lighthouse
        lighthouse_type_map = {
            'Document': 'Document',
            'Stylesheet': 'Stylesheet',
            'Script': 'Script',
            'Image': 'Image',
            'Font': 'Font',
            'Media': 'Media',
            'TextTrack': 'Media',
            'XHR': 'XHR',
            'Fetch': 'Fetch',
            'EventSource': 'XHR',
            'WebSocket': 'WebSocket',
            'Manifest': 'Manifest',
            'SignedExchange': 'Document',
            'Ping': 'Other',
            'CSPViolationReport': 'Other',
            'Other': 'Other'
        }

        # First try Lighthouse resource type
        if resource_type in lighthouse_type_map:
            return lighthouse_type_map[resource_type]

        # Fallback to MIME type detection
        if mime_type:
            if mime_type.startswith('text/html'):
                return 'Document'
            elif mime_type.startswith('text/css'):
                return 'Stylesheet'
            elif 'javascript' in mime_type or mime_type.startswith('application/javascript'):
                return 'Script'
            elif mime_type.startswith('image/'):
                return 'Image'
            elif 'font' in mime_type:
                return 'Font'
            elif mime_type.startswith(('video/', 'audio/')):
                return 'Media'
            elif mime_type.startswith('application/json'):
                return 'XHR'

        # Fallback to URL-based detection
        url_lower = url.lower()
        if any(ext in url_lower for ext in ['.css']):
            return 'Stylesheet'
        elif any(ext in url_lower for ext in ['.js', '.mjs']):
            return 'Script'
        elif any(ext in url_lower for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico']):
            return 'Image'
        elif any(ext in url_lower for ext in ['.woff', '.woff2', '.ttf', '.otf', '.eot']):
            return 'Font'
        elif any(ext in url_lower for ext in ['.mp4', '.webm', '.ogg', '.mp3', '.wav']):
            return 'Media'
        elif '/api/' in url_lower or url_lower.endswith('.json'):
            return 'XHR'

        return 'Other'

    def extract_timing_details(self, request_data):
        """Extract detailed timing information from request data"""
        timing = {}

        # Extract timing details if available in the request
        if 'timing' in request_data:
            timing_data = request_data['timing']

            # DNS lookup time
            if 'dnsStart' in timing_data and 'dnsEnd' in timing_data:
                timing['dns'] = max(0, (timing_data['dnsEnd'] - timing_data['dnsStart']) * 1000)

            # Connection time
            if 'connectStart' in timing_data and 'connectEnd' in timing_data:
                timing['connect'] = max(0, (timing_data['connectEnd'] - timing_data['connectStart']) * 1000)

            # SSL handshake time
            if 'sslStart' in timing_data and 'sslEnd' in timing_data and timing_data['sslStart'] > 0:
                timing['ssl'] = max(0, (timing_data['sslEnd'] - timing_data['sslStart']) * 1000)

            # Request/response time
            if 'sendStart' in timing_data and 'receiveHeadersEnd' in timing_data:
                timing['request'] = max(0, (timing_data['receiveHeadersEnd'] - timing_data['sendStart']) * 1000)

        return timing

    def format_bytes(self, bytes_value):
        """Format bytes into human readable format"""
        if bytes_value == 0:
            return "0B"

        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f}TB"


def main():
    parser = argparse.ArgumentParser(description="GTMetrix-Style Website Performance Checker")
    parser.add_argument("--urls", nargs="+", help="List of URLs to analyze")
    parser.add_argument("--file", help="File containing URLs (CSV, JSON, or TXT)")
    parser.add_argument("--output", help="Output directory for reports")
    parser.add_argument("--devices", nargs="+", choices=['desktop', 'mobile'],
                       default=['desktop', 'mobile'], help="Device modes to test")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--runs", type=int, default=3, help="Number of Lighthouse runs per URL for averaging")
    parser.add_argument("--crux-api-key", help="Chrome User Experience Report API key")

    args = parser.parse_args()

    # Set CrUX API key if provided
    if args.crux_api_key:
        global CRUX_API_KEY
        CRUX_API_KEY = args.crux_api_key

    # Initialize checker
    checker = GTMetrixStyleChecker(output_dir=args.output)

    # Load URLs
    urls = []

    if args.file:
        urls = checker.load_urls_from_file(args.file)
    elif args.urls:
        urls = [{'url': url, 'name': url} for url in args.urls]
    else:
        print("Please provide URLs either via --urls or --file parameter")
        return

    if not urls:
        print("No valid URLs found")
        return

    # Run analysis
    results = checker.run_analysis(urls, args.devices, args.workers, args.runs)

    if results:
        print(f"\nAnalysis completed successfully!")
        print(f"Report available at: {results['report_path']}")
        print(f"Report directory: {results['report_dir']}")
    else:
        print("Analysis failed")


if __name__ == "__main__":
    main()
