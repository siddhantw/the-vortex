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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
import seaborn as sns
from bs4 import BeautifulSoup
from jinja2 import Template
from selenium import webdriver

# Import the historical data manager
from historical_data_manager import HistoricalDataManager

# Paths to Lighthouse and Node.js
LIGHTHOUSE_PATH = os.path.expanduser('~/.nvm/versions/node/v23.1.0/bin/lighthouse')
NODE_PATH = os.path.expanduser('~/.nvm/versions/node/v23.1.0/bin/node')

# Google PageSpeed Insights API key (encoded for security)
ENCODED_API_KEY = "QUl6YVN5Q3c4Q3c1TmF5RWdIYzJ4bnRWRXdqN2J6dzdqd1l6TUU4"


# Function to get the API key
def get_api_key():
    # Check if API key is available in environment variable (more secure option)
    env_api_key = os.environ.get('PAGESPEED_API_KEY')
    if env_api_key:
        return env_api_key
    # Fallback to the encoded key
    return base64.b64decode(ENCODED_API_KEY).decode('utf-8')


# List of brands and their URLs
BRANDS = [
    {"name": "Bluehost", "url": "https://www.bluehost.com/"},
    {"name": "Domain", "url": "https://www.domain.com/"},
    {"name": "HostGator", "url": "https://www.hostgator.com/"},
    {"name": "Network Solutions", "url": "https://www.networksolutions.com/"},
    {"name": "Register", "url": "https://www.register.com/"},
    {"name": "Web", "url": "https://www.web.com/"}
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for detailed traceability
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Log to console
)


# Function to run Lighthouse and return the report as JSON
def run_lighthouse(url, is_mobile=False, mode=None, page=None, retries=3, runs=3):
    # Set the appropriate mode if not provided
    if mode is None:
        mode = 'mobile' if is_mobile else 'desktop'
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument('--disable-gpu')
    chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

    # Add specific options to improve INP measurement accuracy
    chrome_options.add_argument("--enable-blink-features=InteractionToNextPaint")
    chrome_options.add_argument("--enable-features=InteractionToNextPaint")

    # Add window size for desktop mode
    if mode == 'desktop':
        chrome_options.add_argument("--window-size=1920,1080")

    # Use a temporary directory for Chrome user data and Lighthouse output
    with tempfile.TemporaryDirectory(prefix=f"chrome_user_data_{os.getpid()}_") as user_data_dir, \
         tempfile.TemporaryDirectory(prefix=f"lighthouse_output_{os.getpid()}_") as output_dir:

        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

        # Join all Chrome options into a single string for the Lighthouse command
        chrome_flags_str = " ".join(chrome_options.arguments)

        # Basic configuration that's common for both modes
        base_command = [
            NODE_PATH, LIGHTHOUSE_PATH,
            url,
            '--output=json',
            '--quiet',
            f'--chrome-flags={chrome_flags_str}',
            # Include all categories, not just performance
            '--only-categories=performance,accessibility,best-practices,seo,pwa',
            # Set the output directory to avoid file permission issues
            f'--output-path={os.path.join(output_dir, "report.json")}',
        ]

        if mode == 'mobile':
            # Mobile mode configuration - keep existing throttling parameters
            command = base_command + [
                '--emulated-form-factor=mobile',
                '--throttling.cpuSlowdownMultiplier=4',
                '--throttling.rttMs=150',
                '--throttling.throughputKbps=1638'
            ]
        else:
            # Desktop mode configuration - keep existing throttling parameters
            command = base_command + [
                '--emulated-form-factor=desktop',
                '--throttling.cpuSlowdownMultiplier=1',
                '--throttling.rttMs=40',
                '--throttling.throughputKbps=10240'
            ]

        logging.info(f"Running Lighthouse in {mode} mode for {url} with {runs} runs")
        logging.debug(f"Lighthouse command: {' '.join(command)}")

        # Store all successful runs
        successful_reports = []

        attempt = 0
        while attempt < retries and len(successful_reports) < runs:
            try:
                logging.info(f"Running Lighthouse for {url} in {mode} mode (Run {len(successful_reports)+1}/{runs}, Attempt {attempt + 1}/{retries})...")

                # Create a specific directory for this run to avoid file conflicts
                run_dir = os.path.join(output_dir, f"run_{len(successful_reports)+1}")
                os.makedirs(run_dir, exist_ok=True)

                # Update the command with the run-specific output path
                run_command = command.copy()
                output_path_index = next((i for i, item in enumerate(run_command) if item.startswith('--output-path=')), None)
                if output_path_index is not None:
                    run_command[output_path_index] = f'--output-path={os.path.join(run_dir, "report.json")}'

                # Run Lighthouse
                result = subprocess.run(run_command, check=True, capture_output=True, text=True)

                # Read the report from the output file
                report_path = os.path.join(run_dir, "report.json")
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()

                    # Parse the report JSON
                    try:
                        report = json.loads(report_content)
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON in Lighthouse output file for {url} in {mode} mode")
                        attempt += 1
                        continue
                else:
                    # If output file doesn't exist but process completed, try to parse stdout
                    try:
                        report = json.loads(result.stdout)
                    except json.JSONDecodeError:
                        logging.error(f"Invalid JSON in Lighthouse stdout for {url} in {mode} mode")
                        attempt += 1
                        continue

                # Verify the report has the correct mode set
                report_mode = report.get('configSettings', {}).get('emulatedFormFactor')
                if (report_mode == 'mobile' and mode != 'mobile') or (report_mode == 'desktop' and mode != 'desktop'):
                    logging.warning(f"Mode mismatch in report: {report_mode} vs requested {mode}")

                # Add specific debug logging for accessibility, best practices, and SEO scores
                categories = report.get('categories', {})
                a11y_score = categories.get('accessibility', {}).get('score', 0) * 100
                bp_score = categories.get('best-practices', {}).get('score', 0) * 100
                seo_score = categories.get('seo', {}).get('score', 0) * 100
                perf_score = categories.get('performance', {}).get('score', 0) * 100
                logging.info(f"RUN {len(successful_reports)+1} SCORES - Mode: {mode}, URL: {url}, Performance: {perf_score:.0f}, A11y: {a11y_score:.0f}, BP: {bp_score:.0f}, SEO: {seo_score:.0f}")

                # Store the mode in the report for later reference
                if 'configSettings' not in report:
                    report['configSettings'] = {}
                report['configSettings']['mode'] = mode

                # Add the report to our successful runs
                successful_reports.append(report)

                # If we have enough runs, break out
                if len(successful_reports) >= runs:
                    break

                # Add delay between runs for the same URL to ensure varied samples
                time.sleep(random.uniform(1.5, 3.0))

            except subprocess.CalledProcessError as e:
                logging.error(f"Error running Lighthouse for {url} in {mode} mode: {e}")
                if hasattr(e, 'stderr') and e.stderr:
                    logging.error(f"Command stderr: {e.stderr}")
                if hasattr(e, 'stdout') and e.stdout:
                    logging.error(f"Command output: {e.stdout}")

                # Log the full command for debugging
                logging.error(f"Command that failed: {' '.join(run_command)}")
                attempt += 1
            except Exception as e:
                logging.error(f"Unexpected error running Lighthouse for {url} in {mode} mode: {e}")
                attempt += 1

            if attempt < retries and len(successful_reports) < runs:
                logging.warning(f"Retrying Lighthouse for {url} in {mode} mode...")
                time.sleep(2)  # Add a short delay before retrying

        if not successful_reports:
            logging.error(f"Failed to run Lighthouse for {url} in {mode} mode after {retries} attempts.")
            return None

        # If we have multiple successful runs, average the results
        if len(successful_reports) > 1:
            return average_lighthouse_reports(successful_reports)

        # Otherwise, return the single successful report
        return successful_reports[0]


def average_lighthouse_reports(reports):
    """
    Average multiple Lighthouse reports to get more accurate results.
    Particularly important for INP and other variability-prone metrics.
    """
    if not reports:
        return None

    # Use the first report as a template
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
        "interactive"
    ]

    # Average the audit scores
    for metric in metrics_to_avg:
        if metric not in avg_report.get("audits", {}):
            continue

        values = []
        numeric_values = []

        # Collect values from all reports
        for report in reports:
            if metric in report.get("audits", {}):
                num_value = report["audits"][metric].get("numericValue")
                if num_value is not None:
                    numeric_values.append(num_value)

        # Calculate average if we have values
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            avg_report["audits"][metric]["numericValue"] = avg_value

            # Update display value based on the average numeric value
            if metric == "cumulative-layout-shift":
                avg_report["audits"][metric]["displayValue"] = f"{avg_value:.3f}"
            elif metric in ["largest-contentful-paint", "first-contentful-paint", "speed-index", "interactive", "interaction-to-next-paint"]:
                avg_report["audits"][metric]["displayValue"] = f"{avg_value/1000:.1f} s"
            elif metric == "total-blocking-time":
                avg_report["audits"][metric]["displayValue"] = f"{avg_value:.0f} ms"

    # Average the category scores
    for category in avg_report.get("categories", {}):
        scores = []
        for report in reports:
            if category in report.get("categories", {}):
                score = report["categories"][category].get("score")
                if score is not None:
                    scores.append(score)

        if scores:
            avg_report["categories"][category]["score"] = sum(scores) / len(scores)

    logging.info(f"Averaged {len(reports)} Lighthouse reports for more accurate results")
    return avg_report


# Function to fetch Google PageSpeed Insights data
def fetch_pagespeed_insights(url, strategy):
    print(f"Fetching PageSpeed Insights for {url} in {strategy} mode...")
    api_key = get_api_key()
    api_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&strategy={strategy}&key={api_key}"
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    print(f"Error fetching PageSpeed Insights for {url}: {response.status_code}")
    return None


# Recursive function to crawl URLs up to max_depth
def crawl_urls(base_url, max_depth, max_workers=6, delay=1, visited_urls=None):
    if max_workers <= 0:
        raise ValueError("max_workers must be greater than 0")

    logging.info(f"Starting crawl with max_depth={max_depth}")

    if visited_urls is None:
        visited_urls = set()  # Initialize visited_urls if not provided

    results = []
    url_queue = Queue()
    url_queue.put((base_url, 0))  # Add the initial URL with depth 0
    visited_urls.add(base_url)  # Mark the base URL as visited immediately

    processed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Initial processing of the base URL
        future = executor.submit(crawl_single_url, base_url, 0, max_depth, visited_urls)
        futures[future] = (base_url, 0)

        # Process URLs as they are discovered
        while futures:
            # Wait for the next future to complete
            done, _ = futures, {}
            for future in as_completed(list(futures.keys())):
                url, depth = futures.pop(future)
                try:
                    new_results, new_links = future.result()
                    if new_results:
                        results.extend(new_results)
                        processed_count += 1

                    # Only process new links if we're below max_depth
                    if depth < max_depth:
                        for link, new_depth in new_links:
                            if link not in visited_urls and new_depth <= max_depth:
                                visited_urls.add(link)
                                future = executor.submit(crawl_single_url, link, new_depth, max_depth, visited_urls)
                                futures[future] = (link, new_depth)

                except Exception as e:
                    logging.error(f"Error processing future for {url}: {e}")

                # Add a small delay to avoid overwhelming the server
                time.sleep(delay)

    logging.info(f"Crawling completed. Processed {processed_count} URLs with max_depth={max_depth}")
    return results


def crawl_single_url(url, current_depth, max_depth, visited_urls):
    """Crawl a single URL and return results and new links to process."""
    logging.info(f"Crawling: {url} at depth {current_depth}/{max_depth}")

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logging.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return [], []

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract links
        links = []
        if current_depth < max_depth:  # Only extract links if we're not at max depth
            raw_links = [a['href'] for a in soup.find_all('a', href=True)]
            # Process links
            for link in raw_links:
                # Standardize link
                full_link = link if link.startswith("http") else requests.compat.urljoin(url, link)
                full_link = full_link.split('#')[0]  # Remove fragments

                # Add link if it's not already visited
                if full_link not in visited_urls:
                    links.append((full_link, current_depth + 1))

        # Return the current URL result and links to process
        return [{"url": url, "depth": current_depth, "status_code": response.status_code}], links

    except Exception as e:
        logging.error(f"Error crawling {url}: {e}")
        return [], []


def clean_metric_value(value):
    if value == "N/A":
        return value
    # Remove unwanted characters and known units
    cleaned_value = value.replace(",", "").replace("\xa0", "").replace("Â", "").replace("s", "").replace("ms",
                                                                                                         "").strip()
    # Handle unexpected units like 'm'
    if cleaned_value.endswith("m"):
        cleaned_value = cleaned_value[:-1]  # Remove the 'm'
    try:
        return float(cleaned_value)
    except ValueError:
        logging.warning(f"Could not convert value to float: {value}")
        return 0.0


def generate_recommendations(metrics):
    recommendations = []

    # FCP Recommendations
    if metrics.get("fcp") != "N/A" and clean_metric_value(metrics["fcp"]) > 2.5:
        recommendations.append("- Optimize critical CSS and JavaScript to improve First Contentful Paint (FCP).")

    # LCP Recommendations
    if metrics.get("lcp") != "N/A" and clean_metric_value(metrics["lcp"]) > 2.5:
        recommendations.append(
            "- Optimize images, improve server response times, and use a Content Delivery Network (CDN) to reduce Largest Contentful Paint (LCP).")

    # CLS Recommendations
    if metrics.get("cls") != "N/A" and clean_metric_value(metrics["cls"]) > 0.1:
        recommendations.append(
            "- Avoid layout shifts by using fixed dimensions for images and ads, and avoid injecting content dynamically above existing content.")

    # INP Recommendations
    if metrics.get("inp") != "N/A" and clean_metric_value(metrics["inp"]) > 200:
        recommendations.append(
            "- Improve Interaction to Next Paint (INP) by optimizing event handlers, using requestAnimationFrame for visual updates, and reducing main thread blocking during interactions.")

    # TBT Recommendations
    if metrics.get("tbt") != "N/A" and clean_metric_value(metrics["tbt"]) > 300:
        recommendations.append(
            "- Reduce unused JavaScript, optimize third-party scripts, and minimize main-thread work to lower Total Blocking Time (TBT).")

    # TTI Recommendations
    if metrics.get("tti") != "N/A" and clean_metric_value(metrics["tti"]) > 5:
        recommendations.append(
            "- Reduce JavaScript execution time, defer non-critical scripts, and use code-splitting to improve Time to Interactive (TTI).")

    # General Recommendations
    if not recommendations:
        recommendations.append("- Your site is performing well. Continue monitoring and optimizing as needed.")

    return "<br>".join(recommendations)


# Helper function to extract INP metric with better fallback handling
def self_extract_inp_metric(audits_data):
    if not audits_data:
        return "N/A"

    # Try all possible field names for INP
    inp_audit = audits_data.get("interaction-to-next-paint")
    if inp_audit:
        logging.debug(f"Found INP audit with title: {inp_audit.get('title')}")
        display_value = inp_audit.get("displayValue")
        if display_value:
            return display_value
        else:
            # If displayValue is missing but we have numericValue, format it ourselves
            numeric_value = inp_audit.get("numericValue")
            if numeric_value is not None:
                # Convert from microseconds to milliseconds if needed
                if numeric_value > 10000:  # Likely in microseconds
                    return f"{numeric_value / 1000:.1f} ms"
                else:
                    return f"{numeric_value:.1f} ms"

    # Check for alternate field names
    alternate_names = ["experimental-interaction-to-next-paint", "max-potential-fid"]
    for name in alternate_names:
        alt_audit = audits_data.get(name)
        if alt_audit and alt_audit.get("displayValue"):
            logging.debug(f"Using alternate metric {name} for INP")
            return alt_audit.get("displayValue")

    # Last resort: check if INP data is in the main metrics section
    metrics_audit = audits_data.get("metrics", {}).get("details", {}).get("items", [{}])[0]
    if metrics_audit:
        for key in ["interactionToNextPaint", "experimental-interaction-to-next-paint"]:
            if key in metrics_audit:
                value = metrics_audit[key]
                if value is not None:
                    logging.debug(f"Found INP in metrics section: {value}")
                    return f"{value:.1f} ms"

    return "N/A"

def extract_metrics_from_lighthouse(report):
    try:
        audits = report.get("audits", {})
        categories = report.get("categories", {})
        mode = report.get('configSettings', {}).get('mode', 'desktop')  # Get the current mode

        # Dump report structure for debugging if categories are still missing
        if not categories or len(categories) < 3:  # Changed from 4 to 3 as PWA might be missing
            logging.debug(f"Report categories: {list(categories.keys())}")
            logging.debug(f"Report structure: {json.dumps(list(report.keys()))}")
            logging.debug(f"Report mode: {mode}")

        # More robust method to get scores from categories
        def safe_score(category_key):
            category = categories.get(category_key, {})
            if not category:
                logging.warning(f"Category {category_key} not found in report for {mode} mode")
                return 0
            score = category.get("score")
            if score is None:
                logging.warning(f"Score not found in category {category_key} for {mode} mode")
                return 0
            return round(score * 100, 0)

        # Handle case where audits might be missing specific metrics
        metrics = {
            "fcp": audits.get("first-contentful-paint", {}).get("displayValue", "N/A"),
            "lcp": audits.get("largest-contentful-paint", {}).get("displayValue", "N/A"),
            "inp": self_extract_inp_metric(audits),
            "tti": audits.get("interactive", {}).get("displayValue", "N/A"),
            "speed_index": audits.get("speed-index", {}).get("displayValue", "N/A"),
            "tbt": audits.get("total-blocking-time", {}).get("displayValue", "N/A"),
            "cls": audits.get("cumulative-layout-shift", {}).get("displayValue", "N/A")
        }

        # Extract additional metrics for new columns with fallbacks
        additional_metrics = {
            "page_load_time": audits.get("load-fast-enough-for-pwa", {}).get("displayValue",
                                                                             audits.get("interactive", {}).get(
                                                                                 "displayValue", "N/A")),
            "server_response_time": audits.get("server-response-time", {}).get("displayValue", "N/A"),
            "render_blocking_resources": audits.get("render-blocking-resources", {}).get("displayValue",
                                                                                         audits.get(
                                                                                             "render-blocking-time",
                                                                                             {}).get("displayValue",
                                                                                                     "N/A")),
            "unused_javascript": audits.get("unused-javascript", {}).get("displayValue", "N/A"),
            "dom_size": audits.get("dom-size", {}).get("displayValue", "N/A"),
            "mainthread_work": audits.get("mainthread-work-breakdown", {}).get("displayValue", "N/A"),
            "diagnostics": audits.get("diagnostics", {}).get("details", {}).get("items", [{}])[0].get("totalByteWeight",
                                                                                                      "N/A")
        }

        # Add a formatted version of the diagnostics information
        try:
            diag_data = audits.get("diagnostics", {}).get("details", {}).get("items", [{}])[0]
            if diag_data:
                diag_items = []
                # Format network requests
                if "numRequests" in diag_data:
                    diag_items.append(f"Requests: {diag_data.get('numRequests')}")
                # Format total byte weight
                if "totalByteWeight" in diag_data:
                    byte_weight = diag_data.get("totalByteWeight")
                    if byte_weight > 1024 * 1024:
                        diag_items.append(f"Total Size: {byte_weight / (1024 * 1024):.2f} MB")
                    else:
                        diag_items.append(f"Total Size: {byte_weight / 1024:.2f} KB")
                # Format main thread time
                if "mainThreadTime" in diag_data:
                    diag_items.append(f"Main Thread: {diag_data.get('mainThreadTime'):.2f} ms")

                additional_metrics["diagnostics_summary"] = "<br>".join(diag_items) if diag_items else "N/A"
            else:
                additional_metrics["diagnostics_summary"] = "N/A"
        except Exception as e:
            logging.warning(f"Error extracting diagnostics summary: {e}")
            additional_metrics["diagnostics_summary"] = "N/A"

        # Calculate Core Web Vitals score (average of LCP, CLS, INP, and TBT)
        cwv_metrics = []

        # Get numeric values for core web vitals
        try:
            lcp_value = audits.get("largest-contentful-paint", {}).get("numericValue", 0) / 1000  # Convert to seconds
            cls_value = audits.get("cumulative-layout-shift", {}).get("numericValue", 0)
            tbt_value = audits.get("total-blocking-time", {}).get("numericValue", 0)
            inp_value = audits.get("interaction-to-next-paint", {}).get("numericValue", 0) / 1000  # Convert to seconds

            # Normalize values to 0-100 scale (higher is better)
            lcp_score = max(0, min(100, 100 - (lcp_value - 2.5) * 40)) if lcp_value > 0 else 0
            cls_score = max(0, min(100, 100 - (cls_value - 0.1) * 400)) if cls_value > 0 else 0
            tbt_score = max(0, min(100, 100 - (tbt_value - 200) / 6)) if tbt_value > 0 else 0
            inp_score = max(0, min(100, 100 - (inp_value - 0.1) * 1000)) if inp_value > 0 else 0

            # Add to metrics list if valid
            if lcp_score > 0:
                cwv_metrics.append(lcp_score)
            if cls_score > 0:
                cwv_metrics.append(cls_score)
            if tbt_score > 0:
                cwv_metrics.append(tbt_score)
            if inp_score > 0:
                cwv_metrics.append(inp_score)

            # Calculate overall CWV score
            cwv_score = round(sum(cwv_metrics) / len(cwv_metrics)) if cwv_metrics else 0
            additional_metrics["core_web_vitals_score"] = cwv_score
        except Exception as e:
            logging.warning(f"Error calculating Core Web Vitals score: {e}")
            additional_metrics["core_web_vitals_score"] = 0

        # Check for empty or invalid metrics and use fallbacks
        for key, value in metrics.items():
            if not value or value == "undefined" or value == "null":
                metrics[key] = "N/A"
                logging.debug(f"Metric {key} had invalid value: {value}, using N/A")

        for key, value in additional_metrics.items():
            if not value or value == "undefined" or value == "null":
                additional_metrics[key] = "N/A"
                logging.debug(f"Additional metric {key} had invalid value: {value}, using N/A")

        # Get scores with better error handling - improved to ensure all categories are captured
        scores = {}
        # Core categories that should always be present
        core_categories = ["performance", "accessibility", "best-practices", "seo"]
        # Optional categories that might be missing
        optional_categories = ["pwa"]

        # Process core categories
        for category_key in core_categories:
            try:
                scores[category_key.replace("-", "_")] = safe_score(category_key)
                logging.debug(
                    f"Extracted {category_key} score: {scores[category_key.replace('-', '_')]} for {mode} mode")
            except Exception as e:
                logging.warning(f"Error getting {category_key} score: {e}")
                scores[category_key.replace("-", "_")] = 0

        # Process optional categories
        for category_key in optional_categories:
            try:
                if category_key in categories:
                    scores[category_key.replace("-", "_")] = safe_score(category_key)
                    logging.debug(
                        f"Extracted optional {category_key} score: {scores[category_key.replace('-', '_')]} for {mode} mode")
                else:
                    logging.debug(f"Optional category {category_key} not found in report for {mode} mode, using default score 0")
                    scores[category_key.replace("-", "_")] = 0
            except Exception as e:
                logging.warning(f"Error getting optional {category_key} score for {mode} mode: {e}")
                scores[category_key.replace("-", "_")] = 0

        # Merge metrics and additional_metrics for return
        metrics.update(additional_metrics)

        # Ensure the mode information is preserved in the returned data
        scores["mode"] = mode

        return metrics, scores
    except Exception as e:
        logging.error(f"Error extracting metrics: {e}")
        import traceback
        logging.error(traceback.format_exc())  # Log the full traceback for debugging
        # Return default empty values instead of failing
        return {
            "fcp": "N/A",
            "lcp": "N/A",
            "inp": "N/A",
            "tti": "N/A",
            "speed_index": "N/A",
            "tbt": "N/A",
            "cls": "N/A",
            "page_load_time": "N/A",
            "core_web_vitals_score": 0,
            "server_response_time": "N/A",
            "render_blocking_resources": "N/A",
            "unused_javascript": "N/A",
            "dom_size": "N/A",
            "mainthread_work": "N/A",
            "diagnostics": "N/A",
            "diagnostics_summary": "N/A"
        }, {
            "performance": 0,
            "accessibility": 0,
            "best_practices": 0,
            "seo": 0,
            "pwa": 0,
            "mode": "unknown"  # Add mode information in the default case too
        }


def generate_html_report(data, output_path):
    print("Generating HTML report...")

    # Get unique brands for the brand filter dropdown
    unique_brands = sorted(list(set(result["brand"] for result in data)))

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>FOS Report - Website Performance Scan with PSI & Lighthouse</title>
        <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css">
        <link rel="stylesheet" href="https://cdn.datatables.net/colreorder/1.5.6/css/dataTables.colReorder.min.css">
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.html5.min.js"></script>
        <script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.print.min.js"></script>
        <script src="https://cdn.datatables.net/colreorder/1.5.6/js/dataTables.colReorder.min.js"></script>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 40px;
                background-color: #f9f9f9;
                color: #333;
            }
        
            h1 {
                text-align: center;
                font-size: 2em;
                color: #4CAF50;
                margin-bottom: 20px;
            }
        
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 1em;
                background-color: #fff;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
        
            th, td {
                padding: 12px 15px;
                text-align: left;
                border: 1px solid #ddd;
            }
        
            th {
                background: linear-gradient(to right, #4CAF50, #81C784);
                color: white;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 1;
            }
        
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        
            tr:hover {
                background-color: #f1f1f1;
            }
        
            .score-poor {
                background-color: #f8d7da !important;
                color: #721c24;
            }
    
            .score-needs-improvement {
                background-color: #fff3cd !important;
                color: #856404;
            }
    
            .score-good {
                background-color: #d1e7dd !important;
                color: #0f5132;
            }
        
            .filter-controls {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .filter-group {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            select {
                padding: 8px 12px;
                font-size: 1em;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            
            .notes {
                margin-top: 20px;
                padding: 10px;
                background-color: #e9f5e9;
                border-left: 4px solid #4CAF50;
                border-radius: 4px;
            }
            
            .cwv-score {
                display: inline-block;
                width: 40px;
                height: 40px;
                line-height: 40px;
                text-align: center;
                border-radius: 50%;
                color: white;
                font-weight: bold;
            }
            
            .cwv-good {
                background-color: #0cce6b;
            }
            
            .cwv-average {
                background-color: #ffa400;
            }
            
            .cwv-poor {
                background-color: #ff4e42;
            }
            
            .column-toggle {
                margin-top: 10px;
                margin-bottom: 20px;
            }
            
            .column-toggle button {
                margin-right: 5px;
                padding: 5px 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            
            .column-toggle button:hover {
                background-color: #3e8e41;
            }
            
            .methodology-section {
                margin-top: 30px;
                padding: 15px;
                background-color: #e6f7ff;
                border-left: 4px solid #1890ff;
                border-radius: 4px;
            }
            
            .thresholds-table {
                width: auto;
                margin: 15px 0;
            }
            
            .thresholds-table th,
            .thresholds-table td {
                padding: 8px 15px;
                text-align: center;
            }
            
            .thresholds-table .good {
                background-color: #d1e7dd;
            }
            
            .thresholds-table .needs-improvement {
                background-color: #fff3cd;
            }
            
            .thresholds-table .poor {
                background-color: #f8d7da;
            }
        </style>
    </head>
    <body>
        <h1>FOS Report - Website Performance Scan with PSI & Lighthouse</h1>
        
        <div class="filter-controls">
            <div class="filter-group">
                <label for="viewFilter">Device:</label>
                <select id="viewFilter">
                    <option value="all" selected>All</option>
                    <option value="desktop">Desktop</option>
                    <option value="mobile">Mobile</option>
                </select>
            </div>
            
            <div class="filter-group">
                <label for="brandFilter">Brand:</label>
                <select id="brandFilter">
                    <option value="all" selected>All</option>
                    {% for brand in brands %}
                    <option value="{{ brand }}">{{ brand }}</option>
                    {% endfor %}
                </select>
            </div>
        </div>
        
        <div class="column-toggle">
            <button id="toggleBasic">Basic Info</button>
            <button id="toggleCoreVitals">Core Web Vitals</button>
            <button id="togglePerformance">Performance Metrics</button>
            <button id="toggleDiagnostics">Diagnostics</button>
            <button id="toggleScores">Audit Scores</button>
            <button id="toggleAll">Show All</button>
        </div>
        
        <div class="notes">
            <p><strong>Note:</strong> The performance scores shown here are based on the official Lighthouse 
            configuration using default settings. Mobile scores are generated using a 4× CPU slowdown and 
            Fast 3G network throttling, while desktop scores use standard desktop throttling. These settings 
            match the same configuration used by Google PageSpeed Insights.</p>
            <p><strong>Core Web Vitals Score</strong> is calculated based on LCP, CLS, INP, and TBT metrics to provide 
            an overall assessment of the page's real-world user experience.</p>
            <p><strong>Interaction to Next Paint (INP)</strong> is now a Core Web Vital metric that replaced 
            First Input Delay (FID). INP measures the time from when a user first interacts with a page to when 
            the browser is able to respond to that interaction by rendering the next frame.</p>
            <p><strong>Recommendations</strong> are provided based on the Lighthouse report and may include
            suggestions for improving performance, accessibility, and best practices.</p>
            <p><strong>Diagnostics Summary</strong> provides insights into the main thread work, server response
            time, and other performance-related metrics. This information can help identify potential bottlenecks
            and areas for optimization.</p>
        </div>
        
        <div class="methodology-section">
            <h3>Core Web Vitals Thresholds</h3>
            <p>The following thresholds are used to evaluate Core Web Vitals performance:</p>
            <table class="thresholds-table">
                <tr>
                    <th>Metric</th>
                    <th class="good">Good</th>
                    <th class="needs-improvement">Needs Improvement</th>
                    <th class="poor">Poor</th>
                </tr>
                <tr>
                    <td>LCP (Largest Contentful Paint)</td>
                    <td class="good">≤ 2.5s</td>
                    <td class="needs-improvement">2.5s - 4s</td>
                    <td class="poor">> 4s</td>
                </tr>
                <tr>
                    <td>CLS (Cumulative Layout Shift)</td>
                    <td class="good">≤ 0.1</td>
                    <td class="needs-improvement">0.1 - 0.25</td>
                    <td class="poor">> 0.25</td>
                </tr>
                <tr>
                    <td>INP (Interaction to Next Paint)</td>
                    <td class="good">≤ 200ms</td>
                    <td class="needs-improvement">200ms - 500ms</td>
                    <td class="poor">> 500ms</td>
                </tr>
                <tr>
                    <td>TBT (Total Blocking Time)</td>
                    <td class="good">≤ 200ms</td>
                    <td class="needs-improvement">200ms - 600ms</td>
                    <td class="poor">> 600ms</td>
                </tr>
            </table>
            <p><strong>Note:</strong> The scores and recommendations are based on the Lighthouse report and may
            vary based on the specific configuration and settings used during the scan. Always refer to the
            official Lighthouse documentation for the most accurate and up-to-date information.</p>
        </div>
        
        <table id="reportTable" class="display" style="width:100%">
            <thead>
                <tr>
                    <!-- Basic Info Columns -->
                    <th data-column-group="basic">Brand</th>
                    <th data-column-group="basic">URL</th>
                    <th data-column-group="basic">Depth</th>
                    <th data-column-group="basic">Mode</th>
                    
                    <!-- Core Web Vitals Columns -->
                    <th data-column-group="core">FCP</th>
                    <th data-column-group="core">LCP</th>
                    <th data-column-group="core">CLS</th>
                    <th data-column-group="core">INP</th>
                    <th data-column-group="core">Core Web Vitals Score</th>
                    
                    <!-- Performance Metrics -->
                    <th data-column-group="performance">TTI</th>
                    <th data-column-group="performance">Speed Index</th>
                    <th data-column-group="performance">TBT</th>
                    <th data-column-group="performance">Page Load Time</th>
                    
                    <!-- Diagnostic Columns -->
                    <th data-column-group="diagnostics">Server Response Time</th>
                    <th data-column-group="diagnostics">Render Blocking Resources</th>
                    <th data-column-group="diagnostics">Unused JavaScript</th>
                    <th data-column-group="diagnostics">DOM Size</th>
                    <th data-column-group="diagnostics">Main Thread Work</th>
                    <th data-column-group="diagnostics">Diagnostics Summary</th>
                    
                    <!-- Lighthouse Score Columns -->
                    <th data-column-group="scores">Performance</th>
                    <th data-column-group="scores">Accessibility</th>
                    <th data-column-group="scores">Best Practices</th>
                    <th data-column-group="scores">SEO</th>
                    <th data-column-group="scores">PWA</th>
                    
                    <!-- Recommendations -->
                    <th data-column-group="basic">Recommendations</th>
                </tr>
            </thead>
            <tbody>
                {% for result in data %}
                <tr class="{{ 'score-poor' if result['scores'].get('performance', 0) < 50 else
                              'score-needs-improvement' if result['scores'].get('performance', 0) < 90 else
                              'score-good' }}">
                    <!-- Basic Info -->
                    <td>{{ result.brand }}</td>
                    <td><a href="{{ result.url }}" target="_blank">{{ result.url }}</a></td>
                    <td>{{ result.depth }}</td>
                    <td>{{ result.mode }}</td>
                    
                    <!-- Core Web Vitals -->
                    <td>{{ result.metrics.fcp }}</td>
                    <td>{{ result.metrics.lcp }}</td>
                    <td>{{ result.metrics.cls }}</td>
                    <td>{{ result.metrics.inp }}</td>
                    <td>
                        {% set cwv_score = result.metrics.get('core_web_vitals_score', 0)|int %}
                        <div class="cwv-score {{ 'cwv-good' if cwv_score >= 90 else
                                                'cwv-average' if cwv_score >= 50 else
                                                'cwv-poor' }}">
                            {{ cwv_score }}
                        </div>
                    </td>
                    
                    <!-- Performance Metrics -->
                    <td>{{ result.metrics.tti }}</td>
                    <td>{{ result.metrics.speed_index }}</td>
                    <td>{{ result.metrics.tbt }}</td>
                    <td>{{ result.metrics.page_load_time }}</td>
                    
                    <!-- Diagnostic Columns -->
                    <td>{{ result.metrics.server_response_time }}</td>
                    <td>{{ result.metrics.render_blocking_resources }}</td>
                    <td>{{ result.metrics.unused_javascript }}</td>
                    <td>{{ result.metrics.dom_size }}</td>
                    <td>{{ result.metrics.mainthread_work }}</td>
                    <td>{{ result.metrics.diagnostics_summary|safe }}</td>
                    
                    <!-- Lighthouse Score Columns -->
                    <td>{{ result.scores.performance }}</td>
                    <td>{{ result.scores.accessibility }}</td>
                    <td>{{ result.scores.best_practices }}</td>
                    <td>{{ result.scores.seo }}</td>
                    <td>{{ result.scores.pwa }}</td>
                    
                    <!-- Recommendations -->
                    <td>{{ result.recommendations|safe }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <script>
            $(document).ready(function() {
                // Initialize DataTable
                const table = $('#reportTable').DataTable({
                    paging: true,
                    searching: true,
                    ordering: true,
                    dom: 'Bfrtip',
                    pageLength: 25,  // Default to 25 entries per page
                    lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "All"]],
                    buttons: ['pageLength', 'copy', 'csv', 'excel', 'pdf', 'print'],
                    colReorder: true,
                    scrollX: true,
                    columnDefs: [
                        { className: "dt-center", targets: [3, 7, 19, 20, 21, 22, 23] }
                    ]
                });

                // Custom filter for view mode (Desktop/Mobile/All)
                $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
                    const selectedView = $('#viewFilter').val().toLowerCase();
                    const rowView = data[3].toLowerCase(); // Index 3 is the 'Mode' column
                    
                    // Filter for view mode
                    const viewMatch = selectedView === 'all' || rowView === selectedView;
                    
                    // Filter for brand
                    const selectedBrand = $('#brandFilter').val().toLowerCase();
                    const rowBrand = data[0].toLowerCase(); // Index 0 is the 'Brand' column
                    const brandMatch = selectedBrand === 'all' || rowBrand === selectedBrand;
                    
                    // Must match both filters
                    return viewMatch && brandMatch;
                });
        
                // Trigger filter on dropdown changes
                $('#viewFilter, #brandFilter').on('change', function() {
                    table.draw();
                });
                
                // Column visibility toggle functions
                function toggleColumnGroup(group, visible) {
                    table.columns().every(function() {
                        const columnHeader = table.column(this).header();
                        if (columnHeader && $(columnHeader).attr('data-column-group') === group) {
                            table.column(this).visible(visible);
                        }
                    });
                }
                
                // Button click handlers for column groups
                $('#toggleBasic').on('click', function() {
                    table.columns().visible(false);
                    toggleColumnGroup('basic', true);
                });
                
                $('#toggleCoreVitals').on('click', function() {
                    table.columns().visible(false);
                    toggleColumnGroup('basic', true);
                    toggleColumnGroup('core', true);
                });
                
                $('#togglePerformance').on('click', function() {
                    table.columns().visible(false);
                    toggleColumnGroup('basic', true);
                    toggleColumnGroup('performance', true);
                });
                
                $('#toggleDiagnostics').on('click', function() {
                    table.columns().visible(false);
                    toggleColumnGroup('basic', true);
                    toggleColumnGroup('diagnostics', true);
                });
                
                $('#toggleScores').on('click', function() {
                    table.columns().visible(false);
                    toggleColumnGroup('basic', true);
                    toggleColumnGroup('scores', true);
                });
                
                $('#toggleAll').on('click', function() {
                    table.columns().visible(true);
                });
        
                // Start with a good default view that shows the most important columns
                $('#toggleScores').click();
            });
        </script>
    </body>
    </html>
    """
    template = Template(html_template)
    html_content = template.render(data=data, brands=unique_brands)
    with open(output_path, 'w', encoding='utf-8') as report_file:  # Ensure UTF-8 encoding
        report_file.write(html_content)
    print(f"HTML report saved to {output_path}")


def visualize_performance_metrics(results, output_path="performance_metrics.png"):
    if not results:
        logging.error("No data available for performance metrics visualization.")
        return

    # Extract data for visualization
    brands = list(set(result["brand"] for result in results))  # Unique brand names
    desktop_scores = []
    mobile_scores = []

    # Group results by brand and mode
    brand_data = {}
    for brand in brands:
        brand_data[brand] = {
            "desktop": 0,
            "mobile": 0,
            "desktop_count": 0,
            "mobile_count": 0
        }

    # Collect and average scores for each brand and mode
    for result in results:
        brand = result["brand"]
        mode = result["mode"]
        score = result["scores"].get("performance", 0)

        if mode == "desktop":
            brand_data[brand]["desktop"] += score
            brand_data[brand]["desktop_count"] += 1
        elif mode == "mobile":
            brand_data[brand]["mobile"] += score
            brand_data[brand]["mobile_count"] += 1

    # Calculate averages and prepare lists for plotting
    desktop_scores = []
    mobile_scores = []
    for brand in brands:
        desktop_count = brand_data[brand]["desktop_count"]
        mobile_count = brand_data[brand]["mobile_count"]

        desktop_avg = brand_data[brand]["desktop"] / desktop_count if desktop_count > 0 else 0
        mobile_avg = brand_data[brand]["mobile"] / mobile_count if mobile_count > 0 else 0

        desktop_scores.append(desktop_avg)
        mobile_scores.append(mobile_avg)

    # Set up the bar width and positions
    bar_width = 0.4
    x = range(len(brands))

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(x, desktop_scores, width=bar_width, label="Desktop", color="blue")
    plt.bar([i + bar_width for i in x], mobile_scores, width=bar_width, label="Mobile", color="orange")

    # Add labels and title
    plt.xlabel("Brands")
    plt.ylabel("Performance Scores")
    plt.title("Performance Metrics: Desktop vs Mobile")
    plt.xticks([i + bar_width / 2 for i in x], brands, rotation=45)
    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100
    plt.legend()

    # Save the chart as an image
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Performance metrics chart saved to {output_path}")
    plt.close()  # Close the plot to free memory


def visualize_other_metrics(results, output_dir="visualizations"):
    if not results:
        logging.error("No data available for other metrics visualization.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for visualization
    brands = list(set(result["brand"] for result in results))
    metrics = {
        "Accessibility": {"desktop": {}, "mobile": {}},
        "Best Practices": {"desktop": {}, "mobile": {}},
        "SEO": {"desktop": {}, "mobile": {}},
        "Core Web Vitals": {"desktop": {}, "mobile": {}}
    }

    # Initialize metrics dictionaries
    for brand in brands:
        for metric_name in metrics:
            metrics[metric_name]["desktop"][brand] = {"total": 0, "count": 0}
            metrics[metric_name]["mobile"][brand] = {"total": 0, "count": 0}

    # Collect scores for each metric by brand and mode
    for result in results:
        brand = result["brand"]
        mode = result["mode"]

        # Add accessibility, best practices, and SEO scores
        metrics["Accessibility"][mode][brand]["total"] += result["scores"].get("accessibility", 0)
        metrics["Accessibility"][mode][brand]["count"] += 1

        metrics["Best Practices"][mode][brand]["total"] += result["scores"].get("best_practices", 0)
        metrics["Best Practices"][mode][brand]["count"] += 1

        metrics["SEO"][mode][brand]["total"] += result["scores"].get("seo", 0)
        metrics["SEO"][mode][brand]["count"] += 1

        # Add Core Web Vitals score
        cwv_score = result["metrics"].get("core_web_vitals_score", 0)
        if isinstance(cwv_score, str):
            try:
                cwv_score = int(cwv_score)
            except:
                cwv_score = 0

        metrics["Core Web Vitals"][mode][brand]["total"] += cwv_score
        metrics["Core Web Vitals"][mode][brand]["count"] += 1

    # Create a bar chart for each metric
    for metric_name, mode_data in metrics.items():
        plt.figure(figsize=(12, 6))

        desktop_scores = []
        mobile_scores = []

        for brand in brands:
            desktop_count = mode_data["desktop"][brand]["count"]
            desktop_avg = mode_data["desktop"][brand]["total"] / desktop_count if desktop_count > 0 else 0
            desktop_scores.append(desktop_avg)

            mobile_count = mode_data["mobile"][brand]["count"]
            mobile_avg = mode_data["mobile"][brand]["total"] / mobile_count if mobile_count > 0 else 0
            mobile_scores.append(mobile_avg)

        # Set up the bar width and positions
        bar_width = 0.4
        x = range(len(brands))

        # Plot bars
        plt.bar(x, desktop_scores, width=bar_width, label="Desktop", color="blue")
        plt.bar([i + bar_width for i in x], mobile_scores, width=bar_width, label="Mobile", color="orange")

        # Add labels and title
        plt.xlabel("Brands")
        plt.ylabel(f"{metric_name} Scores")
        plt.title(f"{metric_name} Scores: Desktop vs Mobile")
        plt.xticks([i + bar_width / 2 for i in x], brands, rotation=45)
        plt.ylim(0, 100)  # Set y-axis limit from 0 to 100
        plt.legend()

        # Save the chart
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{metric_name.replace(' ', '_')}.png")
        plt.savefig(output_path)
        logging.info(f"{metric_name} chart saved to {output_path}")
        plt.close()


def process_brand(brand, max_depth, max_workers):
    logging.info(f"Processing brand: {brand['name']} ({brand['url']}) with max_depth={max_depth}")
    visited_urls = set()
    crawled_data = crawl_urls(brand["url"], max_depth, max_workers, delay=1, visited_urls=visited_urls)

    if not crawled_data:
        logging.warning(f"No pages crawled for brand: {brand['name']}")
        return []

    logging.info(f"Completed crawling for {brand['name']}. Total pages crawled: {len(crawled_data)}")
    brand_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for mode in ['desktop', 'mobile']:
            for page in crawled_data:
                logging.info(f"Submitting Lighthouse task for {page['url']} in {mode} mode.")
                futures.append(
                    executor.submit(run_lighthouse, page["url"], is_mobile=(mode=='mobile'), mode=mode, page=page))
                page["depth"] = page.get("depth", 0)  # Ensure depth is captured
                # Add a longer delay between submissions to ensure accurate results
                time.sleep(random.uniform(2.0, 4.0))  # Increased delay to reduce load and improve accuracy

        for future in as_completed(futures):
            try:
                report = future.result()
                if report:
                    # Get the mode from the report's configSettings
                    mode = report.get('configSettings', {}).get('mode', 'desktop')
                    logging.info(f"Processing Lighthouse report for mode: {mode}")

                    # Validate the report structure before processing
                    if "audits" not in report or "categories" not in report:
                        logging.warning(f"Lighthouse report is missing required sections for {mode} mode. Skipping.")
                        continue

                    # Extract metrics with mode awareness
                    metrics, scores = extract_metrics_from_lighthouse(report)

                    # Log the scores for debugging
                    logging.debug(f"Extracted scores for {mode} mode: {scores}")

                    # Ensure all required fields are present
                    required_metrics = ["fcp", "lcp", "cls", "tti", "speed_index", "tbt", "inp",
                                        "page_load_time", "core_web_vitals_score", "server_response_time",
                                        "render_blocking_resources", "unused_javascript", "dom_size",
                                        "mainthread_work", "diagnostics_summary"]
                    for metric in required_metrics:
                        if metric not in metrics:
                            metrics[metric] = "N/A"
                            logging.warning(f"Missing required metric: {metric}, adding default N/A value")

                    # Make sure core_web_vitals_score is an integer for correct display
                    if 'core_web_vitals_score' in metrics:
                        try:
                            metrics['core_web_vitals_score'] = int(metrics['core_web_vitals_score'])
                        except (ValueError, TypeError):
                            metrics['core_web_vitals_score'] = 0

                    recommendations = generate_recommendations(metrics)

                    # Find the page that this report corresponds to
                    page_url = report.get('requestedUrl') or report.get('finalUrl')
                    page = next((p for p in crawled_data if p['url'] == page_url),
                                {'url': page_url, 'depth': 0})  # Fallback if page not found

                    # Ensure all required score fields are present
                    required_scores = ["performance", "accessibility", "best_practices", "seo", "pwa", "mode"]
                    for score in required_scores:
                        if score not in scores:
                            # Use the mode from the report for the 'mode' score field
                            if score == "mode":
                                scores[score] = mode
                            else:
                                scores[score] = 0
                                logging.warning(f"Missing required score: {score}, adding default 0 value")

                    # Ensure the mode from scores matches the actual mode from the report
                    if scores.get("mode") != mode:
                        logging.warning(f"Score mode mismatch: {scores.get('mode')} vs {mode}, fixing...")
                        scores["mode"] = mode

                    brand_results.append({
                        "brand": brand["name"],
                        "url": page["url"],
                        "mode": mode,  # Use the mode from the report
                        "depth": page.get("depth", 0),  # Include depth in results
                        "metrics": metrics,
                        "scores": {k: round(float(v)) if k != "mode" else v for k, v in scores.items()},  # Ensure scores are numbers, except mode
                        "recommendations": recommendations
                    })
                    logging.info(f"Completed Lighthouse analysis for {page['url']} in {mode} mode.")
                    # Add delay after processing each URL to prevent overloading
                    time.sleep(random.uniform(1.0, 2.0))
                else:
                    logging.warning(f"No Lighthouse report generated for a page in {mode} mode.")
            except Exception as e:
                logging.error(f"Error processing Lighthouse task: {e}")
                logging.debug("Stack trace:", exc_info=True)  # Add stack trace for more debugging info

    # Check if we got any results for this brand
    if not brand_results:
        logging.warning(f"No valid Lighthouse results for brand: {brand['name']}")
    else:
        logging.info(f"Collected {len(brand_results)} valid Lighthouse results for brand: {brand['name']}")

    return brand_results


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Website Performance Analysis")
    parser.add_argument("--max_depth", type=int, default=0, help="Maximum depth for crawling (default: 2)")
    parser.add_argument("--audit_delay", type=float, default=1.5, help="Delay between audits in seconds (default: 1.5)")
    parser.add_argument("--api_key", type=str, help="Google PageSpeed Insights API key (overrides encoded key)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of retries for failed Lighthouse runs (default: 3)")
    parser.add_argument("--show_dashboard", action="store_true", help="Show historical performance dashboard after run")
    args = parser.parse_args()

    # If API key is provided via command line, set it as an environment variable
    if args.api_key:
        os.environ['PAGESPEED_API_KEY'] = args.api_key
        logging.info("Using API key provided via command line arguments")

    # Set the max_depth from command line args
    max_depth = args.max_depth
    audit_delay = args.audit_delay
    retries = args.retries
    logging.info(f"Command line argument max_depth: {max_depth}")
    logging.info(f"Audit delay between requests: {audit_delay} seconds")
    logging.info(f"Number of retries for failed Lighthouse runs: {retries}")
    max_workers = 6  # Ensure max_workers is greater than 0

    # Set seed for reproducible random delays
    random.seed(int(time.time()))

    start_time = time.time()
    results = []

    logging.info("Starting the performance analysis process...")
    logging.info(f"Max depth for crawling: {max_depth}")
    logging.info(f"Max workers for crawling: {max_workers}")

    # Process each brand in parallel but with even smaller max_workers to ensure accuracy
    max_workers = min(3, len(BRANDS))  # Limit to 3 parallel brand processes
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_brand, brand, max_depth, max_workers): brand for brand in BRANDS}
        for future in as_completed(futures):
            try:
                brand_results = future.result()  # Correctly retrieve the result from the future
                if brand_results:
                    results.extend(brand_results)
                    logging.info(f"Completed processing for brand: {futures[future]['name']}")
                else:
                    logging.warning(f"No results for brand: {futures[future]['name']}")
                # Add delay between processing brands
                time.sleep(audit_delay)
            except Exception as e:
                logging.error(f"Error processing brand {futures[future]['name']}: {e}")

    # Only generate report if we have results
    if not results:
        logging.error("No data collected. Ensure the URLs are reachable and Lighthouse is configured correctly.")
        return

    # Log summary of results
    logging.info(f"Analysis completed. Total results: {len(results)}")
    brands_collected = set(result["brand"] for result in results)
    logging.info(f"Data collected for brands: {', '.join(brands_collected)}")
    modes_collected = set(result["mode"] for result in results)
    logging.info(f"Data collected for modes: {', '.join(modes_collected)}")

    # Initialize historical data manager
    from historical_data_manager import HistoricalDataManager
    data_manager = HistoricalDataManager()

    # Set up output directories
    current_report_dir = data_manager.current_report_dir
    os.makedirs(current_report_dir, exist_ok=True)

    # Generate the HTML report in the current report directory
    output_path = os.path.join(current_report_dir, "fos_performance_report.html")
    generate_html_report(results, output_path)

    try:
        # Create visualizations directory in the current report directory
        viz_dir = os.path.join(current_report_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        # Generate visualizations
        visualize_performance_metrics(results, os.path.join(viz_dir, "performance_metrics.png"))
        visualize_other_metrics(results, viz_dir)

        logging.info("Performance visualizations generated successfully")
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        import traceback
        logging.error(traceback.format_exc())

    # Convert results to a DataFrame for historical data storage
    metrics_data = []
    for result in results:
        metrics_row = {
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'brand': result['brand'],
            'mode': result['mode'],
            'url': result['url']
        }

        # Add metrics
        for metric_key, metric_value in result['metrics'].items():
            metrics_row[metric_key] = metric_value

        # Add scores
        for score_key, score_value in result['scores'].items():
            if score_key != 'mode':  # Skip mode as we already have it
                metrics_row[f'{score_key}_score'] = score_value

        metrics_data.append(metrics_row)

    metrics_df = pd.DataFrame(metrics_data)

    # Save the current run data to historical storage
    archived_report_dir = data_manager.save_current_run_data(metrics_df)

    # Generate time series charts from all historical data
    data_manager.generate_historical_charts(current_report_dir)

    # Show the dashboard if requested
    if args.show_dashboard:
        try:
            from performance_dashboard import PerformanceDashboard
            dashboard = PerformanceDashboard()
            dashboard.generate_dashboard()
            logging.info("Performance dashboard generated. Run 'python performance_dashboard.py' to view it.")
        except Exception as e:
            logging.error(f"Error generating dashboard: {e}")

    end_time = time.time()
    logging.info(f"Performance analysis completed in {end_time - start_time:.2f} seconds.")
    logging.info(f"Report saved to {output_path}")
    logging.info(f"Historical data archived to {archived_report_dir}")
    logging.info("To view historical data dashboard, run: python performance_dashboard.py")


if __name__ == "__main__":
    main()
