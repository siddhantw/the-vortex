#!/usr/bin/env python3
"""
Enhanced GTMetrix Historical Data Manager
A specialized data manager for handling performance data specifically for the Enhanced GTMetrix dashboard.
This module focuses on fetching and processing data from the performance_reports directory.
"""

import os
import glob
import json
import logging
import datetime
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


class EnhancedGTMetrixHistoricalDataManager:
    """
    Specialized data manager for handling historical performance data for GTMetrix-style dashboard.
    Fetches all data from performance_reports directory and processes it for display in the dashboard.
    """

    def __init__(self, base_dir=None):
        """
        Initialize the historical data manager with the specified base directory.

        Args:
            base_dir (str, optional): Base directory for performance data.
                If None, will use the default performance_reports directory.
        """
        if base_dir:
            self.base_dir = base_dir
        else:
            # Get the path to the jarvis-test-automation repository
            current_file = os.path.abspath(__file__)
            scripts_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            repo_dir = os.path.dirname(scripts_dir)
            self.base_dir = os.path.join(repo_dir, "performance_reports")

        self.historical_data_dir = os.path.join(self.base_dir, "historical_data")
        self.current_reports_dir = os.path.join(self.base_dir, "current")
        self.history_file = os.path.join(self.historical_data_dir, "performance_history.csv")
        self.brand_map_file = os.path.join(self.base_dir, "brand_mapping.json")
        self.crux_file_candidates = [
            os.path.join(self.base_dir, "crux.json"),
            os.path.join(self.base_dir, "current", "crux.json"),
            os.path.join(self.historical_data_dir, "crux.json"),
        ]
        self._brand_map = self._load_brand_mapping()

        # Ensure directories exist
        os.makedirs(self.historical_data_dir, exist_ok=True)
        os.makedirs(self.current_reports_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )

        logging.info(f"Enhanced GTMetrix Historical Data Manager initialized")
        logging.info(f"Base directory: {self.base_dir}")
        logging.info(f"Historical data directory: {self.historical_data_dir}")

    def _load_brand_mapping(self) -> Dict[str, str]:
        """Load optional brand mapping from base_dir/brand_mapping.json.
        Format: { "domain_or_pattern": "Brand Name" }
        Pattern can be a substring; simplest matching: if pattern in domain -> brand.
        """
        try:
            if os.path.exists(self.brand_map_file):
                with open(self.brand_map_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        logging.info(f"Loaded brand mapping with {len(data)} entries")
                        return {str(k).lower(): str(v) for k, v in data.items()}
        except Exception as e:
            logging.warning(f"Failed to load brand mapping: {e}")
        return {}

    def _infer_brand_from_domain(self, domain: str) -> str:
        """Infer brand from domain using mapping; fall back to second-level label capitalized."""
        if not domain:
            return "Unknown"
        dom = domain.lower()
        # Exact match first
        if dom in self._brand_map:
            return self._brand_map[dom]
        # Substring match
        for pattern, brand in self._brand_map.items():
            if pattern and pattern in dom:
                return brand
        # Fallback: use main domain label
        parts = dom.split('.')
        if len(parts) >= 2:
            core = parts[-2]
        else:
            core = parts[0]
        return core.replace('-', ' ').title()

    def _url_parts(self, url: str) -> Dict[str, Any]:
        """Return parsed parts: domain, path, brand."""
        try:
            from urllib.parse import urlparse
            p = urlparse(url)
            domain = p.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            path = p.path or '/'
            brand = self._infer_brand_from_domain(domain)
            return {"domain": domain, "path": path, "brand": brand}
        except Exception:
            return {"domain": "", "path": "/", "brand": "Unknown"}

    def scan_performance_reports(self):
        """
        Scan the performance_reports directory to find all available performance data.
        This includes both the historical_data directory and individual report directories.

        Returns:
            list: A list of dictionaries containing metadata about each report found
        """
        reports = []

        # Pattern to match report directories in the base directory
        report_pattern = os.path.join(self.base_dir, "performance_report_*")
        report_dirs = glob.glob(report_pattern)
        logging.info(f"Found {len(report_dirs)} report directories in base directory")

        # Also check historical data directory for report subdirectories
        historical_pattern = os.path.join(self.historical_data_dir, "report_*")
        historical_dirs = glob.glob(historical_pattern)
        logging.info(f"Found {len(historical_dirs)} report directories in historical directory")

        # Get current reports directory - safely handle if it doesn't exist or is empty
        current_reports = []
        if os.path.exists(self.current_reports_dir):
            try:
                current_reports = [os.path.join(self.current_reports_dir, d) for d in os.listdir(self.current_reports_dir)
                              if os.path.isdir(os.path.join(self.current_reports_dir, d))]
                logging.info(f"Found {len(current_reports)} report directories in current directory")
            except Exception as e:
                logging.warning(f"Error accessing current reports directory: {e}")

        # Try recursive search if direct glob didn't find reports
        if not report_dirs:
            logging.info("No reports found with direct pattern, trying recursive search")
            for root, dirs, _ in os.walk(self.base_dir):
                for dir_name in dirs:
                    if dir_name.startswith("performance_report_"):
                        full_path = os.path.join(root, dir_name)
                        if full_path not in report_dirs:
                            report_dirs.append(full_path)
            logging.info(f"Found {len(report_dirs)} report directories after recursive search")

        all_report_dirs = report_dirs + historical_dirs + current_reports
        logging.info(f"Found {len(all_report_dirs)} total potential report directories to scan")

        # Continue with rest of the function
        for report_dir in all_report_dirs:
            try:
                # Extract date from directory name
                dir_name = os.path.basename(report_dir)

                # Different naming formats: performance_report_DATE, report_DATE, or just DATE
                if dir_name.startswith("performance_report_"):
                    date_part = dir_name.split("performance_report_", 1)[1]
                elif dir_name.startswith("report_"):
                    date_part = dir_name.split("report_", 1)[1]
                else:
                    date_part = dir_name

                if "_" in date_part:
                    date_str = date_part.split("_")[0]  # Get date part if there's time included
                else:
                    date_str = date_part

                # Find report data files - search recursively with a max depth of 2
                report_files = []
                metadata_files = []
                results_files = []

                # Walk through the directory and find JSON files
                for root, dirs, files in os.walk(report_dir):
                    # Only go 2 levels deep maximum
                    depth = root.replace(report_dir, '').count(os.path.sep)
                    if depth > 2:
                        continue

                    for file in files:
                        if file.endswith('.json'):
                            file_path = os.path.join(root, file)
                            report_files.append(file_path)
                            if 'metadata' in file.lower():
                                metadata_files.append(file_path)
                            if 'result' in file.lower():
                                results_files.append(file_path)

                # Always add this directory to our reports list, even if it doesn't have JSON files
                report_info = {
                    "directory": report_dir,
                    "date": date_str,
                    "has_metadata": len(metadata_files) > 0,
                    "has_results": len(results_files) > 0,
                    "report_files": report_files,
                    "metadata_files": metadata_files,
                    "results_files": results_files
                }
                reports.append(report_info)
                if not report_files and not metadata_files and not results_files:
                    logging.debug(f"No JSON files found in {report_dir}, but adding directory to reports list anyway")
            except Exception as e:
                logging.warning(f"Error processing report directory {report_dir}: {str(e)}")

        logging.info(f"Found {len(reports)} performance report directories with JSON data")
        return reports

    def extract_performance_data(self, report_info):
        """
        Extract performance metrics from a report directory

        Args:
            report_info (dict): Report metadata from scan_performance_reports

        Returns:
            dict: Extracted performance metrics or None if extraction failed
        """
        try:
            # Check for metadata files first as they usually contain the most structured data
            if report_info["metadata_files"]:
                for metadata_file in report_info["metadata_files"]:
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                        # Extract key performance metrics
                        perf_data = self._extract_from_metadata(metadata, report_info["date"])
                        if perf_data:
                            return perf_data
                    except Exception as e:
                        logging.warning(f"Error extracting data from {metadata_file}: {str(e)}")

            # Check results files if metadata didn't yield anything
            if report_info["results_files"]:
                for results_file in report_info["results_files"]:
                    try:
                        with open(results_file, 'r') as f:
                            results = json.load(f)

                        # Extract key performance metrics
                        perf_data = self._extract_from_results(results, report_info["date"])
                        if perf_data:
                            return perf_data
                    except Exception as e:
                        logging.warning(f"Error extracting data from {results_file}: {str(e)}")

            # If no structured files worked, try any JSON files
            for report_file in report_info["report_files"]:
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)

                    # Try to extract data from general report file
                    perf_data = self._extract_from_general_report(report_data, report_info["date"])
                    if perf_data:
                        return perf_data
                except Exception as e:
                    logging.warning(f"Error extracting data from {report_file}: {str(e)}")

            return None
        except Exception as e:
            logging.error(f"Error extracting performance data: {str(e)}")
            return None

    def _inject_brand_and_path(self, perf_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure brand, domain, and page path fields are present based on URL."""
        url = perf_data.get("url")
        if url:
            parts = self._url_parts(url)
            if "name" not in perf_data or not perf_data["name"]:
                perf_data["name"] = parts["domain"]
            perf_data.setdefault("brand", parts["brand"])
            perf_data.setdefault("domain", parts["domain"])
            perf_data.setdefault("page", parts["path"])  # URL path
        return perf_data

    def _extract_from_metadata(self, metadata, date_str):
        """Extract performance data from metadata file format"""
        perf_data = {
            "date": date_str,
            "device_mode": metadata.get("device_mode", "desktop")
        }

        # Try to extract site information
        if "url" in metadata:
            perf_data["url"] = metadata["url"]
            perf_data["name"] = self._extract_domain_name(metadata["url"])
        elif "site" in metadata:
            if isinstance(metadata["site"], dict) and "url" in metadata["site"]:
                perf_data["url"] = metadata["site"]["url"]
                perf_data["name"] = metadata["site"].get("name", self._extract_domain_name(metadata["site"]["url"]))
            else:
                perf_data["url"] = str(metadata["site"])
                perf_data["name"] = self._extract_domain_name(str(metadata["site"]))

        # Try to extract performance scores
        if "lighthouse" in metadata:
            lighthouse = metadata["lighthouse"]
            if isinstance(lighthouse, dict):
                perf_data["performance_score"] = lighthouse.get("performance", 0) * 100
                perf_data["accessibility_score"] = lighthouse.get("accessibility", 0) * 100
                perf_data["best_practices_score"] = lighthouse.get("best-practices", 0) * 100
                perf_data["seo_score"] = lighthouse.get("seo", 0) * 100

        # Extract Web Vitals metrics
        if "metrics" in metadata:
            metrics = metadata["metrics"]
            if isinstance(metrics, dict):
                perf_data["lcp"] = metrics.get("lcp", "N/A")
                perf_data["fcp"] = metrics.get("fcp", "N/A")
                perf_data["cls"] = metrics.get("cls", "N/A")
                perf_data["inp"] = metrics.get("inp", "N/A")
                perf_data["ttfb"] = metrics.get("ttfb", "N/A")
                perf_data["tbt"] = metrics.get("tbt", "N/A")
                perf_data["speed_index"] = metrics.get("speed_index", "N/A")

        # Alternative paths for Web Vitals metrics
        if "web_vitals" in metadata:
            web_vitals = metadata["web_vitals"]
            if isinstance(web_vitals, dict):
                if "lcp" not in perf_data or perf_data["lcp"] == "N/A":
                    perf_data["lcp"] = web_vitals.get("LCP", "N/A")
                if "fcp" not in perf_data or perf_data["fcp"] == "N/A":
                    perf_data["fcp"] = web_vitals.get("FCP", "N/A")
                if "cls" not in perf_data or perf_data["cls"] == "N/A":
                    perf_data["cls"] = web_vitals.get("CLS", "N/A")
                if "inp" not in perf_data or perf_data["inp"] == "N/A":
                    perf_data["inp"] = web_vitals.get("INP", "N/A")
                if "ttfb" not in perf_data or perf_data["ttfb"] == "N/A":
                    perf_data["ttfb"] = web_vitals.get("TTFB", "N/A")

        return self._inject_brand_and_path(perf_data)

    def _extract_from_results(self, results, date_str):
        """Extract performance data from results file format"""
        perf_data = {
            "date": date_str,
            "device_mode": "desktop"  # Default to desktop if not specified
        }

        # Check if device mode is specified
        if "device" in results:
            perf_data["device_mode"] = results["device"].lower()
        elif "userAgent" in results:
            if "mobile" in results["userAgent"].lower():
                perf_data["device_mode"] = "mobile"

        # Try to extract site information
        if "url" in results:
            perf_data["url"] = results["url"]
            perf_data["name"] = self._extract_domain_name(results["url"])
        elif "finalUrl" in results:
            perf_data["url"] = results["finalUrl"]
            perf_data["name"] = self._extract_domain_name(results["finalUrl"])

        # Try to extract performance scores
        if "categories" in results:
            categories = results["categories"]
            if isinstance(categories, dict):
                if "performance" in categories:
                    perf_data["performance_score"] = categories["performance"].get("score", 0) * 100
                if "accessibility" in categories:
                    perf_data["accessibility_score"] = categories["accessibility"].get("score", 0) * 100
                if "best-practices" in categories:
                    perf_data["best_practices_score"] = categories["best-practices"].get("score", 0) * 100
                if "seo" in categories:
                    perf_data["seo_score"] = categories["seo"].get("score", 0) * 100

        # Extract Web Vitals metrics from audits
        if "audits" in results:
            audits = results["audits"]

            if "largest-contentful-paint" in audits:
                perf_data["lcp"] = audits["largest-contentful-paint"].get("numericValue", "N/A") / 1000

            if "first-contentful-paint" in audits:
                perf_data["fcp"] = audits["first-contentful-paint"].get("numericValue", "N/A") / 1000

            if "cumulative-layout-shift" in audits:
                perf_data["cls"] = audits["cumulative-layout-shift"].get("numericValue", "N/A")

            if "interactive" in audits:
                perf_data["inp"] = audits["interactive"].get("numericValue", "N/A") / 1000

            if "server-response-time" in audits:
                perf_data["ttfb"] = audits["server-response-time"].get("numericValue", "N/A") / 1000

            if "total-blocking-time" in audits:
                perf_data["tbt"] = audits["total-blocking-time"].get("numericValue", "N/A")

            if "speed-index" in audits:
                perf_data["speed_index"] = audits["speed-index"].get("numericValue", "N/A") / 1000

        return self._inject_brand_and_path(perf_data)

    def _extract_from_general_report(self, report_data, date_str):
        """Extract performance data from general report file format"""
        # First try metadata extraction approach
        perf_data = self._extract_from_metadata(report_data, date_str)
        if self._is_valid_perf_data(perf_data):
            return perf_data

        # Then try results extraction approach
        perf_data = self._extract_from_results(report_data, date_str)
        if self._is_valid_perf_data(perf_data):
            return perf_data

        # If both failed, try more generic extraction
        perf_data = {
            "date": date_str,
            "device_mode": "desktop"
        }

        # Look for URL at various levels
        if "url" in report_data:
            perf_data["url"] = report_data["url"]
            perf_data["name"] = self._extract_domain_name(report_data["url"])

        # Look for performance metrics in any top-level keys that might contain them
        for key in ["performance", "metrics", "statistics", "scores", "results"]:
            if key in report_data and isinstance(report_data[key], dict):
                metrics = report_data[key]

                # Try common performance metric names
                for metric_name in ["performance", "performance_score", "performanceScore"]:
                    if metric_name in metrics:
                        score = metrics[metric_name]
                        # Normalize to 0-100 scale
                        if isinstance(score, (int, float)):
                            perf_data["performance_score"] = 100 * score if score <= 1 else score

                # Try Web Vitals
                for web_vital in ["lcp", "fcp", "cls", "inp", "ttfb", "tbt", "speed_index"]:
                    if web_vital in metrics:
                        perf_data[web_vital] = metrics[web_vital]

        perf_data = self._inject_brand_and_path(perf_data)
        return perf_data if self._is_valid_perf_data(perf_data) else None

    def _is_valid_perf_data(self, perf_data):
        """Check if performance data has minimum required fields"""
        if not perf_data:
            return False

        # Minimum required fields
        required = ["date", "name", "url"]

        # At least one performance metric should be present
        metrics = ["performance_score", "lcp", "fcp", "cls"]

        has_required = all(field in perf_data for field in required)
        has_metrics = any(metric in perf_data for metric in metrics)

        return has_required and has_metrics

    def _extract_domain_name(self, url):
        """Extract a readable domain name from a URL"""
        if not url:
            return "unknown"

        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]

            # If domain has subdomains, just use the main domain
            parts = domain.split('.')
            if len(parts) > 2:
                # Handle special cases like co.uk
                if parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu'] and len(parts[-1]) <= 3:
                    return f"{parts[-3]}.{parts[-2]}.{parts[-1]}"
                return f"{parts[-2]}.{parts[-1]}"

            return domain
        except Exception:
            # If parsing fails, just return the URL or a portion of it
            if len(url) > 30:
                return url[:30] + "..."
            return url

    def load_historical_data(self):
        """
        Load all historical performance data from both the history file and by scanning
        all performance report directories.

        Returns:
            pandas.DataFrame: DataFrame containing all historical performance data
        """
        all_data = []

        # First, try to load from the history file if it exists
        if os.path.exists(self.history_file):
            try:
                history_df = pd.read_csv(self.history_file)
                logging.info(f"Loaded {len(history_df)} records from history file")
                all_data.append(history_df)
            except Exception as e:
                logging.warning(f"Failed to load history file: {str(e)}")

        # Then scan for all performance report directories
        reports = self.scan_performance_reports()
        extracted_data = []

        for report_info in reports:
            logging.debug(f"Processing report: {report_info['directory']}")
            data = self.extract_performance_data(report_info)
            if data:
                extracted_data.append(data)
                logging.debug(f"Successfully extracted data from {report_info['directory']}")
            else:
                logging.debug(f"No valid data extracted from {report_info['directory']}")

        if extracted_data:
            extracted_df = pd.DataFrame(extracted_data)
            logging.info(f"Extracted {len(extracted_df)} records from report directories")
            all_data.append(extracted_df)
        
        # Also check for existing performance history files
        additional_csv_files = []
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.csv') and ('performance' in file.lower() or 'history' in file.lower()):
                    file_path = os.path.join(root, file)
                    if file_path != self.history_file:  # Don't double count the main history file
                        additional_csv_files.append(file_path)
        
        for csv_file in additional_csv_files:
            try:
                additional_df = pd.read_csv(csv_file)
                logging.info(f"Loaded {len(additional_df)} records from additional CSV: {csv_file}")
                all_data.append(additional_df)
            except Exception as e:
                logging.warning(f"Error loading CSV file {csv_file}: {str(e)}")
        
        # If we still couldn't find any data, try a more aggressive approach
        if not extracted_data and len(all_data) <= 1:  # Only history file or empty
            logging.info("No data extracted from standard report format, trying alternative approach")
            fallback_data = self._extract_from_fallback_sources()
            if fallback_data is not None and not fallback_data.empty:
                logging.info(f"Found {len(fallback_data)} records using fallback method")
                all_data.append(fallback_data)

        if not all_data:
            logging.warning("No historical data found")
            return pd.DataFrame()

        # Combine all data sources
        combined_df = pd.concat(all_data, ignore_index=True)

        # Convert date strings to datetime objects
        combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')

        # Ensure brand/domain/page fields exist
        for col in ['brand', 'domain', 'page']:
            if col not in combined_df.columns:
                combined_df[col] = None

        # Derive brand/domain/page from URL where values are missing or empty
        try:
            if 'url' in combined_df.columns:
                mask = combined_df['url'].notna()
                # Precompute parsed parts to avoid repeated parsing
                derived = combined_df.loc[mask, 'url'].apply(lambda u: self._url_parts(str(u)))
                for idx, parts in derived.items():
                    # Fill brand if missing/empty
                    try:
                        val = combined_df.at[idx, 'brand'] if 'brand' in combined_df.columns else None
                        if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                            combined_df.at[idx, 'brand'] = parts.get('brand')
                    except Exception:
                        combined_df.at[idx, 'brand'] = parts.get('brand')
                    # Fill domain if missing/empty
                    try:
                        val = combined_df.at[idx, 'domain'] if 'domain' in combined_df.columns else None
                        if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                            combined_df.at[idx, 'domain'] = parts.get('domain')
                    except Exception:
                        combined_df.at[idx, 'domain'] = parts.get('domain')
                    # Fill page/path if missing/empty
                    try:
                        val = combined_df.at[idx, 'page'] if 'page' in combined_df.columns else None
                        if pd.isna(val) or (isinstance(val, str) and not val.strip()):
                            combined_df.at[idx, 'page'] = parts.get('path')
                    except Exception:
                        combined_df.at[idx, 'page'] = parts.get('path')
        except Exception:
            # Non-fatal: continue without derived values
            pass

        # Handle duplicate entries by keeping the most complete one
        combined_df = self._handle_duplicates(combined_df)

        # Sort by date
        combined_df = combined_df.sort_values('date')

        logging.info(f"Total historical records: {len(combined_df)}")
        return combined_df

    def load_crux_data(self) -> Dict[str, Any]:
        """
        Load CrUX data if a local crux.json exists. Supports a few common formats:
        - CrUX API-like output with record.metrics.* histograms and optional formFactor breakdowns
        - Simplified custom file with top-level metrics

        Returns a dict like:
        {
          "source": "file path",
          "updated": "YYYY-MM-DD",
          "overall": { metric: {"good": pct, "ni": pct, "poor": pct} },
          "by_form_factor": { "mobile": {metric: {...}}, "desktop": {...} }
        }
        """
        crux = {}
        path = None
        for cand in self.crux_file_candidates:
            if os.path.exists(cand):
                path = cand
                break
        if not path:
            return {}
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            crux['source'] = path
            crux['updated'] = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d')

            def bins_to_dist(bins):
                # Convert histogram bins (start,end,density) to Good/NI/Poor based on CWV thresholds
                # bins densities usually sum ~1.0; return percentages (0-100)
                good = ni = poor = 0.0
                for b in bins:
                    start = b.get('start', 0)
                    end = b.get('end', None)
                    d = float(b.get('density', 0) or 0)
                    # Map by metric externally using thresholds; we'll just return raw bins here
                return bins

            def metric_hist_to_percentages(metric_key: str, hist) -> Dict[str, float]:
                # hist expected list of bins with start/end/density
                # Apply CWV thresholds
                good = 0.0
                ni = 0.0
                poor = 0.0
                for b in (hist or []):
                    start = b.get('start', 0)
                    end = b.get('end', None)
                    d = float(b.get('density', 0) or 0.0)
                    if metric_key == 'lcp':
                        # thresholds in seconds: good <=2.5, ni <=4.0
                        # bins start/end in seconds typically; if ms encountered, they tend to be large numbers; handle both
                        start_s = start / 1000.0 if start > 100 else start
                        end_s = (end / 1000.0) if (end and end > 100) else end
                        if end_s is None or end_s <= 2.5:
                            good += d
                        elif end_s <= 4.0:
                            ni += d
                        else:
                            poor += d
                    elif metric_key == 'fcp':
                        start_s = start / 1000.0 if start > 100 else start
                        end_s = (end / 1000.0) if (end and end > 100) else end
                        if end_s is None or end_s <= 1.8:
                            good += d
                        elif end_s <= 3.0:
                            ni += d
                        else:
                            poor += d
                    elif metric_key == 'cls':
                        # good <=0.1, ni <=0.25
                        end_v = end if end is not None else 10
                        if end_v <= 0.1:
                            good += d
                        elif end_v <= 0.25:
                            ni += d
                        else:
                            poor += d
                    elif metric_key == 'inp':
                        # good <=200ms, ni <=500ms
                        start_ms = start if start < 10000 else start  # assume ms
                        end_ms = end if end is not None else 100000
                        if end_ms <= 200:
                            good += d
                        elif end_ms <= 500:
                            ni += d
                        else:
                            poor += d
                total = good + ni + poor
                if total <= 0:
                    return {"good": 0.0, "ni": 0.0, "poor": 0.0}
                return {"good": round(good*100/total, 2), "ni": round(ni*100/total, 2), "poor": round(poor*100/total, 2)}

            def extract_overall(record: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
                out = {}
                metrics = record.get('metrics') or {}
                mapping = {
                    'LARGEST_CONTENTFUL_PAINT_MS': 'lcp',
                    'LARGEST_CONTENTFUL_PAINT': 'lcp',
                    'FIRST_CONTENTFUL_PAINT_MS': 'fcp',
                    'FIRST_CONTENTFUL_PAINT': 'fcp',
                    'CUMULATIVE_LAYOUT_SHIFT_SCORE': 'cls',
                    'CUMULATIVE_LAYOUT_SHIFT': 'cls',
                    'INTERACTION_TO_NEXT_PAINT': 'inp',
                    'INTERACTION_TO_NEXT_PAINT_MS': 'inp'
                }
                for k, v in metrics.items():
                    mk = mapping.get(k)
                    if not mk:
                        continue
                    hist = None
                    if isinstance(v, dict) and 'histogram' in v:
                        hist = v['histogram']
                    elif isinstance(v, dict) and 'histograms' in v:
                        hist = v['histograms']
                    if hist:
                        out[mk] = metric_hist_to_percentages(mk, hist)
                return out

            # Try to detect CrUX API structure
            if isinstance(data, dict) and 'record' in data:
                crux['overall'] = extract_overall(data.get('record', {}))
                # Form factor
                form_factors = {}
                # Look for breakdowns
                for ff in ['PHONE', 'DESKTOP', 'TABLET', 'ALL_CLASSIC']:
                    rec = data.get(ff.lower()) or data.get(ff) or data.get('records', {}).get(ff, {})
                    if isinstance(rec, dict) and 'metrics' in rec:
                        form_factors[ff.lower()] = extract_overall(rec)
                if form_factors:
                    crux['by_form_factor'] = form_factors
            else:
                # Fallback simple structure: expect top-level keys like lcp/cls/inp each with histogram bins
                simple = {}
                for mk in ['lcp', 'fcp', 'cls', 'inp']:
                    if mk in data and isinstance(data[mk], list):
                        simple[mk] = metric_hist_to_percentages(mk, data[mk])
                if simple:
                    crux['overall'] = simple
            return crux
        except Exception as e:
            logging.warning(f"Failed to load CrUX data from {path}: {e}")
            return {}

    def _extract_from_fallback_sources(self):
        """
        Fallback method to extract performance data from alternative sources
        when standard extraction fails
        
        Returns:
            pandas.DataFrame: DataFrame with extracted data or None
        """
        extracted_data = []
        
        # 1. Try looking for structured performance data in the performance_reports directory
        logging.info("Scanning for performance data files in main directory")
        
        # Look for any JSON files in the main performance_reports directory and subdirectories
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.json') and ('performance' in file.lower() or 
                                            'metrics' in file.lower() or 
                                            'report' in file.lower() or
                                            'lighthouse' in file.lower() or
                                            'web_vitals' in file.lower()):
                    try:
                        file_path = os.path.join(root, file)
                        logging.debug(f"Trying to extract data from: {file_path}")
                        
                        # Extract date from directory or file name
                        dir_name = os.path.basename(os.path.dirname(file_path))
                        date_match = None
                        
                        # Try to extract date from directory name first
                        for pattern in ["performance_report_", "report_"]:
                            if pattern in dir_name:
                                parts = dir_name.split(pattern)[1].split("_")
                                if len(parts[0]) == 8:  # YYYYMMDD format
                                    date_match = parts[0]
                                    break
                        
                        # If no date found in directory name, try file name
                        if not date_match and "_" in file:
                            parts = file.split("_")
                            for part in parts:
                                if len(part) == 8 and part.isdigit():  # YYYYMMDD format
                                    date_match = part
                                    break
                        
                        # If still no date, use directory modification time
                        if not date_match:
                            mod_time = os.path.getmtime(file_path)
                            date_match = datetime.datetime.fromtimestamp(mod_time).strftime("%Y%m%d")
                        
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Try to extract performance data
                        if isinstance(data, dict):
                            perf_data = {}
                            
                            # Set date and try to find URL and name
                            perf_data["date"] = date_match
                            
                            # Look for URL at any level in the JSON
                            url = self._find_in_nested_dict(data, ["url", "finalUrl", "page_url", "site"])
                            if url:
                                if isinstance(url, dict) and "url" in url:
                                    url = url["url"]
                                perf_data["url"] = url
                                perf_data["name"] = self._extract_domain_name(str(url))
                            else:
                                # If no URL found, use filename as a fallback
                                perf_data["url"] = file
                                perf_data["name"] = "Unknown Site"
                            
                            # Set default device mode
                            perf_data["device_mode"] = "desktop"
                            
                            # Look for device info
                            device = self._find_in_nested_dict(data, ["device", "deviceType", "form_factor"])
                            if device:
                                if isinstance(device, str) and "mobile" in device.lower():
                                    perf_data["device_mode"] = "mobile"
                            
                            # Look for performance scores
                            scores = self._find_in_nested_dict(data, ["scores", "lighthouse", "metrics", "performance"])
                            if scores and isinstance(scores, dict):
                                if "performance" in scores:
                                    score = scores["performance"]
                                    perf_data["performance_score"] = 100 * score if score <= 1 else score
                                elif "score" in scores:
                                    score = scores["score"]
                                    perf_data["performance_score"] = 100 * score if score <= 1 else score
                            
                            # Look for web vitals
                            for metric in ["lcp", "fcp", "cls", "inp", "ttfb", "tbt", "speed_index"]:
                                metric_value = self._find_in_nested_dict(data, [metric, f"{metric}_value", metric.upper()])
                                if metric_value is not None:
                                    perf_data[metric] = metric_value
                            
                            # Inject brand and path
                            perf_data = self._inject_brand_and_path(perf_data)

                            # Check if we have minimum viable data
                            if "date" in perf_data and "name" in perf_data and (
                                "performance_score" in perf_data or 
                                "lcp" in perf_data or 
                                "fcp" in perf_data or
                                "cls" in perf_data
                            ):
                                extracted_data.append(perf_data)
                                logging.debug(f"Successfully extracted data from {file_path}")
                    
                    except Exception as e:
                        logging.debug(f"Error extracting data from {file}: {str(e)}")
        
        if extracted_data:
            return pd.DataFrame(extracted_data)
        
        return None
    
    def _find_in_nested_dict(self, data, possible_keys):
        """
        Search for any of the possible keys in a nested dictionary
        
        Args:
            data (dict): Dictionary to search in
            possible_keys (list): List of possible key names to look for
            
        Returns:
            The value if found, None otherwise
        """
        if not isinstance(data, dict):
            return None
        
        # Check direct keys first
        for key in possible_keys:
            if key in data:
                return data[key]
        
        # Then check nested dictionaries (one level deep)
        for value in data.values():
            if isinstance(value, dict):
                for key in possible_keys:
                    if key in value:
                        return value[key]
            
        # Check two levels deep
        for value1 in data.values():
            if isinstance(value1, dict):
                for value2 in value1.values():
                    if isinstance(value2, dict):
                        for key in possible_keys:
                            if key in value2:
                                return value2[key]
        
        return None

    def _handle_duplicates(self, df):
        """Handle duplicate entries by keeping the most complete one"""
        if df.empty:
            return df

        # Count non-null values in each row
        df['completeness'] = df.notna().sum(axis=1)

        # Sort by completeness (descending) and date (ascending)
        df = df.sort_values(['name', 'url', 'date', 'device_mode', 'completeness'],
                          ascending=[True, True, True, True, False])

        # Drop duplicates, keeping the first occurrence (which will be the most complete one)
        df = df.drop_duplicates(subset=['name', 'url', 'date', 'device_mode'], keep='first')

        # Drop the completeness column
        df = df.drop(columns=['completeness'])

        return df
