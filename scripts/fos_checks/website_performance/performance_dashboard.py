"""
Dashboard for viewing historical FOS website performance data
This script creates a web interface to view all past performance reports and time series charts with interactive filtering capabilities.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import glob
import webbrowser
import http.server
import socketserver
from pathlib import Path
import datetime
import logging
import shutil
import json
from dateutil.relativedelta import relativedelta
import re

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from historical_data_manager import HistoricalDataManager
from time_series_charts import (
    create_core_metrics_chart,
    create_lighthouse_scores_chart,
    create_core_web_vitals_chart,
    create_diagnostics_summary_chart  # Import the diagnostics chart function
)


class PerformanceDashboard:
    """Dashboard for viewing historical website performance data"""

    def __init__(self):
        """Initialize the performance dashboard"""
        self.data_manager = HistoricalDataManager()
        self.dashboard_dir = os.path.join(self.data_manager.base_dir, "dashboard")
        self.port = 8000
        self.js_dir = os.path.join(self.dashboard_dir, "js")
        self.css_dir = os.path.join(self.dashboard_dir, "css")
        self.data_dir = os.path.join(self.dashboard_dir, "data")
        # Flag to check if we should preserve the existing HTML
        self.preserve_existing_html = True

        # Create necessary directories
        for directory in [self.dashboard_dir, self.js_dir, self.css_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )

    def prepare_data_for_visualizations(self, historical_df):
        """
        Process and clean historical data for visualizations

        Args:
            historical_df: DataFrame with historical performance data

        Returns:
            Dictionary with processed data for different visualizations
        """
        # Make a copy to avoid modifying the original data
        df = historical_df.copy()

        # First, convert string metrics to numeric values
        numeric_columns = ['fcp', 'lcp', 'cls', 'inp', 'ttfb', 'tbt', 'tti',
                          'performance_score', 'accessibility_score',
                          'best_practices_score', 'seo_score', 'core_web_vitals_score']

        for col in numeric_columns:
            if col in df.columns:
                # Try to convert string values (like "2.3 s") to numeric
                if df[col].dtype == 'object':
                    df[col] = df[col].apply(lambda x:
                        # Handle values with spaces like "2.3 s"
                        float(str(x).split()[0]) if isinstance(x, str) and ' ' in x
                        # Handle values without spaces like "130ms"
                        else (float(''.join(c for c in str(x) if c.isdigit() or c == '.')) if isinstance(x, str) and any(unit in x for unit in ['ms', 's', 'MB', 'KB'])
                        # Handle direct numeric values
                        else (float(x) if isinstance(x, (int, float)) or
                              (isinstance(x, str) and x.replace('.', '', 1).isdigit())
                        else np.nan)))

                    # Special handling for INP - if the column is 'inp', values less than 10 are likely in seconds, convert to milliseconds
                    if col == 'inp':
                        df[col] = df[col].apply(lambda x: x * 1000 if isinstance(x, (int, float)) and x < 10 else x)

        # Process diagnostics_summary column if it exists
        if 'diagnostics_summary' in df.columns:
            # Extract page size and request count from diagnostics_summary
            def extract_diagnostics_values(summary):
                if not isinstance(summary, str):
                    return None, None

                # Initialize values
                page_size_kb = None
                request_count = None

                # Extract request count using regex
                request_match = re.search(r'Requests:\s*(\d+)', summary, re.IGNORECASE)
                if request_match:
                    request_count = int(request_match.group(1))

                # Extract page size using regex
                size_match = re.search(r'Total Size:\s*([\d.]+)\s*([KkMmGg][Bb])', summary, re.IGNORECASE)
                if size_match:
                    value = float(size_match.group(1))
                    unit = size_match.group(2).lower()

                    # Convert to KB
                    if unit.startswith('m'):  # MB
                        page_size_kb = value * 1024
                    elif unit.startswith('k'):  # KB
                        page_size_kb = value
                    elif unit.startswith('g'):  # GB
                        page_size_kb = value * 1024 * 1024

                return page_size_kb, request_count

            # Apply extraction to all rows
            extracted = df['diagnostics_summary'].apply(extract_diagnostics_values)

            # Create new columns with extracted values
            df['page_size_kb'] = extracted.apply(lambda x: x[0])
            df['request_count'] = extracted.apply(lambda x: x[1])

            # Log extraction results
            valid_size_count = df['page_size_kb'].notna().sum()
            valid_request_count = df['request_count'].notna().sum()
            logging.info(f"Extracted {valid_size_count} page sizes and {valid_request_count} request counts from diagnostics_summary")

        # Convert date to datetime if it's not already
        if 'date' in df.columns and df['date'].dtype == 'object':
            df['date'] = pd.to_datetime(df['date'])

        # Check if we have timestamp information in the data
        has_time_info = False
        if 'timestamp' in df.columns:
            has_time_info = True
        elif 'datetime' in df.columns:
            has_time_info = True
            # Rename datetime to timestamp for consistency
            df.rename(columns={'datetime': 'timestamp'}, inplace=True)

        # Extract timestamp from report filenames if not already in the data
        if not has_time_info:
            # Try to get the report timestamps from the filenames in the format report_YYYYMMDD_HHMMSS
            # First, check if we have a 'report' or 'filename' column that might contain this info
            if 'report' in df.columns and df['report'].str.match(r'report_\d{8}_\d{6}').any():
                # Extract timestamp from report filenames
                df['timestamp'] = df['report'].apply(lambda x:
                    pd.to_datetime(x.split('report_')[1], format='%Y%m%d_%H%M%S')
                    if 'report_' in str(x) else pd.NaT)
                has_time_info = True
            elif 'filename' in df.columns and df['filename'].str.match(r'report_\d{8}_\d{6}').any():
                # Extract timestamp from filenames
                df['timestamp'] = df['filename'].apply(lambda x:
                    pd.to_datetime(x.split('report_')[1], format='%Y%m%d_%H%M%S')
                    if 'report_' in str(x) else pd.NaT)
                has_time_info = True

        # If we still don't have timestamp info, try to get it from the report directories
        if not has_time_info:
            # Get the report directories and their timestamps
            report_dirs = self.data_manager.get_report_directories()
            report_timestamps = {}

            for report_dir in report_dirs:
                report_name = os.path.basename(report_dir)
                if report_name.startswith("report_") and len(report_name) >= 16:
                    date_str = report_name[7:]  # Remove "report_"
                    try:
                        # Parse the timestamp from the directory name
                        timestamp = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                        report_timestamps[report_name] = timestamp
                    except ValueError:
                        continue

            # If we have report timestamps, add timestamp to the data
            if report_timestamps:
                # Add an initial timestamp column with date's values
                df['timestamp'] = df['date']

                # Try to match each row to a report timestamp
                # This approach assumes each date in the data corresponds to a specific report
                # For each unique date, find the closest report timestamp
                for date in df['date'].unique():
                    closest_timestamp = min(report_timestamps.values(),
                                       key=lambda x: abs((pd.to_datetime(date) - x).total_seconds()))

                    # Set the timestamp for all rows with this date
                    df.loc[df['date'] == date, 'timestamp'] = closest_timestamp

                has_time_info = True

        # Format dates with timestamps for better presentation in the dashboard
        if has_time_info:
            # Create a date string that includes the time component
            df['date_with_time'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Use this new column for time series data
            time_column = 'date_with_time'
        else:
            # If we don't have timestamp information, use the regular date column
            time_column = 'date'

        # Processed data dict
        viz_data = {}

        # 1. Time series data by brand and mode
        if not df.empty and time_column in df.columns:
            # Group by date (with time if available), brand, and mode
            time_series = df.groupby([time_column, 'brand', 'mode'])[numeric_columns].mean().reset_index()

            # Replace NaN values with None (which will be converted to null in JSON)
            time_series = time_series.replace({np.nan: None})

            # Rename the time column back to 'date' for consistency with frontend
            time_series.rename(columns={time_column: 'date'}, inplace=True)

            viz_data['time_series'] = time_series.to_dict(orient='records')

            # Flag indicating if hourly data is available
            viz_data['has_hourly_data'] = has_time_info

            # Get unique dates, brands, and modes for filters
            if has_time_info:
                viz_data['unique_dates'] = sorted(df[time_column].unique().tolist())
            else:
                viz_data['unique_dates'] = sorted(df['date'].dt.strftime('%Y-%m-%d').unique().tolist())

            viz_data['unique_brands'] = sorted(df['brand'].unique().tolist())
            viz_data['unique_modes'] = sorted(df['mode'].unique().tolist())

        # 2. Prepare competitive analysis data
        if not df.empty and 'brand' in df.columns:
            # Get the most recent date
            latest_date = df['date'].max()
            latest_df = df[df['date'] == latest_date]

            # Calculate average performance by brand and mode
            brand_performance = latest_df.groupby(['brand', 'mode'])[numeric_columns].mean().reset_index()

            # Replace NaN values with None
            brand_performance = brand_performance.replace({np.nan: None})

            # Make sure all data is JSON serializable
            for col in brand_performance.columns:
                if brand_performance[col].dtype == 'datetime64[ns]' or isinstance(brand_performance[col].iloc[0], pd.Timestamp):
                    brand_performance[col] = brand_performance[col].dt.strftime('%Y-%m-%d')

            viz_data['brand_performance'] = brand_performance.to_dict(orient='records')

        # 3. Core Web Vitals pass rate calculation
        if not df.empty:
            # Core Web Vitals thresholds
            # LCP: good <= 2.5s, needs improvement <= 4s, poor > 4s
            # CLS: good <= 0.1, needs improvement <= 0.25, poor > 0.25
            # INP: good <= 200ms, needs improvement <= 500ms, poor > 500ms

            cwv_metrics = {}

            # Calculate pass rates for each metric by brand
            for brand in df['brand'].unique():
                brand_df = df[df['brand'] == brand]

                # LCP pass rate (good <= 2.5s)
                if 'lcp' in brand_df:
                    lcp_total = brand_df['lcp'].count()
                    lcp_pass = brand_df[brand_df['lcp'] <= 2.5]['lcp'].count() if lcp_total > 0 else 0
                    lcp_rate = (lcp_pass / lcp_total * 100) if lcp_total > 0 else 0
                else:
                    lcp_rate = 0

                # CLS pass rate (good <= 0.1)
                if 'cls' in brand_df:
                    cls_total = brand_df['cls'].count()
                    cls_pass = brand_df[brand_df['cls'] <= 0.1]['cls'].count() if cls_total > 0 else 0
                    cls_rate = (cls_pass / cls_total * 100) if cls_total > 0 else 0
                else:
                    cls_rate = 0

                # INP pass rate (good <= 200ms, converting to seconds if needed)
                if 'inp' in brand_df:
                    inp_total = brand_df['inp'].count()

                    # First, ensure INP values are in milliseconds (convert from seconds if needed)
                    # Create a temporary column for comparison where small values (< 10) are treated as seconds
                    # and converted to milliseconds
                    brand_df['inp_ms'] = brand_df['inp'].apply(
                        lambda x: x * 1000 if isinstance(x, (int, float)) and x < 10 else x
                    )

                    # Now use the converted values for comparison
                    inp_pass = brand_df[brand_df['inp_ms'] <= 200]['inp'].count() if inp_total > 0 else 0
                    inp_rate = (inp_pass / inp_total * 100) if inp_total > 0 else 0
                else:
                    inp_rate = 0

                # Store brand metrics
                cwv_metrics[brand] = {
                    'lcp_pass_rate': float(lcp_rate),  # Ensure values are native Python types
                    'cls_pass_rate': float(cls_rate),
                    'inp_pass_rate': float(inp_rate),
                    'overall_pass_rate': float((lcp_rate + cls_rate + inp_rate) / 3)
                }

            viz_data['cwv_pass_rates'] = cwv_metrics

        # 4. Prepare performance score distribution data
        if not df.empty and 'performance_score' in df.columns:
            # Create performance score buckets
            buckets = {
                'Poor (<50)': [0, 49],
                'Needs Improvement (50-89)': [50, 89],
                'Good (90-100)': [90, 100]
            }

            # Calculate distributions by brand
            score_distribution = {}
            for brand in df['brand'].unique():
                brand_df = df[df['brand'] == brand]
                distribution = {}
                for label, [min_val, max_val] in buckets.items():
                    count = brand_df[(brand_df['performance_score'] >= min_val) &
                                   (brand_df['performance_score'] <= max_val)].shape[0]
                    distribution[label] = int(count)  # Ensure count is a native Python int

                score_distribution[brand] = distribution

            viz_data['score_distribution'] = score_distribution

        # 5. Generate business insights
        viz_data['insights'] = self._generate_business_insights(df)

        return viz_data

    def _generate_business_insights(self, df):
        """
        Generate actionable business insights from performance data

        Args:
            df: DataFrame with cleaned performance data

        Returns:
            List of insight dictionaries with title, description, and impact
        """
        insights = []

        if df.empty:
            return insights

        try:
            # 1. Identify the slowest loading brand (highest average LCP)
            if 'lcp' in df.columns and 'brand' in df.columns:
                brand_lcp = df.groupby('brand')['lcp'].mean().reset_index()
                slowest_brand = brand_lcp.loc[brand_lcp['lcp'].idxmax()]

                insights.append({
                    'title': 'Slowest Loading Brand',
                    'description': f"{slowest_brand['brand']} has the slowest average loading time (LCP: {slowest_brand['lcp']:.2f}s).",
                    'impact': 'High',
                    'recommendation': f"Optimize images and server response time for {slowest_brand['brand']} to improve user experience and reduce bounce rates."
                })

            # 2. Identify the brand with the best Core Web Vitals
            if 'core_web_vitals_score' in df.columns and 'brand' in df.columns:
                brand_cwv = df.groupby('brand')['core_web_vitals_score'].mean().reset_index()
                best_brand = brand_cwv.loc[brand_cwv['core_web_vitals_score'].idxmax()]

                insights.append({
                    'title': 'Brand With Best Core Web Vitals',
                    'description': f"{best_brand['brand']} has the best Core Web Vitals score ({best_brand['core_web_vitals_score']:.1f}).",
                    'impact': 'Medium',
                    'recommendation': f"Study the implementation of {best_brand['brand']} as a benchmark for other brands."
                })

            # 3. Mobile vs Desktop performance gap
            if 'performance_score' in df.columns and 'mode' in df.columns:
                mode_perf = df.groupby('mode')['performance_score'].mean().reset_index()

                if len(mode_perf) == 2:  # Both desktop and mobile data available
                    desktop = mode_perf[mode_perf['mode'] == 'desktop']['performance_score'].iloc[0]
                    mobile = mode_perf[mode_perf['mode'] == 'mobile']['performance_score'].iloc[0]
                    gap = desktop - mobile

                    if gap > 15:  # Significant gap
                        insights.append({
                            'title': 'Significant Mobile Performance Gap',
                            'description': f"Mobile performance scores are {gap:.1f} points lower than desktop scores.",
                            'impact': 'High',
                            'recommendation': "Focus on mobile optimization to address the performance gap. Consider implementing AMP pages or mobile-specific optimizations."
                        })

            # 4. Trend analysis - is performance improving?
            if 'date' in df.columns and 'performance_score' in df.columns:
                # Convert date to datetime if needed
                if df['date'].dtype == 'object':
                    df['date'] = pd.to_datetime(df['date'])

                # Group by date
                date_perf = df.groupby('date')['performance_score'].mean().reset_index()
                date_perf = date_perf.sort_values('date')

                if len(date_perf) >= 3:  # At least 3 data points for trend
                    oldest = date_perf.iloc[0]['performance_score']
                    latest = date_perf.iloc[-1]['performance_score']
                    change = latest - oldest

                    if change >= 5:
                        insights.append({
                            'title': 'Performance Improving',
                            'description': f"Overall performance has improved by {change:.1f} points since {date_perf.iloc[0]['date'].strftime('%Y-%m-%d')}.",
                            'impact': 'Medium',
                            'recommendation': "Continue current optimization efforts and focus on maintaining the positive trend."
                        })
                    elif change <= -5:
                        insights.append({
                            'title': 'Performance Declining',
                            'description': f"Overall performance has declined by {abs(change):.1f} points since {date_perf.iloc[0]['date'].strftime('%Y-%m-%d')}.",
                            'impact': 'High',
                            'recommendation': "Investigate recent changes that may have caused performance regression. Consider reverting changes or implementing new optimizations."
                        })

            # 5. Identify poor accessibility scores
            if 'accessibility_score' in df.columns and 'brand' in df.columns:
                # Get latest data only for current state
                latest_date = df['date'].max()
                latest_df = df[df['date'] == latest_date]

                brand_a11y = latest_df.groupby('brand')['accessibility_score'].mean().reset_index()
                poor_a11y_brands = brand_a11y[brand_a11y['accessibility_score'] < 75]

                if not poor_a11y_brands.empty:
                    brands_list = ", ".join(poor_a11y_brands['brand'].tolist())
                    insights.append({
                        'title': 'Poor Accessibility',
                        'description': f"The following brands have low accessibility scores: {brands_list}.",
                        'impact': 'Medium',
                        'recommendation': "Conduct an accessibility audit and address issues to improve inclusivity and comply with accessibility standards."
                    })

            # 6. SEO optimization opportunities
            if 'seo_score' in df.columns and 'brand' in df.columns:
                # Get latest data only for current state
                latest_date = df['date'].max()
                latest_df = df[df['date'] == latest_date]

                brand_seo = latest_df.groupby('brand')['seo_score'].mean().reset_index()
                poor_seo_brands = brand_seo[brand_seo['seo_score'] < 80]

                if not poor_seo_brands.empty:
                    brands_list = ", ".join(poor_seo_brands['brand'].tolist())
                    insights.append({
                        'title': 'SEO Improvement Needed',
                        'description': f"The following brands have SEO opportunities: {brands_list}.",
                        'impact': 'Medium',
                        'recommendation': "Optimize meta tags, descriptions, and heading structure to improve search engine visibility."
                    })

            # 7. Interaction to Next Paint (INP) optimization opportunities
            if 'inp' in df.columns and 'brand' in df.columns:
                # Group by brand and calculate average INP
                brand_inp = df.groupby('brand')['inp'].mean().reset_index()

                # Identify brands with poor INP (>200ms is considered slow according to Core Web Vitals)
                poor_inp_brands = brand_inp[brand_inp['inp'] > 200]

                if not poor_inp_brands.empty:
                    worst_brand = poor_inp_brands.loc[poor_inp_brands['inp'].idxmax()]
                    inp_value = worst_brand['inp']
                    # Convert to milliseconds if in seconds
                    if inp_value < 10:  # likely in seconds
                        inp_value = inp_value * 1000

                    insights.append({
                        'title': 'Poor Interaction Performance',
                        'description': f"{worst_brand['brand']} has an extremely slow Interaction to Next Paint (INP) of {inp_value:.0f}ms, which significantly impacts user experience.",
                        'impact': 'High',
                        'recommendation': "Optimize JavaScript execution, reduce unused code, and implement code-splitting to improve interactivity. Consider using Web Workers for heavy computations."
                    })

            # 8. NEW INSIGHT: Identify FCP optimization opportunities for the slowest loading brands
            if 'fcp' in df.columns and 'brand' in df.columns:
                # Group by brand and calculate average FCP
                brand_fcp = df.groupby('brand')['fcp'].mean().reset_index()

                # Identify brands with poor FCP (>2.5s is considered slow)
                poor_fcp_brands = brand_fcp[brand_fcp['fcp'] > 2.5]

                if not poor_fcp_brands.empty:
                    worst_fcp_brand = poor_fcp_brands.loc[poor_fcp_brands['fcp'].idxmax()]

                    insights.append({
                        'title': 'First Contentful Paint Optimization',
                        'description': f"{worst_fcp_brand['brand']} has a slow First Contentful Paint (FCP) of {worst_fcp_brand['fcp']:.2f}s, affecting perceived loading speed.",
                        'impact': 'Medium',
                        'recommendation': "Optimize critical rendering path, reduce render-blocking resources, and implement resource prioritization to improve initial page rendering."
                    })

        except Exception as e:
            logging.error(f"Error generating business insights: {e}")
            insights.append({
                'title': 'Data Analysis Error',
                'description': "There was an error analyzing the performance data for insights.",
                'impact': 'Unknown',
                'recommendation': "Check the data quality and try again."
            })

        return insights

    def generate_dashboard(self):
        """Generate the dashboard HTML and assets"""
        # Get historical data
        historical_df = self.data_manager.load_historical_data()
        if historical_df.empty:
            logging.error("No historical data found. Run check_fos_performance.py first.")
            return False

        # Get list of report directories
        report_dirs = self.data_manager.get_report_directories()

        # Create time series charts for the dashboard (static versions as fallback)
        charts_dir = os.path.join(self.dashboard_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        # Generate new time series charts based on all historical data
        try:
            create_core_metrics_chart(historical_df, charts_dir)
            create_lighthouse_scores_chart(historical_df, charts_dir)
            create_core_web_vitals_chart(historical_df, charts_dir)
            create_diagnostics_summary_chart(historical_df, charts_dir)  # Generate diagnostics summary chart
            logging.info(f"Generated static time series charts in {charts_dir}")
        except Exception as e:
            logging.error(f"Error generating time series charts: {e}")

        # Create report links HTML
        report_links_html = ""

        # Create a reports directory in the dashboard folder to store symlinks or copies
        reports_symlink_dir = os.path.join(self.dashboard_dir, "reports")
        os.makedirs(reports_symlink_dir, exist_ok=True)

        # Log total number of report directories found
        logging.info(f"Found {len(report_dirs)} report directories: {[os.path.basename(d) for d in report_dirs]}")

        for report_dir in report_dirs:
            report_name = os.path.basename(report_dir)
            # Convert report_YYYYMMDD_HHMMSS to a more readable format
            if report_name.startswith("report_") and len(report_name) >= 16:
                date_str = report_name[7:]  # Remove "report_"
                try:
                    date_obj = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                    readable_date = date_obj.strftime("%B %d, %Y at %H:%M:%S")
                except ValueError:
                    readable_date = report_name

                # Check if HTML report exists in this directory
                html_reports = glob.glob(os.path.join(report_dir, "*.html"))
                logging.info(f"Found {len(html_reports)} HTML reports in {report_name}: {[os.path.basename(h) for h in html_reports]}")

                if html_reports:
                    # Create a symlink or copy the report to the dashboard/reports directory
                    for html_report in html_reports:
                        html_filename = os.path.basename(html_report)
                        report_symlink_path = os.path.join(reports_symlink_dir, f"{report_name}_{html_filename}")

                        # Try to create symlink first, if fails, create a copy
                        try:
                            if os.path.exists(report_symlink_path):
                                os.remove(report_symlink_path)
                            os.symlink(html_report, report_symlink_path)
                            logging.info(f"Created symlink for {html_filename} to {report_symlink_path}")
                        except (OSError, AttributeError) as e:
                            # If symlink fails (e.g., on Windows), copy the file
                            logging.info(f"Symlink failed ({str(e)}), creating copy instead")
                            shutil.copy2(html_report, report_symlink_path)
                            logging.info(f"Copied {html_filename} to {report_symlink_path}")

                        # Use relative path from dashboard root to the report
                        relative_path = f"reports/{os.path.basename(report_symlink_path)}"
                        report_links_html += f'<div class="report-link"><a href="{relative_path}" target="_blank">Report from {readable_date}</a></div>\n'
                else:
                    logging.warning(f"No HTML reports found in {report_dir}")

                    # Try to copy the main report to this directory if it doesn't have HTML files
                    main_report_path = os.path.join(self.data_manager.base_dir, "current", "fos_performance_report.html")
                    if os.path.exists(main_report_path):
                        # Create a copy of the main report in the report directory
                        dest_report_path = os.path.join(report_dir, "fos_performance_report.html")
                        try:
                            shutil.copy2(main_report_path, dest_report_path)
                            logging.info(f"Copied main report to {dest_report_path}")

                            # Now create a symlink or copy for the dashboard
                            html_filename = os.path.basename(dest_report_path)
                            report_symlink_path = os.path.join(reports_symlink_dir, f"{report_name}_{html_filename}")

                            try:
                                if os.path.exists(report_symlink_path):
                                    os.remove(report_symlink_path)
                                os.symlink(dest_report_path, report_symlink_path)
                                logging.info(f"Created symlink for {html_filename} to {report_symlink_path}")
                            except (OSError, AttributeError) as e:
                                # If symlink fails, create a copy
                                logging.info(f"Symlink failed ({str(e)}), creating copy instead")
                                shutil.copy2(dest_report_path, report_symlink_path)
                                logging.info(f"Copied {html_filename} to {report_symlink_path}")

                            # Add to report links HTML
                            relative_path = f"reports/{os.path.basename(report_symlink_path)}"
                            report_links_html += f'<div class="report-link"><a href="{relative_path}" target="_blank">Report from {readable_date}</a></div>\n'
                        except Exception as e:
                            logging.error(f"Failed to copy main report to report directory: {e}")
                    else:
                        logging.warning(f"Main report not found at {main_report_path}. Cannot add HTML report to {report_dir}")

        # If we found no reports with HTML files, copy the main report to each archived directory
        if not report_links_html:
            logging.info("No HTML reports found in any directory, copying the main report")
            main_report_path = os.path.join(self.data_manager.base_dir, "current", "fos_performance_report.html")
            if os.path.exists(main_report_path):
                for report_dir in report_dirs:
                    report_name = os.path.basename(report_dir)
                    if report_name.startswith("report_"):
                        date_str = report_name[7:]
                        try:
                            date_obj = datetime.datetime.strptime(date_str, "%Y%m%d_%H%M%S")
                            readable_date = date_obj.strftime("%B %d, %Y at %H:%M:%S")
                        except ValueError:
                            readable_date = report_name

                        # Copy the main report to the dashboard/reports directory
                        report_copy_path = os.path.join(reports_symlink_dir, f"{report_name}_fos_performance_report.html")
                        shutil.copy2(main_report_path, report_copy_path)

                        # Use relative path from dashboard root
                        relative_path = f"reports/{os.path.basename(report_copy_path)}"
                        report_links_html += f'<div class="report-link"><a href="{relative_path}" target="_blank">Report from {readable_date}</a></div>\n'
            else:
                logging.warning(f"Main report not found at {main_report_path}. Cannot copy to report directories.")

        # If still no report links, provide a message
        if not report_links_html:
            report_links_html = "<div class='no-reports'>No historical reports found</div>"

        # Prepare visualization data and save to JSON
        viz_data = self.prepare_data_for_visualizations(historical_df)
        viz_data_path = os.path.join(self.data_dir, "dashboard_data.json")
        with open(viz_data_path, 'w') as f:
            json.dump(viz_data, f)
        logging.info(f"Saved visualization data to {viz_data_path}")

        # Copy the necessary JavaScript and CSS files
        self._ensure_assets_available()

        # Generate the main dashboard HTML with interactive features
        dashboard_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>FOS Website Performance Dashboard</title>
            <link rel="stylesheet" href="css/dashboard.css">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <header class="dashboard-header">
                <div>
                    <h1 class="dashboard-title">FOS Website Performance Dashboard</h1>
                    <p class="dashboard-subtitle">Tracking and analyzing performance metrics across all brands</p>
                </div>
                <div class="last-updated">
                    Last updated: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}
                </div>
            </header>
            
            <div id="loading-indicator" class="loading-indicator">
                <div class="spinner"></div>
                <p>Loading dashboard data...</p>
            </div>
            
            <div id="error-message" class="error-message">
                <h3>Error Loading Dashboard Data</h3>
                <p>There was a problem loading the dashboard data.</p>
                <p id="error-details"></p>
            </div>
            
            <div id="dashboard-content" style="display: none;">
                <div class="filters-container">
                    <div class="filters-title">
                        <h2>Dashboard Filters</h2>
                        <button class="filter-toggle" id="toggle-filters">Hide Filters</button>
                    </div>
                    
                    <div class="filter-grid" id="filters-section">
                        <div class="filter-section">
                            <h3>Time Range</h3>
                            <select id="time-range">
                                <option value="1m">Last Month</option>
                                <option value="3m" selected>Last 3 Months</option>
                                <option value="6m">Last 6 Months</option>
                                <option value="1y">Last Year</option>
                                <option value="all">All Time</option>
                            </select>
                        </div>
                        
                        <div class="filter-section">
                            <h3>Brand</h3>
                            <div class="filter-options" id="brand-filter">
                                <!-- Brands will be added dynamically via JavaScript -->
                            </div>
                        </div>
                        
                        <div class="filter-section">
                            <h3>Device Mode</h3>
                            <select id="device-mode">
                                <option value="all" selected>All Devices</option>
                                <option value="desktop">Desktop Only</option>
                                <option value="mobile">Mobile Only</option>
                            </select>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-content">
                    <!-- Key Performance Indicators -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Key Performance Indicators</h2>
                        </div>
                        <div class="kpi-container" id="kpi-container">
                            <div class="kpi-card">
                                <div class="kpi-label">Average Performance Score</div>
                                <div class="kpi-value" id="kpi-performance">-</div>
                                <div class="kpi-trend" id="kpi-performance-trend"></div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-label">Core Web Vitals Pass Rate</div>
                                <div class="kpi-value" id="kpi-cwv">-</div>
                                <div class="cwv-pass-progress">
                                    <div class="cwv-pass-progress-bar" id="cwv-progress-bar" style="width: 0%"></div>
                                </div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-label">Average LCP</div>
                                <div class="kpi-value" id="kpi-lcp">-</div>
                                <div class="kpi-trend" id="kpi-lcp-trend"></div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-label">Average FCP</div>
                                <div class="kpi-value" id="kpi-fcp">-</div>
                                <div class="kpi-trend" id="kpi-fcp-trend"></div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-label">Average CLS</div>
                                <div class="kpi-value" id="kpi-cls">-</div>
                                <div class="kpi-trend" id="kpi-cls-trend"></div>
                            </div>
                            
                            <div class="kpi-card">
                                <div class="kpi-label">Average INP</div>
                                <div class="kpi-value" id="kpi-inp">-</div>
                                <div class="kpi-trend" id="kpi-inp-trend"></div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Performance Score Chart -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Performance Score Trend</h2>
                            <div class="chart-controls">
                                <label for="score-chart-view-type">View: </label>
                                <select id="score-chart-view-type" onchange="updateScoreChart(dashboardState.filteredData)">
                                    <option value="daily">Daily</option>
                                    <option value="hourly">Hourly</option>
                                    <option value="weekly">Weekly</option>
                                    <option value="monthly">Monthly</option>
                                    <option value="yearly">Yearly</option>
                                </select>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="performance-scores-chart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Core Web Vitals Chart -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Core Web Vitals by Brand</h2>
                        </div>
                        <div class="chart-container">
                            <canvas id="core-web-vitals-chart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Diagnostics Summary Chart -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Diagnostics Summary: Page Size vs Requests</h2>
                        </div>
                        <div class="chart-container">
                            <canvas id="diagnostics-summary-chart"></canvas>
                        </div>
                    </div>
                    
                    <!-- Business Insights Section -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Business Insights & Recommendations</h2>
                        </div>
                        <div class="business-insights" id="business-insights">
                            <!-- Insights will be populated dynamically via JavaScript -->
                        </div>
                    </div>
                    
                    <!-- Performance Metrics Table -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Detailed Performance Metrics</h2>
                        </div>
                        <div style="overflow-x: auto;">
                            <table class="performance-table" id="performance-table">
                                <thead>
                                    <tr>
                                        <th>Brand</th>
                                        <th>Mode</th>
                                        <th>Performance</th>
                                        <th>LCP</th>
                                        <th>FCP</th>
                                        <th>CLS</th>
                                        <th>INP</th>
                                        <th>Core Web Vitals</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Table rows will be populated via JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <!-- Historical Reports Section -->
                    <div class="card full-width">
                        <div class="card-header">
                            <h2 class="card-title">Historical Reports</h2>
                        </div>
                        <div class="report-links-container">
                            {report_links_html}
                        </div>
                    </div>
                </div>
            </div>
            
            <footer class="footer">
                <p>FOS Website Performance Monitoring System &copy; {datetime.datetime.now().year}</p>
            </footer>
            
            <script src="js/dashboard.js"></script>
        </body>
        </html>
        """

        # Write the dashboard HTML to a file
        dashboard_path = os.path.join(self.dashboard_dir, "index.html")
        with open(dashboard_path, "w") as f:
            f.write(dashboard_html)

        logging.info(f"Enhanced dashboard generated at {dashboard_path}")
        return True

    def _ensure_assets_available(self):
        """Ensure all necessary asset files are available in the dashboard directory"""
        # Check if JS file exists, if not create a placeholder
        js_path = os.path.join(self.js_dir, "dashboard.js")
        css_path = os.path.join(self.css_dir, "dashboard.css")

        # The paths to the files we just created with our enhancements
        current_js_path = os.path.join(os.path.dirname(__file__), "dashboard", "js", "dashboard.js")
        current_css_path = os.path.join(os.path.dirname(__file__), "dashboard", "css", "dashboard.css")

        try:
            # Copy JS file if it exists
            if os.path.exists(current_js_path):
                shutil.copy2(current_js_path, js_path)
                logging.info(f"Copied dashboard.js from {current_js_path} to {js_path}")
            else:
                logging.warning(f"JS file not found at {current_js_path}. Check that it's being created correctly.")

            # Copy CSS file if it exists
            if os.path.exists(current_css_path):
                shutil.copy2(current_css_path, css_path)
                logging.info(f"Copied dashboard.css from {current_css_path} to {css_path}")
            else:
                logging.warning(f"CSS file not found at {current_css_path}. Check that it's being created correctly.")

        except Exception as e:
            logging.error(f"Error copying asset files: {e}")
            # Create placeholders if files don't exist
            os.makedirs(os.path.dirname(js_path), exist_ok=True)
            os.makedirs(os.path.dirname(css_path), exist_ok=True)

            if not os.path.exists(js_path):
                with open(js_path, 'w') as f:
                    f.write('// Dashboard JavaScript - Placeholder')

            if not os.path.exists(css_path):
                with open(css_path, 'w') as f:
                    f.write('/* Dashboard CSS - Placeholder */')

    def serve_dashboard(self):
        """Serve the dashboard on a local web server"""
        if not self.generate_dashboard():
            return False

        # Start HTTP server
        handler = http.server.SimpleHTTPRequestHandler
        os.chdir(self.dashboard_dir)

        try:
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                url = f"http://localhost:{self.port}"
                logging.info(f"Dashboard is available at {url}")

                # Open web browser
                webbrowser.open(url)

                # Serve until interrupted
                logging.info("Press Ctrl+C to stop the server")
                httpd.serve_forever()
        except KeyboardInterrupt:
            logging.info("Server stopped")
        except Exception as e:
            logging.error(f"Error serving dashboard: {e}")
            return False

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOS Website Performance Dashboard")
    parser.add_argument("--generate-only", action="store_true", help="Only generate the dashboard without serving it")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve the dashboard on")
    parser.add_argument("--jenkins", action="store_true", help="Generate dashboard in Jenkins mode with relative paths")

    args = parser.parse_args()

    dashboard = PerformanceDashboard()
    dashboard.port = args.port

    if args.generate_only:
        dashboard.generate_dashboard()
    else:
        dashboard.serve_dashboard()
