"""
Time series visualization functions for website performance metrics
This file contains functions for creating time series charts from historical performance data.
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import json
import re

def create_core_metrics_chart(df, output_dir):
    """Create time series charts for core performance metrics"""
    # Ensure numeric columns
    metrics = ['fcp', 'lcp', 'tti']
    for metric in metrics:
        # Convert string values like "2.3 s" to numeric 2.3
        if df[metric].dtype == 'object':
            df[metric] = df[metric].apply(lambda x:
                float(x.split()[0]) if isinstance(x, str) and len(x.split()) > 1
                else (float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').isdigit())
                else np.nan))

    # Group by date and brand, calculate averages
    metrics_data = df.groupby(['date', 'brand', 'mode'])[metrics].mean().reset_index()

    # Create chart for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))

        # Plot desktop data with solid lines
        desktop_data = metrics_data[metrics_data['mode'] == 'desktop']
        for brand in desktop_data['brand'].unique():
            brand_data = desktop_data[desktop_data['brand'] == brand]
            plt.plot(brand_data['date'], brand_data[metric],
                    marker='o', linestyle='-', label=f"{brand} - Desktop")

        # Plot mobile data with dashed lines
        mobile_data = metrics_data[metrics_data['mode'] == 'mobile']
        for brand in mobile_data['brand'].unique():
            brand_data = mobile_data[mobile_data['brand'] == brand]
            plt.plot(brand_data['date'], brand_data[metric],
                    marker='s', linestyle='--', label=f"{brand} - Mobile")

        # Add thresholds if applicable
        if metric == 'lcp':
            plt.axhline(y=2.5, color='g', linestyle='-', alpha=0.3, label='Good (≤2.5s)')
            plt.axhline(y=4.0, color='r', linestyle='-', alpha=0.3, label='Poor (>4.0s)')
        elif metric == 'fcp':
            plt.axhline(y=1.8, color='g', linestyle='-', alpha=0.3, label='Good (≤1.8s)')
            plt.axhline(y=3.0, color='r', linestyle='-', alpha=0.3, label='Poor (>3.0s)')

        # Use different titles based on metric
        metric_names = {
            'fcp': 'First Contentful Paint',
            'lcp': 'Largest Contentful Paint',
            'tti': 'Time to Interactive'
        }

        plt.title(f'{metric_names.get(metric, metric.upper())} Trend Over Time by Brand')
        plt.xlabel('Date')
        plt.ylabel('Time (seconds)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric}_trend.png'))
        plt.close()


def create_lighthouse_scores_chart(df, output_dir):
    """Create time series charts for Lighthouse scores"""
    score_columns = ['performance_score', 'accessibility_score', 'best_practices_score', 'seo_score']

    # Group by date and brand, calculate averages
    scores_data = df.groupby(['date', 'brand', 'mode'])[score_columns].mean().reset_index()

    # Create chart for each score
    for score in score_columns:
        plt.figure(figsize=(12, 6))

        # Plot desktop data with solid lines
        desktop_data = scores_data[scores_data['mode'] == 'desktop']
        for brand in desktop_data['brand'].unique():
            brand_data = desktop_data[desktop_data['brand'] == brand]
            plt.plot(brand_data['date'], brand_data[score],
                    marker='o', linestyle='-', label=f"{brand} - Desktop")

        # Plot mobile data with dashed lines
        mobile_data = scores_data[scores_data['mode'] == 'mobile']
        for brand in mobile_data['brand'].unique():
            brand_data = mobile_data[mobile_data['brand'] == brand]
            plt.plot(brand_data['date'], brand_data[score],
                    marker='s', linestyle='--', label=f"{brand} - Mobile")

        # Add threshold lines for good/poor scores
        plt.axhline(y=90, color='g', linestyle='-', alpha=0.3, label='Good (≥90)')
        plt.axhline(y=50, color='r', linestyle='-', alpha=0.3, label='Poor (<50)')

        # Use friendly names for titles
        score_names = {
            'performance_score': 'Performance',
            'accessibility_score': 'Accessibility',
            'best_practices_score': 'Best Practices',
            'seo_score': 'SEO'
        }

        plt.title(f'{score_names.get(score, score.replace("_score", "").title())} Score Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score (0-100)')
        plt.ylim(0, 100)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{score.replace("_score", "")}_score_trend.png'))
        plt.close()


def create_core_web_vitals_chart(df, output_dir):
    """Create time series chart for Core Web Vitals score"""
    # Ensure core_web_vitals_score is numeric
    df['core_web_vitals_score'] = pd.to_numeric(df['core_web_vitals_score'], errors='coerce')

    # Group by date and brand, calculate average CWV score
    cwv_data = df.groupby(['date', 'brand', 'mode'])['core_web_vitals_score'].mean().reset_index()

    plt.figure(figsize=(12, 6))

    # Plot desktop data with solid lines
    desktop_data = cwv_data[cwv_data['mode'] == 'desktop']
    for brand in desktop_data['brand'].unique():
        brand_data = desktop_data[desktop_data['brand'] == brand]
        plt.plot(brand_data['date'], brand_data['core_web_vitals_score'],
                marker='o', linestyle='-', label=f"{brand} - Desktop")

    # Plot mobile data with dashed lines
    mobile_data = cwv_data[cwv_data['mode'] == 'mobile']
    for brand in mobile_data['brand'].unique():
        brand_data = mobile_data[mobile_data['brand'] == brand]
        plt.plot(brand_data['date'], brand_data['core_web_vitals_score'],
                marker='s', linestyle='--', label=f"{brand} - Mobile")

    # Add threshold lines
    plt.axhline(y=90, color='g', linestyle='-', alpha=0.3, label='Good (≥90)')
    plt.axhline(y=50, color='r', linestyle='-', alpha=0.3, label='Poor (<50)')

    plt.title('Core Web Vitals Score Trend Over Time by Brand')
    plt.xlabel('Date')
    plt.ylabel('Core Web Vitals Score (0-100)')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'core_web_vitals_score_trend.png'))
    plt.close()

def create_diagnostics_summary_chart(df, output_dir):
    """Create a scatter plot of page size vs number of requests for each brand/mode combination"""
    # Check if page_size and request_count columns exist
    required_columns = ['page_size', 'request_count']
    alternative_columns = {
        'page_size': ['page_weight', 'total_byte_weight', 'transfer_size'],
        'request_count': ['requests', 'num_requests']
    }

    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Check for the Diagnostics Summary column
    has_diagnostics_summary = 'diagnostics_summary' in df.columns

    # If we have the Diagnostics Summary column, extract page size and request count from it
    if has_diagnostics_summary:
        logging.info("Found diagnostics_summary column, extracting page size and request data")

        def extract_from_diagnostics(diagnostic_str):
            """Extract page size and request count from diagnostics summary string"""
            if not isinstance(diagnostic_str, str):
                return None, None

            size_kb = None
            requests = None

            # Handle the specific format: 'Requests: 293' and 'Total Size: 3.82 MB'
            # Look for Requests pattern first
            request_patterns = [
                r'Requests:\s*(\d+)',
                r'requests:\s*(\d+)',
                r'total requests:\s*(\d+)',
                r'request count:\s*(\d+)',
                r'num requests:\s*(\d+)'
            ]

            for pattern in request_patterns:
                match = re.search(pattern, diagnostic_str, re.IGNORECASE)
                if match:
                    requests = int(match.group(1))
                    logging.debug(f"Extracted request count: {requests}")
                    break

            # Look for patterns like "Total Size: 3.82 MB" with proper unit handling
            size_patterns = [
                r'Total Size:\s*([\d.]+)\s*([KkMmGg][Bb])',
                r'Page Size:\s*([\d.]+)\s*([KkMmGg][Bb])',
                r'Transfer Size:\s*([\d.]+)\s*([KkMmGg][Bb])',
                r'Page Weight:\s*([\d.]+)\s*([KkMmGg][Bb])',
                # More generic patterns as fallback
                r'size:\s*([\d.]+)\s*([KkMmGg][Bb])',
                r'([\d.]+)\s*([KkMmGg][Bb])'
            ]

            for pattern in size_patterns:
                match = re.search(pattern, diagnostic_str, re.IGNORECASE)
                if match:
                    value = float(match.group(1))
                    unit = match.group(2).lower()

                    # Convert to KB
                    if unit.startswith('m'):  # MB
                        size_kb = value * 1024
                        logging.debug(f"Extracted page size: {value} {unit} -> {size_kb} KB")
                    elif unit.startswith('k'):  # KB
                        size_kb = value
                        logging.debug(f"Extracted page size: {value} {unit}")
                    elif unit.startswith('g'):  # GB
                        size_kb = value * 1024 * 1024
                        logging.debug(f"Extracted page size: {value} {unit} -> {size_kb} KB")
                    else:  # Assume bytes
                        size_kb = value / 1024
                        logging.debug(f"Extracted page size: {value} {unit} -> {size_kb} KB")
                    break

            # Handle HTML break tags if present in the string
            if not requests or not size_kb:
                parts = diagnostic_str.split("<br>")
                for part in parts:
                    if not requests:
                        for pattern in request_patterns:
                            match = re.search(pattern, part, re.IGNORECASE)
                            if match:
                                requests = int(match.group(1))
                                break

                    if not size_kb:
                        for pattern in size_patterns:
                            match = re.search(pattern, part, re.IGNORECASE)
                            if match:
                                value = float(match.group(1))
                                unit = match.group(2).lower()

                                # Convert to KB
                                if unit.startswith('m'):  # MB
                                    size_kb = value * 1024
                                elif unit.startswith('k'):  # KB
                                    size_kb = value
                                elif unit.startswith('g'):  # GB
                                    size_kb = value * 1024 * 1024
                                else:  # Assume bytes
                                    size_kb = value / 1024
                                break

            return size_kb, requests

        # Extract page size and requests from diagnostics_summary column
        extracted_data = df['diagnostics_summary'].apply(extract_from_diagnostics)

        # Create new columns from extracted data
        df['page_size_kb'] = extracted_data.apply(lambda x: x[0])
        df['request_count'] = extracted_data.apply(lambda x: x[1])

        # Log how many values were successfully extracted
        valid_size_count = df['page_size_kb'].notna().sum()
        valid_request_count = df['request_count'].notna().sum()
        logging.info(f"Successfully extracted {valid_size_count} page sizes and {valid_request_count} request counts from diagnostics_summary")

    # Try to find or create the required columns from alternatives if not already present
    for req_col, alt_cols in alternative_columns.items():
        if req_col not in df.columns:
            # Try alternative column names
            found = False
            for alt_col in alt_cols:
                if alt_col in df.columns:
                    df[req_col] = df[alt_col]
                    found = True
                    break

            if not found:
                logging.warning(f"Could not find {req_col} or its alternatives. Diagnostics chart may be incomplete.")
                # Create empty column as placeholder
                df[req_col] = np.nan

    # Convert page_size to numeric KB if it's a string
    if 'page_size' in df.columns and df['page_size'].dtype == 'object':
        def convert_to_kb(size_str):
            if not isinstance(size_str, str):
                return size_str

            size_str = size_str.lower()
            if 'kb' in size_str:
                return float(size_str.replace('kb', '').strip())
            elif 'mb' in size_str:
                return float(size_str.replace('mb', '').strip()) * 1024
            elif 'b' in size_str and 'kb' not in size_str and 'mb' not in size_str:
                return float(size_str.replace('b', '').strip()) / 1024
            else:
                try:
                    return float(size_str) / 1024  # Assume bytes if no unit
                except:
                    return np.nan

        df['page_size_kb'] = df['page_size'].apply(convert_to_kb)
    elif 'page_size' in df.columns and pd.api.types.is_numeric_dtype(df['page_size']):
        # If page_size is already numeric, convert from bytes to KB
        df['page_size_kb'] = df['page_size'] / 1024

    # Convert request_count to numeric if it's a string
    if 'request_count' in df.columns and df['request_count'].dtype == 'object':
        df['request_count'] = pd.to_numeric(df['request_count'], errors='coerce')

    # Consolidate page size and request count data - prefer diagnostics_summary extraction if available
    if 'page_size' in df.columns and df['page_size_kb'].isna().any():
        df.loc[df['page_size_kb'].isna(), 'page_size_kb'] = df.loc[df['page_size_kb'].isna(), 'page_size']

    # Filter out rows with missing data
    diagnostic_data = df.dropna(subset=['page_size_kb', 'request_count']).copy()

    if diagnostic_data.empty:
        logging.warning("No valid diagnostic data for page size vs requests chart.")
        return

    # Group by brand and mode, calculate latest values
    latest_date = diagnostic_data['date'].max()
    latest_data = diagnostic_data[diagnostic_data['date'] == latest_date]

    # If no data for the latest date, use all data
    if latest_data.empty:
        logging.warning("No data for the latest date. Using all available data for diagnostics chart.")
        latest_data = diagnostic_data

    # Calculate average page size and request count for each brand/mode
    brand_mode_data = latest_data.groupby(['brand', 'mode']).agg({
        'page_size_kb': 'mean',
        'request_count': 'mean'
    }).reset_index()

    # Log the actual values being generated
    for _, row in brand_mode_data.iterrows():
        logging.info(f"Diagnostics data: {row['brand']} ({row['mode']}): Page size = {row['page_size_kb']:.2f} KB, Requests = {row['request_count']:.0f}")

    # Save diagnostics data to a JSON file for the dashboard to use
    diagnostics_json_path = os.path.join(output_dir, 'diagnostics_data.json')
    brand_mode_data.to_json(diagnostics_json_path, orient='records')
    logging.info(f"Saved diagnostics data to {diagnostics_json_path}")

    # Create the scatter plot
    plt.figure(figsize=(14, 8))

    # Desktop points - circles
    desktop_data = brand_mode_data[brand_mode_data['mode'] == 'desktop']
    if not desktop_data.empty:
        plt.scatter(
            desktop_data['request_count'],
            desktop_data['page_size_kb'],
            s=100,
            alpha=0.7,
            marker='o',
            c=np.arange(len(desktop_data)),
            cmap='viridis',
            label='Desktop'
        )

        # Add brand labels to points
        for i, row in desktop_data.iterrows():
            plt.text(row['request_count'] * 1.02, row['page_size_kb'] * 1.02,
                    row['brand'], fontsize=9)

    # Mobile points - triangles
    mobile_data = brand_mode_data[brand_mode_data['mode'] == 'mobile']
    if not mobile_data.empty:
        plt.scatter(
            mobile_data['request_count'],
            mobile_data['page_size_kb'],
            s=100,
            alpha=0.7,
            marker='^',
            c=np.arange(len(mobile_data) + len(desktop_data)),
            cmap='viridis',
            label='Mobile'
        )

        # Add brand labels to points
        for i, row in mobile_data.iterrows():
            plt.text(row['request_count'] * 1.02, row['page_size_kb'] * 1.02,
                    row['brand'], fontsize=9)

    plt.title('Page Size vs Number of Requests by Brand and Mode')
    plt.xlabel('Number of Requests')
    plt.ylabel('Page Size (KB)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a logarithmic trend line if enough data
    if len(brand_mode_data) > 3:
        x = brand_mode_data['request_count']
        y = brand_mode_data['page_size_kb']
        z = np.polyfit(x, np.log(y), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = np.exp(p(x_trend))
        plt.plot(x_trend, y_trend, "r--", alpha=0.7, label="Trend")

    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'page_size_vs_requests.png')
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(f"Diagnostics summary chart saved to {output_path}")

    # Also save a copy of the data as CSV for easier debugging
    csv_path = os.path.join(output_dir, 'diagnostics_data.csv')
    brand_mode_data.to_csv(csv_path, index=False)
    logging.info(f"Diagnostics data also saved as CSV to {csv_path}")

    return brand_mode_data

def create_time_series_charts(df, output_dir):
    """Create all time series charts for website performance metrics"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Creating core metrics charts...")
    create_core_metrics_chart(df, output_dir)

    logging.info("Creating Lighthouse scores charts...")
    create_lighthouse_scores_chart(df, output_dir)

    logging.info("Creating Core Web Vitals score chart...")
    create_core_web_vitals_chart(df, output_dir)

    logging.info("Creating diagnostics summary chart...")
    create_diagnostics_summary_chart(df, output_dir)

    logging.info("Time series charts created successfully.")

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('website_performance_data.csv')  # Load your data here
    output_dir = 'time_series_charts'
    create_time_series_charts(df, output_dir)
