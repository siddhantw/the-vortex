"""
Historical data management for FOS website performance metrics
This module handles storing, retrieving, and managing historical performance data.
"""

import os
import json
import pandas as pd
import logging
import datetime
import glob
from pathlib import Path


class HistoricalDataManager:
    """Manages historical performance data for FOS websites"""

    def __init__(self, base_dir=None):
        """
        Initialize the historical data manager

        Args:
            base_dir: Base directory for historical data storage. If None, uses default location.
        """
        if base_dir is None:
            # Default location is in the project directory
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            project_dir = current_dir.parent.parent.parent  # Go up three levels
            self.base_dir = os.path.join(project_dir, "reports", "website_performance")
        else:
            self.base_dir = base_dir

        # Ensure the directory structure exists
        self.archive_dir = os.path.join(self.base_dir, "historical_data")
        self.current_report_dir = os.path.join(self.base_dir, "current")

        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(self.current_report_dir, exist_ok=True)

        # Initialize the data file
        self.data_file_path = os.path.join(self.archive_dir, "performance_history.csv")
        self.initialize_data_file_if_needed()

        logging.info(f"Historical data manager initialized with base directory: {self.base_dir}")

    def initialize_data_file_if_needed(self):
        """Create the data file if it doesn't exist"""
        if not os.path.exists(self.data_file_path):
            # Create an empty dataframe with the required columns
            columns = [
                'date', 'brand', 'mode', 'url',
                'fcp', 'lcp', 'cls', 'inp', 'ttfb', 'tbt', 'tti',
                'performance_score', 'accessibility_score', 'best_practices_score',
                'seo_score', 'core_web_vitals_score'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.data_file_path, index=False)
            logging.info(f"Created new historical data file at {self.data_file_path}")

    def save_current_run_data(self, metrics_df):
        """
        Save the current run data to historical storage

        Args:
            metrics_df: DataFrame with current run metrics data
        """
        # Add timestamp to the data if not already present
        if 'date' not in metrics_df.columns:
            metrics_df['date'] = datetime.datetime.now().strftime('%Y-%m-%d')

        # Save to archive
        if os.path.exists(self.data_file_path):
            historical_df = pd.read_csv(self.data_file_path)
            combined_df = pd.concat([historical_df, metrics_df], ignore_index=True)
            combined_df.to_csv(self.data_file_path, index=False)
        else:
            metrics_df.to_csv(self.data_file_path, index=False)

        logging.info(f"Added {len(metrics_df)} new records to historical data")

        # Create a timestamped directory for the current report
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_dir = os.path.join(self.archive_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)

        # Save the current metrics data in the timestamped directory
        metrics_df.to_csv(os.path.join(report_dir, "metrics.csv"), index=False)

        # Copy any charts/images from the current report directory
        if os.path.exists(self.current_report_dir):
            for file in glob.glob(os.path.join(self.current_report_dir, "*.png")):
                filename = os.path.basename(file)
                command = f"cp '{file}' '{os.path.join(report_dir, filename)}'"
                os.system(command)

        logging.info(f"Archived current report to {report_dir}")
        return report_dir

    def load_historical_data(self):
        """
        Load all historical performance data

        Returns:
            DataFrame with all historical data
        """
        if os.path.exists(self.data_file_path):
            return pd.read_csv(self.data_file_path)
        else:
            logging.warning(f"Historical data file not found at {self.data_file_path}")
            return pd.DataFrame()

    def get_report_directories(self):
        """
        Get list of all report directories

        Returns:
            List of report directory paths sorted by date (newest first)
        """
        if not os.path.exists(self.archive_dir):
            return []

        report_dirs = [d for d in os.listdir(self.archive_dir)
                      if d.startswith("report_") and
                      os.path.isdir(os.path.join(self.archive_dir, d))]

        # Sort by timestamp (newest first)
        report_dirs.sort(reverse=True)

        return [os.path.join(self.archive_dir, d) for d in report_dirs]

    def generate_historical_charts(self, output_dir):
        """
        Generate time series charts from historical data

        Args:
            output_dir: Directory to save the generated charts
        """
        from time_series_charts import (
            create_core_metrics_chart,
            create_lighthouse_scores_chart,
            create_core_web_vitals_chart
        )

        historical_df = self.load_historical_data()
        if historical_df.empty:
            logging.warning("No historical data available for generating charts")
            return

        # Generate charts
        create_core_metrics_chart(historical_df, output_dir)
        create_lighthouse_scores_chart(historical_df, output_dir)
        create_core_web_vitals_chart(historical_df, output_dir)

        logging.info(f"Generated historical charts in {output_dir}")
