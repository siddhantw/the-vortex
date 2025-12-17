#!/usr/bin/env python3
"""
Enhanced GTMetrix-Style Performance Dashboard
A comprehensive dashboard for viewing historical performance data with advanced features including
Core Web Vitals grades, performance structure analysis, speed visualization, CrUX metrics,
waterfall charts, page load videos, and detailed report history.
"""

import os
import sys
import json
import logging
import datetime
import argparse
import http.server
import socketserver
import webbrowser
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import numpy as np
# Configure matplotlib to use non-interactive backend to prevent hanging
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from dateutil import parser
from dateutil.relativedelta import relativedelta

# Add parent directory to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
scripts_root = os.path.dirname(parent_dir)  # /scripts
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if scripts_root not in sys.path:
    sys.path.append(scripts_root)

from enhanced_gtmetrix_historical_data_manager import EnhancedGTMetrixHistoricalDataManager
# Optional Azure OpenAI for AI insights
try:
    import importlib
    _mod = importlib.import_module('gen_ai.azure_openai_client')
    AzureOpenAIClient = getattr(_mod, 'AzureOpenAIClient', None)
except Exception:
    AzureOpenAIClient = None


class EnhancedPerformanceDashboard:
    """Enhanced dashboard for comprehensive performance analysis and historical tracking"""
    
    def __init__(self, base_dir=None):
        """Initialize the enhanced performance dashboard"""
        self.data_manager = EnhancedGTMetrixHistoricalDataManager(base_dir=base_dir)
        self.dashboard_dir = os.path.join(self.data_manager.base_dir, "enhanced_dashboard")
        self.port = 8001
        # Store last generated data for HTML rendering
        self.dashboard_data = None
        
        # Create dashboard directories
        self.js_dir = os.path.join(self.dashboard_dir, "js")
        self.css_dir = os.path.join(self.dashboard_dir, "css")
        self.data_dir = os.path.join(self.dashboard_dir, "data")
        self.charts_dir = os.path.join(self.dashboard_dir, "charts")
        self.assets_dir = os.path.join(self.dashboard_dir, "assets")
        
        for directory in [self.dashboard_dir, self.js_dir, self.css_dir, 
                         self.data_dir, self.charts_dir, self.assets_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Performance thresholds for grading
        self.performance_thresholds = {
            'lcp': {'good': 2.5, 'needs_improvement': 4.0},
            'fcp': {'good': 1.8, 'needs_improvement': 3.0},
            'cls': {'good': 0.1, 'needs_improvement': 0.25},
            'inp': {'good': 200, 'needs_improvement': 500},
            'ttfb': {'good': 0.8, 'needs_improvement': 1.8},
            'tbt': {'good': 200, 'needs_improvement': 600},
            'speed_index': {'good': 3.4, 'needs_improvement': 5.8}
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        
        logging.info(f"Enhanced dashboard initialized at: {self.dashboard_dir}")

    def calculate_cwv_grade(self, lcp, fcp, cls, inp):
        """Calculate Core Web Vitals grade based on thresholds"""
        scores = []
        
        # LCP Score
        if lcp != "N/A" and lcp is not None:
            try:
                lcp_val = float(lcp)
                if lcp_val <= 2.5:
                    scores.append(100)
                elif lcp_val <= 4.0:
                    scores.append(75)
                else:
                    scores.append(50)
            except (ValueError, TypeError):
                pass
        
        # FCP Score
        if fcp != "N/A" and fcp is not None:
            try:
                fcp_val = float(fcp)
                if fcp_val <= 1.8:
                    scores.append(100)
                elif fcp_val <= 3.0:
                    scores.append(75)
                else:
                    scores.append(50)
            except (ValueError, TypeError):
                pass
        
        # CLS Score
        if cls != "N/A" and cls is not None:
            try:
                cls_val = float(cls)
                if cls_val <= 0.1:
                    scores.append(100)
                elif cls_val <= 0.25:
                    scores.append(75)
                else:
                    scores.append(50)
            except (ValueError, TypeError):
                pass
        
        # INP Score
        if inp != "N/A" and inp is not None:
            try:
                inp_val = float(inp)
                if inp_val <= 200:
                    scores.append(100)
                elif inp_val <= 500:
                    scores.append(75)
                else:
                    scores.append(50)
            except (ValueError, TypeError):
                pass
        
        if not scores:
            return 'N/A', 0
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 90:
            return 'A', avg_score
        elif avg_score >= 80:
            return 'B', avg_score
        elif avg_score >= 70:
            return 'C', avg_score
        elif avg_score >= 60:
            return 'D', avg_score
        else:
            return 'F', avg_score

    def _safe_numeric_conversion(self, value):
        """Safely convert value to numeric, return 'N/A' if conversion fails"""
        if value is None or value == '' or value == 'N/A':
            return 'N/A'
        try:
            return float(value)
        except (ValueError, TypeError):
            return 'N/A'

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types and other objects"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN values
            if np.isnan(obj):
                return "N/A"
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif pd.isna(obj):
            return "N/A"
        else:
            return str(obj)

    def generate_performance_trend_charts(self):
        """Generate comprehensive performance trend charts"""
        historical_df = self.data_manager.load_historical_data()
        if historical_df.empty:
            logging.warning("No historical data available for trend charts - generating placeholder charts")
            self.create_placeholder_charts()
            # Also attempt CrUX chart even if site data missing
            self.create_crux_chart()
            return
        
        historical_df['date'] = pd.to_datetime(historical_df['date'])
        self.create_performance_score_trend_chart(historical_df)
        self.create_cwv_trend_chart(historical_df)
        self.create_grade_distribution_chart(historical_df)
        self.create_performance_matrix_chart(historical_df)
        self.create_speed_index_chart(historical_df)
        # New enhanced charts
        self.create_crux_chart()
        self.create_brand_trend_charts(historical_df)
        logging.info("Performance trend charts generated successfully")

    def create_placeholder_charts(self):
        """Create placeholder charts when no data is available"""
        # Create placeholder performance trend chart
        plt.figure(figsize=(14, 8))
        plt.text(0.5, 0.5, 'No Performance Data Available\nRun some performance tests to see trends here', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title('Performance Scores Trend Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'performance_trend.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create placeholder CWV chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        for ax, title in zip([ax1, ax2, ax3, ax4], 
                           ['Largest Contentful Paint (LCP)', 'First Contentful Paint (FCP)', 
                            'Cumulative Layout Shift (CLS)', 'Interaction to Next Paint (INP)']):
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'cwv_trend.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create placeholder grade distribution chart
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'No Grade Data Available\nRun performance tests to see grade distribution', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title('Core Web Vitals Grade Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Device Mode', fontsize=12)
        plt.ylabel('Number of Tests', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'grade_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create placeholder performance matrix chart
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, 'No Matrix Data Available\nRun tests for multiple sites to see performance comparison', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title('Performance Score Matrix by Site and Device', fontsize=16, fontweight='bold')
        plt.xlabel('Device Mode', fontsize=12)
        plt.ylabel('Website', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'performance_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

        # Create placeholder speed index chart
        plt.figure(figsize=(14, 8))
        plt.text(0.5, 0.5, 'No Speed Index Data Available\nRun performance tests to see speed distribution', 
                ha='center', va='center', fontsize=16, transform=plt.gca().transAxes)
        plt.title('Speed Index Distribution by Device and Site', fontsize=16, fontweight='bold')
        plt.xlabel('Device Mode', fontsize=12)
        plt.ylabel('Speed Index (seconds)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'speed_index.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create placeholder CrUX chart
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No CrUX data found (crux.json)', ha='center', va='center', fontsize=14)
        plt.title('CrUX Distributions (Good/Needs Improvement/Poor)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'crux_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Create placeholder brand trends chart
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No brand data available', ha='center', va='center', fontsize=14)
        plt.title('Brand Performance Trends')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'brand_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logging.info("Placeholder charts created successfully")

    def create_performance_score_trend_chart(self, df):
        """Create performance score trend chart"""
        plt.figure(figsize=(14, 8))
        
        # Group by date and calculate average scores
        daily_scores = df.groupby(['date', 'device_mode']).agg({
            'performance_score': 'mean',
            'accessibility_score': 'mean',
            'best_practices_score': 'mean',
            'seo_score': 'mean'
        }).reset_index()
        
        # Plot trends for each score type
        for device_mode in daily_scores['device_mode'].unique():
            device_data = daily_scores[daily_scores['device_mode'] == device_mode]
            
            plt.plot(device_data['date'], device_data['performance_score'], 
                    marker='o', label=f'Performance ({device_mode})', linewidth=2)
            plt.plot(device_data['date'], device_data['accessibility_score'], 
                    marker='s', label=f'Accessibility ({device_mode})', linewidth=2)
            plt.plot(device_data['date'], device_data['best_practices_score'], 
                    marker='^', label=f'Best Practices ({device_mode})', linewidth=2)
            plt.plot(device_data['date'], device_data['seo_score'], 
                    marker='d', label=f'SEO ({device_mode})', linewidth=2)
        
        plt.title('Performance Scores Trend Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.charts_dir, 'performance_trend.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_cwv_trend_chart(self, df):
        """Create Core Web Vitals trend chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by date and calculate average metrics
        daily_metrics = df.groupby(['date', 'device_mode']).agg({
            'lcp': 'mean',
            'fcp': 'mean',
            'cls': 'mean',
            'inp': 'mean'
        }).reset_index()
        
        # LCP Trend
        for device_mode in daily_metrics['device_mode'].unique():
            device_data = daily_metrics[daily_metrics['device_mode'] == device_mode]
            ax1.plot(device_data['date'], device_data['lcp'], 
                    marker='o', label=device_mode, linewidth=2)
        
        ax1.axhline(y=2.5, color='green', linestyle='--', alpha=0.7, label='Good (≤2.5s)')
        ax1.axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='Needs Improvement (≤4.0s)')
        ax1.set_title('Largest Contentful Paint (LCP) Trend')
        ax1.set_ylabel('Seconds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # FCP Trend
        for device_mode in daily_metrics['device_mode'].unique():
            device_data = daily_metrics[daily_metrics['device_mode'] == device_mode]
            ax2.plot(device_data['date'], device_data['fcp'], 
                    marker='s', label=device_mode, linewidth=2)
        
        ax2.axhline(y=1.8, color='green', linestyle='--', alpha=0.7, label='Good (≤1.8s)')
        ax2.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, label='Needs Improvement (≤3.0s)')
        ax2.set_title('First Contentful Paint (FCP) Trend')
        ax2.set_ylabel('Seconds')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # CLS Trend
        for device_mode in daily_metrics['device_mode'].unique():
            device_data = daily_metrics[daily_metrics['device_mode'] == device_mode]
            ax3.plot(device_data['date'], device_data['cls'], 
                    marker='^', label=device_mode, linewidth=2)
        
        ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Good (≤0.1)')
        ax3.axhline(y=0.25, color='orange', linestyle='--', alpha=0.7, label='Needs Improvement (≤0.25)')
        ax3.set_title('Cumulative Layout Shift (CLS) Trend')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # INP Trend
        for device_mode in daily_metrics['device_mode'].unique():
            device_data = daily_metrics[daily_metrics['device_mode'] == device_mode]
            ax4.plot(device_data['date'], device_data['inp'], 
                    marker='d', label=device_mode, linewidth=2)
        
        ax4.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Good (≤200ms)')
        ax4.axhline(y=500, color='orange', linestyle='--', alpha=0.7, label='Needs Improvement (≤500ms)')
        ax4.set_title('Interaction to Next Paint (INP) Trend')
        ax4.set_ylabel('Milliseconds')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'cwv_trend.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_grade_distribution_chart(self, df):
        """Create grade distribution chart"""
        # Calculate grades for each record
        grades = []
        for _, row in df.iterrows():
            grade, _ = self.calculate_cwv_grade(row.get('lcp'), row.get('fcp'), 
                                              row.get('cls'), row.get('inp'))
            grades.append(grade)
        
        df['cwv_grade'] = grades
        
        # Create distribution chart
        plt.figure(figsize=(12, 8))
        
        # Grade distribution by device mode
        grade_counts = df.groupby(['device_mode', 'cwv_grade']).size().unstack(fill_value=0)
        
        grade_counts.plot(kind='bar', stacked=True, 
                         color=['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#F44336'],
                         alpha=0.8)
        
        plt.title('Core Web Vitals Grade Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Device Mode', fontsize=12)
        plt.ylabel('Number of Tests', fontsize=12)
        plt.legend(title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.charts_dir, 'grade_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_performance_matrix_chart(self, df):
        """Create performance comparison matrix heatmap"""
        # Create pivot table for performance scores
        pivot_data = df.pivot_table(
            values='performance_score',
            index='name',
            columns='device_mode',
            aggfunc='mean'
        ).fillna(0)
        
        plt.figure(figsize=(10.0, max(8.0, float(len(pivot_data)) * 0.5)))

        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=75, vmin=0, vmax=100,
                   cbar_kws={'label': 'Performance Score'})
        
        plt.title('Performance Score Matrix by Site and Device', fontsize=16, fontweight='bold')
        plt.xlabel('Device Mode', fontsize=12)
        plt.ylabel('Website', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.charts_dir, 'performance_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_speed_index_chart(self, df):
        """Create speed index visualization"""
        plt.figure(figsize=(14, 8))
        
        # Filter out N/A values and convert to numeric
        speed_data = df[df['speed_index'] != 'N/A'].copy()
        if not speed_data.empty:
            speed_data['speed_index'] = pd.to_numeric(speed_data['speed_index'], errors='coerce')
            speed_data = speed_data.dropna(subset=['speed_index'])
            
            # Create violin plot
            import seaborn as sns
            sns.violinplot(data=speed_data, x='device_mode', y='speed_index', 
                          hue='name', palette='Set3')
            
            plt.axhline(y=3.4, color='green', linestyle='--', alpha=0.7, label='Good (≤3.4s)')
            plt.axhline(y=5.8, color='orange', linestyle='--', alpha=0.7, label='Needs Improvement (≤5.8s)')
            
            plt.title('Speed Index Distribution by Device and Site', fontsize=16, fontweight='bold')
            plt.xlabel('Device Mode', fontsize=12)
            plt.ylabel('Speed Index (seconds)', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.charts_dir, 'speed_index.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()

    def create_crux_chart(self):
        """Create CrUX distribution stacked bars if crux.json available, else compute from history."""
        crux = None
        # Prefer already prepared dashboard data
        if isinstance(self.dashboard_data, dict):
            crux = self.dashboard_data.get('crux')
        # Fallback to loading from file
        if not crux or not crux.get('overall'):
            crux = self.data_manager.load_crux_data()
        # Final fallback: compute from historical data
        if not crux or not crux.get('overall'):
            df = self.data_manager.load_historical_data()
            crux = self.compute_crux_overview_from_df(df)
        metrics = ['lcp', 'fcp', 'cls', 'inp']
        labels = {'lcp': 'LCP', 'fcp': 'FCP', 'cls': 'CLS', 'inp': 'INP'}
        if not crux or 'overall' not in crux:
            # Create placeholder
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No CrUX data found (crux.json)', ha='center', va='center', fontsize=14)
            plt.title('CrUX Distributions (Good/Needs Improvement/Poor)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, 'crux_overview.png'), dpi=300, bbox_inches='tight')
            plt.close()
            return
        # Prepare data
        overall = crux.get('overall', {})
        x = np.arange(len(metrics))
        good = [overall.get(m, {}).get('good', 0) for m in metrics]
        ni = [overall.get(m, {}).get('ni', 0) for m in metrics]
        poor = [overall.get(m, {}).get('poor', 0) for m in metrics]
        width = 0.6
        plt.figure(figsize=(10, 6))
        plt.bar(x, good, width, label='Good', color='#10b981')
        plt.bar(x, ni, width, bottom=good, label='Needs Improvement', color='#f59e0b')
        bottom_poor = [g + n for g, n in zip(good, ni)]
        plt.bar(x, poor, width, bottom=bottom_poor, label='Poor', color='#ef4444')
        plt.xticks(x, [labels[m] for m in metrics])
        plt.ylabel('Percentage')
        title = 'CrUX Distributions (Good/NI/Poor)'
        if crux.get('updated'):
            title += f" • Updated: {crux['updated']}"
        if crux.get('source') == 'computed_from_history':
            title += ' • (from history)'
        plt.title(title)
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.charts_dir, 'crux_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_brand_trend_charts(self, df: pd.DataFrame):
        """Create small-multiple charts for top brands by number of tests"""
        if 'brand' not in df.columns or df['brand'].dropna().empty:
            # Placeholder
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No brand data available', ha='center', va='center', fontsize=14)
            plt.title('Brand Performance Trends')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, 'brand_trends.png'), dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Work on a copy and ensure needed columns exist
        df_plot = df.copy()
        # Ensure datetime for sorting/plotting
        try:
            df_plot['date'] = pd.to_datetime(df_plot['date'], errors='coerce')
        except Exception:
            pass

        # Ensure cwv_score exists; compute if missing
        if 'cwv_score' not in df_plot.columns:
            def _row_cwv_score(row):
                try:
                    _, score = self.calculate_cwv_grade(row.get('lcp'), row.get('fcp'), row.get('cls'), row.get('inp'))
                    return float(score) if score is not None else 0.0
                except Exception:
                    return 0.0
            try:
                df_plot['cwv_score'] = df_plot.apply(_row_cwv_score, axis=1)
            except Exception:
                # As a last resort, fill with zeros to avoid KeyError
                df_plot['cwv_score'] = 0.0

        # Determine top brands
        brand_counts = df_plot['brand'].value_counts().head(6)
        top_brands = list(brand_counts.index)
        if not top_brands:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, 'No brand data available', ha='center', va='center', fontsize=14)
            plt.title('Brand Performance Trends')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.charts_dir, 'brand_trends.png'), dpi=300, bbox_inches='tight')
            plt.close()
            return

        # Create grid
        rows = int(np.ceil(len(top_brands) / 3))
        cols = min(3, len(top_brands))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows), squeeze=False)
        last_idx = -1
        for idx, brand in enumerate(top_brands):
            last_idx = idx
            r = idx // cols
            c = idx % cols
            ax = axes[r][c]
            sub = df_plot[df_plot['brand'] == brand].copy()
            sub = sub.sort_values('date')

            # Performance trend (if available)
            if 'performance_score' in sub.columns:
                ps = pd.to_numeric(sub['performance_score'], errors='coerce')
                ax.plot(sub['date'], ps, marker='o', linewidth=1.8, label='Performance')

            # CWV score trend (always available now)
            cs = pd.to_numeric(sub['cwv_score'], errors='coerce')
            ax.plot(sub['date'], cs, marker='s', linewidth=1.5, label='CWV score')

            ax.set_title(str(brand))
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for j in range(last_idx + 1, rows*cols):
            r = j // cols
            c = j % cols
            axes[r][c].axis('off')

        # Build a consolidated legend from first populated axis
        handles, labels = [], []
        for r in range(rows):
            for c in range(cols):
                h, l = axes[r][c].get_legend_handles_labels()
                if h and l:
                    handles, labels = h, l
                    break
            if handles:
                break
        if handles:
            fig.legend(handles, labels, loc='upper right')

        fig.suptitle('Brand Performance & CWV Score Trends', y=0.98, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig(os.path.join(self.charts_dir, 'brand_trends.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def compute_crux_overview_from_df(self, df: pd.DataFrame) -> dict:
        """Compute CrUX-like Good/NI/Poor distributions from historical metrics as a fallback.
        Returns a dict with an 'overall' section similar to load_crux_data output.
        """
        if df is None or df.empty:
            return {}
        result = {}
        # We will compute distributions for these metrics only
        specs = {
            'lcp': {'good': 2.5, 'ni': 4.0, 'unit': 's'},
            'fcp': {'good': 1.8, 'ni': 3.0, 'unit': 's'},
            'cls': {'good': 0.1, 'ni': 0.25, 'unit': ''},
            'inp': {'good': 200, 'ni': 500, 'unit': 'ms'}
        }
        overall = {}
        for m, t in specs.items():
            if m not in df.columns:
                continue
            series = df[m]
            # Convert to numeric, ignore 'N/A'
            if series.dtype == object:
                series = series[series != 'N/A']
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            if numeric.empty:
                overall[m] = {'good': 0.0, 'ni': 0.0, 'poor': 0.0}
                continue
            good = ni = poor = 0
            if m in ('lcp', 'fcp'):
                good = (numeric <= t['good']).sum()
                ni = ((numeric > t['good']) & (numeric <= t['ni'])).sum()
                poor = (numeric > t['ni']).sum()
            elif m == 'cls':
                good = (numeric <= t['good']).sum()
                ni = ((numeric > t['good']) & (numeric <= t['ni'])).sum()
                poor = (numeric > t['ni']).sum()
            elif m == 'inp':
                # inp is already expected in ms in our df based on compute; if seconds, values would be small; keep simple
                good = (numeric <= t['good']).sum()
                ni = ((numeric > t['good']) & (numeric <= t['ni'])).sum()
                poor = (numeric > t['ni']).sum()
            total = max(int(good + ni + poor), 1)
            overall[m] = {
                'good': round(100.0 * good / total, 2),
                'ni': round(100.0 * ni / total, 2),
                'poor': round(100.0 * poor / total, 2)
            }
        if not overall:
            return {}
        return {
            'overall': overall,
            'source': 'computed_from_history',
            'updated': datetime.datetime.now().strftime('%Y-%m-%d')
        }

    def prepare_dashboard_data(self):
        """Prepare comprehensive data for the dashboard"""
        historical_df = self.data_manager.load_historical_data()
        if historical_df.empty:
            logging.warning("No historical data available - generating sample dashboard")
            # Still include CrUX data if present for the UI
            sample = self.generate_sample_dashboard_data()
            sample['crux'] = self.data_manager.load_crux_data()
            data_file = os.path.join(self.data_dir, 'dashboard_data.json')
            with open(data_file, 'w') as f:
                json.dump(sample, f, indent=2, default=self._json_serializer)
            # Store for server-side rendering and inline fallback
            self.dashboard_data = sample
            return sample

        # Convert date column to datetime with robust error handling
        historical_df['date'] = pd.to_datetime(historical_df['date'], errors='coerce')
        # Remove rows with invalid dates
        historical_df = historical_df.dropna(subset=['date'])
        
        if historical_df.empty:
            logging.warning("No valid dates found in historical data")
            return self.generate_sample_dashboard_data()
        
        # Ensure brand/domain/page columns exist
        for col in ['brand', 'domain', 'page']:
            if col not in historical_df.columns:
                historical_df[col] = None

        # Ensure numeric columns are properly converted
        numeric_columns = ['performance_score', 'accessibility_score', 'best_practices_score', 'seo_score']
        for col in numeric_columns:
            if col in historical_df.columns:
                historical_df[col] = pd.to_numeric(historical_df[col], errors='coerce')
        
        # Handle Core Web Vitals metrics with proper numeric conversion
        cwv_columns = ['lcp', 'fcp', 'cls', 'inp', 'ttfb', 'tbt', 'speed_index']
        for col in cwv_columns:
            if col in historical_df.columns:
                # Convert to numeric, keeping "N/A" as string for non-numeric values
                historical_df[col] = historical_df[col].apply(self._safe_numeric_conversion)
        
        # Calculate grades with error handling
        grades = []
        for _, row in historical_df.iterrows():
            try:
                grade, score = self.calculate_cwv_grade(row.get('lcp'), row.get('fcp'), 
                                                      row.get('cls'), row.get('inp'))
                grades.append({'grade': grade, 'score': score})
            except Exception as e:
                logging.warning(f"Error calculating CWV grade for row: {e}")
                grades.append({'grade': 'N/A', 'score': 0})
        
        historical_df['cwv_grade'] = [g['grade'] for g in grades]
        historical_df['cwv_score'] = [g['score'] for g in grades]
        
        # Prepare summary statistics with safe numeric operations
        total_sites = len(historical_df['name'].unique()) if 'name' in historical_df.columns else 0
        total_tests = len(historical_df)
        
        # Calculate averages with error handling
        perf_scores = historical_df['performance_score'].dropna()
        avg_performance_score = float(perf_scores.mean()) if not perf_scores.empty else 0
        
        cwv_scores = historical_df['cwv_score'].dropna()
        avg_cwv_score = float(cwv_scores.mean()) if not cwv_scores.empty else 0
        
        latest_test_date = historical_df['date'].max().strftime('%Y-%m-%d')
        grade_distribution = historical_df['cwv_grade'].value_counts().to_dict()
        
        # Brand level summary
        brand_summary = self.get_brand_summary(historical_df)
        forecasts = self.compute_forecasts(historical_df)
        comparisons = self.compute_period_comparisons(historical_df)
        crux = self.data_manager.load_crux_data()
        # Fallback: compute CrUX-like overview from historical data if crux.json missing
        if not crux or not isinstance(crux, dict) or not crux.get('overall'):
            crux = self.compute_crux_overview_from_df(historical_df)

        dashboard_data = {
            'summary': {
                'total_sites': total_sites,
                'total_tests': total_tests,
                'avg_performance_score': avg_performance_score,
                'avg_cwv_score': avg_cwv_score,
                'latest_test_date': latest_test_date,
                'grade_distribution': grade_distribution
            },
            'trends': {
                'performance_trend': self.get_trend_data(historical_df, 'performance_score'),
                'cwv_trend': self.get_cwv_trend_data(historical_df),
                'monthly_summary': self.get_monthly_summary(historical_df)
            },
            'brands': sorted([str(b) for b in historical_df['brand'].dropna().unique()]) if 'brand' in historical_df.columns else [],
            'brand_summary': brand_summary,
            'forecasts': forecasts,
            'comparisons': comparisons,
            'crux': crux,
            'sites': self.get_sites_data(historical_df),
            'recent_tests': self.get_recent_tests(historical_df),
            'performance_insights': self.generate_performance_insights(historical_df)
        }
        
        # AI insights (optional)
        ai_insights = self.generate_ai_insights(historical_df, dashboard_data)
        if ai_insights:
            dashboard_data['ai_insights'] = ai_insights

        # Save to JSON file with custom serialization
        data_file = os.path.join(self.data_dir, 'dashboard_data.json')
        with open(data_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=self._json_serializer)
        
        # Store for server-side rendering and inline fallback
        self.dashboard_data = dashboard_data

        logging.info(f"Dashboard data prepared and saved to {data_file}")
        return dashboard_data

    def compute_period_comparisons(self, df: pd.DataFrame) -> dict:
        """Compare last 30 days vs previous 30 days overall and by brand"""
        now = df['date'].max()
        if pd.isna(now):
            return {}
        start_30 = now - pd.Timedelta(days=30)
        prev_start_30 = now - pd.Timedelta(days=60)
        prev_end_30 = now - pd.Timedelta(days=30)

        def agg(sub):
            ps = pd.to_numeric(sub['performance_score'], errors='coerce').dropna()
            cs = pd.to_numeric(sub['cwv_score'], errors='coerce').dropna()
            return {
                'performance_score': float(ps.mean()) if not ps.empty else 0,
                'cwv_score': float(cs.mean()) if not cs.empty else 0,
                'tests': int(len(sub))
            }

        current = agg(df[(df['date'] >= start_30)])
        previous = agg(df[(df['date'] >= prev_start_30) & (df['date'] < prev_end_30)])

        brand_comp = {}
        if 'brand' in df.columns:
            for b, g in df.groupby('brand'):
                if pd.isna(b):
                    continue
                cur = agg(g[g['date'] >= start_30])
                prev = agg(g[(g['date'] >= prev_start_30) & (g['date'] < prev_end_30)])
                brand_comp[str(b)] = {
                    'current': cur,
                    'previous': prev,
                    'delta_performance': cur['performance_score'] - prev['performance_score'],
                    'delta_cwv': cur['cwv_score'] - prev['cwv_score']
                }

        return {
            'overall': {
                'current': current,
                'previous': previous,
                'delta_performance': current['performance_score'] - previous['performance_score'],
                'delta_cwv': current['cwv_score'] - previous['cwv_score']
            },
            'by_brand': brand_comp
        }

    def compute_forecasts(self, df: pd.DataFrame) -> dict:
        """Forecast next month's average performance and CWV overall and by brand using simple linear regression on monthly means."""
        def monthly_series(data: pd.DataFrame, col: str):
            tmp = data.copy()
            tmp['month'] = tmp['date'].dt.to_period('M')
            tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
            s = tmp.groupby('month')[col].mean().dropna()
            if s.empty:
                return None
            s.index = s.index.to_timestamp()
            return s

        def forecast_series(s: pd.Series, periods: int = 1):
            if s is None or len(s) < 2:
                return None
            x = np.arange(len(s))
            y = s.values
            try:
                coeffs = np.polyfit(x, y, deg=1)
                next_x = len(s)
                y_next = coeffs[0] * next_x + coeffs[1]
                return float(y_next)
            except Exception:
                return None

        forecasts = {'overall': {}, 'by_brand': {}}
        for col in ['performance_score', 'cwv_score']:
            s = monthly_series(df, col)
            forecasts['overall'][col] = forecast_series(s)

        if 'brand' in df.columns:
            for b, g in df.groupby('brand'):
                if pd.isna(b):
                    continue
                brand_pred = {}
                for col in ['performance_score', 'cwv_score']:
                    s = monthly_series(g, col)
                    brand_pred[col] = forecast_series(s)
                forecasts['by_brand'][str(b)] = brand_pred
        return forecasts

    def generate_sample_dashboard_data(self):
        """Generate sample dashboard data when no historical data exists"""
        sample_data = {
            'summary': {
                'total_sites': 0,
                'total_tests': 0,
                'avg_performance_score': 0,
                'avg_cwv_score': 0,
                'latest_test_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'grade_distribution': {}
            },
            'trends': {
                'performance_trend': [],
                'cwv_trend': {'lcp': [], 'fcp': [], 'cls': [], 'inp': []},
                'monthly_summary': []
            },
            'brands': [],
            'brand_summary': {},
            'forecasts': {},
            'comparisons': {},
            'sites': [],
            'recent_tests': [],
            'performance_insights': [
                {
                    'type': 'info',
                    'title': 'Welcome to Performance Dashboard',
                    'description': 'No performance data available yet. Run some performance tests to see results here.',
                    'action': 'Use the GTMetrix-style performance checker to generate your first performance reports.'
                }
            ]
        }
        
        # Save to JSON file
        data_file = os.path.join(self.data_dir, 'dashboard_data.json')
        with open(data_file, 'w') as f:
            json.dump(sample_data, f, indent=2, default=str)
        
        logging.info(f"Sample dashboard data generated and saved to {data_file}")
        return sample_data

    def get_trend_data(self, df, metric):
        """Get trend data for a specific metric"""
        if metric not in df.columns:
            return []
        
        # Filter out non-numeric values
        df_filtered = df[df[metric].notna()]
        df_filtered = df_filtered[pd.to_numeric(df_filtered[metric], errors='coerce').notna()]
        
        if df_filtered.empty:
            return []
        
        trend_data = df_filtered.groupby(['date', 'device_mode'])[metric].mean().reset_index()
        
        # Convert to regular Python types for JSON serialization
        records = []
        for _, row in trend_data.iterrows():
            metric_value = row[metric]
            # Ensure no NaN values
            if pd.isna(metric_value) or np.isnan(metric_value):
                continue
                
            records.append({
                'date': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                'device_mode': str(row['device_mode']),
                metric: float(metric_value)
            })
        
        return records

    def get_cwv_trend_data(self, df):
        """Get Core Web Vitals trend data"""
        cwv_metrics = ['lcp', 'fcp', 'cls', 'inp']
        trend_data = {}
        
        for metric in cwv_metrics:
            if metric not in df.columns:
                trend_data[metric] = []
                continue
                
            # Filter out non-numeric values (keep only numeric CWV values)
            df_filtered = df[df[metric] != 'N/A']
            df_filtered = df_filtered[pd.to_numeric(df_filtered[metric], errors='coerce').notna()]
            
            if df_filtered.empty:
                trend_data[metric] = []
                continue
            
            metric_trend = df_filtered.groupby(['date', 'device_mode'])[metric].mean().reset_index()
            
            # Convert to regular Python types for JSON serialization
            records = []
            for _, row in metric_trend.iterrows():
                metric_value = row[metric]
                # Ensure no NaN values
                if pd.isna(metric_value) or np.isnan(metric_value):
                    continue
                    
                records.append({
                    'date': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                    'device_mode': str(row['device_mode']),
                    metric: float(metric_value)
                })
            
            trend_data[metric] = records
        
        return trend_data

    def get_monthly_summary(self, df):
        """Get monthly performance summary"""
        if df.empty:
            return []
        
        df_copy = df.copy()
        df_copy['month'] = df_copy['date'].dt.to_period('M')
        
        # Ensure numeric columns are properly converted
        numeric_columns = ['performance_score', 'cwv_score']
        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        monthly_data = df_copy.groupby('month').agg({
            'performance_score': 'mean',
            'cwv_score': 'mean',
            'name': 'nunique'
        }).reset_index()
        
        # Convert to regular Python types for JSON serialization
        records = []
        for _, row in monthly_data.iterrows():
            records.append({
                'month': str(row['month']),
                'performance_score': float(row['performance_score']) if not pd.isna(row['performance_score']) else 0,
                'cwv_score': float(row['cwv_score']) if not pd.isna(row['cwv_score']) else 0,
                'name': int(row['name'])
            })
        
        return records

    def get_brand_summary(self, df: pd.DataFrame) -> dict:
        """Aggregate metrics by brand for quick slicing on frontend"""
        if df.empty or 'brand' not in df.columns:
            return {}
        out = {}
        for brand, g in df.groupby('brand'):
            if pd.isna(brand):
                continue
            ps = pd.to_numeric(g['performance_score'], errors='coerce').dropna()
            cs = pd.to_numeric(g['cwv_score'], errors='coerce').dropna()
            mob = pd.to_numeric(g[g['device_mode'] == 'mobile']['performance_score'], errors='coerce').dropna()
            desk = pd.to_numeric(g[g['device_mode'] == 'desktop']['performance_score'], errors='coerce').dropna()
            out[str(brand)] = {
                'avg_performance': float(ps.mean()) if not ps.empty else 0,
                'avg_cwv': float(cs.mean()) if not cs.empty else 0,
                'tests': int(len(g)),
                'sites': int(g['name'].nunique()),
                'mobile_avg_performance': float(mob.mean()) if not mob.empty else None,
                'desktop_avg_performance': float(desk.mean()) if not desk.empty else None
            }
        return out

    def get_sites_data(self, df):
        """Get detailed data for each site"""
        if df.empty or 'name' not in df.columns:
            return []
        
        sites_data = []
        
        for site_name in df['name'].unique():
            site_data = df[df['name'] == site_name]
            if site_data.empty:
                continue
                
            # Get the most recent test for this site
            try:
                latest_test = site_data.loc[site_data['date'].idxmax()]
            except (KeyError, ValueError):
                latest_test = site_data.iloc[-1]  # Fallback to last row
            
            # Calculate performance trend
            performance_scores = pd.to_numeric(site_data['performance_score'], errors='coerce').dropna()
            if len(performance_scores) >= 2:
                trend = 'improving' if performance_scores.iloc[-1] > performance_scores.iloc[0] else 'declining'
            else:
                trend = 'stable'
            
            # Safe data extraction with proper type conversion
            site_info = {
                'name': str(site_name),
                'brand': str(latest_test.get('brand', 'Unknown')),
                'domain': str(latest_test.get('domain', '')),
                'page': str(latest_test.get('page', '')),
                'url': str(latest_test.get('url', '')),
                'latest_grade': str(latest_test.get('cwv_grade', 'N/A')),
                'latest_score': float(latest_test.get('cwv_score', 0)) if pd.notna(latest_test.get('cwv_score')) else 0,
                'performance_score': float(latest_test.get('performance_score', 0)) if pd.notna(latest_test.get('performance_score')) else 0,
                'lcp': self._safe_metric_value(latest_test.get('lcp')),
                'fcp': self._safe_metric_value(latest_test.get('fcp')),
                'cls': self._safe_metric_value(latest_test.get('cls')),
                'inp': self._safe_metric_value(latest_test.get('inp')),
                'total_tests': int(len(site_data)),
                'avg_performance': float(performance_scores.mean()) if not performance_scores.empty else 0,
                'trend': trend
            }
            
            sites_data.append(site_info)
        
        return sites_data

    def _safe_metric_value(self, value):
        """Safely convert metric value to appropriate type"""
        if value is None or value == '' or value == 'N/A':
            return 'N/A'
        if pd.isna(value):
            return 'N/A'
        try:
            float_val = float(value)
            if np.isnan(float_val):
                return 'N/A'
            return float_val
        except (ValueError, TypeError):
            return 'N/A'

    def get_recent_tests(self, df, limit=10):
        """Get recent test results"""
        if df.empty:
            return []
        
        recent = df.nlargest(limit, 'date')
        
        # Convert to regular Python types for JSON serialization
        records = []
        for _, row in recent.iterrows():
            records.append({
                'date': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                'brand': str(row.get('brand', 'Unknown')),
                'domain': str(row.get('domain', '')),
                'page': str(row.get('page', '')),
                'name': str(row.get('name', '')),
                'device_mode': str(row.get('device_mode', '')),
                'performance_score': float(row.get('performance_score', 0)) if pd.notna(row.get('performance_score')) else 0,
                'cwv_grade': str(row.get('cwv_grade', 'N/A')),
                'cwv_score': float(row.get('cwv_score', 0)) if pd.notna(row.get('cwv_score')) else 0
            })
        
        return records

    def generate_performance_insights(self, df):
        """Generate actionable performance insights"""
        insights = []
        
        if df.empty:
            insights.append({
                'type': 'info',
                'title': 'No Data Available',
                'description': 'No performance data available for analysis.',
                'action': 'Run performance tests to generate insights.'
            })
            return insights
        
        # Site with most improvement needed
        if 'name' in df.columns and 'performance_score' in df.columns:
            try:
                perf_scores = pd.to_numeric(df['performance_score'], errors='coerce')
                valid_data = df[perf_scores.notna()]
                
                if not valid_data.empty:
                    worst_performers = valid_data.groupby('name')['performance_score'].apply(
                        lambda x: pd.to_numeric(x, errors='coerce').mean()
                    ).nsmallest(3)
                    
                    if not worst_performers.empty:
                        insights.append({
                            'type': 'warning',
                            'title': 'Sites Needing Attention',
                            'description': f'These sites have the lowest average performance scores: {", ".join(worst_performers.index)}',
                            'action': 'Review and optimize these sites for better performance'
                        })
            except Exception as e:
                logging.warning(f"Error generating performance insights: {e}")
        
        # CLS issues
        if 'cls' in df.columns:
            try:
                cls_issues = df[df['cls'] != 'N/A']
                if not cls_issues.empty:
                    cls_numeric = pd.to_numeric(cls_issues['cls'], errors='coerce')
                    valid_cls = cls_issues[cls_numeric.notna()]
                    
                    if not valid_cls.empty:
                        poor_cls = valid_cls[cls_numeric > 0.25]
                        if not poor_cls.empty:
                            insights.append({
                                'type': 'error',
                                'title': 'Layout Shift Issues',
                                'description': f'{len(poor_cls)} tests show poor Cumulative Layout Shift scores',
                                'action': 'Fix layout shifts by setting image dimensions and reserving space for dynamic content'
                            })
            except Exception as e:
                logging.warning(f"Error analyzing CLS data: {e}")
        
        # Mobile vs Desktop performance gap
        if 'device_mode' in df.columns and 'performance_score' in df.columns:
            try:
                mobile_data = df[df['device_mode'] == 'mobile']
                desktop_data = df[df['device_mode'] == 'desktop']
                
                if not mobile_data.empty and not desktop_data.empty:
                    mobile_scores = pd.to_numeric(mobile_data['performance_score'], errors='coerce').dropna()
                    desktop_scores = pd.to_numeric(desktop_data['performance_score'], errors='coerce').dropna()
                    
                    if not mobile_scores.empty and not desktop_scores.empty:
                        mobile_avg = mobile_scores.mean()
                        desktop_avg = desktop_scores.mean()
                        
                        if mobile_avg < desktop_avg - 10:
                            insights.append({
                                'type': 'info',
                                'title': 'Mobile Performance Gap',
                                'description': f'Mobile performance is {desktop_avg - mobile_avg:.1f} points lower than desktop',
                                'action': 'Optimize mobile experience with responsive images and mobile-first design'
                            })
            except Exception as e:
                logging.warning(f"Error analyzing mobile vs desktop performance: {e}")
        
        # Brand-level attention
        if 'brand' in df.columns:
            try:
                brand_summary = self.get_brand_summary(df)
                if brand_summary:
                    worst_brands = sorted(brand_summary.items(), key=lambda x: x[1].get('avg_performance', 0))[:3]
                    if worst_brands:
                        names = [b[0] for b in worst_brands]
                        insights.append({
                            'type': 'warning',
                            'title': 'Brands Needing Attention',
                            'description': f"Lowest average performance across brands: {', '.join(names)}",
                            'action': 'Prioritize these brands for performance optimization sprints'
                        })
            except Exception as e:
                logging.warning(f"Error generating brand insights: {e}")

        # If no insights were generated, add a default message
        if not insights:
            insights.append({
                'type': 'info',
                'title': 'Performance Analysis Complete',
                'description': 'Your sites are performing well within expected parameters.',
                'action': 'Continue monitoring performance and consider running more tests for better insights.'
            })
        
        return insights

    def generate_ai_insights(self, df: pd.DataFrame, dashboard_data: dict):
        """Use Azure OpenAI (if configured) to generate structured, actionable insights.
        Returns a list of {type,title,description,action} dicts or None.
        """
        try:
            if AzureOpenAIClient is None:
                return None
            client = AzureOpenAIClient()
            if not client.is_configured():
                return None
            # Build compact summary for prompt
            brand_summary = dashboard_data.get('brand_summary', {})
            comparisons = dashboard_data.get('comparisons', {})
            forecasts = dashboard_data.get('forecasts', {})
            grade_dist = dashboard_data.get('summary', {}).get('grade_distribution', {})

            # Limit brands to top 8 by tests for brevity
            top_brands = sorted(brand_summary.items(), key=lambda x: x[1].get('tests', 0), reverse=True)[:8]
            brand_lines = [f"{b}: perf {v.get('avg_performance', 0):.1f}, cwv {v.get('avg_cwv', 0):.1f}, tests {v.get('tests', 0)}" for b, v in top_brands]

            system = {
                'role': 'system',
                'content': (
                    'You are a senior Technical SEO and Web Performance engineer. '
                    'Analyze Core Web Vitals (LCP,FCP,CLS,INP), performance scores, mobile vs desktop gaps, '
                    'brand-level aggregates, month-over-month trends, and forecasts. '
                    'Return 5-8 prioritized, actionable insights as JSON array. '
                    'Each item: {"type":"info|warning|error","title":"...","description":"...","action":"..."}. '
                    'Be concise, specific, and avoid generic advice. '
                    'Prefer concrete remediation steps (e.g., reduce hero image LCP via AVIF + preconnect, defer non-critical JS).'
                )
            }
            user = {
                'role': 'user',
                'content': (
                    f"Overall: avg performance {dashboard_data['summary'].get('avg_performance_score',0):.1f}, "
                    f"avg CWV {dashboard_data['summary'].get('avg_cwv_score',0):.1f}.\n"
                    f"Grade distribution: {json.dumps(grade_dist)}\n"
                    f"Brands: {', '.join(brand_lines)}\n"
                    f"Comparisons (30d vs prev): {json.dumps(comparisons)[:1200]}\n"
                    f"Forecasts next month: {json.dumps(forecasts)[:800]}\n"
                    "Return ONLY JSON array as specified."
                )
            }
            raw = client.chat_completion_create(messages=[system, user], temperature=0.2, max_tokens=600)
            content = raw['choices'][0]['message']['content'] if raw and raw.get('choices') else ''
            # Try parse JSON array
            insights = None
            try:
                # Extract JSON if wrapped in markdown
                content_stripped = content.strip()
                if content_stripped.startswith('```'):
                    content_stripped = content_stripped.strip('`')
                    # remove possible json label
                    if content_stripped.startswith('json'):
                        content_stripped = content_stripped[4:]
                insights = json.loads(content_stripped)
            except Exception:
                pass
            if not isinstance(insights, list):
                # Fallback: wrap as single info
                return [{
                    'type': 'info',
                    'title': 'AI Insights',
                    'description': content[:1000],
                    'action': 'Review the above AI summary and plan actions.'
                }]
            # Validate items
            cleaned = []
            for item in insights:
                if not isinstance(item, dict):
                    continue
                cleaned.append({
                    'type': item.get('type', 'info'),
                    'title': str(item.get('title', 'Insight'))[:120],
                    'description': str(item.get('description', ''))[:800],
                    'action': str(item.get('action', ''))[:300]
                })
            return cleaned[:10]
        except Exception as e:
            logging.warning(f"AI insights generation failed: {e}")
            return None

    def get_historical_reports_data(self):
        """Get historical reports data for dashboard display"""
        try:
            # Get all available reports from the data manager
            reports = self.data_manager.scan_performance_reports()

            # Filter to only include reports from the main performance_reports folder that have performance_report.html
            filtered_reports = []
            for report in reports:
                dir_path = report.get('directory', '')

                # Check if the directory is directly under the main performance_reports folder (not in subfolders)
                if not dir_path.startswith(self.data_manager.base_dir):
                    continue

                # Skip reports in subdirectories like historical_data or current
                if '/historical_data/' in dir_path or '/current/' in dir_path or '/enhanced_dashboard/' in dir_path:
                    continue

                # Check if performance_report.html exists in this directory
                html_file = os.path.join(dir_path, 'performance_report.html')
                if not os.path.exists(html_file):
                    continue

                # If we got here, this report meets our criteria
                filtered_reports.append(report)

            # Sort reports by date (most recent first)
            reports_with_dates = []
            for report in filtered_reports:
                date_obj = None
                date_str = report.get('date', '')

                try:
                    # Handle different date formats
                    if len(date_str) == 8:  # YYYYMMDD format
                        date_obj = datetime.datetime.strptime(date_str, '%Y%m%d')
                    elif len(date_str) == 15 and '_' in date_str:  # YYYYMMDD_HHMMSS format
                        date_obj = datetime.datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                    else:
                        # Try to parse with dateutil
                        date_obj = parser.parse(date_str)
                except (ValueError, TypeError):
                    continue # Skip if date is invalid

                reports_with_dates.append({
                    'date': date_obj,
                    'date_str': date_str,
                    'directory': report.get('directory'),
                    'has_metadata': report.get('has_metadata', False),
                    'has_results': report.get('has_results', False),
                    'report_files': report.get('report_files', []),
                    'html_file': os.path.join(report.get('directory'), 'performance_report.html')
                })

            # Sort by date descending (most recent first)
            reports_with_dates.sort(key=lambda x: x['date'], reverse=True)

            return reports_with_dates[:50]  # Limit to 50 most recent reports

        except Exception as e:
            logging.error(f'Error getting historical reports data: {e}')
            return []

    def generate_historical_reports_html(self):
        """Generate HTML for displaying historical reports"""
        reports = self.get_historical_reports_data()

        if not reports:
            return '''
            <div class="no-reports">
                <p>📊 No historical performance reports found with performance_report.html.</p>
                <p>Generate performance reports with HTML output to see them listed here.</p>
            </div>
            '''

        html = f'''
        <div class="reports-grid">
            <div class="reports-header">
                <h3>Available Performance Reports ({len(reports)} total)</h3>
                <p>Performance reports with HTML output</p>
            </div>
            <div class="reports-list">
        '''

        for report in reports:
            # Format date for display
            formatted_date = report['date'].strftime('%B %d, %Y, %I:%M %p')

            # Get relative path for the HTML file
            rel_path = os.path.relpath(report['html_file'], self.dashboard_dir)

            # Create report card
            html += f'''
            <div class="report-card">
                <div class="report-header">
                    <div class="report-date">
                        <span class="date">{formatted_date}</span>
                    </div>
                    <div class="report-status">
                        ✅ HTML Report Available
                    </div>
                </div>
                <div class="report-content">
                    <div class="report-info">
                        <p><strong>Report ID:</strong> {report['date_str']}</p>
                        <p><strong>Location:</strong> {os.path.basename(report['directory'])}</p>
                    </div>
                    <div class="report-actions">
                        <a href="{rel_path}" class="btn btn-primary" target="_blank">
                            📊 View Performance Report
                        </a>
                    </div>
                </div>
            </div>
            '''

        html += '''
            </div>
        </div>
        '''

        return html

    def create_dashboard_assets(self):
        """Create CSS and JavaScript assets for the dashboard"""
        # CSS
        css_content = """
        :root {
            --primary-color: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --background-color: #f8fafc;
            --card-background: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--background-color);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, #3b82f6 100%);
            color: white;
            padding: 40px;
            border-radius: 16px;
            margin-bottom: 30px;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
        }

        .dashboard-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .dashboard-header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: var(--card-background);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }

        .stat-card h3 {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 15px;
            font-weight: 600;
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .stat-change {
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .stat-change.positive {
            color: var(--success-color);
        }

        .stat-change.negative {
            color: var(--error-color);
        }

        .grade-a { color: var(--success-color); }
        .grade-b { color: #22c55e; }
        .grade-c { color: var(--warning-color); }
        .grade-d { color: #f97316; }
        .grade-f { color: var(--error-color); }

        .card {
            background: var(--card-background);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }

        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .chart-container {
            position: relative;
            margin: 20px 0;
            text-align: center;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .insights-container {
            display: grid;
            gap: 15px;
        }

        .insight {
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid;
            background: #f8fafc;
        }

        .insight.warning {
            border-left-color: var(--warning-color);
            background: #fffbeb;
        }

        .insight.error {
            border-left-color: var(--error-color);
            background: #fef2f2;
        }

        .insight.info {
            border-left-color: var(--primary-color);
            background: #eff6ff;
        }

        .insight h4 {
            margin-bottom: 8px;
            font-weight: 600;
        }

        .insight p {
            margin-bottom: 10px;
            color: var(--text-secondary);
        }

        .insight .action {
            font-weight: 500;
            color: var(--text-primary);
        }

        .sites-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .sites-table th,
        .sites-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .sites-table th {
            background: #f8fafc;
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .sites-table tr:hover {
            background: #f8fafc;
        }

        .grade-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.8rem;
            text-align: center;
            min-width: 30px;
        }

        .grade-badge.A { background: #dcfce7; color: #166534; }
        .grade-badge.B { background: #ecfdf5; color: #15803d; }
        .grade-badge.C { background: #fef3c7; color: #92400e; }
        .grade-badge.D { background: #fed7aa; color: #9a3412; }
        .grade-badge.F { background: #fee2e2; color: #991b1b; }

        .filter-section {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .filter-group label {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .filter-group select,
        .filter-group input {
            padding: 8px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.9rem;
        }

        .footer {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-secondary);
            border-top: 1px solid var(--border-color);
            margin-top: 50px;
        }

        /* Historical Reports Styles */
        .historical-reports-container {
            margin-top: 20px;
        }

        .historical-reports-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .historical-reports-table th,
        .historical-reports-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }

        .historical-reports-table th {
            background: #f8fafc;
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .historical-reports-table tr:hover {
            background: #f8fafc;
        }

        .view-report-btn {
            display: inline-block;
            padding: 6px 12px;
            background: var(--primary-color);
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }

        .view-report-btn:hover {
            background: #1d4ed8;
            color: white;
            text-decoration: none;
        }

        .no-report-link {
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-style: italic;
        }

        .no-data-message {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-secondary);
        }

        .no-data-message p {
            margin: 10px 0;
            font-size: 1.1rem;
        }

        .no-data-message p:first-child {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .error-message {
            background: #fee2e2;
            color: #991b1b;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #fecaca;
            margin: 20px 0;
        }

        .error-message h3 {
            margin-bottom: 10px;
            font-weight: 600;
        }

        .error-message p {
            margin: 5px 0;
        }

        .reports-grid {
            margin-top: 20px;
        }

        .reports-header {
            margin-bottom: 25px;
            text-align: center;
        }

        .reports-header h3 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 10px;
        }

        .reports-header p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }

        .reports-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }

        .report-card {
            background: var(--card-background);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--border-color);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .report-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }

        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--border-color);
        }

        .report-date {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }

        .report-date .date {
            font-weight: 600;
            color: var(--text-primary);
            font-size: 0.95rem;
        }

        .report-date .time {
            font-size: 0.8rem;
            color: var(--text-secondary);
        }

        .report-status {
            font-size: 0.85rem;
            font-weight: 500;
            padding: 4px 8px;
            border-radius: 4px;
            background: #f1f5f9;
            color: var(--text-secondary);
        }

        .report-content {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .report-info {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .report-info p {
            margin: 0;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .report-info strong {
            color: var(--text-primary);
            font-weight: 600;
        }

        .report-actions {
            display: flex;
            gap: 10px;
        }

        .btn {
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            text-align: center;
            transition: all 0.2s ease;
            border: none;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
        }

        .btn-primary:hover {
            background: #1d4ed8;
            color: white;
            text-decoration: none;
        }

        .btn-secondary {
            background: #f8fafc;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: #e2e8f0;
            color: var(--text-primary);
            text-decoration: none;
        }

        .no-reports {
            text-align: center;
            padding: 40px 20px;
            color: var(--text-secondary);
        }

        .no-reports p {
            margin: 10px 0;
            font-size: 1.1rem;
        }

        .no-reports p:first-child {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .metric-pill { padding: 2px 6px; border-radius: 10px; font-size: 0.8rem; font-weight: 600; }
        .metric-good { background: #dcfce7; color: #166534; }
        .metric-ni { background: #fef3c7; color: #92400e; }
        .metric-poor { background: #fee2e2; color: #991b1b; }

        @media (max-width: 768px) {
            .dashboard-container {
                padding: 10px;
            }
            
            .dashboard-header {
                padding: 20px;
            }
            
            .dashboard-header h1 {
                font-size: 2rem;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .card {
                padding: 20px;
            }
            
            .historical-reports-table {
                font-size: 0.9rem;
            }
            
            .historical-reports-table th,
            .historical-reports-table td {
                padding: 10px 8px;
            }
            
            .reports-list {
                grid-template-columns: 1fr;
            }
        }
        """
        
        with open(os.path.join(self.css_dir, 'dashboard.css'), 'w') as f:
            f.write(css_content)
        
        # JavaScript
        js_content = """
        // Dashboard JavaScript functionality
        class PerformanceDashboard {
            constructor() {
                this.data = null;
                this.filtered = { sites: [], recent_tests: [] };
                this.init();
            }

            async init() {
                try {
                    console.log('Initializing dashboard...');
                    await this.loadData();
                    console.log('Data loaded successfully:', this.data);
                    this.populateFilters();
                    this.applyFilters();
                    this.renderDashboard();
                    this.setupEventListeners();
                    console.log('Dashboard initialized successfully');
                } catch (error) {
                    console.error('Failed to initialize dashboard:', error);
                    this.showError('Failed to load dashboard data: ' + error.message);
                }
            }

            async loadData() {
                console.log('Loading data from data/dashboard_data.json...');
                try {
                    const response = await fetch('data/dashboard_data.json?t=' + Date.now());
                    console.log('Fetch response status:', response.status);
                    if (!response.ok) {
                        throw new Error('Failed to load data: ' + response.status);
                    }
                    this.data = await response.json();
                    console.log('Data loaded via fetch:', this.data);
                } catch (e) {
                    console.warn('Fetch failed, trying inline fallback...', e);
                    if (window.__DASHBOARD_DATA__) {
                        this.data = window.__DASHBOARD_DATA__;
                        console.log('Data loaded from inline fallback');
                    } else {
                        throw e;
                    }
                }
                this.filtered.sites = [...(this.data.sites || [])];
                this.filtered.recent_tests = [...(this.data.recent_tests || [])];
                // Update last updated if summary has it
                const last = (this.data && this.data.summary && this.data.summary.latest_test_date) ? this.data.summary.latest_test_date : null;
                const lastEl = document.getElementById('last-updated');
                if (lastEl) lastEl.textContent = last || new Date().toLocaleString();
            }

            populateFilters() {
                const brandSel = document.getElementById('filter-brand');
                const domainSel = document.getElementById('filter-domain');
                const deviceSel = document.getElementById('filter-device');
                if (!brandSel) return;
                // Brands
                const brands = (this.data.brands || []).slice().sort();
                brandSel.innerHTML = '<option value="">All brands</option>' + brands.map(b => `<option value="${b}">${b}</option>`).join('');
                // Domains
                this.populateDomainOptions();
                // Devices
                if (deviceSel) {
                    deviceSel.innerHTML = ["", "desktop", "mobile"].map(v => `<option value="${v}">${v ? v : 'All devices'}</option>`).join('');
                }
            }

            populateDomainOptions() {
                const brandSel = document.getElementById('filter-brand');
                const domainSel = document.getElementById('filter-domain');
                if (!domainSel) return;
                const brand = brandSel ? brandSel.value : '';
                let domains = (this.data.sites || []).filter(s => !brand || s.brand === brand).map(s => s.domain);
                domains = Array.from(new Set(domains.filter(Boolean))).sort();
                domainSel.innerHTML = '<option value="">All domains</option>' + domains.map(d => `<option value="${d}">${d}</option>`).join('');
            }

            renderDashboard() {
                console.log('Rendering dashboard...');
                this.updateSummaryStats();
                this.renderInsights();
                this.renderBrandSummary();
                this.renderSitesTable();
                this.renderRecentTests();                
                console.log('Dashboard rendered successfully');
            }

            updateSummaryStats() {
                const totalSitesEl = document.getElementById('total-sites');
                const totalTestsEl = document.getElementById('total-tests');
                const avgPerfEl = document.getElementById('avg-performance');
                const avgCwvEl = document.getElementById('avg-cwv');
                const sites = this.filtered.sites || [];
                if (totalSitesEl) totalSitesEl.textContent = sites.length;
                const totalTests = (this.data && this.data.summary && this.data.summary.total_tests) ? this.data.summary.total_tests : 0;
                if (totalTestsEl) totalTestsEl.textContent = totalTests;
                const perf = sites.map(s => s.performance_score).filter(v => typeof v === 'number' && !isNaN(v));
                const cwv = sites.map(s => s.latest_score).filter(v => typeof v === 'number' && !isNaN(v));
                const avgPerf = perf.length ? Math.round(perf.reduce((a,b)=>a+b,0)/perf.length) : 0;
                const avgCwv = cwv.length ? Math.round(cwv.reduce((a,b)=>a+b,0)/cwv.length) : 0;
                if (avgPerfEl) avgPerfEl.textContent = avgPerf;
                if (avgCwvEl) avgCwvEl.textContent = avgCwv;
            }

            renderInsights() {
                console.log('Rendering insights...');
                const container = document.getElementById('insights-container');
                if (!container) {
                    console.log('Insights container not found');
                    return;
                }
                container.innerHTML = '';
                const source = this.data.ai_insights && this.data.ai_insights.length ? this.data.ai_insights : this.data.performance_insights;
                (source || []).forEach(insight => {
                    const insightElement = document.createElement('div');
                    insightElement.className = `insight ${insight.type}`;
                    insightElement.innerHTML = `
                        <h4>${insight.title}</h4>
                        <p>${insight.description}</p>
                        <div class="action">${insight.action}</div>
                    `;
                    container.appendChild(insightElement);
                });
                console.log('Insights rendered successfully');
            }

            renderBrandSummary() {
                const container = document.getElementById('brand-summary');
                if (!container) return;
                const brandSel = document.getElementById('filter-brand');
                const brand = brandSel ? brandSel.value : '';
                const summary = this.data && this.data.brand_summary ? this.data.brand_summary : {};
                const entries = Object.entries(summary).filter(([b]) => !brand || b === brand);
                if (!entries.length) {
                    container.innerHTML = '<div class="no-data-message">No brand summary available.</div>';
                    return;
                }
                const rows = entries.map(([b, v]) => {
                    const ap = (v && v.avg_performance != null) ? Number(v.avg_performance).toFixed(1) : '-';
                    const ac = (v && v.avg_cwv != null) ? Number(v.avg_cwv).toFixed(1) : '-';
                    const mp = (v && v.mobile_avg_performance != null) ? Number(v.mobile_avg_performance).toFixed(1) : '-';
                    const dp = (v && v.desktop_avg_performance != null) ? Number(v.desktop_avg_performance).toFixed(1) : '-';
                    const sites = (v && v.sites != null) ? v.sites : '-';
                    const tests = (v && v.tests != null) ? v.tests : '-';
                    return `
                    <tr>
                        <td>${b}</td>
                        <td>${ap}</td>
                        <td>${ac}</td>
                        <td>${mp}</td>
                        <td>${dp}</td>
                        <td>${sites}</td>
                        <td>${tests}</td>
                    </tr>`;
                }).join('');
                container.innerHTML = `
                    <div style="overflow-x:auto">
                    <table class="sites-table">
                        <thead>
                            <tr>
                                <th>Brand</th>
                                <th>Avg Perf</th>
                                <th>Avg CWV</th>
                                <th>Mobile Perf</th>
                                <th>Desktop Perf</th>
                                <th>Sites</th>
                                <th>Tests</th>
                            </tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                    </div>
                `;
            }

            renderSitesTable() {
                console.log('Rendering sites table...');
                const tbody = document.getElementById('sites-table-body');
                if (!tbody) {
                    console.log('Sites table body not found');
                    return;
                }

                tbody.innerHTML = '';
                
                this.filtered.sites.forEach(site => {
                    const lcpClass = this.metricClass('lcp', site.lcp);
                    const fcpClass = this.metricClass('fcp', site.fcp);
                    const clsClass = this.metricClass('cls', site.cls);
                    const inpClass = this.metricClass('inp', site.inp);
                    const lcpText = (typeof site.lcp === 'number') ? site.lcp.toFixed(2) + 's' : site.lcp;
                    const fcpText = (typeof site.fcp === 'number') ? site.fcp.toFixed(2) + 's' : site.fcp;
                    const clsText = (typeof site.cls === 'number') ? site.cls.toFixed(3) : site.cls;
                    const inpText = (typeof site.inp === 'number') ? Math.round(site.inp) + 'ms' : site.inp;
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>
                            <div style="font-weight: 600;">${site.name}</div>
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">${site.brand} • ${site.domain}${site.page}</div>
                            <div style="font-size: 0.8rem, color: var(--text-secondary);">${site.url}</div>
                        </td>
                        <td><span class="grade-badge ${site.latest_grade}">${site.latest_grade}</span></td>
                        <td>${Math.round(site.latest_score)}</td>
                        <td>${Math.round(site.performance_score)}</td>
                        <td><span class="metric-pill ${lcpClass}">${lcpText}</span></td>
                        <td><span class="metric-pill ${fcpClass}">${fcpText}</span></td>
                        <td><span class="metric-pill ${clsClass}">${clsText}</span></td>
                        <td><span class="metric-pill ${inpClass}">${inpText}</span></td>
                        <td>${site.total_tests}</td>
                        <td>
                            <span class="stat-change ${site.trend === 'improving' ? 'positive' : (site.trend === 'declining' ? 'negative' : '')}">
                                ${site.trend === 'improving' ? '↗' : (site.trend === 'declining' ? '↘' : '•')} ${site.trend}
                            </span>
                        </td>
                    `;
                    tbody.appendChild(row);
                });
            }

            renderRecentTests() {
                console.log('Rendering recent tests...');
                const container = document.getElementById('recent-tests');
                if (!container) {
                    console.log('Recent tests container not found');
                    return;
                }

                container.innerHTML = '';
                
                this.filtered.recent_tests.slice(0, 5).forEach(test => {
                    const testElement = document.createElement('div');
                    testElement.className = 'recent-test';
                    testElement.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border-color);">
                            <div>
                                <div style="font-weight: 600;">${test.name}</div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary);">${test.brand} • ${test.domain}${test.page}</div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary);">${test.device_mode} • ${new Date(test.date).toLocaleDateString()}</div>
                            </div>
                            <div style="text-align: right;">
                                <span class="grade-badge ${test.cwv_grade}">${test.cwv_grade}</span>
                                <div style="font-size: 0.8rem; color: var(--text-secondary);">${Math.round(test.performance_score)}% performance</div>
                            </div>
                        </div>
                    `;
                    container.appendChild(testElement);
                });
                
                console.log('Recent tests rendered successfully');
            }

            renderHistoricalReports() {
                console.log('Rendering historical reports...');
                const container = document.getElementById('historical-reports');
                if (!container) {
                    console.log('Historical reports container not found');
                    return;
                }
                // The content is now server-rendered from Python.
                console.log('Historical reports are rendered server-side.');
            }

            setupEventListeners() {
                // Add event listeners for filters and interactions
                document.addEventListener('change', (e) => {
                    if (e.target.matches('.filter-select') || e.target.matches('.filter-input')) {
                        if (e.target.id === 'filter-brand') this.populateDomainOptions();
                        this.applyFilters();
                        this.renderDashboard();
                    }
                });
                const exportBtn = document.getElementById('export-csv');
                if (exportBtn) {
                    exportBtn.addEventListener('click', () => this.exportCSV());
                }
            }

            applyFilters() {
                const brand = (document.getElementById('filter-brand') || {}).value || '';
                const domain = (document.getElementById('filter-domain') || {}).value || '';
                const pathQuery = (document.getElementById('filter-path') || {}).value || '';
                const from = (document.getElementById('filter-from') || {}).value || '';
                const to = (document.getElementById('filter-to') || {}).value || '';
                const device = (document.getElementById('filter-device') || {}).value || '';
                
                // Filter sites by brand/domain/path
                this.filtered.sites = (this.data.sites || []).filter(s => {
                    if (brand && s.brand !== brand) return false;
                    if (domain && s.domain !== domain) return false;
                    if (pathQuery && !(s.page || '').toLowerCase().includes(pathQuery.toLowerCase())) return false;
                    return true;
                });
                
                // Filter recent tests by all filters including date and device
                const fromDate = from ? new Date(from) : null;
                const toDate = to ? new Date(to) : null;
                this.filtered.recent_tests = (this.data.recent_tests || []).filter(t => {
                    if (brand && t.brand !== brand) return false;
                    if (domain && t.domain !== domain) return false;
                    if (pathQuery && !(t.page || '').toLowerCase().includes(pathQuery.toLowerCase())) return false;
                    if (device && t.device_mode !== device) return false;
                    const d = new Date(t.date);
                    if (fromDate && d < fromDate) return false;
                    if (toDate && d > toDate) return false;
                    return true;
                });
            }

            exportCSV() {
                const headers = ['name','brand','domain','page','url','latest_grade','latest_score','performance_score','lcp','fcp','cls','inp','total_tests','trend'];
                const rows = this.filtered.sites.map(s => headers.map(h => (s[h] !== undefined ? s[h] : '')).join(','));
                const csv = [headers.join(','), ...rows].join('\n');
                const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'performance_sites.csv';
                a.click();
                URL.revokeObjectURL(url);
            }

            showError(message) {
                console.error('Dashboard error:', message);
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error-message';
                errorDiv.style.cssText = 'background: #fee2e2; color: #991b1b; padding: 20px; margin: 20px; border-radius: 8px; border: 1px solid #fecaca;';
                errorDiv.innerHTML = `
                    <h3>Error</h3>
                    <p>${message}</p>
                `;
                document.body.insertBefore(errorDiv, document.body.firstChild);
            }

        }

        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, initializing dashboard...');
            new PerformanceDashboard();
        });
        """
        
        with open(os.path.join(self.js_dir, 'dashboard.js'), 'w') as f:
            f.write(js_content)
        
        logging.info("Dashboard assets created successfully")

    def generate_html_dashboard(self):
        """Generate the main HTML dashboard"""
        historical_reports_html = self.generate_historical_reports_html()
        # Pull summary numbers for server-rendered placeholders and inline data
        summary = (self.dashboard_data or {}).get('summary', {}) if isinstance(self.dashboard_data, dict) else {}
        total_sites = summary.get('total_sites', '-')
        total_tests = summary.get('total_tests', '-')
        avg_perf = summary.get('avg_performance_score', '-')
        avg_cwv = summary.get('avg_cwv_score', '-')
        # Round numbers if present
        def fmt(v):
            try:
                return str(int(round(float(v))))
            except Exception:
                return '-'
        total_sites_txt = str(total_sites) if isinstance(total_sites, int) else fmt(total_sites)
        total_tests_txt = str(total_tests) if isinstance(total_tests, int) else fmt(total_tests)
        avg_perf_txt = fmt(avg_perf)
        avg_cwv_txt = fmt(avg_cwv)
        # Inline fallback JSON for file:// usage
        inline_json = json.dumps(self.dashboard_data or {}, indent=2, default=self._json_serializer)
        html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Enhanced Performance Dashboard</title>
    <link rel=\"stylesheet\" href=\"css/dashboard.css\">\n    <link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap\" rel=\"stylesheet\">
</head>
<body>
    <div class=\"dashboard-container\"> 
        <!-- Header -->
        <div class=\"dashboard-header\">
            <h1>Enhanced Performance Dashboard</h1>
            <p>Comprehensive website performance monitoring and analysis</p>
            <div style=\"margin-top: 15px; font-size: 0.9rem; opacity: 0.8;\">
                Last updated: <span id=\"last-updated\"></span>
            </div>
        </div>

        <!-- Filters -->
        <div class=\"filter-section\">
            <div class=\"filter-group\">
                <label for=\"filter-brand\">Brand</label>
                <select id=\"filter-brand\" class=\"filter-select\"></select>
            </div>
            <div class=\"filter-group\">
                <label for=\"filter-domain\">Domain</label>
                <select id=\"filter-domain\" class=\"filter-select\"></select>
            </div>
            <div class=\"filter-group\">
                <label for=\"filter-path\">Path contains</label>
                <input id=\"filter-path\" type=\"text\" placeholder=\"/blog\" class=\"filter-input\" />
            </div>
            <div class=\"filter-group\">
                <label for=\"filter-device\">Device</label>
                <select id=\"filter-device\" class=\"filter-select\"></select>
            </div>
            <div class=\"filter-group\">
                <label for=\"filter-from\">From</label>
                <input id=\"filter-from\" type=\"date\" class=\"filter-input\" />
            </div>
            <div class=\"filter-group\">
                <label for=\"filter-to\">To</label>
                <input id=\"filter-to\" type=\"date\" class=\"filter-input\" />
            </div>
            <div class=\"filter-group\" style=\"align-self:flex-end;\">
                <button id=\"export-csv\" class=\"btn btn-secondary\">⬇ Export CSV</button>
            </div>
        </div>

        <!-- Summary Statistics -->
        <div class=\"stats-grid\">
            <div class=\"stat-card\">
                <h3>Total Sites Monitored</h3>
                <div class=\"stat-value\" id=\"total-sites\">{total_sites_txt}</div>
                <div class=\"stat-change positive\">
                    <span>↗</span> Active monitoring
                </div>
            </div>
            <div class=\"stat-card\">
                <h3>Total Performance Tests</h3>
                <div class=\"stat-value\" id=\"total-tests\">{total_tests_txt}</div>
                <div class=\"stat-change positive\">
                    <span>↗</span> Historical data
                </div>
            </div>
            <div class=\"stat-card\">
                <h3>Avg Performance Score</h3>
                <div class=\"stat-value\" id=\"avg-performance\">{avg_perf_txt}</div>
                <div class=\"stat-change\">
                    <span>📊</span> Overall health
                </div>
            </div>
            <div class=\"stat-card\">
                <h3>Avg Core Web Vitals</h3>
                <div class=\"stat-value\" id=\"avg-cwv\">{avg_cwv_txt}</div>
                <div class=\"stat-change\">
                    <span>⚡</span> User experience
                </div>
            </div>
        </div>

        <!-- Performance Trends -->
        <div class=\"card\">
            <div class=\"card-header\">
                <h2 class=\"card-title\">Performance Trends</h2>
            </div>
            <div class=\"chart-container\">
                <img src=\"charts/performance_trend.png\" alt=\"Performance Trend Chart\" style=\"max-width: 100%;\">
            </div>
        </div>

        <!-- Core Web Vitals Trends -->
        <div class=\"card\">
            <div class=\"card-header\">
                <h2 class=\"card-title\">Core Web Vitals Trends</h2>
            </div>
            <div class=\"chart-container\">
                <img src=\"charts/cwv_trend.png\" alt=\"Core Web Vitals Trend Chart\" style=\"max-width: 100%;\">
            </div>
        </div>

        <!-- Grade Distribution -->
        <div class=\"card\">
            <div class=\"card-header\">
                <h2 class=\"card-title\">Performance Grade Distribution</h2>
            </div>
            <div class=\"chart-container\">
                <img src=\"charts/grade_distribution.png\" alt=\"Grade Distribution Chart\" style=\"max-width: 100%;\">
            </div>
        </div>

        <!-- Performance Matrix -->
        <div class=\"card\">
            <div class=\"card-header\">
                <h2 class=\"card-title\">Performance Matrix</h2>
            </div>
            <div class=\"chart-container\">
                <img src=\"charts/performance_matrix.png\" alt=\"Performance Matrix Chart\" style=\"max-width: 100%;\">
            </div>
        </div>

        <!-- Speed Index Visualization -->
        <div class=\"card\">
            <div class=\"card-header\">
                <h2 class=\"card-title\">Speed Index Distribution</h2>
            </div>
            <div class=\"chart-container\">
                <img src=\"charts/speed_index.png\" alt=\"Speed Index Chart\" style=\"max-width: 100%;\">
            </div>
        </div>

        <!-- CrUX Overview -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">CrUX Overview</h2>
            </div>
            <div class="chart-container">
                <img src="charts/crux_overview.png" alt="CrUX Overview Chart" style="max-width: 100%;">
            </div>
        </div>

        <!-- Brand Trends -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Brand Trends</h2>
            </div>
            <div class="chart-container">
                <img src="charts/brand_trends.png" alt="Brand Trends Chart" style="max-width: 100%;">
            </div>
        </div>
        
        <!-- Brand Summary -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Brand Summary</h2>
            </div>
            <div id="brand-summary"></div>
        </div>

        <!-- Performance Insights -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Performance Insights & Recommendations</h2>
            </div>
            <div class="insights-container" id="insights-container">
                <!-- Insights will be loaded dynamically -->
            </div>
        </div>

        <!-- Sites Overview Table -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Sites Overview</h2>
            </div>
            <div style="overflow-x: auto;">
                <table class="sites-table">
                    <thead>
                        <tr>
                            <th>Site</th>
                            <th>CWV Grade</th>
                            <th>CWV Score</th>
                            <th>Performance</th>
                            <th>LCP</th>
                            <th>FCP</th>
                            <th>CLS</th>
                            <th>INP</th>
                            <th>Tests</th>
                            <th>Trend</th>
                        </tr>
                    </thead>
                    <tbody id="sites-table-body">
                        <!-- Site data will be loaded dynamically -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Recent Tests -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Recent Tests</h2>
            </div>
            <div id="recent-tests">
                <!-- Recent tests will be loaded dynamically -->
            </div>
        </div>

        <!-- Historical Reports -->
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Historical Reports</h2>
            </div>
            <div id="historical-reports">
                {historical_reports_html}
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Enhanced Performance Dashboard &copy; 2025</p>
        <p>Comprehensive website performance monitoring and analysis</p>
    </div>

    <script>window.__DASHBOARD_DATA__ = {inline_json};</script>
    <script src=\"js/dashboard.js\"></script>
    <script>
        // Set last updated time from summary if available, else fallback to now
        (function() {{
            try {{
                var d = window.__DASHBOARD_DATA__ || {{}};
                var t = d.summary && d.summary.latest_test_date ? d.summary.latest_test_date : new Date().toLocaleString();
                document.getElementById('last-updated').textContent = t;
            }} catch (e) {{
                document.getElementById('last-updated').textContent = new Date().toLocaleString();
            }}
        }})();
    </script>
</body>
</html>
        """
        
        dashboard_path = os.path.join(self.dashboard_dir, 'index.html')
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"HTML dashboard generated: {dashboard_path}")
        return dashboard_path

    def generate_dashboard(self):
        """Generate the complete enhanced dashboard"""
        logging.info("Generating enhanced performance dashboard...")
        
        # Prepare data
        dashboard_data = self.prepare_dashboard_data()

        if not dashboard_data:
            logging.error("No data available for dashboard generation")
            return False

        # Generate charts
        self.generate_performance_trend_charts()

        # Create assets
        self.create_dashboard_assets()

        # Generate HTML
        dashboard_path = self.generate_html_dashboard()

        logging.info(f"Enhanced dashboard generated successfully: {dashboard_path}")
        return True
    def serve_dashboard(self):
        """Serve the dashboard on a local web server"""
        if not self.generate_dashboard():
            logging.error("Failed to generate dashboard")
            return False
        
        # Start HTTP server
        import threading
        import time
        
        # Change to dashboard directory
        original_dir = os.getcwd()
        os.chdir(self.dashboard_dir)

        try:
            # Create a simple HTTP request handler with CORS headers
            class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
                def end_headers(self):
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    super().end_headers()
                
                def log_message(self, format, *args):
                    # Override to reduce logging noise
                    pass
            
            with socketserver.TCPServer(("", self.port), CORSHTTPRequestHandler) as httpd:
                url = f"http://localhost:{self.port}"
                logging.info(f"Enhanced dashboard server started at {url}")
                
                # Try to open browser
                def open_browser():
                    time.sleep(1)  # Give server time to start
                    try:
                        webbrowser.open(url)
                        logging.info(f"Browser opened to {url}")
                    except Exception as e:
                        logging.warning(f"Could not open browser automatically: {e}")
                
                browser_thread = threading.Thread(target=open_browser)
                browser_thread.daemon = True
                browser_thread.start()
                
                print(f"\n🚀 Enhanced Performance Dashboard is running!")
                print(f"📊 Dashboard URL: {url}")
                print(f"📁 Dashboard directory: {self.dashboard_dir}")
                print(f"🔄 Refresh the page if you see cached data")
                print(f"\nPress Ctrl+C to stop the server...")
                
                httpd.serve_forever()
        
        except KeyboardInterrupt:
            logging.info("Dashboard server stopped by user")
        except Exception as e:
            logging.error(f"Error serving dashboard: {str(e)}")
            return False
        finally:
            os.chdir(original_dir)
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Enhanced GTMetrix-Style Performance Dashboard")
    parser.add_argument("--generate-only", action="store_true", 
                       help="Only generate the dashboard without serving it")
    parser.add_argument("--port", type=int, default=8001, 
                       help="Port to serve the dashboard on")
    parser.add_argument("--base-dir", 
                       help="Base directory for performance data")
    
    args = parser.parse_args()
    
    # Initialize dashboard
    dashboard = EnhancedPerformanceDashboard(base_dir=args.base_dir)
    dashboard.port = args.port
    
    if args.generate_only:
        success = dashboard.generate_dashboard()
        if success:
            print(f"✅ Dashboard generated successfully!")
            print(f"📁 Location: {dashboard.dashboard_dir}")
        else:
            print("❌ Failed to generate dashboard")
    else:
        dashboard.serve_dashboard()


if __name__ == "__main__":
    main()
