"""
Enhanced Time Series Charts Module for Performance Dashboard
This module creates comprehensive time series charts for performance data visualization
with GTMetrix-style features and advanced analytics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import logging


def create_comprehensive_performance_trend(df, output_dir):
    """Create comprehensive performance trend chart with multiple metrics"""
    if df.empty:
        logging.warning("No data available for performance trend chart")
        return
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    fig.suptitle('Comprehensive Performance Trends Over Time', fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors for different device modes
    colors = {'desktop': '#4285F4', 'mobile': '#EA4335'}
    
    # 1. Performance Score Trend
    ax1 = axes[0, 0]
    for device_mode in df['device_mode'].unique():
        device_data = df[df['device_mode'] == device_mode]
        if not device_data.empty and 'performance_score' in device_data.columns:
            daily_avg = device_data.groupby('date')['performance_score'].mean()
            ax1.plot(daily_avg.index, daily_avg.values, 
                    color=colors.get(device_mode, '#666'), marker='o', 
                    label=f'{device_mode.title()}', linewidth=3, markersize=5)
    
    ax1.set_title('Performance Score Trend', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance Score (0-100)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Add performance zones
    ax1.axhspan(90, 100, alpha=0.1, color='green', label='Excellent (90-100)')
    ax1.axhspan(75, 90, alpha=0.1, color='yellow', label='Good (75-89)')
    ax1.axhspan(50, 75, alpha=0.1, color='orange', label='Needs Improvement (50-74)')
    ax1.axhspan(0, 50, alpha=0.1, color='red', label='Poor (0-49)')
    
    # 2. LCP Trend
    ax2 = axes[0, 1]
    for device_mode in df['device_mode'].unique():
        device_data = df[df['device_mode'] == device_mode]
        if not device_data.empty and 'lcp' in device_data.columns:
            lcp_data = device_data[device_data['lcp'] != 'N/A'].copy()
            if not lcp_data.empty:
                lcp_data['lcp'] = pd.to_numeric(lcp_data['lcp'], errors='coerce')
                lcp_data = lcp_data.dropna(subset=['lcp'])
                if not lcp_data.empty:
                    daily_avg = lcp_data.groupby('date')['lcp'].mean()
                    ax2.plot(daily_avg.index, daily_avg.values, 
                            color=colors.get(device_mode, '#666'), marker='o', 
                            label=f'{device_mode.title()}', linewidth=3, markersize=5)
    
    ax2.axhline(y=2.5, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Good (≤2.5s)')
    ax2.axhline(y=4.0, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Needs Improvement (≤4.0s)')
    ax2.set_title('Largest Contentful Paint (LCP)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Seconds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. FCP Trend
    ax3 = axes[1, 0]
    for device_mode in df['device_mode'].unique():
        device_data = df[df['device_mode'] == device_mode]
        if not device_data.empty and 'fcp' in device_data.columns:
            fcp_data = device_data[device_data['fcp'] != 'N/A'].copy()
            if not fcp_data.empty:
                fcp_data['fcp'] = pd.to_numeric(fcp_data['fcp'], errors='coerce')
                fcp_data = fcp_data.dropna(subset=['fcp'])
                if not fcp_data.empty:
                    daily_avg = fcp_data.groupby('date')['fcp'].mean()
                    ax3.plot(daily_avg.index, daily_avg.values, 
                            color=colors.get(device_mode, '#666'), marker='s', 
                            label=f'{device_mode.title()}', linewidth=3, markersize=5)
    
    ax3.axhline(y=1.8, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Good (≤1.8s)')
    ax3.axhline(y=3.0, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Needs Improvement (≤3.0s)')
    ax3.set_title('First Contentful Paint (FCP)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Seconds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. CLS Trend
    ax4 = axes[1, 1]
    for device_mode in df['device_mode'].unique():
        device_data = df[df['device_mode'] == device_mode]
        if not device_data.empty and 'cls' in device_data.columns:
            cls_data = device_data[device_data['cls'] != 'N/A'].copy()
            if not cls_data.empty:
                cls_data['cls'] = pd.to_numeric(cls_data['cls'], errors='coerce')
                cls_data = cls_data.dropna(subset=['cls'])
                if not cls_data.empty:
                    daily_avg = cls_data.groupby('date')['cls'].mean()
                    ax4.plot(daily_avg.index, daily_avg.values, 
                            color=colors.get(device_mode, '#666'), marker='^', 
                            label=f'{device_mode.title()}', linewidth=3, markersize=5)
    
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Good (≤0.1)')
    ax4.axhline(y=0.25, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Needs Improvement (≤0.25)')
    ax4.set_title('Cumulative Layout Shift (CLS)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. INP Trend
    ax5 = axes[2, 0]
    for device_mode in df['device_mode'].unique():
        device_data = df[df['device_mode'] == device_mode]
        if not device_data.empty and 'inp' in device_data.columns:
            inp_data = device_data[device_data['inp'] != 'N/A'].copy()
            if not inp_data.empty:
                inp_data['inp'] = pd.to_numeric(inp_data['inp'], errors='coerce')
                inp_data = inp_data.dropna(subset=['inp'])
                if not inp_data.empty:
                    daily_avg = inp_data.groupby('date')['inp'].mean()
                    ax5.plot(daily_avg.index, daily_avg.values, 
                            color=colors.get(device_mode, '#666'), marker='d', 
                            label=f'{device_mode.title()}', linewidth=3, markersize=5)
    
    ax5.axhline(y=200, color='green', linestyle='--', alpha=0.8, linewidth=2, label='Good (≤200ms)')
    ax5.axhline(y=500, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Needs Improvement (≤500ms)')
    ax5.set_title('Interaction to Next Paint (INP)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Milliseconds')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. All Lighthouse Categories
    ax6 = axes[2, 1]
    lighthouse_metrics = ['performance_score', 'accessibility_score', 'best_practices_score', 'seo_score']
    lighthouse_colors = ['#4285F4', '#34A853', '#FBBC04', '#EA4335']
    
    desktop_data = df[df['device_mode'] == 'desktop']
    for metric, color in zip(lighthouse_metrics, lighthouse_colors):
        if metric in desktop_data.columns and not desktop_data.empty:
            daily_avg = desktop_data.groupby('date')[metric].mean()
            ax6.plot(daily_avg.index, daily_avg.values, 
                    color=color, marker='o', label=metric.replace('_score', '').replace('_', ' ').title(),
                    linewidth=2, markersize=4)
    
    ax6.set_title('All Lighthouse Categories (Desktop)', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Score (0-100)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 100)
    
    # Format x-axis for all subplots
    for ax in axes.flat:
        if len(df) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df['date'].unique()) // 10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "comprehensive_performance_trend.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logging.info(f"Comprehensive performance trend chart saved to {output_path}")


def create_performance_grade_timeline(df, output_dir):
    """Create timeline showing performance grades over time"""
    if df.empty:
        logging.warning("No data available for grade timeline chart")
        return
    
    # Calculate grades for each record
    def calculate_grade(performance_score):
        if performance_score >= 90:
            return 'A'
        elif performance_score >= 80:
            return 'B'
        elif performance_score >= 70:
            return 'C'
        elif performance_score >= 60:
            return 'D'
        else:
            return 'F'
    
    df['grade'] = df['performance_score'].apply(calculate_grade)
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Define grade colors
    grade_colors = {'A': '#4CAF50', 'B': '#8BC34A', 'C': '#FF9800', 'D': '#FF5722', 'F': '#F44336'}
    grade_values = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    
    # Plot for each site
    sites = df['name'].unique()
    y_positions = np.arange(len(sites))
    
    for i, site in enumerate(sites):
        site_data = df[df['name'] == site].sort_values('date')
        
        for j, (_, row) in enumerate(site_data.iterrows()):
            color = grade_colors.get(row['grade'], '#666')
            marker = 'o' if row['device_mode'] == 'desktop' else 's'
            size = 100 if row['device_mode'] == 'desktop' else 80
            
            plt.scatter(row['date'], i + (0.1 if row['device_mode'] == 'desktop' else -0.1), 
                       c=color, marker=marker, s=size, alpha=0.8,
                       label=f"Grade {row['grade']}" if j == 0 else "")
    
    plt.yticks(y_positions, sites)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Websites', fontsize=12)
    plt.title('Performance Grade Timeline by Website', fontsize=16, fontweight='bold')
    
    # Create custom legend
    legend_elements = [plt.scatter([], [], c=color, marker='o', s=100, label=f'Grade {grade}') 
                      for grade, color in grade_colors.items()]
    legend_elements.extend([
        plt.scatter([], [], c='gray', marker='o', s=100, label='Desktop'),
        plt.scatter([], [], c='gray', marker='s', s=80, label='Mobile')
    ])
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "performance_grade_timeline.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Performance grade timeline saved to {output_path}")


def create_core_web_vitals_heatmap(df, output_dir):
    """Create heatmap showing Core Web Vitals performance across sites and time"""
    if df.empty:
        logging.warning("No data available for Core Web Vitals heatmap")
        return
    
    # Prepare data for heatmap
    cwv_metrics = ['lcp', 'fcp', 'cls', 'inp']
    
    # Calculate pass/fail for each metric
    def calculate_pass_fail(value, metric):
        if value == 'N/A' or pd.isna(value):
            return 0  # No data
        
        try:
            numeric_value = float(value)
            thresholds = {
                'lcp': 2.5,
                'fcp': 1.8,
                'cls': 0.1,
                'inp': 200
            }
            return 1 if numeric_value <= thresholds.get(metric, float('inf')) else -1
        except:
            return 0
    
    # Create pivot table for heatmap
    heatmap_data = []
    for site in df['name'].unique():
        site_data = df[df['name'] == site]
        for device_mode in ['desktop', 'mobile']:
            device_data = site_data[site_data['device_mode'] == device_mode]
            if not device_data.empty:
                latest_data = device_data.loc[device_data['date'].idxmax()]
                row = [f"{site} ({device_mode})"]
                for metric in cwv_metrics:
                    if metric in latest_data:
                        pass_fail = calculate_pass_fail(latest_data[metric], metric)
                        row.append(pass_fail)
                    else:
                        row.append(0)
                heatmap_data.append(row)
    
    if not heatmap_data:
        logging.warning("No data available for heatmap")
        return
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(heatmap_data, columns=['Site'] + [m.upper() for m in cwv_metrics])
    heatmap_df = heatmap_df.set_index('Site')
    
    # Create heatmap
    plt.figure(figsize=(10, max(8, len(heatmap_df) * 0.5)))
    
    # Custom colormap: Red for fail, Yellow for no data, Green for pass
    colors = ['#F44336', '#FFC107', '#4CAF50']  # Red, Yellow, Green
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    sns.heatmap(heatmap_df, annot=True, cmap=cmap, center=0, 
               cbar_kws={'label': 'Performance Status'},
               fmt='d', linewidths=0.5)
    
    plt.title('Core Web Vitals Pass/Fail Status Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Core Web Vitals Metrics', fontsize=12)
    plt.ylabel('Websites (Device Mode)', fontsize=12)
    
    # Add colorbar labels
    colorbar = plt.gca().collections[0].colorbar
    colorbar.set_ticks([-1, 0, 1])
    colorbar.set_ticklabels(['Fail', 'No Data', 'Pass'])
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "cwv_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Core Web Vitals heatmap saved to {output_path}")


def create_performance_distribution_chart(df, output_dir):
    """Create distribution chart showing performance score distribution"""
    if df.empty:
        logging.warning("No data available for performance distribution chart")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Create subplots for different views
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Score Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram by device mode
    ax1 = axes[0, 0]
    for device_mode in df['device_mode'].unique():
        device_data = df[df['device_mode'] == device_mode]['performance_score']
        ax1.hist(device_data, bins=20, alpha=0.7, label=f'{device_mode.title()}', 
                edgecolor='black', linewidth=0.5)
    
    ax1.set_title('Performance Score Distribution by Device')
    ax1.set_xlabel('Performance Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add grade boundaries
    grade_boundaries = [50, 60, 70, 80, 90]
    for boundary in grade_boundaries:
        ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
    
    # 2. Box plot by site
    ax2 = axes[0, 1]
    sites = df['name'].unique()
    site_scores = [df[df['name'] == site]['performance_score'].values for site in sites]
    
    box_plot = ax2.boxplot(site_scores, labels=[site[:15] + '...' if len(site) > 15 else site for site in sites])
    ax2.set_title('Performance Score Distribution by Site')
    ax2.set_xlabel('Websites')
    ax2.set_ylabel('Performance Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Color boxes based on median performance
    for patch, scores in zip(box_plot['boxes'], site_scores):
        if len(scores) > 0:
            median_score = np.median(scores)
            if median_score >= 90:
                patch.set_facecolor('#4CAF50')
            elif median_score >= 80:
                patch.set_facecolor('#8BC34A')
            elif median_score >= 70:
                patch.set_facecolor('#FF9800')
            elif median_score >= 60:
                patch.set_facecolor('#FF5722')
            else:
                patch.set_facecolor('#F44336')
            patch.set_alpha(0.7)
    
    # 3. Violin plot comparing desktop vs mobile
    ax3 = axes[1, 0]
    violin_data = [df[df['device_mode'] == mode]['performance_score'].values 
                   for mode in ['desktop', 'mobile']]
    
    violins = ax3.violinplot(violin_data, positions=[1, 2], showmeans=True, showmedians=True)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Desktop', 'Mobile'])
    ax3.set_title('Performance Score Distribution: Desktop vs Mobile')
    ax3.set_ylabel('Performance Score')
    ax3.grid(True, alpha=0.3)
    
    # Color violins
    colors = ['#4285F4', '#EA4335']
    for violin, color in zip(violins['bodies'], colors):
        violin.set_facecolor(color)
        violin.set_alpha(0.7)
    
    # 4. Grade distribution pie chart
    ax4 = axes[1, 1]
    grade_counts = df['performance_score'].apply(lambda x: 
        'A (90-100)' if x >= 90 else
        'B (80-89)' if x >= 80 else
        'C (70-79)' if x >= 70 else
        'D (60-69)' if x >= 60 else
        'F (0-59)'
    ).value_counts()
    
    colors_pie = ['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#F44336']
    ax4.pie(grade_counts.values, labels=grade_counts.index, autopct='%1.1f%%',
           colors=colors_pie[:len(grade_counts)], startangle=90)
    ax4.set_title('Performance Grade Distribution')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "performance_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Performance distribution chart saved to {output_path}")


def create_speed_metrics_comparison(df, output_dir):
    """Create comprehensive speed metrics comparison chart"""
    if df.empty:
        logging.warning("No data available for speed metrics comparison")
        return
    
    # Speed-related metrics
    speed_metrics = ['fcp', 'lcp', 'ttfb', 'tti', 'speed_index']
    available_metrics = [metric for metric in speed_metrics if metric in df.columns]
    
    if not available_metrics:
        logging.warning("No speed metrics available for comparison")
        return
    
    # Create radar chart for latest data
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    fig.suptitle('Speed Metrics Comparison: Desktop vs Mobile', fontsize=16, fontweight='bold')
    
    # Get latest data for each site
    latest_data = df.loc[df.groupby(['name', 'device_mode'])['date'].idxmax()]
    
    # Normalize metrics for radar chart (lower is better, so we invert)
    def normalize_metric(values, metric):
        """Normalize metrics to 0-100 scale (100 = best performance)"""
        if metric in ['fcp', 'lcp']:
            # Good: ≤2.5s for LCP, ≤1.8s for FCP
            threshold = 2.5 if metric == 'lcp' else 1.8
            return np.maximum(0, 100 - (values / threshold) * 50)
        elif metric == 'ttfb':
            # Good: ≤0.8s
            return np.maximum(0, 100 - (values / 0.8) * 50)
        elif metric in ['tti', 'speed_index']:
            # Good: ≤3.8s for TTI, ≤3.4s for Speed Index
            threshold = 3.8 if metric == 'tti' else 3.4
            return np.maximum(0, 100 - (values / threshold) * 50)
        else:
            # Default normalization
            return 100 - np.minimum(100, values)
    
    # Prepare data for radar charts
    for ax_idx, device_mode in enumerate(['desktop', 'mobile']):
        ax = axes[ax_idx]
        device_data = latest_data[latest_data['device_mode'] == device_mode]
        
        if device_data.empty:
            continue
        
        # Calculate angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the plot
        
        # Plot each site
        for site in device_data['name'].unique()[:5]:  # Limit to 5 sites for clarity
            site_data = device_data[device_data['name'] == site]
            if not site_data.empty:
                values = []
                for metric in available_metrics:
                    metric_value = site_data[metric].iloc[0]
                    if metric_value != 'N/A' and not pd.isna(metric_value):
                        try:
                            numeric_value = float(metric_value)
                            normalized_value = normalize_metric(np.array([numeric_value]), metric)[0]
                            values.append(normalized_value)
                        except:
                            values.append(0)
                    else:
                        values.append(0)
                
                values = np.concatenate((values, [values[0]]))  # Close the plot
                
                ax.plot(angles, values, 'o-', linewidth=2, label=site[:20])
                ax.fill(angles, values, alpha=0.25)
        
        ax.set_thetagrids(angles[:-1] * 180/np.pi, available_metrics)
        ax.set_ylim(0, 100)
        ax.set_title(f'{device_mode.title()} Performance', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(0.1, 0.1))
        ax.grid(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "speed_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Speed metrics comparison chart saved to {output_path}")


def create_monthly_performance_summary(df, output_dir):
    """Create monthly performance summary with trends and insights"""
    if df.empty:
        logging.warning("No data available for monthly summary")
        return
    
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Group by month and calculate statistics
    monthly_stats = df.groupby('month').agg({
        'performance_score': ['mean', 'std', 'count'],
        'lcp': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'fcp': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'cls': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'inp': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'name': 'nunique'  # Number of unique sites tested
    }).reset_index()
    
    # Flatten column names
    monthly_stats.columns = ['month', 'perf_mean', 'perf_std', 'test_count', 
                           'lcp_mean', 'fcp_mean', 'cls_mean', 'inp_mean', 'site_count']
    
    # Create comprehensive monthly summary
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle('Monthly Performance Summary and Trends', fontsize=16, fontweight='bold')
    
    months_str = monthly_stats['month'].astype(str)
    
    # 1. Performance Score Trend with Error Bars
    ax1 = axes[0, 0]
    ax1.errorbar(months_str, monthly_stats['perf_mean'], 
                yerr=monthly_stats['perf_std'], fmt='o-', capsize=5, 
                linewidth=3, markersize=8, color='#4285F4')
    ax1.set_title('Monthly Performance Score Trend')
    ax1.set_ylabel('Performance Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Core Web Vitals Monthly Trends
    ax2 = axes[0, 1]
    ax2.plot(months_str, monthly_stats['lcp_mean'], 'o-', label='LCP (s)', linewidth=2, markersize=6)
    ax2.plot(months_str, monthly_stats['fcp_mean'], 's-', label='FCP (s)', linewidth=2, markersize=6)
    ax2.plot(months_str, monthly_stats['cls_mean'] * 10, '^-', label='CLS (×10)', linewidth=2, markersize=6)  # Scale CLS
    ax2.set_title('Core Web Vitals Monthly Trends')
    ax2.set_ylabel('Seconds / Scaled Score')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Testing Activity
    ax3 = axes[1, 0]
    bars = ax3.bar(months_str, monthly_stats['test_count'], color='#34A853', alpha=0.7)
    ax3.set_title('Monthly Testing Activity')
    ax3.set_ylabel('Number of Tests')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, monthly_stats['test_count']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5, str(int(count)),
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Sites Monitored
    ax4 = axes[1, 1]
    ax4.plot(months_str, monthly_stats['site_count'], 'o-', 
            linewidth=3, markersize=8, color='#FF9800')
    ax4.set_title('Number of Sites Monitored')
    ax4.set_ylabel('Unique Sites')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance Distribution by Month
    ax5 = axes[2, 0]
    monthly_grades = []
    grade_months = []
    
    for month in df['month'].unique():
        month_data = df[df['month'] == month]
        grades = month_data['performance_score'].apply(lambda x: 
            'A' if x >= 90 else 'B' if x >= 80 else 'C' if x >= 70 else 'D' if x >= 60 else 'F')
        grade_counts = grades.value_counts()
        
        for grade in ['A', 'B', 'C', 'D', 'F']:
            if grade in grade_counts:
                monthly_grades.append(grade_counts[grade])
                grade_months.append((str(month), grade))
    
    # Create stacked bar chart for grades
    grade_data = {}
    for month_str in months_str:
        grade_data[month_str] = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'F': 0}
    
    for (month, grade), count in zip(grade_months, monthly_grades):
        if month in grade_data:
            grade_data[month][grade] = count
    
    bottom_values = np.zeros(len(months_str))
    colors = ['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#F44336']
    
    for i, grade in enumerate(['A', 'B', 'C', 'D', 'F']):
        values = [grade_data[month][grade] for month in months_str]
        ax5.bar(months_str, values, bottom=bottom_values, 
               label=f'Grade {grade}', color=colors[i], alpha=0.8)
        bottom_values += values
    
    ax5.set_title('Monthly Grade Distribution')
    ax5.set_ylabel('Number of Tests')
    ax5.legend()
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Performance Improvement/Decline Indicators
    ax6 = axes[2, 1]
    if len(monthly_stats) > 1:
        # Calculate month-over-month changes
        perf_changes = monthly_stats['perf_mean'].diff()
        colors_change = ['green' if change > 0 else 'red' if change < 0 else 'gray' 
                        for change in perf_changes[1:]]
        
        bars = ax6.bar(months_str[1:], perf_changes[1:], color=colors_change, alpha=0.7)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_title('Month-over-Month Performance Change')
        ax6.set_ylabel('Performance Score Change')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, change in zip(bars, perf_changes[1:]):
            if not pd.isna(change):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.1 if height > 0 else -0.3), 
                        f'{change:+.1f}', ha='center', va='bottom' if height > 0 else 'top',
                        fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Month-over-Month Performance Change')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "monthly_performance_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Monthly performance summary saved to {output_path}")


def create_all_enhanced_charts(df, output_dir):
    """Create all enhanced performance charts"""
    logging.info("Creating enhanced performance charts...")
    
    if df.empty:
        logging.warning("No data available for chart generation")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create all charts
        create_comprehensive_performance_trend(df, output_dir)
        create_performance_grade_timeline(df, output_dir)
        create_core_web_vitals_heatmap(df, output_dir)
        create_performance_distribution_chart(df, output_dir)
        create_speed_metrics_comparison(df, output_dir)
        create_monthly_performance_summary(df, output_dir)
        
        logging.info(f"All enhanced charts created successfully in {output_dir}")
        
    except Exception as e:
        logging.error(f"Error creating enhanced charts: {str(e)}")
        raise
