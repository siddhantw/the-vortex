"""
FOS (Front of Site) Checks Module
Comprehensive web quality assurance and monitoring tools

This module integrates various FOS checking capabilities:
1. Console JS Errors/Warnings Detection
2. Media Elements Analysis
3. Web Crawler for Broken/Dead Links
4. Accessibility Audit using Axe-core
5. Enhanced GTMetrix Performance Monitoring
6. Network Calls Response Time Analysis
"""

import streamlit as st
import pandas as pd
import subprocess
import os
import sys
import json
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from pathlib import Path
import tempfile
import threading
import queue
import io
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import requests

# Fix the Azure OpenAI import
try:
    # Add the path to import azure_openai_client
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from azure_openai_client import AzureOpenAIClient
    AI_AVAILABLE = True
    azure_openai_client = AzureOpenAIClient()
except ImportError as e:
    AI_AVAILABLE = False
    azure_openai_client = None
    print(f"Warning: Azure OpenAI client not available: {e}")

# Add the scripts directory to the path for imports
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import existing web crawler functionality
try:
    from fos_checks.website_crawler.web_crawler_all_brands import crawl_website, get_all_links
except ImportError:
    # Fallback if import fails
    crawl_website = None
    get_all_links = None

# Import notification system
try:
    import notifications
except ImportError:
    notifications = None

# Global variable to store crawler results
crawler_results = []

# Check if Azure OpenAI is available
AI_AVAILABLE = True
try:
    from azure_openai_client import AzureOpenAIClient
except ImportError:
    AI_AVAILABLE = False

# Analysis execution functions
def run_console_error_analysis(urls, max_depth, timeout, headless, log_levels, predefined_urls):
    """Execute console error analysis"""
    with st.spinner("Analyzing console errors and warnings..."):
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Add console logging for progress tracking
            print(f"[FOS Console Analysis] Starting console error analysis")
            print(f"[FOS Console Analysis] Parameters: max_depth={max_depth}, timeout={timeout}, headless={headless}, log_levels={log_levels}")
            print(f"[FOS Console Analysis] Predefined URL set: {predefined_urls}")
            print(f"[FOS Console Analysis] Input URLs: {urls}")

            script_path = os.path.join(script_dir, "fos_checks", "website_console_logs", "capture_js_errors_warnings_console_logs.py")
            print(f"[FOS Console Analysis] Script path: {script_path}")
            print(f"[FOS Console Analysis] Script exists: {os.path.exists(script_path)}")

            # Update status based on URL set selection
            if predefined_urls == "Custom":
                status_text.text(f"Running console error analysis on {len(urls)} custom URLs...")
            elif predefined_urls == "All Brands":
                status_text.text("Running console error analysis on all brand URLs...")
            elif predefined_urls == "Critical Pages Only":
                status_text.text("Running console error analysis on critical pages...")

            progress_bar.progress(0.3)

            # Since the original script doesn't support custom URLs, we need to create a temporary modified version
            # or run individual URL analysis. For all scenarios, use our custom implementation for better control
            if predefined_urls == "Custom" or predefined_urls == "Critical Pages Only":
                print(f"[FOS Console Analysis] Running custom URL analysis for: {urls}")
                # Run our own console error analysis for the specific URLs
                console_data = run_custom_console_analysis(urls, max_depth, headless, log_levels)
            else:
                # For "All Brands", also use custom implementation to ensure proper screenshot handling
                print(f"[FOS Console Analysis] Running custom analysis for all brands to ensure screenshot capture")
                console_data = run_custom_console_analysis(urls, max_depth, headless, log_levels)

            if notifications:
                notifications.add_notification(
                    module_name="fos_checks",
                    status="success",
                    message=f"Console analysis completed for {predefined_urls.lower()} URLs",
                    details=f"Found {len(console_data)} console entries"
                )

            # Display results
            if console_data:
                print(f"[FOS Console Analysis] Displaying results table with {len(console_data)} entries")

                # Save results to session state for later viewing - replace instead of extend for current run
                st.session_state.fos_console_results = console_data  # Replace with current results

                # Also save to a historical list that accumulates over time
                if 'fos_console_historical' not in st.session_state:
                    st.session_state.fos_console_historical = []
                st.session_state.fos_console_historical.extend(console_data)

                progress_bar.progress(1.0)
                st.success(f"Console error analysis completed for {predefined_urls.lower()}. Found {len(console_data)} console entries.")
            else:
                print(f"[FOS Console Analysis] No console data found")
                # Clear current results if no data found
                st.session_state.fos_console_results = []
                progress_bar.progress(1.0)
                st.info("No console errors or warnings found.")

        except Exception as e:
            print(f"[FOS Console Analysis] Top-level error in run_console_error_analysis: {str(e)}")
            st.error(f"Failed to run console error analysis: {str(e)}")
            if notifications:
                notifications.add_notification(
                    module_name="fos_checks",
                    status="error",
                    message="Console error analysis failed",
                    details=str(e)
                )

def run_media_optimization_analysis():
    """Run AI analysis on media optimization opportunities"""
    try:
        st.subheader("🖼️ AI Analysis of Media Optimization")

        media_data = load_media_data()
        if not media_data:
            st.warning("No media data available for analysis. Run a media analysis first.")
            return

        with st.spinner("Analyzing media optimization with AI..."):
            media_summary = prepare_media_summary(media_data)

            prompt = f"""
            Analyze the following media elements data and provide optimization recommendations:
            
            {media_summary}
            
            Please provide:
            1. Specific file size optimization opportunities
            2. Image format recommendations
            3. Lazy loading implementation suggestions
            4. Alt text improvement recommendations
            5. Overall performance impact assessment
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            st.markdown("### Media Optimization Analysis")
            st.markdown(analysis_result)

            if st.button("Download Media Optimization Report"):
                generate_media_optimization_report(media_data, analysis_result)

    except Exception as e:
        st.error(f"Error during media optimization analysis: {str(e)}")

def prepare_crawler_summary(crawler_data):
    """Prepare crawler data summary for AI analysis"""
    try:
        if not crawler_data:
            return "No crawler data available."

        df = pd.DataFrame(crawler_data)

        summary = []
        summary.append(f"Crawler Analysis Summary:")
        summary.append(f"Total URLs Analyzed: {len(df)}")

        if 'status_code' in df.columns:
            status_counts = df['status_code'].value_counts()
            for status, count in status_counts.items():
                summary.append(f"Status Code {status}: {count}")

        if 'broken_links' in df.columns:
            broken_count = df['broken_links'].sum()
            summary.append(f"Total Broken Links Found: {broken_count}")

        if 'redirects' in df.columns:
            redirect_count = df['redirects'].sum()
            summary.append(f"Total Redirects Found: {redirect_count}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing crawler summary: {str(e)}"

def run_crawler_insights_analysis():
    """Run AI analysis on crawler insights"""
    try:
        st.subheader("🔗 AI Analysis of Crawler Insights")

        crawler_data = load_crawler_data()
        if not crawler_data:
            st.warning("No crawler data available for analysis. Run a link crawler first.")
            return

        with st.spinner("Analyzing crawler insights with AI..."):
            crawler_summary = prepare_crawler_summary(crawler_data)

            prompt = f"""
            Analyze the following crawler data and provide actionable insights:
            
            {crawler_summary}
            
            Please provide:
            1. Common broken link patterns
            2. Navigation issues and recommendations
            3. SEO impact assessment
            4. Suggestions for improving site structure
            5. Prioritized list of critical issues to address
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            st.markdown("### Crawler Insights Analysis")
            st.markdown(analysis_result)

            if st.button("Download Crawler Insights Report"):
                generate_crawler_report(crawler_data, analysis_result)

    except Exception as e:
        st.error(f"Error during crawler insights analysis: {str(e)}")

def run_accessibility_insights_analysis():
    """Run AI analysis on accessibility insights"""
    try:
        st.subheader("♿ AI Analysis of Accessibility Insights")

        accessibility_data = load_accessibility_data()
        if not accessibility_data:
            st.warning("No accessibility data available for analysis. Run an accessibility audit first.")
            return

        with st.spinner("Analyzing accessibility insights with AI..."):
            accessibility_summary = prepare_accessibility_summary(accessibility_data)

            prompt = f"""
            Analyze the following accessibility audit results and provide actionable insights:
            
            {accessibility_summary}
            
            Please provide:
            1. Priority remediation steps for critical issues
            2. Common accessibility patterns that need improvement
            3. WCAG compliance recommendations
            4. Implementation guidelines for fixes
            5. Impact assessment on user experience
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            st.markdown("### Accessibility Insights Analysis")
            st.markdown(analysis_result)

            if st.button("Download Accessibility Insights Report"):
                generate_accessibility_report(accessibility_data, analysis_result)

    except Exception as e:
        st.error(f"Error during accessibility insights analysis: {str(e)}")

def run_performance_recommendations_analysis():
    """Run AI analysis on performance recommendations"""
    try:
        st.subheader("⚡ AI Analysis of Performance Recommendations")

        performance_data = load_performance_data()
        if not performance_data:
            st.warning("No performance data available for analysis. Run a performance analysis first.")
            return

        with st.spinner("Analyzing performance recommendations with AI..."):
            performance_summary = prepare_performance_summary(performance_data)

            prompt = f"""
            Analyze the following performance metrics and provide optimization recommendations:
            
            {performance_summary}
            
            Please provide:
            1. Core Web Vitals improvement strategies
            2. Resource loading optimization recommendations
            3. Caching and CDN implementation suggestions
            4. Code splitting and bundling recommendations
            5. Server-side optimization opportunities
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            st.markdown("### Performance Recommendations Analysis")
            st.markdown(analysis_result)

            if st.button("Download Performance Recommendations Report"):
                generate_performance_report(performance_data, analysis_result)

    except Exception as e:
        st.error(f"Error during performance recommendations analysis: {str(e)}")

def run_network_analysis_insights():
    """Run AI analysis on network analysis insights"""
    try:
        st.subheader("🌐 AI Analysis of Network Analysis Insights")

        network_data = load_network_data()
        if not network_data:
            st.warning("No network data available for analysis. Run a network analysis first.")
            return

        with st.spinner("Analyzing network analysis insights with AI..."):
            network_summary = prepare_network_summary(network_data)

            prompt = f"""
            Analyze the following network analysis results and provide actionable insights:
            
            {network_summary}
            
            Please provide:
            1. Network performance optimization strategies
            2. API response time improvement recommendations
            3. Resource loading efficiency suggestions
            4. Security header implementation guidelines
            5. CDN performance enhancement tips
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            st.markdown("### Network Analysis Insights")
            st.markdown(analysis_result)

            if st.button("Download Network Analysis Report"):
                generate_network_report(network_data, analysis_result)

    except Exception as e:
        st.error(f"Error during network analysis insights: {str(e)}")

def prepare_accessibility_summary(accessibility_data):
    """Prepare accessibility data summary for AI analysis"""
    try:
        if not accessibility_data:
            return "No accessibility data available."

        df = pd.DataFrame(accessibility_data)

        summary = []
        summary.append(f"Accessibility Audit Summary:")
        summary.append(f"Total Issues: {len(df)}")

        if 'impact' in df.columns:
            impact_counts = df['impact'].value_counts()
            for impact, count in impact_counts.items():
                summary.append(f"{impact.title()} Issues: {count}")

        if 'rule_id' in df.columns:
            top_issues = df['rule_id'].value_counts().head(5)
            summary.append(f"\nTop 5 Most Common Issues:")
            for rule, count in top_issues.items():
                summary.append(f"- {rule}: {count} occurrences")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing accessibility summary: {str(e)}"

def prepare_network_summary(network_data):
    """Prepare network data summary for AI analysis"""
    try:
        if not network_data:
            return "No network data available."

        df = pd.DataFrame(network_data)

        summary = []
        summary.append(f"Network Analysis Summary:")
        summary.append(f"Total URLs Analyzed: {len(df)}")

        if 'response_time' in df.columns:
            avg_response_time = df['response_time'].mean()
            summary.append(f"Average Response Time: {avg_response_time:.2f} ms")

        if 'status_code' in df.columns:
            status_counts = df['status_code'].value_counts()
            for status, count in status_counts.items():
                summary.append(f"Status Code {status}: {count}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing network summary: {str(e)}"

def prepare_performance_summary(performance_data):
    """Prepare performance data summary for AI analysis"""
    try:
        if not performance_data:
            return "No performance data available."

        df = pd.DataFrame(performance_data)

        summary = []
        summary.append(f"Performance Analysis Summary:")
        summary.append(f"Total URLs Analyzed: {len(df)}")

        if 'load_time' in df.columns:
            avg_load_time = df['load_time'].mean()
            summary.append(f"Average Load Time: {avg_load_time:.2f} seconds")

        if 'performance_score' in df.columns:
            avg_score = df['performance_score'].mean()
            summary.append(f"Average Performance Score: {avg_score:.1f}")

        if 'largest_contentful_paint' in df.columns:
            avg_lcp = df['largest_contentful_paint'].mean()
            summary.append(f"Average LCP: {avg_lcp:.2f} seconds")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing performance summary: {str(e)}"

def prepare_console_error_summary(console_data):
    """Prepare console error data summary for AI analysis"""
    try:
        if not console_data:
            return "No console error data available."

        df = pd.DataFrame(console_data)

        summary = []
        summary.append(f"Console Error Analysis Summary:")
        summary.append(f"Total Console Entries: {len(df)}")

        if 'Level' in df.columns:
            level_counts = df['Level'].value_counts()
            for level, count in level_counts.items():
                summary.append(f"{level} Level: {count}")

        if 'Brand' in df.columns:
            brand_counts = df['Brand'].value_counts()
            summary.append(f"\nIssues by Brand:")
            for brand, count in brand_counts.items():
                summary.append(f"- {brand}: {count} issues")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing console error summary: {str(e)}"

def get_ai_analysis(prompt):
    """Get AI analysis using Azure OpenAI"""
    try:
        if not AI_AVAILABLE or azure_openai_client is None:
            return "AI analysis is not available. Please check Azure OpenAI configuration."

        # Use the correct method from your AzureOpenAIClient
        response = azure_openai_client.get_completion(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.7
        )

        return response

    except Exception as e:
        st.error(f"Error getting AI analysis: {str(e)}")
        return f"Unable to get AI analysis due to error: {str(e)}"

def show_ui():
    """Main UI function for FOS Checks module"""
    st.markdown("# 🔍 FOS (Front of Site) Checks")
    st.markdown("""
    Comprehensive web quality assurance and monitoring tools for detecting issues,
    analyzing performance, and ensuring optimal user experience across your websites.
    """)

    # Create tabs for different FOS check categories
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚨 Console Errors",
        "🖼️ Media Analysis",
        "🔗 Link Crawler",
        "♿ Accessibility",
        "⚡ Performance",
        "🌐 Network Analysis",
        "🤖 AI Insights"
    ])

    with tab1:
        show_console_errors_ui()

    with tab2:
        show_media_analysis_ui()

    with tab3:
        show_link_crawler_ui()

    with tab4:
        show_accessibility_ui()

    with tab5:
        show_performance_ui()

    with tab6:
        show_network_analysis_ui()

    with tab7:
        show_ai_insights_ui()

def show_console_errors_ui():
    """UI for Console JS Errors/Warnings Detection"""
    st.markdown("## 🚨 Console JS Errors & Warnings Detection")
    st.markdown("Detect JavaScript errors and warnings in browser console logs across your websites.")

    col1, col2 = st.columns([2, 1])

    with col1:
        urls_input = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://subdomain.example.com",
            height=120,
            help="Enter the URLs you want to check for console errors"
        )

        max_depth = st.slider("Crawl Depth", 0, 3, 1, help="How deep to crawl for additional pages")

        # Log Level Selection
        st.markdown("### 📊 Log Level Selection")
        level_col1, level_col2, level_col3 = st.columns(3)

        with level_col1:
            include_severe = st.checkbox("🔴 Severe", value=True, help="Include severe/critical console errors")
        with level_col2:
            include_warnings = st.checkbox("🟡 Warning", value=True, help="Include warning messages")
        with level_col3:
            include_info = st.checkbox("🔵 Info", value=False, help="Include informational messages")

        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            timeout = st.number_input("Timeout (seconds)", 5, 60, 30)
            headless = st.checkbox("Run in headless mode", value=True)

    with col2:
        st.markdown("### Quick Actions")

        predefined_urls = st.selectbox(
            "Use predefined URL set",
            ["Custom", "All Brands", "Critical Pages Only"],
            help="Select a predefined set of URLs to test"
        )

        if predefined_urls == "All Brands":
            urls_input = """https://www.bluehost.com/
https://www.domain.com/
https://www.hostgator.com/
https://www.networksolutions.com/"""
            st.info("Loaded all brand URLs")
        elif predefined_urls == "Critical Pages Only":
            urls_input = """https://www.bluehost.com/
https://www.networksolutions.com/"""
            st.info("Loaded critical pages URLs")

        # Show selected log levels
        st.markdown("### 🎯 Selected Log Levels")
        selected_levels = []
        if include_severe:
            selected_levels.append("🔴 Severe")
        if include_warnings:
            selected_levels.append("🟡 Warning")
        if include_info:
            selected_levels.append("🔵 Info")

        if selected_levels:
            st.success(f"Including: {', '.join(selected_levels)}")
        else:
            st.warning("⚠️ No log levels selected!")

    # Validation before starting analysis
    if st.button("🔍 Start Console Error Analysis", type="primary"):
        if not (include_severe or include_warnings or include_info):
            st.error("❌ Please select at least one log level (Severe, Warning, or Info) to analyze.")
        elif predefined_urls == "Custom" and urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            log_levels = {
                'severe': include_severe,
                'warning': include_warnings,
                'info': include_info
            }
            run_console_error_analysis(urls, max_depth, timeout, headless, log_levels, predefined_urls)
        elif predefined_urls in ["All Brands", "Critical Pages Only"]:
            # For predefined sets, pass the selection type to the function
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            log_levels = {
                'severe': include_severe,
                'warning': include_warnings,
                'info': include_info
            }
            run_console_error_analysis(urls, max_depth, timeout, headless, log_levels, predefined_urls)
        else:
            st.error("Please enter at least one URL to analyze or select a predefined URL set")

    # Display recent results
    display_console_error_results()

def show_media_analysis_ui():
    """UI for Media Elements Analysis"""
    st.markdown("## 🖼️ Media Elements Analysis")
    st.markdown("Analyze images and media elements across your websites for optimization opportunities.")

    col1, col2 = st.columns([2, 1])

    with col1:
        urls_input = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://subdomain.example.com",
            height=120,
            key="media_urls"
        )

        analysis_options = st.multiselect(
            "Analysis Options",
            ["Image sizes", "Alt text validation", "Lazy loading check", "Format optimization", "Broken images"],
            default=["Image sizes", "Alt text validation", "Broken images"],
            help="Select which media elements to analyze"
        )

    with col2:
        st.markdown("### Analysis Settings")

        max_images = st.number_input("Max images per page", 10, 500, 100)
        min_file_size = st.number_input("Min file size (KB)", 0, 1000, 10)

    if st.button("🖼️ Start Media Analysis", type="primary", key="media_analysis"):
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            run_media_analysis(urls, analysis_options, max_images, min_file_size)
        else:
            st.error("Please enter at least one URL to analyze")

    # Display recent results
    display_media_analysis_results()

def show_link_crawler_ui():
    """UI for Web Crawler Broken/Dead Links Detection"""
    st.markdown("## 🔗 Link Crawler & Dead Link Detection")
    st.markdown("Crawl websites to identify broken links, dead pages, and navigation issues.")

    col1, col2 = st.columns([2, 1])

    # Initialize all variables to avoid UnboundLocalError
    base_url = ""
    max_pages = 50
    urls_input = ""
    sitemap_url = ""

    with col1:
        crawl_mode = st.radio(
            "Crawl Mode",
            ["Single URL", "Multiple URLs", "Sitemap"],
            help="Choose how to specify URLs for crawling"
        )

        if crawl_mode == "Single URL":
            base_url = st.text_input("Base URL", placeholder="https://example.com")
            max_pages = st.number_input("Max pages to crawl", 1, 1000, 50)
        elif crawl_mode == "Multiple URLs":
            urls_input = st.text_area(
                "Website URLs (one per line)",
                placeholder="https://example.com\nhttps://subdomain.example.com",
                height=120,
                key="crawler_urls"
            )
        else:  # Sitemap
            sitemap_url = st.text_input("Sitemap URL", placeholder="https://example.com/sitemap.xml")

        crawl_depth = st.slider("Crawl Depth", 0, 5, 2)

        check_options = st.multiselect(
            "Check Options",
            ["Internal links", "External links", "Images", "CSS files", "JS files", "Media files"],
            default=["Internal links", "External links", "Images"],
            help="Select which types of links to check"
        )

    with col2:
        st.markdown("### Crawler Settings")

        concurrent_requests = st.slider("Concurrent requests", 1, 20, 5)
        request_timeout = st.number_input("Request timeout (seconds)", 5, 60, 15)

        follow_redirects = st.checkbox("Follow redirects", value=True)
        check_ssl = st.checkbox("Verify SSL certificates", value=True)

        exclude_patterns = st.text_area(
            "Exclude patterns (regex)",
            placeholder=".*\\.pdf$\n.*\\.zip$",
            height=80,
            help="Regular expressions for URLs to exclude"
        )

    if st.button("🔗 Start Link Crawling", type="primary", key="link_crawler"):
        if crawl_mode == "Single URL" and base_url.strip():
            run_link_crawler(base_url, max_pages, crawl_depth, check_options,
                           concurrent_requests, request_timeout, follow_redirects,
                           check_ssl, exclude_patterns)
        elif crawl_mode == "Multiple URLs" and urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            for url in urls:
                run_link_crawler(url, max_pages, crawl_depth, check_options,
                               concurrent_requests, request_timeout, follow_redirects,
                               check_ssl, exclude_patterns)
        elif crawl_mode == "Sitemap" and sitemap_url.strip():
            run_sitemap_crawler(sitemap_url, check_options, concurrent_requests,
                              request_timeout, follow_redirects, check_ssl, exclude_patterns)
        else:
            st.error("Please provide the required URL information")

    # Display recent results
    display_link_crawler_results()

def show_accessibility_ui():
    """UI for Accessibility Audit using Axe-core"""
    st.markdown("## ♿ Accessibility Audit")
    st.markdown("Comprehensive accessibility testing using Axe-core to ensure WCAG compliance.")

    col1, col2 = st.columns([2, 1])

    with col1:
        urls_input = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://subdomain.example.com",
            height=120,
            key="accessibility_urls"
        )

        audit_options = st.multiselect(
            "Audit Categories",
            ["WCAG 2.0 Level A", "WCAG 2.0 Level AA", "WCAG 2.1 Level A", "WCAG 2.1 Level AA",
             "WCAG 2.2 Level A", "WCAG 2.2 Level AA", "Best Practices", "Experimental"],
            default=["WCAG 2.2 Level AA", "Best Practices"],
            help="Select accessibility standards to test against"
        )

        include_elements = st.multiselect(
            "Include Elements",
            ["Images", "Forms", "Navigation", "Tables", "Media", "Interactive elements"],
            default=["Images", "Forms", "Navigation"],
            help="Specific elements to focus accessibility testing on"
        )

    with col2:
        st.markdown("### Audit Settings")

        severity_filter = st.multiselect(
            "Issue Severity",
            ["Critical", "Serious", "Moderate", "Minor"],
            default=["Critical", "Serious"],
            help="Filter issues by severity level"
        )

        generate_report = st.checkbox("Generate detailed report", value=True)
        include_screenshots = st.checkbox("Include screenshots", value=False)

    if st.button("♿ Start Accessibility Audit", type="primary", key="accessibility_audit"):
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            run_accessibility_audit(urls, audit_options, include_elements, severity_filter,
                                  generate_report, include_screenshots)
        else:
            st.error("Please enter at least one URL to audit")

    # Display recent results
    display_accessibility_results()

def show_performance_ui():
    """UI for Enhanced GTMetrix Performance Monitoring"""
    st.markdown("## ⚡ Performance Monitoring")
    st.markdown("Advanced performance testing using GTMetrix and Core Web Vitals analysis.")

    col1, col2 = st.columns([2, 1])

    with col1:
        performance_mode = st.radio(
            "Performance Test Mode",
            ["Quick Test", "Comprehensive Analysis", "Historical Comparison"],
            help="Choose the type of performance analysis"
        )

        urls_input = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://subdomain.example.com",
            height=120,
            key="performance_urls"
        )

        metrics_to_analyze = st.multiselect(
            "Performance Metrics",
            ["Page Load Time", "Core Web Vitals", "GTMetrix Scores", "Resource Analysis",
             "Network Performance", "Mobile Performance"],
            default=["Page Load Time", "Core Web Vitals", "GTMetrix Scores"],
            help="Select which performance metrics to analyze"
        )

    with col2:
        st.markdown("### Test Settings")

        test_location = st.selectbox(
            "Test Location",
            ["London, UK", "Dallas, USA", "Sydney, Australia", "Vancouver, Canada"],
            help="Geographic location for performance testing"
        )

        browser = st.selectbox("Browser", ["Chrome", "Firefox"], index=0)

        device_type = st.selectbox(
            "Device Type",
            ["Desktop", "Mobile", "Both"],
            help="Device type for performance testing"
        )

        adblock_enabled = st.checkbox("Enable Ad Blocking", value=False)

        if performance_mode == "Historical Comparison":
            days_back = st.number_input("Days to compare", 1, 90, 7)

    if st.button("⚡ Start Performance Analysis", type="primary", key="performance_analysis"):
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            run_performance_analysis(urls, performance_mode, metrics_to_analyze, test_location,
                                   browser, device_type, adblock_enabled,
                                   days_back if performance_mode == "Historical Comparison" else None)
        else:
            st.error("Please enter at least one URL to analyze")

    # Display recent results
    display_performance_results()

def show_network_analysis_ui():
    """UI for Network Calls Response Time Analysis"""
    st.markdown("## 🌐 Network Analysis")
    st.markdown("Monitor network calls, API response times, and resource loading performance.")

    col1, col2 = st.columns([2, 1])

    with col1:
        analysis_type = st.radio(
            "Analysis Type",
            ["Resource Loading", "API Response Times", "Network Security Headers", "CDN Performance"],
            help="Choose the type of network analysis"
        )

        urls_input = st.text_area(
            "Website URLs (one per line)",
            placeholder="https://example.com\nhttps://subdomain.example.com",
            height=120,
            key="network_urls"
        )

        if analysis_type == "API Response Times":
            api_endpoints = st.text_area(
                "API Endpoints (one per line)",
                placeholder="/api/users\n/api/products",
                height=80,
                help="Specific API endpoints to test"
            )

        network_metrics = st.multiselect(
            "Network Metrics",
            ["Response Time", "Transfer Size", "Compression", "Caching", "Security Headers", "SSL/TLS"],
            default=["Response Time", "Transfer Size", "Security Headers"],
            help="Select which network metrics to analyze"
        )

    with col2:
        st.markdown("### Analysis Settings")

        test_iterations = st.number_input("Test iterations", 1, 10, 3)
        timeout_threshold = st.number_input("Timeout threshold (seconds)", 5, 60, 30)

        include_resources = st.multiselect(
            "Include Resources",
            ["Images", "CSS", "JavaScript", "Fonts", "Videos", "Documents"],
            default=["Images", "CSS", "JavaScript"],
            help="Resource types to include in analysis"
        )

        geographical_testing = st.checkbox("Multi-location testing", value=False)

        if geographical_testing:
            test_locations = st.multiselect(
                "Test Locations",
                ["US East", "US West", "Europe", "Asia Pacific"],
                default=["US East", "Europe"]
            )

    if st.button("🌐 Start Network Analysis", type="primary", key="network_analysis"):
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            run_network_analysis(urls, analysis_type, network_metrics, test_iterations,
                               timeout_threshold, include_resources, geographical_testing,
                               test_locations if geographical_testing else None,
                               api_endpoints if analysis_type == "API Response Times" else None)
        else:
            st.error("Please enter at least one URL to analyze")

    # Display recent results
    display_network_analysis_results()

def run_ai_console_error_analysis():
    """Run AI analysis on console error patterns"""
    try:
        st.subheader("🔍 AI Analysis of Console Error Patterns")

        with st.spinner("Analyzing console errors with AI..."):
            # Load console error data
            if 'fos_console_results' not in st.session_state or not st.session_state.fos_console_results:
                st.warning("No console error data available. Please run a console error analysis first.")
                return

            console_data = st.session_state.fos_console_results

            # Prepare data for AI analysis
            prompt = f"""
            Analyze the following console error data to find common patterns and root causes:

            {json.dumps(console_data, indent=2)}

            Please provide:
            1. Common error patterns
            2. Root cause analysis
            3. Recommendations for remediation
            4. Impact assessment on user experience
            5. Prioritized list of critical issues to address
            6. Suggestions for improving site stability
            7. Any specific brand-related issues
            8. Trends over time if applicable
            9. Potential security implications of the errors
            10. Recommendations for monitoring and alerting
            11. Best practices for preventing similar issues in the future
            12. Any additional insights based on the error data
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            # Display results
            st.markdown("### Console Error Analysis Results")
            st.markdown(analysis_result)

            if st.button("Download Console Error Report"):
                generate_console_error_report(console_data, analysis_result)

    except Exception as e:
        st.error(f"Error during AI analysis: {str(e)}")

def show_ai_insights_ui():
    """UI for AI-powered insights and analysis"""
    st.markdown("## 🤖 AI-Powered Insights")
    st.markdown("Get intelligent analysis and recommendations based on your FOS check results.")

    # Check data availability
    has_console_data = 'fos_console_results' in st.session_state and st.session_state.fos_console_results
    has_media_data = 'fos_media_results' in st.session_state and st.session_state.fos_media_results
    has_crawler_data = 'crawler_results' in st.session_state and st.session_state.crawler_results
    has_accessibility_data = 'accessibility_results' in st.session_state and st.session_state.accessibility_results
    has_performance_data = 'performance_results' in st.session_state and st.session_state.performance_results
    has_network_data = 'network_results' in st.session_state and st.session_state.network_results

    # Data availability status
    st.markdown("### 📊 Available Data for Analysis")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        status = "✅ Available" if has_console_data else "❌ No Data"
        st.markdown(f"**Console Errors:** {status}")
    with col2:
        status = "✅ Available" if has_media_data else "❌ No Data"
        st.markdown(f"**Media Elements:** {status}")
    with col3:
        status = "✅ Available" if has_crawler_data else "❌ No Data"
        st.markdown(f"**Crawler Data:** {status}")
    with col4:
        status = "✅ Available" if has_accessibility_data else "❌ No Data"
        st.markdown(f"**Accessibility:** {status}")
    with col5:
        status = "✅ Available" if has_performance_data else "❌ No Data"
        st.markdown(f"**Performance:** {status}")
    with col6:
        status = "✅ Available" if has_network_data else "❌ No Data"
        st.markdown(f"**Network Analysis:** {status}")

    if not any([has_console_data, has_media_data, has_crawler_data, has_accessibility_data, has_performance_data, has_network_data]):
        st.info("🔍 Run some FOS checks first to enable AI analysis. Use the other tabs to generate data.")
        return

    st.markdown("""
    ### 🎯 AI Analysis Types
    Choose from the following AI-powered analysis options:
    """)

    col1, col2 = st.columns([2, 1])
    with col1:
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Console Error Patterns",
                "Media Optimization",
                "Crawler Insights",
                "Accessibility Insights",
                "Performance Recommendations",
                "Network Analysis Insights",
                "Cross-Analysis Correlations"
            ],
            key="ai_analysis_type"
        )

        if st.button("Run AI Analysis"):
            if analysis_type == "Console Error Patterns":
                if has_console_data:
                    run_ai_console_error_analysis()
                else:
                    st.warning("No console error data available for analysis.")
            elif analysis_type == "Media Optimization":
                if has_media_data:
                    run_media_optimization_analysis()
                else:
                    st.warning("No media data available for analysis.")
            elif analysis_type == "Crawler Insights":
                if has_crawler_data:
                    run_crawler_insights_analysis()
                else:
                    st.warning("No crawler data available for analysis.")
            elif analysis_type == "Accessibility Insights":
                if has_accessibility_data:
                    run_accessibility_insights_analysis()
                else:
                    st.warning("No accessibility data available for analysis.")
            elif analysis_type == "Performance Recommendations":
                if has_crawler_data:
                    run_performance_recommendations_analysis()
                else:
                    st.warning("No performance data available for analysis.")
            elif analysis_type == "Network Analysis Insights":
                if has_network_data:
                    run_network_analysis_insights()
                else:
                    st.warning("No network data available for analysis.")
            elif analysis_type == "Cross-Analysis Correlations":
                run_cross_analysis_correlations()
            else:
                st.error("Invalid analysis type selected.")
    with col2:
        st.markdown("### AI Analysis Options")
        st.markdown("""
        - **Console Error Patterns**: Analyze JavaScript errors to find common patterns and root causes.
        - **Media Optimization**: Get AI-driven recommendations for optimizing images and media files.
        - **Crawler Insights**: Gain insights from web crawler data to improve site navigation and SEO.
        - **Accessibility Insights**: Understand accessibility issues and receive remediation steps.
        - **Performance Recommendations**: Receive actionable suggestions for improving page performance.
        - **Network Analysis Insights**: Analyze network calls and API performance for optimization.
        - **Cross-Analysis Correlations**: Find relationships between different FOS check results.
        """)

def display_accessibility_results():
    """Display recent accessibility audit results"""
    if 'accessibility_results' in st.session_state and st.session_state.accessibility_results:
        st.markdown("### Recent Accessibility Audit Results")

        results = st.session_state.accessibility_results
        df = pd.DataFrame(results)

        # Summary statistics
        total_issues = len(df)
        critical_issues = len(df[df['impact'] == 'critical'])
        serious_issues = len(df[df['impact'] == 'serious'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Issues", total_issues)
        with col2:
            st.metric("Critical Issues", critical_issues)
        with col3:
            st.metric("Serious Issues", serious_issues)

        # Display results table
        st.dataframe(df, use_container_width=True)

        # Download option
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Accessibility Results",
            data=csv_data,
            file_name=f"accessibility_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No recent accessibility audit results. Run an accessibility audit to see results here.")

def run_performance_analysis(urls, performance_mode, metrics_to_analyze, test_location, browser, device_type, adblock_enabled, days_back=None):
    """Run performance analysis (placeholder implementation)"""
    with st.spinner("Running performance analysis..."):
        try:
            # This is a placeholder implementation
            st.info("Performance analysis functionality is being implemented. This would integrate with GTMetrix API or similar services.")

            # Mock results for demonstration
            results = []
            for url in urls:
                results.append({
                    'url': url,
                    'load_time': 2.5 + (hash(url) % 3),
                    'page_size': 1500 + (hash(url) % 1000),
                    'gtmetrix_grade': 'B',
                    'performance_score': 85 + (hash(url) % 15),
                    'test_location': test_location,
                    'device_type': device_type,
                    'timestamp': datetime.now().isoformat()
                })

            # Store results
            if 'performance_results' not in st.session_state:
                st.session_state.performance_results = []
            st.session_state.performance_results.extend(results)

            st.success(f"Performance analysis completed for {len(urls)} URLs")

        except Exception as e:
            st.error(f"Error running performance analysis: {str(e)}")

def display_performance_results():
    """Display recent performance analysis results"""
    if 'performance_results' in st.session_state and st.session_state.performance_results:
        st.markdown("### Recent Performance Analysis Results")

        results = st.session_state.performance_results
        df = pd.DataFrame(results)

        # Summary statistics
        avg_load_time = df['load_time'].mean()
        avg_performance_score = df['performance_score'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("URLs Tested", len(df))
        with col2:
            st.metric("Avg Load Time", f"{avg_load_time:.2f}s")
        with col3:
            st.metric("Avg Performance Score", f"{avg_performance_score:.1f}")

        # Display results
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent performance analysis results. Run a performance analysis to see results here.")

def run_network_analysis(urls, analysis_type, network_metrics, test_iterations, timeout_threshold, include_resources, geographical_testing, test_locations=None, api_endpoints=None):
    """Run network analysis (placeholder implementation)"""
    with st.spinner("Running network analysis..."):
        try:
            st.info("Network analysis functionality is being implemented. This would analyze network calls, response times, and resource loading.")

            # Mock results
            results = []
            for url in urls:
                results.append({
                    'url': url,
                    'response_time': 150 + (hash(url) % 200),
                    'total_requests': 25 + (hash(url) % 20),
                    'total_size': 800 + (hash(url) % 500),
                    'ssl_grade': 'A',
                    'cdn_performance': 'Good',
                    'timestamp': datetime.now().isoformat()
                })

            # Store results
            if 'network_results' not in st.session_state:
                st.session_state.network_results = []
            st.session_state.network_results.extend(results)

            st.success(f"Network analysis completed for {len(urls)} URLs")

        except Exception as e:
            st.error(f"Error running network analysis: {str(e)}")

def display_network_analysis_results():
    """Display recent network analysis results"""
    if 'network_results' in st.session_state and st.session_state.network_results:
        st.markdown("### Recent Network Analysis Results")

        results = st.session_state.network_results
        df = pd.DataFrame(results)

        # Summary statistics
        avg_response_time = df['response_time'].mean()
        avg_requests = df['total_requests'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("URLs Tested", len(df))
        with col2:
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")
        with col3:
            st.metric("Avg Requests", f"{avg_requests:.0f}")

        # Display results
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent network analysis results. Run a network analysis to see results here.")

def run_cross_analysis_correlations():
    """Run AI analysis to find correlations between different FOS check results."""
    try:
        st.subheader("🔗 Cross-Analysis Correlations")

        # Check if we have data to analyze
        available_data = {
            'console_errors': load_console_error_data(),
            'media_elements': load_media_data(),
            'crawler_data': load_crawler_data(),
            'accessibility_issues': load_accessibility_data(),
            'performance_metrics': load_performance_data(),
            'network_results': load_network_data()
        }

        if not any(available_data.values()):
            st.warning("No data available for cross-analysis. Run some checks first.")
            return

        with st.spinner("Analyzing correlations with AI..."):
            # Prepare data summary for analysis
            data_summary = prepare_cross_analysis_summary(available_data)

            # AI analysis prompt
            prompt = f"""
            Analyze the following FOS check results and find correlations:

            {data_summary}

            Please provide:
            1. Correlations between console errors and performance metrics
            2. Media elements impact on accessibility issues
            3. Performance bottlenecks related to media loading
            4. Recommendations for improving overall site health based on correlations
            5. Any brand-specific trends or issues observed
            6. Suggestions for prioritizing fixes based on impact
            7. Insights on how different checks relate to each other
            8. Potential security implications of the findings
            9. Steps for monitoring and alerting based on correlations
            10. Best practices for preventing similar issues in the future
            11. Any additional insights based on the data provided
            12. Trends over time if applicable
            13. Suggestions for improving site stability and user experience
            """

            # Get AI analysis
            ai_client = AzureOpenAIClient()
            analysis_result = ai_client.generate_response(prompt)

            # Display results
            st.markdown("### Cross-Analysis Results")
            st.markdown(analysis_result)

            if st.button("Download Cross-Analysis Report"):
                generate_cross_analysis_report(available_data, analysis_result)

    except Exception as e:
        st.error(f"Error during cross-analysis: {str(e)}")

def prepare_cross_analysis_summary(available_data):
    """Prepare summary of available data for cross-analysis."""
    try:
        summary = []

        if available_data['console_errors']:
            console_summary = prepare_console_error_summary(available_data['console_errors'])
            summary.append(f"Console Errors:\n{console_summary}")

        if available_data['media_elements']:
            media_summary = prepare_media_summary(available_data['media_elements'])
            summary.append(f"\nMedia Elements:\n{media_summary}")

        if available_data['crawler_data']:
            crawler_summary = prepare_crawler_summary(available_data['crawler_data'])
            summary.append(f"\nCrawler Data:\n{crawler_summary}")

        if available_data['accessibility_issues']:
            accessibility_summary = prepare_accessibility_summary(available_data['accessibility_issues'])
            summary.append(f"\nAccessibility Issues:\n{accessibility_summary}")

        if available_data['performance_metrics']:
            performance_summary = prepare_performance_summary(available_data['performance_metrics'])
            summary.append(f"\nPerformance Metrics:\n{performance_summary}")

        if available_data['network_results']:
            network_summary = prepare_network_summary(available_data['network_results'])
            summary.append(f"\nNetwork Analysis:\n{network_summary}")

        if not summary:
            return "No data available for cross-analysis."

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing cross-analysis summary: {str(e)}"

def prepare_console_error_summary(console_data):
    """Prepare console error data summary for AI analysis"""
    try:
        if not console_data:
            return "No console error data available."

        summary = []
        total_errors = 0
        error_types = {}

        for result in console_data:
            if 'results' in result:
                errors = result['results']
                total_errors += len(errors)

                for error in errors:
                    error_type = error.get('type', 'Unknown')
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1

        summary.append(f"Console Error Analysis Summary:")
        summary.append(f"Total Errors: {total_errors}")
        for et, count in error_types.items():
            summary.append(f"{et}: {count}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing console error summary: {str(e)}"

def prepare_accessibility_summary(accessibility_data):
    """Prepare accessibility data summary for AI analysis"""
    try:
        if not accessibility_data:
            return "No accessibility data available."

        summary = []
        total_issues = 0
        impact_counts = {}

        for result in accessibility_data:
            if 'results' in result:
                issues = result['results']
                total_issues += len(issues)

                for issue in issues:
                    impact = issue.get('impact', 'Unknown')
                    if impact not in impact_counts:
                        impact_counts[impact] = 0
                    impact_counts[impact] += 1

        summary.append(f"Accessibility Analysis Summary:")
        summary.append(f"Total Issues: {total_issues}")
        for imp, count in impact_counts.items():
            summary.append(f"{imp.capitalize()}: {count}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing accessibility summary: {str(e)}"

def prepare_performance_summary(performance_data):
    """Prepare performance data summary for AI analysis"""
    try:
        if not performance_data:
            return "No performance data available."

        summary = []
        total_tests = len(performance_data)
        avg_load_time = sum(result['load_time_ms'] for result in performance_data) / total_tests if total_tests > 0 else 0
        success_rate = (len([r for r in performance_data if r['status_code'] == 200]) / total_tests * 100) if total_tests > 0 else 0

        summary.append(f"Performance Analysis Summary:")
        summary.append(f"Total Tests: {total_tests}")
        summary.append(f"Average Load Time: {avg_load_time:.2f} ms")
        summary.append(f"Success Rate: {success_rate:.2f}%")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing performance summary: {str(e)}"

def run_custom_console_analysis(urls, max_depth, headless, log_levels):
    """Run console error analysis for custom URLs"""
    try:
        from selenium import webdriver
        from selenium.common.exceptions import WebDriverException
        import tempfile
        import shutil

        print(f"[Custom Console Analysis] Starting analysis for {len(urls)} URLs with max_depth={max_depth}")

        all_js_errors = []

        for i, url in enumerate(urls):
            print(f"[Custom Console Analysis] Processing URL {i+1}/{len(urls)}: {url}")

            # Setup Chrome options
            chrome_options = webdriver.ChromeOptions()
            if headless:
                chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--disable-background-timer-throttling')
            chrome_options.add_argument('--disable-renderer-backgrounding')
            chrome_options.add_argument('--disable-backgrounding-occluded-windows')
            chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

            # Create temporary directory manually for better control
            user_data_dir = tempfile.mkdtemp(prefix=f"chrome_user_data_{os.getpid()}_")
            chrome_options.add_argument(f'--user-data-dir={user_data_dir}')

            driver = None
            try:
                driver = webdriver.Chrome(options=chrome_options)

                visited_urls = set()
                urls_to_visit = [(url, 0)]
                brand_name = urlparse(url).netloc.split('.')[1].capitalize()
                if brand_name.lower() == "networksolutions":
                    brand_name = "Network Solutions"

                while urls_to_visit:
                    current_url, depth = urls_to_visit.pop(0)
                    if depth > max_depth or current_url in visited_urls:
                        continue

                    visited_urls.add(current_url)
                    print(f"[Custom Console Analysis] Visiting: {current_url} at depth {depth}")

                    try:
                        driver.get(current_url)
                        time.sleep(2)

                        # Get console logs with proper categorization
                        try:
                            logs = driver.get_log('browser')
                            # Filter for all meaningful console messages and maintain proper levels
                            filtered_logs = []
                            for entry in logs:
                                if entry['message'] and not entry['message'].startswith('favicon'):
                                    # Map Chrome log levels to proper categories
                                    level_mapping = {
                                        'SEVERE': 'SEVERE',
                                        'WARNING': 'WARNING',
                                        'INFO': 'INFO'
                                    }

                                    mapped_level = level_mapping.get(entry['level'], 'INFO')

                                    # Include based on user's log level selections
                                    should_include = False
                                    if mapped_level == 'SEVERE' and log_levels['severe']:
                                        should_include = True
                                    elif mapped_level == 'WARNING' and log_levels['warning']:
                                        should_include = True
                                    elif mapped_level == 'INFO' and log_levels['info']:
                                        should_include = True

                                    if should_include:
                                        filtered_logs.append({
                                            "level": mapped_level,
                                            "message": entry['message'],
                                            "timestamp": entry['timestamp'],
                                            "original_level": entry['level']
                                        })

                            if filtered_logs:
                                screenshot_filename = f"screenshot_{urlparse(current_url).netloc.replace('.', '_')}_{depth}_{int(time.time())}.png"
                                screenshot_path = os.path.join("screenshots", screenshot_filename)
                                os.makedirs("screenshots", exist_ok=True)
                                driver.save_screenshot(screenshot_path)

                                for log in filtered_logs:
                                    all_js_errors.append({
                                        "Brand": brand_name,
                                        "URL": current_url,
                                        "Depth": depth,
                                        "Level": log['level'],  # Proper categorization
                                        "Message": log['message'],
                                        "Timestamp": datetime.fromtimestamp(log['timestamp']/1000).strftime('%Y-%m-%d %H:%M:%S'),
                                        "Screenshot": screenshot_path,
                                        "Original_Level": log['original_level']
                                    })
                            else:
                                print(f"[Custom Console Analysis] No JS errors found on {current_url}")

                        except WebDriverException as e:
                            print(f"[Custom Console Analysis] Error fetching console logs from {current_url}: {e}")

                        # Get additional links to crawl if depth allows
                        if depth < max_depth:
                            try:
                                soup = BeautifulSoup(driver.page_source, "html.parser")
                                links = [urljoin(url, a.get('href')) for a in soup.find_all('a', href=True)]
                                for link in links:
                                    parsed_link = urlparse(link)
                                    if parsed_link.netloc == urlparse(url).netloc and link not in visited_urls:
                                        urls_to_visit.append((link, depth + 1))
                            except Exception as e:
                                print(f"[Custom Console Analysis] Error getting links from {current_url}: {e}")

                    except Exception as e:
                        print(f"[Custom Console Analysis] Error processing {current_url}: {e}")

                    time.sleep(1)

            finally:
                # Ensure proper cleanup
                if driver:
                    try:
                        driver.quit()
                        print(f"[Custom Console Analysis] Driver closed for URL {i+1}")
                    except Exception as e:
                        print(f"[Custom Console Analysis] Error closing driver: {e}")

                # Wait a moment for Chrome to fully close
                time.sleep(2)

                # Clean up temporary directory
                try:
                    if os.path.exists(user_data_dir):
                        shutil.rmtree(user_data_dir, ignore_errors=True)
                        print(f"[Custom Console Analysis] Cleaned up temp directory for URL {i+1}")
                except Exception as e:
                    print(f"[Custom Console Analysis] Warning: Could not clean up temp directory {user_data_dir}: {e}")

        print(f"[Custom Console Analysis] Analysis completed. Found {len(all_js_errors)} console entries.")
        return all_js_errors

    except Exception as e:
        print(f"[Custom Console Analysis] Error in custom console analysis: {e}")
        st.error(f"Error running custom console analysis: {e}")
        return []

def run_media_analysis(urls, analysis_options, max_images, min_file_size):
    """Execute media elements analysis using the actual media_elements_reviewer.py script"""
    with st.spinner("Analyzing media elements..."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Display the fetched parameters for verification
            st.info(f"""
            **Analysis Parameters:**
            - Website URLs: {len(urls)} URLs
            - Max images per page: {max_images}
            - Min file size (KB): {min_file_size}
            - Analysis Options: {', '.join(analysis_options)}
            """)

            script_path = os.path.join(script_dir, "fos_checks", "website_crawler", "media_elements_reviewer.py")

            if not os.path.exists(script_path):
                st.error(f"Media analysis script not found at: {script_path}")
                return

            results = []
            total_urls = len(urls)

            # Create a temporary modified version of the script for custom URLs
            temp_script_path = create_custom_media_script(urls, analysis_options, max_images, min_file_size)

            try:
                status_text.text("Running media analysis with user parameters...")
                progress_bar.progress(0.3)

                # Run the modified script with user parameters
                cmd = [
                    sys.executable, temp_script_path,
                    "--max_depth", "2",  # Default crawl depth
                    "--large_size", str(min_file_size)  # Use min_file_size as threshold
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minute timeout
                progress_bar.progress(0.8)

                if result.returncode == 0:
                    # Parse the results and check for generated files
                    html_report_path, excel_report_path = check_generated_reports()

                    if html_report_path and os.path.exists(html_report_path):
                        # Display the HTML report in Streamlit
                        display_html_media_report(html_report_path, analysis_options, max_images, min_file_size)

                        # Also parse Excel data for session state storage
                        if excel_report_path and os.path.exists(excel_report_path):
                            results = parse_excel_media_results(excel_report_path)
                    else:
                        # Fallback to custom analysis if script doesn't generate expected output
                        results = run_custom_media_analysis(urls, analysis_options, max_images, min_file_size)

                    progress_bar.progress(1.0)
                else:
                    error_msg = result.stderr or "Unknown error occurred"
                    st.error(f"Error running media analysis: {error_msg}")
                    # Fallback to custom analysis
                    results = run_custom_media_analysis(urls, analysis_options, max_images, min_file_size)

            finally:
                # Clean up temporary script
                if os.path.exists(temp_script_path):
                    os.remove(temp_script_path)

                # Save results to session state with metadata
                if results or html_report_path:
                    if 'fos_media_results' not in st.session_state:
                        st.session_state.fos_media_results = []

                    # Add metadata to results
                    analysis_metadata = {
                        'timestamp': datetime.now().isoformat(),
                        'urls': urls,
                        'max_images_per_page': max_images,
                        'min_file_size_kb': min_file_size,
                        'analysis_options': analysis_options,
                        'total_media_elements': len(results),
                        'html_report_path': html_report_path,
                        'excel_report_path': excel_report_path
                    }

                    st.session_state.fos_media_results.extend([{
                        'metadata': analysis_metadata,
                        'results': results
                    }])

                    # Export results if requested and not already generated
                    if not html_report_path:
                        export_media_results(results, analysis_metadata, "HTML")

                    if notifications:
                        notifications.add_notification(
                            module_name="fos_checks",
                            status="success",
                            message=f"Media analysis completed for {len(urls)} URLs",
                            details=f"Analyzed {len(results)} media elements with {len(analysis_options)} analysis types"
                        )

                    st.success(f"Media analysis completed for {total_urls} URLs. Generated comprehensive HTML report with filtering and export capabilities.")
                else:
                    st.info("No media elements found to analyze.")

        except Exception as e:
            st.error(f"Failed to run media analysis: {str(e)}")
            # Fallback to custom analysis
            try:
                results = run_custom_media_analysis(urls, analysis_options, max_images, min_file_size)
                if results:
                    display_media_results_table(results, analysis_options, max_images, min_file_size)
            except Exception as fallback_error:
                st.error(f"Fallback analysis also failed: {str(fallback_error)}")

def create_custom_media_script(urls, analysis_options, max_images, min_file_size):
    """Create a temporary version of the media script with custom URLs"""
    try:
        # Read the original script
        original_script_path = os.path.join(script_dir, "fos_checks", "website_crawler", "media_elements_reviewer.py")

        with open(original_script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()

        # Replace the brand_urls list with user-provided URLs
        urls_list_str = ',\n        '.join([f'"{url}"' for url in urls])

        # Find and replace the brand_urls section
        import re
        pattern = r'brand_urls = \[(.*?)\]'
        replacement = f'brand_urls = [\n        {urls_list_str}\n    ]'

        modified_content = re.sub(pattern, replacement, script_content, flags=re.DOTALL)

        # Modify the LARGE_IMAGE_SIZE based on min_file_size
        modified_content = re.sub(
            r'LARGE_IMAGE_SIZE = \d+ \* 1024',
            f'LARGE_IMAGE_SIZE = {min_file_size} * 1024',
            modified_content
        )

        # Add analysis options handling (modify the analysis logic if needed)
        # This would require more complex modifications based on analysis_options

        # Create temporary script file
        temp_script_path = os.path.join(tempfile.gettempdir(), f"custom_media_reviewer_{int(time.time())}.py")

        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        return temp_script_path

    except Exception as e:
        st.error(f"Error creating custom script: {str(e)}")
        return None

def check_generated_reports():
    """Check for generated HTML and Excel reports"""
    try:
        # Look for recently generated files
        current_dir = os.getcwd()

        # Check for HTML report
        html_files = [
            "media_elements_report.html",
            os.path.join(current_dir, "media_elements_report.html")
        ]

        html_report_path = None
        for path in html_files:
            if os.path.exists(path):
                # Check if file was modified recently (within last 5 minutes)
                file_time = os.path.getmtime(path)
                current_time = time.time()
                if current_time - file_time < 300:  # 5 minutes
                    html_report_path = path
                    break

        # Check for Excel report
        excel_files = [
            "media_elements_review.xlsx",
            os.path.join(current_dir, "media_elements_review.xlsx")
        ]

        excel_report_path = None
        for path in excel_files:
            if os.path.exists(path):
                # Check if file was modified recently
                file_time = os.path.getmtime(path)
                current_time = time.time()
                if current_time - file_time < 300:  # 5 minutes
                    excel_report_path = path
                    break

        return html_report_path, excel_report_path

    except Exception as e:
        st.warning(f"Error checking for generated reports: {str(e)}")
        return None, None

def display_html_media_report(html_report_path, analysis_options, max_images, min_file_size):
    """Display the generated HTML report with enhancements"""
    try:
        # Show the parameters that were used
        st.markdown("### 📋 Analysis Configuration Used")
        params_col1, params_col2 = st.columns(2)

        with params_col1:
            st.info(f"""
            **Analysis Parameters:**
            - **Max Images per Page:** {max_images}
            - **Min File Size:** {min_file_size} KB
            """)

        with params_col2:
            st.info(f"""
            **Analysis Options Selected:**
            {chr(10).join([f"• {option}" for option in analysis_options])}
            """)

        # Read and modify the HTML report to add Streamlit-specific enhancements
        with open(html_report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Add custom CSS for better Streamlit integration
        enhanced_html = add_streamlit_enhancements(html_content, analysis_options, max_images, min_file_size)

        # Display the enhanced HTML report
        st.markdown("### 🖼️ Interactive Media Analysis Report")
        st.markdown("**Features:** Filter by brand/section/format, search, sort, export to Excel/CSV/PDF, image previews")

        # Use components to display the full HTML report
        try:
            st.components.v1.html(enhanced_html, height=800, scrolling=True)
        except AttributeError:
            # Fallback if st.components.v1 is not available
            st.markdown("### Media Analysis Report")
            st.markdown("*Interactive HTML report generated - download below to view full report*")

        # Provide download links
        st.markdown("### 📥 Download Reports")
        download_col1, download_col2 = st.columns(2)

        with download_col1:
            # Offer HTML report download
            with open(html_report_path, 'rb') as f:
                html_data = f.read()
            st.download_button(
                label="📄 Download HTML Report",
                data=html_data,
                file_name=f"media_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

        with download_col2:
            # Check for Excel report
            excel_path = html_report_path.replace('.html', '.xlsx')
            if os.path.exists(excel_path):
                with open(excel_path, 'rb') as f:
                    excel_data = f.read()
                st.download_button(
                    label="📊 Download Excel Report",
                    data=excel_data,
                    file_name=f"media_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Error displaying HTML report: {str(e)}")

def add_streamlit_enhancements(html_content, analysis_options, max_images, min_file_size):
    """Add Streamlit-specific enhancements to the HTML report"""
    try:
        # Add custom header with analysis parameters
        analysis_header = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin-bottom: 20px; border-radius: 8px;">
            <h2>🖼️ Media Analysis Report - Generated by Jarvis FOS Checks</h2>
            <div style="display: flex; gap: 30px; margin-top: 15px;">
                <div>
                    <strong>Max Images per Page:</strong> {max_images}
                </div>
                <div>
                    <strong>Min File Size:</strong> {min_file_size} KB
                </div>
                <div>
                    <strong>Analysis Options:</strong> {', '.join(analysis_options)}
                </div>
                <div>
                    <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </div>
        """

        # Insert the header after the body tag
        html_content = html_content.replace('<body>', f'<body>{analysis_header}')

        # Add custom CSS for better integration
        custom_css = """
        <style>
            .streamlit-enhancement {
                font-family: 'Source Sans Pro', sans-serif;
            }
            .dataTables_wrapper {
                margin-top: 20px;
            }
            .btn-group {
                margin: 10px 0;
            }
            .export-buttons {
                margin: 15px 0;
                text-align: center;
            }
            .export-buttons button {
                margin: 0 5px;
                padding: 8px 16px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            .export-buttons button:hover {
                background: #764ba2;
            }
            .analysis-summary {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #667eea;
            }
            /* Modal styles for full-size screenshot viewing */
            .modal {
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.8);
            }
            .modal-content {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                max-width: 90%;
                max-height: 90%;
            }
            .modal-image {
                width: 100%;
                height: auto;
                border-radius: 8px;
            }
            .close {
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }
            .close:hover {
                color: #667eea;
            }
        </style>
        """

        # Insert custom CSS before closing head tag
        html_content = html_content.replace('</head>', f'{custom_css}</head>')

        # Add analysis summary section
        summary_section = f"""
        <div class="analysis-summary">
            <h3>📊 Analysis Summary</h3>
            <p>This report was generated using the following parameters:</p>
            <ul>
                <li><strong>Image Analysis Types:</strong> {', '.join(analysis_options)}</li>
                <li><strong>Size Threshold:</strong> Minimum {min_file_size} KB file size</li>
                <li><strong>Page Limit:</strong> Maximum {max_images} images per page</li>
            </ul>
            <p><em>Use the filters and search functionality below to explore the results. All data can be exported using the buttons in the table toolbar.</em></p>
        </div>
        """

        # Insert summary before the filters section
        html_content = html_content.replace('<div class="filters">', f'{summary_section}<div class="filters">')

        return html_content
    except Exception as e:
        st.warning(f"Error enhancing HTML report: {str(e)}")
        return html_content

def parse_excel_media_results(excel_path):
    """Parse Excel report to extract media analysis results"""
    try:
        # Read the Excel file
        df = pd.read_excel(excel_path, sheet_name='All Images')

        # Convert DataFrame to list of dictionaries
        results = df.to_dict('records')

        return results

    except Exception as e:
        st.warning(f"Error parsing Excel results: {str(e)}")
        return []

# Output parsing functions
def parse_console_output_from_file():
    """Parse console error results from generated files"""
    try:
        results = []

        # Check for HTML report in multiple locations, prioritizing root directory
        html_report_paths = [
            "js_errors_and_warnings_report.html",  # Root directory (most common)
            os.path.join(os.getcwd(), "js_errors_and_warnings_report.html"),  # Current working directory
            os.path.join(script_dir, "js_errors_and_warnings_report.html"),  # Script directory
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "js_errors_and_warnings_report.html")  # Same directory as this file
        ]

        found_html_report = None
        for report_path in html_report_paths:
            if os.path.exists(report_path):
                found_html_report = os.path.abspath(report_path)  # Get absolute path
                print(f"[FOS Console Analysis] Found HTML report at: {found_html_report}")
                break

        # Look for CSV files in multiple directories with better patterns
        csv_files = []
        search_dirs = [
            os.getcwd(),  # Current working directory first
            script_dir,   # Script directory second
            os.path.dirname(os.path.abspath(__file__))  # Same directory as this file
        ]

        for search_dir in search_dirs:
            try:
                if os.path.exists(search_dir):
                    for file in os.listdir(search_dir):
                        # Look for CSV files with console error patterns
                        if (file.endswith('.csv') and
                            ('js_error' in file.lower() or 'console' in file.lower() or
                             'error' in file.lower() or 'warning' in file.lower())):
                            csv_file_path = os.path.join(search_dir, file)
                            csv_files.append(csv_file_path)
                            print(f"[FOS Console Analysis] Found CSV file: {csv_file_path}")
            except Exception as e:
                print(f"[FOS Console Analysis] Could not search directory {search_dir}: {e}")

        # Parse CSV files if found (preferred method)
        if csv_files:
            for csv_file in csv_files:
                try:
                    print(f"[FOS Console Analysis] Parsing CSV file: {csv_file}")
                    df = pd.read_csv(csv_file)
                    print(f"[FOS Console Analysis] CSV columns: {df.columns.tolist()}")
                    print(f"[FOS Console Analysis] CSV shape: {df.shape}")

                    for _, row in df.iterrows():
                        # Handle NaN values properly and validate data
                        brand = str(row.get('Brand', '')).strip() if pd.notna(row.get('Brand')) else 'Unknown'
                        url = str(row.get('URL', '')).strip() if pd.notna(row.get('URL')) else 'Unknown'
                        timestamp = str(row.get('Timestamp', '')).strip() if pd.notna(row.get('Timestamp')) else datetime.now().isoformat()
                        level = str(row.get('Level', '')).strip() if pd.notna(row.get('Level')) else 'UNKNOWN'
                        message = str(row.get('Message', '')).strip() if pd.notna(row.get('Message')) else 'No message'
                        screenshot = str(row.get('Screenshot', '')).strip() if pd.notna(row.get('Screenshot')) else 'No screenshot'
                        depth = int(row.get('Depth', 0)) if pd.notna(row.get('Depth')) and str(row.get('Depth')).isdigit() else 0

                        # Only add entries with meaningful data
                        if brand != 'Unknown' and url != 'Unknown' and message != 'No message':
                            results.append({
                                'Brand': brand,
                                'URL': url,
                                'Depth': depth,
                                'Level': level,
                                'Message': message,
                                'Timestamp': timestamp,
                                'Screenshot': screenshot,
                                'url': url,  # For backward compatibility
                                'timestamp': timestamp,  # For backward compatibility
                                'level': level,  # For backward compatibility
                                'message': message,  # For backward compatibility
                                'source': screenshot  # For backward compatibility
                            })

                    print(f"[FOS Console Analysis] Successfully parsed {len([r for r in results if 'CSV' not in str(r)])} valid entries from CSV")
                except Exception as e:
                    print(f"[FOS Console Analysis] Could not parse CSV file {csv_file}: {e}")

        # If no CSV files found but HTML report exists, try to extract meaningful data
        elif found_html_report:
            try:
                with open(found_html_report, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                print(f"[FOS Console Analysis] HTML report size: {len(html_content)} characters")

                # Check if HTML contains actual error data by looking for table rows
                import re

                # Look for data rows in the HTML table (excluding header)
                table_rows = re.findall(r'<tr[^>]*>.*?</tr>', html_content, re.DOTALL)
                data_rows = [row for row in table_rows if '<th>' not in row.lower()]  # Exclude header rows

                for i, row in enumerate(data_rows):
                    # Extract basic info from table row
                    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
                    if len(cells) >= 5:
                        brand = re.sub(r'<[^>]+>', '', cells[0]).strip()
                        url = re.sub(r'<[^>]+>', '', cells[1]).strip()
                        depth = re.sub(r'<[^>]+>', '', cells[2]).strip()
                        level = re.sub(r'<[^>]+>', '', cells[3]).strip()
                        message = re.sub(r'<[^>]+>', '', cells[4]).strip()

                        # Convert depth to integer
                        try:
                            depth = int(depth) if depth.isdigit() else 0
                        except:
                            depth = 0

                        # Only add entries with meaningful data
                        if (brand and brand != 'Unknown' and
                            url and url != 'Unknown' and
                            message and message != 'No message' and len(message) > 10):

                            results.append({
                                'Brand': brand,
                                'URL': url,
                                'Depth': depth,
                                'Level': level,
                                'Message': message,
                                'Timestamp': datetime.now().isoformat(),
                                'Screenshot': found_html_report,
                                'url': url,  # For backward compatibility
                                'timestamp': datetime.now().isoformat(),  # For backward compatibility
                                'level': level,  # For backward compatibility
                                'message': message,  # For backward compatibility
                                'source': found_html_report  # For backward compatibility
                            })
            except Exception as e:
                print(f"[FOS Console Analysis] Could not parse HTML report: {e}")
                return []
        else:
            print(f"[FOS Console Analysis] No output files found - script may have failed")
            print(f"[FOS Console Analysis] Searched in directories: {[os.path.abspath(p) for p in [os.getcwd(), script_dir]]}")
            return []

        print(f"[FOS Console Analysis] Total parsed results: {len(results)}")

        # Add some sample output for debugging
        if results:
            print(f"[FOS Console Analysis] Sample result: {results[0]}")

        return results

    except Exception as e:
        print(f"[FOS Console Analysis] Error in parse_console_output_from_file: {e}")
        st.warning(f"Could not parse console output files: {e}")
        return []

def parse_console_output(output, url):
    """Parse console error output into structured data"""
    # This would parse the actual output from the console error script
    # For now, return mock data structure
    return [
        {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'message': 'Sample console error',
            'source': 'main.js:123'
        }
    ]

def parse_media_output(output, url, analysis_options):
    """Parse media analysis output into structured data"""
    return [
        {
            'url': url,
            'image_url': 'https://example.com/image.jpg',
            'alt_text': 'Sample image',
            'file_size': '125KB',
            'dimensions': '800x600',
            'format': 'JPEG',
            'loading': 'eager'
        }
    ]

def parse_crawler_output(output, url):
    """Parse crawler output into structured data"""
    return [
        {
            'source_url': url,
            'target_url': 'https://example.com/page1',
            'status': 'working',
            'response_code': 200,
            'response_time': '0.5s'
        }
    ]

def parse_accessibility_output(output, url):
    """Parse accessibility output into structured data"""
    return [
        {
            'url': url,
            'rule': 'color-contrast',
            'severity': 'serious',
            'description': 'Color contrast issue detected',
            'element': 'button.primary',
            'help_url': 'https://dequeuniversity.com/rules/axe/4.4/color-contrast'
        }
    ]

def parse_performance_output(output, url):
    """Parse performance output into structured data"""
    return [
        {
            'url': url,
            'load_time': 2.5,
            'gtmetrix_grade': 'A',
            'page_size': '1.2MB',
            'requests': 45,
            'largest_contentful_paint': 1.8,
            'cumulative_layout_shift': 0.1
        }
    ]

def parse_network_output(output, url):
    """Parse network analysis output into structured data"""
    return [
        {
            'url': url,
            'resource_type': 'image',
            'resource_url': 'https://example.com/image.jpg',
            'response_time': 150,
            'size': '45KB',
            'status_code': 200
        }
    ]

def display_console_results_table(results, key_suffix=""):
    """Display console error results in a beautified HTML report with filters and export"""
    if not results:
        st.info("No console errors found.")
        return

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Ensure all required columns exist
    required_cols = ['Brand', 'URL', 'Depth', 'Level', 'Message', 'Timestamp', 'Screenshot']
    for col in required_cols:
        if col not in df.columns:
            if col == 'Brand':
                df[col] = 'Unknown'
            elif col == 'URL':
                df[col] = df.get('url', 'Unknown')
            elif col == 'Depth':
                df[col] = 0
            elif col == 'Level':
                df[col] = df.get('level', 'SEVERE')
            elif col == 'Message':
                df[col] = df.get('message', 'No message')
            elif col == 'Timestamp':
                df[col] = df.get('timestamp', datetime.now().isoformat())
            elif col == 'Screenshot':
                df[col] = df.get('source', 'No screenshot')

    # Clean and normalize the Level column to handle mixed data types
    df['Level'] = df['Level'].fillna('UNKNOWN').astype(str)
    df['Brand'] = df['Brand'].fillna('Unknown').astype(str)
    df['URL'] = df['URL'].fillna('Unknown').astype(str)
    df['Message'] = df['Message'].fillna('No message').astype(str)
    df['Timestamp'] = df['Timestamp'].fillna(datetime.now().isoformat()).astype(str)
    df['Screenshot'] = df['Screenshot'].fillna('No screenshot').astype(str)

    st.markdown("## 🚨 Console Error Analysis Report")

    # Filter and Sort Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        brand_filter = st.selectbox(
            "Filter by Brand",
            ["All"] + sorted(df['Brand'].unique().tolist()),
            key=f"console_brand_filter{key_suffix}",
            help="Select a specific brand to filter console errors"
        )

    with col2:
        # Get unique level values, handle any remaining NaN or mixed types
        level_values = df['Level'].dropna().astype(str).unique().tolist()
        level_values = [str(val) for val in level_values if str(val) != 'nan']
        level_filter = st.selectbox(
            "Filter by Level",
            ["All"] + sorted(level_values),
            key=f"console_level_filter{key_suffix}",
            help="Filter by console error severity level"
        )

    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Timestamp", "Brand", "URL", "Level", "Depth"],
            key=f"console_sort_by{key_suffix}",
            help="Choose field to sort results by"
        )

    with col4:
        sort_order = st.selectbox(
            "Sort Order",
            ["Descending", "Ascending"],
            key=f"console_sort_order{key_suffix}",
            help="Choose ascending or descending sort order"
        )

    # Apply filters with improved accuracy and error handling
    filtered_df = df.copy()
    original_count = len(df)

    # Improved filter logic with proper dataframe handling
    if brand_filter != "All":
        # Convert to string and handle NaN values
        filtered_df['Brand'] = filtered_df['Brand'].astype(str).fillna('Unknown')
        mask_brand = filtered_df['Brand'].str.strip().str.lower() == brand_filter.strip().lower()
        filtered_df = filtered_df[mask_brand]

    if level_filter != "All":
        # Convert to string and handle NaN values
        filtered_df['Level'] = filtered_df['Level'].astype(str).fillna('UNKNOWN')
        mask_level = filtered_df['Level'].str.strip().str.upper() == level_filter.strip().upper()
        filtered_df = filtered_df[mask_level]

    # Show filter status
    if original_count != len(filtered_df):
        st.info(f"🔍 Filtered from {original_count} to {len(filtered_df)} results based on selected filters")

    # Apply sorting
    ascending = sort_order == "Ascending"
    if sort_by == "Timestamp":
        filtered_df['sort_timestamp'] = pd.to_datetime(filtered_df['Timestamp'], errors='coerce')
        filtered_df = filtered_df.sort_values('sort_timestamp', ascending=ascending)
        filtered_df = filtered_df.drop('sort_timestamp', axis=1)
    else:
        filtered_df = filtered_df.sort_values(sort_by, ascending=ascending)

    # Export Controls
    st.markdown("### 📥 Export Options")
    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📊 Export to CSV",
            data=csv,
            file_name=f"console_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"export_csv{key_suffix}"
        )

    with export_col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="📋 Export to JSON",
            data=json_data,
            file_name=f"console_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"export_json{key_suffix}"
        )

    with export_col3:
        html_report = generate_beautified_html_report(filtered_df)
        st.download_button(
            label="📄 Export to HTML",
            data=html_report,
            file_name=f"console_errors_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key=f"export_html{key_suffix}"
        )

    # Summary Statistics
    st.markdown("### 📊 Summary Statistics")
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Total Issues", len(filtered_df))
    with summary_col2:
        severe_count = len(filtered_df[filtered_df['Level'] == 'SEVERE'])
        st.metric("Severe Issues", severe_count)
    with summary_col3:
        unique_brands = len(filtered_df['Brand'].unique())
        st.metric("Brands Affected", unique_brands)
    with summary_col4:
        unique_urls = len(filtered_df['URL'].unique())
        st.metric("URLs Affected", unique_urls)

    # Beautified HTML Report Display
    st.markdown("### 🎨 Detailed Console Error Report")

    # Generate and display the beautified HTML report
    html_content = generate_beautified_html_report(filtered_df)
    st.components.v1.html(html_content, height=600, scrolling=True)

    # Show individual error details in expandable sections with pagination
    st.markdown("### 🔍 Error Details")

    # Pagination controls
    total_items = len(filtered_df)
    items_per_page = 50
    total_pages = (total_items + items_per_page - 1) // items_per_page

    if total_pages > 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            page = st.selectbox(
                f"Page (showing {items_per_page} items per page)",
                range(1, total_pages + 1),
                key=f"error_details_page_{key_suffix}"
            )
    else:
        page = 1

    # Calculate start and end indices for current page
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)

    # Display page info
    if total_pages > 1:
        st.info(f"Showing items {start_idx + 1}-{end_idx} of {total_items} total errors (Page {page} of {total_pages})")

    # Get the current page data
    current_page_df = filtered_df.iloc[start_idx:end_idx]

    for idx, row in current_page_df.iterrows():
        with st.expander(f"🚨 {row['Brand']} - {row['Level']} - {row['URL'][:50]}..."):
            error_col1, error_col2 = st.columns([2, 1])

            with error_col1:
                st.markdown(f"**URL:** {row['URL']}")
                # Display full message without truncation
                st.markdown("**Message:**")
                st.text_area("Console Error Message", value=row['Message'], height=100, disabled=True, key=f"message_{idx}_{key_suffix}", label_visibility="collapsed")
                st.markdown(f"**Timestamp:** {row['Timestamp']}")
                st.markdown(f"**Depth:** {row['Depth']}")

            with error_col2:
                # Check if screenshot is a valid file path (not HTML report path)
                screenshot_path = row['Screenshot']
                if (screenshot_path and
                    screenshot_path != 'No screenshot' and
                    not str(screenshot_path).endswith('.html') and
                    os.path.exists(screenshot_path)):
                    st.markdown("**Screenshot:**")
                    try:
                        st.image(screenshot_path, width=200)
                    except Exception as e:
                        st.text(f"Error loading screenshot: {str(e)}")
                else:
                    st.text("No screenshot available")

def generate_beautified_html_report(df):
    """Generate a beautified HTML report for console errors"""

    # Generate table rows first
    table_rows = ""
    for _, row in df.iterrows():
        level_class = f"level-{row['Level'].lower()}" if row['Level'].lower() in ['severe', 'warning', 'info'] else "level-severe"

        # Escape HTML characters in content
        brand = str(row['Brand']).replace('<', '&lt;').replace('>', '&gt;')
        url = str(row['URL']).replace('<', '&lt;').replace('>', '&gt;')
        message = str(row['Message']).replace('<', '&lt;').replace('>', '&gt;')
        timestamp = str(row['Timestamp'])

        # Handle screenshot display - show actual thumbnail or placeholder
        screenshot_cell = ""
        screenshot_path = row['Screenshot']
        if (screenshot_path and
            screenshot_path != 'No screenshot' and
            not str(screenshot_path).endswith('.html') and
            os.path.exists(screenshot_path)):
            # Convert image to base64 for embedding in HTML
            try:
                import base64
                with open(screenshot_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    screenshot_cell = f'''
                    <div class="screenshot-container">
                        <img src="data:image/png;base64,{img_data}"
                             class="screenshot-thumbnail"
                             alt="Screenshot"
                             onclick="openScreenshot(this.src)"
                             title="Click to view full size">
                        <div class="screenshot-status">📸 Available</div>
                    </div>
                    '''
            except Exception as e:
                screenshot_cell = '<div class="screenshot-error">📸 Error loading</div>'
        else:
            screenshot_cell = '<div class="no-screenshot">❌ None</div>'

        table_rows += f"""
        <tr>
            <td class="brand-cell">{brand}</td>
            <td class="url-cell" title="{url}">{url}</td>
            <td><span class="depth-badge">{row['Depth']}</span></td>
            <td><span class="{level_class}">{row['Level']}</span></td>
            <td class="message-cell" title="{message}">{message}</td>
            <td class="timestamp-cell">{timestamp}</td>
            <td class="screenshot-cell">{screenshot_cell}</td>
        </tr>
        """

    # Calculate statistics
    total_errors = len(df)
    severe_count = len(df[df['Level'] == 'SEVERE'])
    brands_affected = len(df['Brand'].unique())
    urls_affected = len(df['URL'].unique())
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create the HTML template with enhanced screenshot functionality
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Console Error Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
        }}
        .stat-item {{
            text-align: center;
            padding: 10px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .stat-label {{
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .table-container {{
            padding: 20px;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            font-size: 0.9em;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 12px 10px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: top;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .brand-cell {{
            font-weight: 600;
            color: #495057;
        }}
        .level-severe {{
            background: #dc3545;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .level-warning {{
            background: #ffc107;
            color: #212529;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .level-info {{
            background: #17a2b8;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .url-cell {{
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .message-cell {{
            width: 100%;
            min-width: 300px;
            font-family: 'Courier New', monospace;
            font-size: 0.8em;
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            word-wrap: break-word;
            white-space: pre-wrap;
            overflow-wrap: break-word;
            hyphens: auto;
        }}
        .timestamp-cell {{
            font-size: 0.8em;
            color: #6c757d;
        }}
        .depth-badge {{
            background: #6f42c1;
            color: white;
            padding: 2px 6px;
            border-radius: 50%;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .screenshot-cell {{
            text-align: center;
            min-width: 120px;
        }}
        .screenshot-container {{
            position: relative;
            display: inline-block;
        }}
        .screenshot-thumbnail {{
            width: 80px;
            height: 60px;
            object-fit: cover;
            border-radius: 4px;
            border: 2px solid #dee2e6;
            cursor: pointer;
            transition: transform 0.2s ease;
        }}
        .screenshot-thumbnail:hover {{
            transform: scale(1.1);
            border-color: #667eea;
        }}
        .screenshot-status {{
            font-size: 0.7em;
            color: #6c757d;
            margin-top: 2px;
        }}
        .no-screenshot {{
            color: #dc3545;
            font-size: 0.8em;
            padding: 10px;
        }}
        .screenshot-error {{
            color: #fd7e14;
            font-size: 0.8em;
            padding: 10px;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
        /* Modal styles for full-size screenshot viewing */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
        }}
        .modal-content {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            max-width: 90%;
            max-height: 90%;
        }}
        .modal-image {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #667eea;
        }}
    </style>
    <script>
        function openScreenshot(src) {{
            var modal = document.getElementById('screenshotModal');
            var modalImg = document.getElementById('modalImage');
            modal.style.display = 'block';
            modalImg.src = src;
        }}

        function closeModal() {{
            document.getElementById('screenshotModal').style.display = 'none';
        }}

        // Close modal when clicking outside of image
        window.onclick = function(event) {{
            var modal = document.getElementById('screenshotModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 Console Error Analysis Report</h1>
            <p>Generated on {timestamp}</p>
        </div>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{total_errors}</div>
                <div class="stat-label">Total Errors</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{severe_count}</div>
                <div class="stat-label">Severe Issues</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{brands_affected}</div>
                <div class="stat-label">Brands Affected</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{urls_affected}</div>
                <div class="stat-label">URLs Affected</div>
            </div>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Brand</th>
                        <th>URL</th>
                        <th>Depth</th>
                        <th>Level</th>
                        <th>Message</th>
                        <th>Timestamp</th>
                        <th>Screenshot</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>🔍 FOS Console Error Analysis Report - Generated by Jarvis Test Automation</p>
        </div>
    </div>

    <!-- Screenshot Modal -->
    <div id="screenshotModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <div class="modal-content">
            <img id="modalImage" class="modal-image" alt="Full size screenshot">
        </div>
    </div>
</body>
</html>"""

    return html_content

# Add web crawler functions
def run_link_crawler(base_url, max_pages, crawl_depth, check_options, concurrent_requests, request_timeout, follow_redirects, check_ssl, exclude_patterns):
    """
    Run the link crawler using existing functionality from web_crawler_all_brands.py
    """
    global crawler_results

    if crawl_website is None:
        st.error("Web crawler functionality not available. Please check the import.")
        return

    with st.spinner(f"Crawling {base_url}..."):
        try:
            # Use existing crawler functionality
            brand_name = base_url
            crawled_data = crawl_website(base_url, brand_name, crawl_depth)

            # Store results globally for display
            crawler_results.extend(crawled_data)

            # Display results
            if crawled_data:
                st.success(f"Successfully crawled {len(crawled_data)} URLs from {base_url}")

                # Create DataFrame for display
                df = pd.DataFrame(crawled_data)

                # Apply filtering based on check_options
                if "Internal links" not in check_options:
                    # Could add logic to filter internal vs external links
                    pass

                # Color code based on status codes
                def color_status(val):
                    if val >= 400:
                        return 'background-color: #ff4444; color: white'
                    elif val >= 300:
                        return 'background-color: #ffaa00; color: black'
                    else:
                        return 'background-color: #44aa44; color: white'

                # Display styled dataframe
                styled_df = df.style.map(color_status, subset=['HTTP Status Code'])
                st.dataframe(styled_df, use_container_width=True)

                # Summary statistics
                total_urls = len(df)
                broken_links = len(df[df['HTTP Status Code'] >= 400])
                redirects = len(df[(df['HTTP Status Code'] >= 300) & (df['HTTP Status Code'] < 400)])

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total URLs", total_urls)
                with col2:
                    st.metric("Broken Links", broken_links)
                with col3:
                    st.metric("Redirects", redirects)

                # Download options
                st.download_button(
                    label="Download Results as CSV",
                    data=df.to_csv(index=False),
                    file_name=f"crawler_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            else:
                st.warning("No URLs were crawled. Please check the URL and try again.")

        except Exception as e:
            st.error(f"Error during crawling: {str(e)}")

def run_sitemap_crawler(sitemap_url, check_options, concurrent_requests, request_timeout, follow_redirects, check_ssl, exclude_patterns):
    """
    Run sitemap crawler by parsing sitemap and then crawling individual URLs
    """
    global crawler_results

    with st.spinner(f"Processing sitemap {sitemap_url}..."):
        try:
            import xml.etree.ElementTree as ET
            import requests

            # Fetch sitemap
            response = requests.get(sitemap_url, timeout=request_timeout)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)

            # Extract URLs from sitemap
            urls = []
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None:
                    urls.append(loc_elem.text)

            if urls:
                st.info(f"Found {len(urls)} URLs in sitemap. Checking status...")

                # Check each URL
                results = []
                progress_bar = st.progress(0)

                for i, url in enumerate(urls[:100]):  # Limit to first 100 URLs
                    try:
                        start_time = time.time()
                        response = requests.get(url, timeout=request_timeout, allow_redirects=follow_redirects)
                        response_time = round(time.time() - start_time, 2)

                        results.append({
                            "Brand URL": sitemap_url,
                            "URLs Captured": url,
                            "Depth": 0,
                            "HTTP Status Code": response.status_code,
                            "Response Time (s)": response_time,
                            "Last Updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

                    except Exception as e:
                        results.append({
                            "Brand URL": sitemap_url,
                            "URLs Captured": url,
                            "Depth": 0,
                            "HTTP Status Code": 0,
                            "Response Time (s)": 0,
                            "Last Updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

                    progress_bar.progress((i + 1) / min(len(urls), 100))

                # Store and display results
                crawler_results.extend(results)

                if results:
                    df = pd.DataFrame(results)

                    # Color code based on status codes
                    def color_status(val):
                        if val >= 400:
                            return 'background-color: #ff4444; color: white'
                        elif val >= 300:
                            return 'background-color: #ffaa00; color: black'
                        else:
                            return 'background-color: #44aa44; color: white'

                    styled_df = df.style.map(color_status, subset=['HTTP Status Code'])
                    st.dataframe(styled_df, use_container_width=True)

                    # Summary statistics
                    total_urls = len(df)
                    broken_links = len(df[df['HTTP Status Code'] >= 400])
                    redirects = len(df[(df['HTTP Status Code'] >= 300) & (df['HTTP Status Code'] < 400)])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total URLs", total_urls)
                    with col2:
                        st.metric("Broken Links", broken_links)
                    with col3:
                        st.metric("Redirects", redirects)

                    st.success(f"Sitemap crawling completed. Checked {len(results)} URLs.")

            else:
                st.warning("No URLs found in the sitemap.")

        except Exception as e:
            st.error(f"Error processing sitemap: {str(e)}")

def display_link_crawler_results():
    """
    Display recent link crawler results
    """
    global crawler_results

    if crawler_results:
        st.markdown("### Recent Crawler Results")

        # Show all results, not just last 50
        df = pd.DataFrame(crawler_results)

        # Add pagination controls
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Total URLs in crawler history:** {len(crawler_results)}")

        with col2:
            results_per_page = st.selectbox(
                "Results per page",
                [25, 50, 100, "All"],
                index=1,
                key="crawler_results_per_page"
            )

        # Apply pagination
        if results_per_page == "All":
            display_df = df
            start_idx = 0
            end_idx = len(df)
        else:
            # Add page selection
            total_pages = (len(df) - 1) // results_per_page + 1 if len(df) > 0 else 1
            page = st.selectbox(
                f"Page (1-{total_pages})",
                range(1, total_pages + 1),
                index=total_pages - 1,  # Start with last page (most recent)
                key="crawler_results_page"
            )

            start_idx = (page - 1) * results_per_page
            end_idx = min(start_idx + results_per_page, len(df))
            display_df = df.iloc[start_idx:end_idx]

        # Summary metrics for current page/view
        total_urls = len(display_df)
        broken_links = len(display_df[display_df['HTTP Status Code'] >= 400])
        success_rate = ((total_urls - broken_links) / total_urls * 100) if total_urls > 0 else 0

        # Show metrics for current page/view
        st.markdown(f"**Showing results {start_idx + 1}-{end_idx} of {len(df)} total**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("URLs in Current View", total_urls)
        with col2:
            st.metric("Broken Links (Current View)", broken_links)
        with col3:
            st.metric("Success Rate (Current View)", f"{success_rate:.1f}%")

        # Overall statistics
        if results_per_page != "All":
            overall_broken = len(df[df['HTTP Status Code'] >= 400])
            overall_success_rate = ((len(df) - overall_broken) / len(df) * 100) if len(df) > 0 else 0

            st.markdown("#### Overall Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total URLs (All Pages)", len(df))
            with col2:
                st.metric("Total Broken Links", overall_broken)
            with col3:
                st.metric("Overall Success Rate", f"{overall_success_rate:.1f}%")

        # Filter options
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Success (2xx)", "Redirects (3xx)", "Client Errors (4xx)", "Server Errors (5xx)"],
            key="crawler_status_filter"
        )

        if status_filter != "All":
            if status_filter == "Success (2xx)":
                display_df = display_df[(display_df['HTTP Status Code'] >= 200) & (display_df['HTTP Status Code'] < 300)]
            elif status_filter == "Redirects (3xx)":
                display_df = display_df[(display_df['HTTP Status Code'] >= 300) & (display_df['HTTP Status Code'] < 400)]
            elif status_filter == "Client Errors (4xx)":
                display_df = display_df[(display_df['HTTP Status Code'] >= 400) & (display_df['HTTP Status Code'] < 500)]
            elif status_filter == "Server Errors (5xx)":
                display_df = display_df[display_df['HTTP Status Code'] >= 500]

        # Display filtered results
        if not display_df.empty:
            # Add color coding for better visualization with proper contrast
            def highlight_status(row):
                if row['HTTP Status Code'] >= 400:
                    return ['background-color: #ffcdd2; color: #c62828; font-weight: bold'] * len(row)  # Light red background, dark red text
                elif row['HTTP Status Code'] >= 300:
                    return ['background-color: #ffe0b2; color: #f57c00; font-weight: bold'] * len(row)  # Light orange background, dark orange text
                else:
                    return ['background-color: #c8e6c9; color: #2e7d32; font-weight: bold'] * len(row)  # Light green background, dark green text

            styled_df = display_df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_df, use_container_width=True)

            st.session_state ['crawler_results'] = display_df.to_dict(orient='records')
            # Display the DataFrame
            st.markdown("### Crawler Results Table")
            st.dataframe(display_df, use_container_width=True)

            # Export option for current view
            if st.button("📥 Export Current View as CSV", key="export_crawler_results"):
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"crawler_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_crawler_csv"
                )
        else:
            st.info("No results match the selected filter.")

        # Clear results option
        if st.button("🗑️ Clear Crawler History", key="clear_crawler_results"):
            crawler_results.clear()
            st.success("Crawler history cleared!")
            st.rerun()
    else:
        st.info("No crawler results available. Run a crawl to see results here.")

def display_console_error_results():
    """Display recent console error results"""
    if 'fos_console_results' in st.session_state and st.session_state.fos_console_results:
        st.markdown("### Recent Console Error Results")
        display_console_results_table(st.session_state.fos_console_results, "_recent")
    else:
        st.markdown(
            '<div style="background-color: var(--primary-color); padding: 10px; border-radius: var(--border-radius-md); color: white;">No recent console error results. Run an analysis to see results here.</div>',
            unsafe_allow_html=True)

def display_media_analysis_results():
    """Display recent media analysis results"""
    if 'fos_media_results' in st.session_state and st.session_state.fos_media_results:
        st.markdown("### Recent Media Analysis Results")
        latest_results = st.session_state.fos_media_results[-1]
        st.json(latest_results['metadata'])
        if latest_results['results']:
            df = pd.DataFrame(latest_results['results'])
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No recent media analysis results. Run an analysis to see results here.")

def run_custom_media_analysis(urls, analysis_options, max_images, min_file_size):
    """Run custom media analysis for the provided URLs"""
    results = []
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        import requests
        from urllib.parse import urljoin, urlparse

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)

        for url in urls:
            try:
                driver.get(url)
                time.sleep(2)

                # Find all images
                images = driver.find_elements(By.TAG_NAME, "img")

                for i, img in enumerate(images[:max_images]):
                    try:
                        src = img.get_attribute("src")
                        alt = img.get_attribute("alt") or ""

                        if src:
                            full_url = urljoin(url, src)

                            # Check file size if requested
                            file_size = 0
                            if "Image sizes" in analysis_options:
                                try:
                                    import requests
                                    response = requests.head(full_url, timeout=5)
                                    file_size = int(response.headers.get('content-length', 0)) / 1024  # KB
                                except:
                                    file_size = 0

                            # Only include if meets minimum size requirement
                            if file_size >= min_file_size or "Image sizes" not in analysis_options:
                                results.append({
                                    'url': url,
                                    'image_url': full_url,
                                    'alt_text': alt,
                                    'file_size_kb': round(file_size, 2),
                                    'has_alt_text': bool(alt.strip()),
                                    'loading': img.get_attribute("loading") or "eager"
                                })
                    except Exception as e:
                        continue

            except Exception as e:
                st.warning(f"Could not analyze media on {url}: {str(e)}")

        driver.quit()

    except Exception as e:
        st.error(f"Error in custom media analysis: {str(e)}")

    return results

def display_media_results_table(results, analysis_options, max_images, min_file_size):
    """Display media analysis results in a table"""
    if results:
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", len(df))
        with col2:
            missing_alt = len(df[~df['has_alt_text']])
            st.metric("Missing Alt Text", missing_alt)
        with col3:
            avg_size = df['file_size_kb'].mean() if 'file_size_kb' in df.columns else 0
            st.metric("Avg Size (KB)", f"{avg_size:.1f}")

def export_media_results(results, metadata, export_format):
    """Export media analysis results"""
    if not results:
        return

    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if export_format == "CSV":
        csv_data = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            f"media_analysis_{timestamp}.csv",
            "text/csv"
        )
    elif export_format == "JSON":
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            f"media_analysis_{timestamp}.json",
            "application/json"
        )

def run_accessibility_audit(urls, audit_options, include_elements, severity_filter, generate_report, include_screenshots):
    """Run accessibility audit using axe-core with Selenium"""
    with st.spinner("Running accessibility audit..."):
        try:
            from axe_selenium_python import Axe
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            import tempfile
            import shutil

            all_results = []
            progress_bar = st.progress(0)

            for i, url in enumerate(urls):
                try:
                    # Update progress
                    progress_bar.progress((i + 1) / len(urls))

                    # Set up headless Chrome
                    chrome_options = webdriver.ChromeOptions()
                    chrome_options.add_argument("--headless=new")
                    chrome_options.add_argument("--disable-gpu")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    chrome_options.add_argument("--window-size=1920,1080")

                    # Create temporary directory for Chrome user data
                    user_data_dir = tempfile.mkdtemp(prefix=f"chrome_accessibility_{os.getpid()}_")
                    chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

                    driver = None
                    try:
                        driver = webdriver.Chrome(options=chrome_options)
                        driver.get(url)
                        time.sleep(3)  # Wait for page to load

                        # Initialize axe
                        axe = Axe(driver)
                        axe.inject()

                        # Configure axe based on user options
                        axe_options = {}

                        # Set rules based on audit options
                        if audit_options:
                            tags = []
                            if "WCAG 2.0 Level A" in audit_options:
                                tags.append("wcag2a")
                            if "WCAG 2.0 Level AA" in audit_options:
                                tags.append("wcag2aa")
                            if "WCAG 2.1 Level A" in audit_options:
                                tags.append("wcag21a")
                            if "WCAG 2.1 Level AA" in audit_options:
                                tags.append("wcag21aa")
                            if "WCAG 2.2 Level A" in audit_options:
                                tags.append("wcag22a")
                            if "WCAG 2.2 Level AA" in audit_options:
                                tags.append("wcag22aa")
                            if "Best Practices" in audit_options:
                                tags.append("best-practice")

                            if tags:
                                axe_options["tags"] = tags

                        # Run the audit
                        results = axe.run(options=axe_options)

                        # Take screenshot if requested
                        screenshot_path = None
                        if include_screenshots:
                            screenshot_dir = "accessibility_screenshots"
                            os.makedirs(screenshot_dir, exist_ok=True)
                            screenshot_filename = f"accessibility_{urlparse(url).netloc.replace('.', '_')}_{int(time.time())}.png"
                            screenshot_path = os.path.join(screenshot_dir, screenshot_filename)
                            driver.save_screenshot(screenshot_path)

                        # Process violations based on severity filter
                        violations = results.get("violations", [])
                        filtered_violations = []

                        for violation in violations:
                            impact = violation.get("impact", "").lower()
                            if not severity_filter or any(sev.lower() in impact for sev in severity_filter):
                                # Extract key information for each violation
                                for node in violation.get("nodes", []):
                                    filtered_violations.append({
                                        'url': url,
                                        'rule_id': violation.get("id", "unknown"),
                                        'impact': violation.get("impact", "unknown"),
                                        'description': violation.get("description", "No description"),
                                        'help': violation.get("help", "No help available"),
                                        'help_url': violation.get("helpUrl", "#"),
                                        'element': node.get("target", ["unknown"])[0] if node.get("target") else "unknown",
                                        'html': node.get("html", "")[:200] + "..." if len(node.get("html", "")) > 200 else node.get("html", ""),
                                        'screenshot': screenshot_path,
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })

                        all_results.extend(filtered_violations)

                        # Store raw results for potential export
                        if not hasattr(st.session_state, 'accessibility_raw_results'):
                            st.session_state.accessibility_raw_results = []

                        st.session_state.accessibility_raw_results.append({
                            'url': url,
                            'timestamp': datetime.now().isoformat(),
                            'results': results,
                            'screenshot': screenshot_path
                        })

                    finally:
                        if driver:
                            driver.quit()

                        # Clean up temporary directory
                        try:
                            if os.path.exists(user_data_dir):
                                shutil.rmtree(user_data_dir, ignore_errors=True)
                        except Exception as e:
                            print(f"Warning: Could not clean up temp directory {user_data_dir}: {e}")

                except Exception as e:
                    st.warning(f"Error auditing {url}: {str(e)}")
                    all_results.append({
                        'url': url,
                        'rule_id': 'audit_error',
                        'impact': 'critical',
                        'description': f'Failed to audit URL: {str(e)}',
                        'help': 'Check URL accessibility and try again',
                        'help_url': '#',
                        'element': 'N/A',
                        'html': '',
                        'screenshot': None,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })

            # Display results
            if all_results:
                # Store results in session state
                st.session_state.accessibility_results = all_results

                # Create DataFrame
                df = pd.DataFrame(all_results)

                # Summary statistics
                total_issues = len(df)
                critical_issues = len(df[df['impact'] == 'critical'])
                serious_issues = len(df[df['impact'] == 'serious'])
                moderate_issues = len(df[df['impact'] == 'moderate'])
                minor_issues = len(df[df['impact'] == 'minor'])

                # Display summary
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Total Issues", total_issues)
                with col2:
                    st.metric("Critical", critical_issues, delta=None if critical_issues == 0 else f"🔴")
                with col3:
                    st.metric("Serious", serious_issues, delta=None if serious_issues == 0 else f"🟠")
                with col4:
                    st.metric("Moderate", moderate_issues, delta=None if moderate_issues == 0 else f"🟡")
                with col5:
                    st.metric("Minor", minor_issues, delta=None if minor_issues == 0 else f"🔵")

                # Color code by impact
                def highlight_impact(row):
                    if row['impact'] == 'critical':
                        return ['background-color: #ffebee; color: #c62828'] * len(row)
                    elif row['impact'] == 'serious':
                        return ['background-color: #fff3e0; color: #f57c00'] * len(row)
                    elif row['impact'] == 'moderate':
                        return ['background-color: #fffde7; color: #f9a825'] * len(row)
                    elif row['impact'] == 'minor':
                        return ['background-color: #e3f2fd; color: #1976d2'] * len(row)
                    else:
                        return [''] * len(row)

                # Display table
                styled_df = df.style.apply(highlight_impact, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                st.success(f"Accessibility audit completed. Found {total_issues} accessibility issues across {len(urls)} URLs.")

                if notifications:
                    notifications.add_notification(
                        module_name="fos_checks",
                        status="success" if critical_issues == 0 else "warning",
                        message=f"Accessibility audit completed for {len(urls)} URLs",
                        details=f"Found {total_issues} issues ({critical_issues} critical, {serious_issues} serious)"
                    )

            else:
                st.success("🎉 No accessibility issues found! All audited pages are compliant.")

        except ImportError:
            st.error("❌ axe-selenium-python is not installed. Please install it using: pip install axe-selenium-python")
        except Exception as e:
            st.error(f"❌ Error running accessibility audit: {str(e)}")
            if notifications:
                notifications.add_notification(
                    module_name="fos_checks",
                    status="error",
                    message="Accessibility audit failed",
                    details=str(e)
                )

def generate_accessibility_html_report(df, urls, audit_options):
    """Generate a comprehensive HTML report for accessibility audit"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Summary statistics
    total_issues = len(df)
    impact_counts = df['impact'].value_counts().to_dict()

    # Top issues by frequency
    top_issues = df.groupby(['rule_id', 'impact', 'description']).size().reset_index(name='count').sort_values('count', ascending=False).head(10)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessibility Audit Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .impact-critical {{ background-color: #dc3545; color: white; }}
        .impact-serious {{ background-color: #fd7e14; color: white; }}
        .impact-moderate {{ background-color: #ffc107; color: black; }}
        .impact-minor {{ background-color: #0dcaf0; color: black; }}
        .metric-card {{ border-left: 4px solid #0d6efd; }}
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="text-center mb-4">♿ Accessibility Audit Report</h1>
        <p class="text-center text-muted">Generated on {timestamp}</p>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>📊 Summary</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-2">
                                <div class="card metric-card">
                                    <div class="card-body text-center">
                                        <h3>{total_issues}</h3>
                                        <p class="mb-0">Total Issues</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card metric-card">
                                    <div class="card-body text-center">
                                        <h3>{impact_counts.get('critical', 0)}</h3>
                                        <p class="mb-0">Critical</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card metric-card">
                                    <div class="card-body text-center">
                                        <h3>{impact_counts.get('serious', 0)}</h3>
                                        <p class="mb-0">Serious</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card metric-card">
                                    <div class="card-body text-center">
                                        <h3>{impact_counts.get('moderate', 0)}</h3>
                                        <p class="mb-0">Moderate</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card metric-card">
                                    <div class="card-body text-center">
                                        <h3>{impact_counts.get('minor', 0)}</h3>
                                        <p class="mb-0">Minor</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-2">
                                <div class="card metric-card">
                                    <div class="card-body text-center">
                                        <h3>{len(urls)}</h3>
                                        <p class="mb-0">URLs Tested</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>🔍 Audit Configuration</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Standards Tested:</strong> {', '.join(audit_options) if audit_options else 'Default WCAG rules'}</p>
                        <p><strong>URLs Audited:</strong></p>
                        <ul>
                            {''.join([f'<li>{url}</li>' for url in urls])}
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5>📋 Detailed Issues</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>URL</th>
                                        <th>Rule</th>
                                        <th>Impact</th>
                                        <th>Description</th>
                                        <th>Element</th>
                                        <th>Help</th>
                                    </tr>
                                </thead>
                                <tbody>"""

    # Add table rows
    for _, row in df.iterrows():
        impact_class = f"impact-{row['impact']}"
        html_content += f"""
                                    <tr>
                                        <td>{row['url']}</td>
                                        <td>{row['rule_id']}</td>
                                        <td><span class="badge {impact_class}">{row['impact'].upper()}</span></td>
                                        <td>{row['description']}</td>
                                        <td><code>{row['element']}</code></td>
                                        <td><a href="{row['help_url']}" target="_blank">{row['help']}</a></td>
                                    </tr>"""

    html_content += """
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    return html_content

# Supporting functions for AI analysis
def load_media_data():
    """Load media data from session state or files."""
    try:
        if 'fos_media_results' in st.session_state and st.session_state.fos_media_results:
            return st.session_state.fos_media_results
        return None
    except Exception:
        return None

def load_accessibility_data():
    """Load accessibility data from session state or files."""
    try:
        if 'accessibility_raw_results' in st.session_state and st.session_state.accessibility_raw_results:
            return st.session_state.accessibility_raw_results
        return None
    except Exception:
        return None

def load_console_error_data():
    """Load console error data from session state or files."""
    try:
        if 'fos_console_results' in st.session_state and st.session_state.fos_console_results:
            return st.session_state.fos_console_results
        return None
    except Exception:
        return None

def load_network_data():
    """Load network data from session state or files."""
    try:
        if 'network_results' in st.session_state and st.session_state.network_results:
            return st.session_state.network_results
        return None
    except Exception:
        return None

def prepare_media_summary(media_data):
    """Prepare media elements data summary for AI analysis."""
    try:
        if not media_data:
            return "No media data available."

        summary = []
        total_elements = 0
        total_size = 0
        missing_alt = 0

        for media_result in media_data:
            if 'results' in media_result:
                elements = media_result['results']
                total_elements += len(elements)

                for element in elements:
                    if 'file_size_kb' in element:
                        total_size += element.get('file_size_kb', 0)
                    if not element.get('has_alt_text', True):
                        missing_alt += 1

        summary.append(f"Media Elements Analysis Summary:")
        summary.append(f"Total Media Elements: {total_elements}")
        summary.append(f"Total Size: {total_size:.2f} KB")
        summary.append(f"Missing Alt Text: {missing_alt}")
        summary.append(
            f"Average Size per Element: {(total_size / total_elements):.2f} KB" if total_elements > 0 else "N/A")

        # Add optimization opportunities
        if total_elements > 0:
            large_files = total_size / total_elements > 100  # Files > 100KB average
            summary.append(f"\nOptimization Opportunities:")
            if large_files:
                summary.append("- Consider image compression and format optimization")
            if missing_alt > 0:
                summary.append(f"- Add alt text to {missing_alt} images for accessibility")
            summary.append("- Implement lazy loading for better performance")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing media summary: {str(e)}"

def load_performance_data():
    """Load performance data from session state or files."""
    try:
        if 'performance_results' in st.session_state and st.session_state.performance_results:
            return st.session_state.performance_results
        return None
    except Exception:
        return None

def prepare_comprehensive_summary(available_data):
    """Prepare a comprehensive summary of available data for AI analysis."""
    try:
        summary = []

        if available_data['console_errors']:
            console_summary = prepare_console_error_summary(available_data['console_errors'])
            summary.append(f"Console Errors:\n{console_summary}")

        if available_data['media_elements']:
            media_summary = prepare_media_summary(available_data['media_elements'])
            summary.append(f"\nMedia Elements:\n{media_summary}")

        if available_data['accessibility_issues']:
            accessibility_summary = prepare_accessibility_summary(available_data['accessibility_issues'])
            summary.append(f"\nAccessibility Issues:\n{accessibility_summary}")

        if available_data['performance_metrics']:
            performance_summary = prepare_performance_summary(available_data['performance_metrics'])
            summary.append(f"\nPerformance Metrics:\n{performance_summary}")

        if available_data['crawler_results']:
            crawler_summary = prepare_correlation_summary(available_data['crawler_results'])
            summary.append(f"\nCrawler Results:\n{crawler_summary}")

        if available_data['network_data']:
            network_summary = prepare_correlation_summary(available_data['network_data'])
            summary.append(f"\nNetwork Data:\n{network_summary}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing comprehensive summary: {str(e)}"

def prepare_correlation_summary(available_data):
    """Prepare correlation summary (placeholder)."""
    return prepare_comprehensive_summary(available_data)

def generate_crawler_report(crawler_data, analysis_result):
    """Generate and download crawler report."""
    try:
        st.success("Crawler report generated! (Download functionality to be implemented)")
    except Exception as e:
        st.error(f"Error generating crawler report: {str(e)}")

def generate_accessibility_report(accessibility_data, analysis_result):
    """Generate and download accessibility report."""
    try:
        st.success("Accessibility report generated! (Download functionality to be implemented)")
    except Exception as e:
        st.error(f"Error generating accessibility report: {str(e)}")

def generate_performance_report(performance_data, analysis_result):
    try:
        # This would load historical data from files or database
        # For now, return None as placeholder
        return None
    except Exception:
        return None

def generate_cross_analysis_report(available_data, analysis_result):
    """Generate and download cross-analysis report."""
    try:
        st.success("Cross-analysis report generated! (Download functionality to be implemented)")
    except Exception as e:
        st.error(f"Error generating cross-analysis report: {str(e)}")

def prepare_cross_analysis_summary(available_data):
    """Prepare summary of available data for cross-analysis."""
    try:
        summary = []

        if available_data['console_errors']:
            console_summary = prepare_console_error_summary(available_data['console_errors'])
            summary.append(f"Console Errors:\n{console_summary}")

        if available_data['media_elements']:
            media_summary = prepare_media_summary(available_data['media_elements'])
            summary.append(f"\nMedia Elements:\n{media_summary}")

        if available_data['crawler_results']:
            crawler_summary = prepare_correlation_summary(available_data['crawler_results'])
            summary.append(f"\nCrawler Results:\n{crawler_summary}")

        if available_data['accessibility_issues']:
            accessibility_summary = prepare_accessibility_summary(available_data['accessibility_issues'])
            summary.append(f"\nAccessibility Issues:\n{accessibility_summary}")

        if available_data['performance_metrics']:
            performance_summary = prepare_performance_summary(available_data['performance_metrics'])
            summary.append(f"\nPerformance Metrics:\n{performance_summary}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing cross-analysis summary: {str(e)}"

def prepare_console_error_summary(console_data):
    """Prepare console error data summary for AI analysis."""
    try:
        if not console_data:
            return "No console error data available."

        summary = []
        total_errors = 0
        error_types = {}

        for result in console_data:
            if 'results' in result:
                errors = result['results']
                total_errors += len(errors)

                for error in errors:
                    error_type = error.get('type', 'Unknown')
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1

        summary.append(f"Console Error Analysis Summary:")
        summary.append(f"Total Errors: {total_errors}")
        for et, count in error_types.items():
            summary.append(f"{et}: {count}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing console error summary: {str(e)}"

def prepare_accessibility_summary(accessibility_data):
    """Prepare accessibility data summary for AI analysis."""
    try:
        if not accessibility_data:
            return "No accessibility data available."

        summary = []
        total_issues = 0
        impact_counts = {}

        for result in accessibility_data:
            if 'results' in result:
                issues = result['results']
                total_issues += len(issues)

                for issue in issues:
                    impact = issue.get('impact', 'Unknown')
                    if impact not in impact_counts:
                        impact_counts[impact] = 0
                    impact_counts[impact] += 1

        summary.append(f"Accessibility Analysis Summary:")
        summary.append(f"Total Issues: {total_issues}")
        for imp, count in impact_counts.items():
            summary.append(f"{imp.capitalize()}: {count}")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing accessibility summary: {str(e)}"

def display_performance_charts(performance_data):
    """Display performance charts based on the performance data."""
    try:
        if not performance_data:
            st.warning("No performance data available for charts.")
            return

        df = pd.DataFrame(performance_data)

        # Load time distribution
        fig = px.histogram(df, x='load_time_ms', nbins=50, title='Load Time Distribution')
        fig.update_layout(xaxis_title='Load Time (ms)', yaxis_title='Frequency')
        st.plotly_chart(fig, use_container_width=True)

        # Status code distribution
        status_counts = df['status_code'].value_counts()
        fig = px.pie(status_counts, values=status_counts.values, names=status_counts.index,
                     title='Status Code Distribution')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying performance charts: {str(e)}")

def display_media_statistics(media_data):
    """Display media statistics based on the media data."""
    try:
        if not media_data:
            st.warning("No media data available for statistics.")
            return

        total_elements = sum(len(result['results']) for result in media_data if 'results' in result)
        total_size = sum(
            element.get('file_size_kb', 0) for result in media_data for element in result.get('results', [])
        )
        avg_size = total_size / total_elements if total_elements > 0 else 0

        st.markdown(f"### Media Statistics")
        st.metric("Total Media Elements", total_elements)
        st.metric("Total Size (KB)", f"{total_size:.2f}")
        st.metric("Average Size per Element (KB)", f"{avg_size:.2f}")

    except Exception as e:
        st.error(f"Error displaying media statistics: {str(e)}")

def prepare_performance_summary(performance_data):
    """Prepare performance data summary for AI analysis."""
    try:
        if not performance_data:
            return "No performance data available."

        summary = []
        total_tests = len(performance_data)
        avg_load_time = sum(result['load_time_ms'] for result in performance_data) / total_tests if total_tests > 0 else 0
        success_rate = (len([r for r in performance_data if r['status_code'] == 200]) / total_tests * 100) if total_tests > 0 else 0

        summary.append(f"Performance Analysis Summary:")
        summary.append(f"Total Tests: {total_tests}")
        summary.append(f"Average Load Time: {avg_load_time:.2f} ms")
        summary.append(f"Success Rate: {success_rate:.2f}%")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing performance summary: {str(e)}"

def display_accessibility_statistics(accessibility_data):
    """Display accessibility statistics based on the accessibility data."""
    try:
        if not accessibility_data:
            st.warning("No accessibility data available for statistics.")
            return

        total_issues = sum(len(result['results']) for result in accessibility_data if 'results' in result)
        impact_counts = {}
        for result in accessibility_data:
            if 'results' in result:
                for issue in result['results']:
                    impact = issue.get('impact', 'Unknown')
                    if impact not in impact_counts:
                        impact_counts[impact] = 0
                    impact_counts[impact] += 1

        st.markdown(f"### Accessibility Statistics")
        st.metric("Total Accessibility Issues", total_issues)
        for impact, count in impact_counts.items():
            st.metric(f"{impact.capitalize()} Issues", count)

    except Exception as e:
        st.error(f"Error displaying accessibility statistics: {str(e)}")

def load_media_data():
    """Load media data from session state or files."""
    try:
        if 'fos_media_results' in st.session_state and st.session_state.fos_media_results:
            return st.session_state.fos_media_results
        return None
    except Exception:
        return None

def generate_media_optimization_report(media_data, analysis_result):
    """Generate and download media optimization report."""
    try:
        report_data = analysis_result.to_csv(index=False)  # Convert the analysis result to CSV format
        st.download_button(
            label="Download Media Optimization Report",
            data=report_data,
            file_name="media_optimization_report.csv",
            mime="text/csv"
        )
        st.success("Media optimization report generated!")
    except Exception as e:
        st.error(f"Error generating media optimization report: {str(e)}")

def load_accessibility_data():
    """Load accessibility data from session state or files."""
    try:
        if 'accessibility_raw_results' in st.session_state and st.session_state.accessibility_raw_results:
            return st.session_state.accessibility_raw_results
        return None
    except Exception:
        return None

def generate_console_error_report(console_data, analysis_result):
    """Generate and download console error report."""
    try:
        report_data = analysis_result.to_csv(index=False)  # Convert the analysis result to CSV format
        st.download_button(
            label="Download Console Error Report",
            data=report_data,
            file_name="console_error_report.csv",
            mime="text/csv"
        )
        st.success("Console error report generated!")
    except Exception as e:
        st.error(f"Error generating console error report: {str(e)}")

def prepare_correlation_summary(crawler_results):
    """Prepare summary of crawler results for correlation analysis."""
    try:
        if not crawler_results:
            return "No crawler results available."

        summary = []
        total_urls = len(crawler_results)
        total_load_time = sum(result['load_time_ms'] for result in crawler_results)
        avg_load_time = total_load_time / total_urls if total_urls > 0 else 0

        summary.append(f"Crawler Analysis Summary:")
        summary.append(f"Total URLs Crawled: {total_urls}")
        summary.append(f"Total Load Time: {total_load_time:.2f} ms")
        summary.append(f"Average Load Time: {avg_load_time:.2f} ms")

        return "\n".join(summary)
    except Exception as e:
        return f"Error preparing correlation summary: {str(e)}"

def load_crawler_data():
    """Load crawler data from session state or files."""
    try:
        # First try to load from session state
        if 'crawler_results' in st.session_state and st.session_state.crawler_results:
            return st.session_state.crawler_results

        # If session state is empty, try to load from Excel file
        excel_file_path = os.path.join(script_dir, "website_urls_crawled_with_depth_and_status_code.xlsx")
        if os.path.exists(excel_file_path):
            # Read all sheets from the Excel file
            all_data = []
            excel_file = pd.ExcelFile(excel_file_path)

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
                # Convert DataFrame to list of dictionaries
                sheet_data = df.to_dict('records')
                all_data.extend(sheet_data)

            # Store in session state for future use
            st.session_state.crawler_results = all_data
            return all_data

        return []
    except Exception as e:
        print(f"Error loading crawler data: {str(e)}")
        return []

def display_correlation_visualizations(crawler_results):
    """Display visualizations for correlation analysis."""
    try:
        if not crawler_results:
            st.warning("No crawler data available for visualizations.")
            return

        df = pd.DataFrame(crawler_results)

        # Load time distribution
        fig = px.histogram(df, x='load_time_ms', nbins=50, title='Crawler Load Time Distribution')
        fig.update_layout(xaxis_title='Load Time (ms)', yaxis_title='Frequency')
        st.plotly_chart(fig, use_container_width=True)

        # Status code distribution
        status_counts = df['status_code'].value_counts()
        fig = px.pie(status_counts, values=status_counts.values, names=status_counts.index,
                     title='Crawler Status Code Distribution')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying correlation visualizations: {str(e)}")

if __name__ == "__main__":
    show_ui()
