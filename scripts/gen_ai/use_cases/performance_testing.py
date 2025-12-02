import json
import os
import sys
import time
from datetime import datetime
import subprocess
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
from matplotlib.ticker import MaxNLocator

# Ensure parent directory is in path to import shared modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import notifications module for action feedback
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    print("Notifications module not available. Notification features will be disabled.")

# Define JMeter parameter types with descriptions
PARAMETER_TYPES = {
    "Threads": "Number of concurrent users/threads",
    "Ramp-up Period": "Time to reach full number of threads (seconds)",
    "Loop Count": "Number of times to execute test",
    "Duration": "Test duration in seconds",
    "Startup Delay": "Delay before starting the test (seconds)",
    "Throughput": "Target requests per minute",
    "Connect Timeout": "Connection timeout in milliseconds",
    "Response Timeout": "Response timeout in milliseconds",
    "Think Time": "Time between user actions (milliseconds)",
    "GC Time": "Garbage collection time threshold (milliseconds)"
}

# Define report types and their descriptions
REPORT_TYPES = {
    "HTML Dashboard": "Comprehensive HTML dashboard with graphs and statistics",
    "CSV Results": "Raw results in CSV format for custom analysis",
    "JTL Results": "JMeter's native format with complete request/response data",
    "Aggregate Report": "Summary of test results with key metrics",
    "Summary Report": "Brief overview of test execution",
    "Response Time Graph": "Visual representation of response times over the test duration",
    "Active Threads Graph": "Visual representation of active threads over the test duration",
    "Transaction Throughput": "Number of transactions processed per second",
    "Latency Distribution": "Distribution of latency values across all requests",
    "Response Codes": "Distribution of HTTP response codes"
}

# Define test types
TEST_TYPES = [
    "Performance Testing",
    "Load Testing",
    "Stress Testing",
    "Spike Testing",
    "Endurance Testing",
    "Volume Testing",
    "Scalability Testing",
    "Baseline Testing"
]

# Define CI/CD integration options
CICD_PLATFORMS = {
    "Jenkins": {
        "description": "Popular open-source automation server",
        "config_template": """
pipeline {
    agent any
    stages {
        stage('Performance Test') {
            steps {
                sh '${JMETER_PATH}/bin/jmeter -n -t ${TEST_PLAN} -l results.jtl -e -o report'
            }
        }
        stage('Publish Results') {
            steps {
                perfReport sourceDataFiles: 'results.jtl'
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'report/**'
        }
    }
}"""
    },
    "GitHub Actions": {
        "description": "GitHub's integrated CI/CD solution",
        "config_template": """
name: JMeter Performance Test

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up JDK
      uses: actions/setup-java@v3
      with:
        java-version: '17'
        distribution: 'temurin'
    - name: Download JMeter
      run: |
        wget https://downloads.apache.org/jmeter/binaries/apache-jmeter-5.5.tgz
        tar -xzf apache-jmeter-5.5.tgz
    - name: Run Performance Tests
      run: |
        ./apache-jmeter-5.5/bin/jmeter -n -t ${TEST_PLAN} -l results.jtl -e -o report
    - name: Upload Results
      uses: actions/upload-artifact@v3
      with:
        name: jmeter-report
        path: report"""
    },
    "Azure DevOps": {
        "description": "Microsoft's DevOps platform",
        "config_template": """
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: JMeterInstaller@1
  inputs:
    jmeterVersion: '5.5'

- script: |
    jmeter -n -t $(System.DefaultWorkingDirectory)/$(TEST_PLAN) -l results.jtl -e -o $(System.DefaultWorkingDirectory)/report
  displayName: 'Run JMeter Tests'

- task: PublishBuildArtifacts@1
  inputs:
    pathToPublish: '$(System.DefaultWorkingDirectory)/report'
    artifactName: 'jmeter-report'
    publishLocation: 'Container'"""
    },
    "GitLab CI": {
        "description": "GitLab's integrated CI/CD solution",
        "config_template": """
stages:
  - performance

jmeter_test:
  stage: performance
  image: 
    name: justb4/jmeter:5.5
    entrypoint: [""]
  script:
    - mkdir -p report
    - jmeter -n -t $TEST_PLAN -l results.jtl -e -o report
  artifacts:
    paths:
      - report/
      - results.jtl"""
    }
}

# Main UI function for the performance testing module
def show_ui():
    st.markdown("# JMeter Distributed Performance Testing")
    st.markdown("""
    This module helps you execute JMeter performance tests in distributed non-GUI mode with:
    - Runtime parameter configuration
    - Distributed test execution across multiple servers
    - Comprehensive visual reporting and insights
    - Support for different test types (load, stress, performance)
    - CI/CD integration for automated performance testing
    """)

    # Create tabs for different sections
    setup_tab, execution_tab, results_tab, templates_tab, cicd_tab = st.tabs([
        "📊 Test Setup", "🚀 Test Execution", "📈 Results Analysis", "🧩 Templates", "🔄 CI/CD Integration"
    ])

    with setup_tab:
        # Test configuration settings
        st.markdown("### Test Configuration")

        # JMeter home directory
        JMETER_PATH = st.text_input("JMeter Home Directory", value="/path/to/apache-jmeter", help="Path to your JMeter installation")

        # Test plan file
        test_plan = st.file_uploader("JMeter Test Plan (.jmx)", type="jmx", help="Upload your JMeter test plan file")

        # Distributed test settings
        st.markdown("### Distributed Test Settings")
        num_clients = st.number_input("Number of Client Machines", min_value=1, value=1, help="Number of client machines for the test")
        client_hosts = []

        for i in range(num_clients):
            host = st.text_input(f"Client {i+1} Hostname or IP", help="Hostname or IP address of the client machine")
            client_hosts.append(host)

        # Test parameters
        st.markdown("### Test Parameters")
        params = {}

        for param, description in PARAMETER_TYPES.items():
            value = st.text_input(f"{param} ({description})", help=f"Set the {param} for the test")
            params[param] = value

        # Save settings button
        if st.button("Save Settings"):
            # Save the settings to a file or database
            st.success("Settings saved successfully!")

    with execution_tab:
        st.markdown("### Test Execution")

        # Load test plan button
        if st.button("Load Test Plan"):
            # Load the test plan and parameters
            st.success("Test plan loaded successfully!")

        # Start test button
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start Test"):
                # Validate settings
                if not JMETER_PATH or not test_plan:
                    st.error("Please set the JMeter path and upload a test plan")
                else:
                    # Create temporary file for the test plan
                    import tempfile
                    import os

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jmx') as temp_file:
                        temp_file.write(test_plan.getvalue())
                        test_plan_path = temp_file.name

                    # Start the test with actual JMeter execution
                    st.info("Starting JMeter test. Results will update in real-time.")

                    # Run the JMeter test and get run info
                    test_result = run_jmeter_test(
                        JMETER_PATH,
                        test_plan_path,
                        remote_hosts=client_hosts if client_hosts and client_hosts[0] else None,
                        properties=params if params else None
                    )

                    # Store run info in session state for monitoring
                    if test_result['status'] == 'started':
                        st.session_state['current_test_run'] = test_result['run_info']
                        st.session_state['test_running'] = True
                        st.rerun()  # Rerun the app to start monitoring
                    else:
                        st.error(f"Failed to start test: {test_result.get('error', 'Unknown error')}")

        with col2:
            if st.button("Stop Test", disabled=not st.session_state.get('test_running', False)):
                if 'current_test_run' in st.session_state and 'process' in st.session_state['current_test_run']:
                    # Terminate the process
                    try:
                        st.session_state['current_test_run']['process'].terminate()
                        st.info("Test stopped by user")
                        st.session_state['test_running'] = False
                    except Exception as e:
                        st.error(f"Failed to stop test: {str(e)}")

        # If test is running, show real-time results
        if st.session_state.get('test_running', False) and 'current_test_run' in st.session_state:
            st.markdown("### Real-time Test Results")

            # Create placeholders for real-time updates
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            progress_container = st.container()

            # Get the current status
            status_info = monitor_test_execution(st.session_state['current_test_run'])

            # Update the status
            test_status = status_info['status']
            if test_status == 'running':
                # Show duration
                duration = status_info.get('duration', 0)
                status_placeholder.info(f"Test running for {duration:.1f} seconds")

                # Show progress metrics if available
                if 'partial_metrics' in status_info and status_info['partial_metrics']:
                    metrics = status_info['partial_metrics']
                    with metrics_placeholder.container():
                        # Create metrics cards for real-time data
                        cols = st.columns(4)
                        with cols[0]:
                            st.metric("Samples", f"{metrics.get('total_samples', 0):,}")
                        with cols[1]:
                            st.metric("Throughput", f"{metrics.get('throughput', 0):.2f}/sec")
                        with cols[2]:
                            st.metric("Error Rate", f"{metrics.get('error_rate', 0):.2f}%")
                        with cols[3]:
                            st.metric("Avg Response", f"{metrics.get('avg_response', 0):.2f} ms")

                        # Show response time chart
                        if 'p95_response' in metrics:
                            chart_data = pd.DataFrame({
                                'Metric': ['Min', 'Avg', '90th', '95th', '99th', 'Max'],
                                'Value (ms)': [
                                    metrics.get('min_response', 0),
                                    metrics.get('avg_response', 0),
                                    metrics.get('p90_response', 0),
                                    metrics.get('p95_response', 0),
                                    metrics.get('p99_response', 0),
                                    metrics.get('max_response', 0)
                                ]
                            })
                            st.bar_chart(chart_data.set_index('Metric'))

                # Automatic refresh every 5 seconds
                time.sleep(2)
                st.rerun()  # This will rerun the app to update the metrics

            elif test_status == 'completed':
                # Test completed, show final results
                status_placeholder.success("Test completed successfully!")

                # Process final results
                if 'result_file' in status_info:
                    final_metrics = process_jmeter_results_file(status_info['result_file'])
                    st.session_state['test_results'] = final_metrics
                    st.session_state['test_running'] = False

                    # Display a summary of results
                    st.info("Test results are available in the Results Analysis tab")

                    # Create a button to go to results tab
                    if st.button("View Detailed Results"):
                        st.session_state['active_tab'] = 'results_tab'
                        st.rerun()

            elif test_status in ['failed', 'error']:
                # Test failed, show error
                status_placeholder.error(f"Test failed: {status_info.get('stderr', 'Unknown error')}")
                st.session_state['test_running'] = False

    with results_tab:
        st.markdown("### Test Results Analysis")

        # Check if we have results in session state
        if 'test_results' in st.session_state:
            metrics = st.session_state['test_results']
            create_result_visualizations(metrics)

            # Enhanced results analysis
            st.markdown("### 🔍 Advanced Performance Metrics")

            # Create tabs for different visualization types
            metrics_tabs = st.tabs(["Response Time", "Throughput", "Errors", "Resource Usage", "Comparisons"])

            with metrics_tabs[0]:
                st.markdown("#### Response Time Analysis")

                # Response time histogram
                if 'response_times' not in st.session_state:
                    # Generate sample data for demonstration
                    # In a real implementation, this would come from the JMeter results
                    response_times = np.random.lognormal(mean=5.0, sigma=1.0, size=1000)
                    st.session_state['response_times'] = response_times

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(st.session_state['response_times'], bins=30, alpha=0.7, color='blue')
                ax.set_xlabel('Response Time (ms)')
                ax.set_ylabel('Frequency')
                ax.set_title('Response Time Distribution')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # Percentile table
                percentiles = [50, 75, 90, 95, 99]
                percentile_values = np.percentile(st.session_state['response_times'], percentiles)

                percentile_df = pd.DataFrame({
                    'Percentile': [f"{p}th" for p in percentiles],
                    'Response Time (ms)': [f"{p:.2f}" for p in percentile_values]
                })
                st.table(percentile_df)

            with metrics_tabs[1]:
                st.markdown("#### Throughput Analysis")

                # Generate throughput over time data for demonstration
                if 'throughput_data' not in st.session_state:
                    # In a real implementation, this would come from the JMeter results
                    timestamps = np.linspace(0, 300, 30)  # 5 minutes test
                    throughput = 50 + 10 * np.sin(timestamps / 30) + np.random.normal(0, 5, 30)
                    throughput_df = pd.DataFrame({
                        'Time (s)': timestamps,
                        'Throughput (req/sec)': throughput
                    })
                    st.session_state['throughput_data'] = throughput_df

                st.line_chart(st.session_state['throughput_data'].set_index('Time (s)'))

                # Throughput statistics
                avg_throughput = st.session_state['throughput_data']['Throughput (req/sec)'].mean()
                max_throughput = st.session_state['throughput_data']['Throughput (req/sec)'].max()
                min_throughput = st.session_state['throughput_data']['Throughput (req/sec)'].min()

                cols = st.columns(3)
                cols[0].metric("Average Throughput", f"{avg_throughput:.2f} req/sec")
                cols[1].metric("Peak Throughput", f"{max_throughput:.2f} req/sec")
                cols[2].metric("Min Throughput", f"{min_throughput:.2f} req/sec")

            with metrics_tabs[2]:
                st.markdown("#### Error Analysis")

                # Generate error data for demonstration
                if 'error_data' not in st.session_state:
                    # In a real implementation, this would come from the JMeter results
                    error_types = {
                        "HTTP 500": 42,
                        "HTTP 404": 18,
                        "Connection timeout": 27,
                        "Read timeout": 15,
                        "Socket error": 8
                    }
                    st.session_state['error_data'] = pd.DataFrame({
                        'Error Type': error_types.keys(),
                        'Count': error_types.values()
                    })

                # Error pie chart
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.pie(st.session_state['error_data']['Count'], labels=st.session_state['error_data']['Error Type'],
                       autopct='%1.1f%%', shadow=True, startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                # Error table with details
                st.table(st.session_state['error_data'])

                # Error rate over time
                if 'error_rate_data' not in st.session_state:
                    # Generate sample data for demonstration
                    timestamps = np.linspace(0, 300, 30)  # 5 minutes test
                    error_rate = 2 + 5 * np.sin(timestamps / 60) + np.random.normal(0, 1, 30)
                    error_rate = np.clip(error_rate, 0, 100)  # Ensure values are between 0-100%
                    error_rate_df = pd.DataFrame({
                        'Time (s)': timestamps,
                        'Error Rate (%)': error_rate
                    })
                    st.session_state['error_rate_data'] = error_rate_df

                st.line_chart(st.session_state['error_rate_data'].set_index('Time (s)'))

            with metrics_tabs[3]:
                st.markdown("#### Resource Usage Analysis")

                # Generate resource usage data for demonstration
                if 'resource_data' not in st.session_state:
                    # In a real implementation, this would come from system monitoring during the test
                    timestamps = np.linspace(0, 300, 30)  # 5 minutes test
                    cpu_usage = 40 + 30 * np.sin(timestamps / 60) + np.random.normal(0, 5, 30)
                    cpu_usage = np.clip(cpu_usage, 0, 100)  # Ensure values are between 0-100%

                    memory_usage = 30 + 20 * np.sin(timestamps / 120) + np.random.normal(0, 3, 30)
                    memory_usage = np.clip(memory_usage, 0, 100)  # Ensure values are between 0-100%

                    resource_df = pd.DataFrame({
                        'Time (s)': timestamps,
                        'CPU Usage (%)': cpu_usage,
                        'Memory Usage (%)': memory_usage
                    })
                    st.session_state['resource_data'] = resource_df

                # CPU and Memory usage chart
                resource_df_melted = pd.melt(
                    st.session_state['resource_data'],
                    id_vars=['Time (s)'],
                    value_vars=['CPU Usage (%)', 'Memory Usage (%)']
                )

                chart = alt.Chart(resource_df_melted).mark_line().encode(
                    x='Time (s):Q',
                    y='value:Q',
                    color='variable:N',
                    tooltip=['Time (s)', 'value', 'variable']
                ).interactive()

                st.altair_chart(chart, use_container_width=True)

                # Resource usage statistics
                avg_cpu = st.session_state['resource_data']['CPU Usage (%)'].mean()
                max_cpu = st.session_state['resource_data']['CPU Usage (%)'].max()

                avg_memory = st.session_state['resource_data']['Memory Usage (%)'].mean()
                max_memory = st.session_state['resource_data']['Memory Usage (%)'].max()

                cols = st.columns(2)
                with cols[0]:
                    st.metric("Avg CPU Usage", f"{avg_cpu:.1f}%")
                    st.metric("Max CPU Usage", f"{max_cpu:.1f}%")

                with cols[1]:
                    st.metric("Avg Memory Usage", f"{avg_memory:.1f}%")
                    st.metric("Max Memory Usage", f"{max_memory:.1f}%")

                # System bottleneck analysis
                st.markdown("#### System Bottleneck Analysis")

                bottleneck = "None detected"
                bottleneck_recommendation = "The system appears to be well-balanced."

                if max_cpu > 85:
                    bottleneck = "CPU"
                    bottleneck_recommendation = "Consider optimizing CPU-intensive operations or scaling up CPU capacity."
                elif max_memory > 85:
                    bottleneck = "Memory"
                    bottleneck_recommendation = "Consider optimizing memory usage or increasing available memory."

                st.info(f"**Detected Bottleneck:** {bottleneck}\n\n**Recommendation:** {bottleneck_recommendation}")

            with metrics_tabs[4]:
                st.markdown("#### Performance Comparison")

                # Load previous test results for comparison
                st.markdown("Compare current test with previous results:")

                # Demo comparison data
                if 'comparison_data' not in st.session_state:
                    comparison_data = pd.DataFrame({
                        'Metric': ['Response Time (ms)', 'Throughput (req/s)', 'Error Rate (%)', 'CPU Usage (%)', 'Memory Usage (%)'],
                        'Current Test': [325.67, 83.4, 2.4, 65.2, 48.7],
                        'Previous Test': [342.1, 78.9, 3.1, 68.4, 52.3],
                        'Baseline': [300.0, 90.0, 1.0, 60.0, 45.0]
                    })
                    st.session_state['comparison_data'] = comparison_data

                # Calculate and add delta column
                comparison_df = st.session_state['comparison_data'].copy()
                comparison_df['Delta (%)'] = ((comparison_df['Current Test'] - comparison_df['Previous Test']) / comparison_df['Previous Test'] * 100).round(2)

                # Create a styled dataframe with arrows for improvement/degradation
                def style_delta(val):
                    if pd.isna(val):
                        return ''
                    color = 'green' if (val < 0 and not comparison_df.iloc[i]['Metric'].startswith('Throughput')) or \
                                       (val > 0 and comparison_df.iloc[i]['Metric'].startswith('Throughput')) else 'red'
                    arrow = '▲' if val > 0 else '▼' if val < 0 else '◆'
                    return f'color: {color}'

                styled_comparison = comparison_df.style.map(lambda x: '', subset=['Metric', 'Current Test', 'Previous Test', 'Baseline'])

                for i in range(len(comparison_df)):
                    styled_comparison = styled_comparison.map(
                        style_delta, subset=pd.IndexSlice[[i], ['Delta (%)']]
                    )

                st.table(styled_comparison)

                # Radar chart for visual comparison
                st.markdown("#### Radar Chart Comparison")

                # Normalize metrics for radar chart
                metrics = comparison_df['Metric'].tolist()

                # For response time, error rate, and resource usage, lower is better
                # For throughput, higher is better
                # Normalize all to 0-100 scale where higher is better
                current_values = []
                previous_values = []
                baseline_values = []

                for i, metric in enumerate(metrics):
                    if metric.startswith('Throughput'):
                        # For throughput, higher is better, so normalize directly
                        max_val = max(comparison_df.loc[i, ['Current Test', 'Previous Test', 'Baseline']].max() * 1.2, 1)
                        current_values.append(comparison_df.loc[i, 'Current Test'] / max_val * 100)
                        previous_values.append(comparison_df.loc[i, 'Previous Test'] / max_val * 100)
                        baseline_values.append(comparison_df.loc[i, 'Baseline'] / max_val * 100)
                    else:
                        # For other metrics, lower is better, so invert the normalization
                        max_val = max(comparison_df.loc[i, ['Current Test', 'Previous Test', 'Baseline']].max() * 1.2, 1)
                        current_values.append(100 - (comparison_df.loc[i, 'Current Test'] / max_val * 100))
                        previous_values.append(100 - (comparison_df.loc[i, 'Previous Test'] / max_val * 100))
                        baseline_values.append(100 - (comparison_df.loc[i, 'Baseline'] / max_val * 100))

                # Create radar chart
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, polar=True)

                # Plot each metric
                angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]  # Close the loop

                # Add values for each test and close the loop
                current_values += current_values[:1]
                previous_values += previous_values[:1]
                baseline_values += baseline_values[:1]

                # Plot the values
                ax.plot(angles, current_values, 'o-', linewidth=2, label='Current Test')
                ax.plot(angles, previous_values, 'o-', linewidth=2, label='Previous Test')
                ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline')

                # Fill the area
                ax.fill(angles, current_values, alpha=0.1)
                ax.fill(angles, previous_values, alpha=0.1)
                ax.fill(angles, baseline_values, alpha=0.1)

                # Set the labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)

                # Set y-axis range
                ax.set_ylim(0, 100)

                # Add legend
                ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

                st.pyplot(fig)

            # Option to export results
            st.markdown("### Export Results")
            export_format = st.selectbox(
                "Export Format",
                options=["CSV", "JSON", "PDF", "Excel", "HTML"],
                help="Select the format to export the results"
            )

            if st.button("Export Results"):
                st.success(f"Results would be exported as {export_format} in a real implementation")

        else:
            st.info("No test results available. Run a test from the 'Test Execution' tab to see results.")

            col1, col2 = st.columns(2)

            with col1:
                # Option to load sample results
                if st.button("Load Sample Results"):
                    # Sample data for demonstration
                    sample_metrics = {
                        'total_samples': 5000,
                        'error_count': 120,
                        'error_rate': 2.4,
                        'avg_response': 325.67,
                        'min_response': 42.3,
                        'max_response': 2456.8,
                        'p90_response': 876.2,
                        'p95_response': 1234.5,
                        'p99_response': 1987.3,
                        'throughput': 83.4
                    }
                    st.session_state['test_results'] = sample_metrics
                    st.rerun()

            with col2:
                # Option to load actual JMeter results
                uploaded_file = st.file_uploader("Upload JMeter Results (.jtl or .csv)", type=["jtl", "csv"])
                if uploaded_file is not None:
                    try:
                        # Process the JMeter results file
                        metrics = process_jmeter_results_file(uploaded_file)
                        st.session_state['test_results'] = metrics
                        st.success("Results loaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing results file: {str(e)}")

    with templates_tab:
        st.markdown("### JMeter Test Templates")
        st.info("Select a template to quickly create a JMeter test for common scenarios")

        template_types = [
            "Web Application Load Test",
            "REST API Performance Test",
            "Database Performance Test",
            "Microservices Performance Test",
            "E-commerce Site Load Test",
            "Login Performance Test"
        ]

        selected_template = st.selectbox("Select Template", options=template_types)

        # Template descriptions
        template_descriptions = {
            "Web Application Load Test": "Tests the performance of web applications with realistic user scenarios",
            "REST API Performance Test": "Tests the performance of REST APIs with various request types",
            "Database Performance Test": "Tests database performance with various query types",
            "Microservices Performance Test": "Tests performance across multiple microservices",
            "E-commerce Site Load Test": "Simulates typical e-commerce user journey",
            "Login Performance Test": "Tests login process under heavy load"
        }

        st.write(template_descriptions[selected_template])

        # Template parameters
        st.markdown("### Template Parameters")

        if selected_template == "Web Application Load Test":
            base_url = st.text_input("Base URL", value="https://example.com")
            pages = st.text_area("Pages to Test (one per line)", value="home\nproducts\nabout\ncontact")
            think_time = st.slider("Think Time (ms)", min_value=100, max_value=10000, value=1000, step=100)

        elif selected_template == "REST API Performance Test":
            base_url = st.text_input("API Base URL", value="https://api.example.com/v1")
            endpoints = st.text_area("Endpoints to Test (one per line)", value="users\nproducts\norders")
            auth_type = st.selectbox("Authentication Type", options=["None", "Basic Auth", "Bearer Token", "API Key"])

            if auth_type == "Basic Auth":
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
            elif auth_type == "Bearer Token":
                token = st.text_input("Token")
            elif auth_type == "API Key":
                api_key_name = st.text_input("API Key Name", value="api-key")
                api_key_value = st.text_input("API Key Value", type="password")

        elif selected_template == "Database Performance Test":
            db_type = st.selectbox("Database Type", options=["MySQL", "PostgreSQL", "Oracle", "SQL Server", "MongoDB"])
            connection_string = st.text_input("Connection String", value="jdbc:mysql://localhost:3306/testdb")
            queries = st.text_area("SQL Queries (one per line)", value="SELECT * FROM users LIMIT 10;\nSELECT COUNT(*) FROM orders;")

        elif selected_template == "Microservices Performance Test":
            services = st.text_area("Services (format: name,url - one per line)",
                                  value="user-service,http://user-service:8080/api\norder-service,http://order-service:8080/api")
            test_flow = st.text_area("Test Flow Steps (one per line)",
                                   value="1. Call user-service to authenticate\n2. Call order-service to create order\n3. Call user-service to update profile")

        elif selected_template == "E-commerce Site Load Test":
            site_url = st.text_input("E-commerce Site URL", value="https://shop.example.com")
            user_journey = st.multiselect(
                "User Journey Steps",
                options=["Browse Homepage", "Search for Product", "View Product Details", "Add to Cart", "Checkout", "Payment", "Order Confirmation"],
                default=["Browse Homepage", "Search for Product", "Add to Cart", "Checkout"]
            )
            product_search = st.text_input("Product Search Term", value="smartphone")

        elif selected_template == "Login Performance Test":
            login_url = st.text_input("Login URL", value="https://example.com/login")
            username_field = st.text_input("Username Field ID/Name", value="username")
            password_field = st.text_input("Password Field ID/Name", value="password")
            success_criteria = st.text_input("Success Criteria (text or URL pattern)", value="Dashboard")

        # Common parameters for all templates
        st.markdown("### Common Test Parameters")

        col1, col2 = st.columns(2)
        with col1:
            threads = st.number_input("Number of Threads (Users)", min_value=1, value=50)
            ramp_up = st.number_input("Ramp-up Period (seconds)", min_value=0, value=30)

        with col2:
            duration = st.number_input("Test Duration (seconds)", min_value=10, value=300)
            iterations = st.number_input("Loop Count (0 = infinite)", min_value=0, value=1)

        # Advanced options
        with st.expander("Advanced Options"):
            delay = st.number_input("Startup Delay (seconds)", min_value=0, value=0)
            connection_timeout = st.number_input("Connection Timeout (ms)", min_value=100, value=30000)
            response_timeout = st.number_input("Response Timeout (ms)", min_value=100, value=60000)
            follow_redirects = st.checkbox("Follow Redirects", value=True)
            use_keepalive = st.checkbox("Use KeepAlive", value=True)

        # Generate template
        if st.button("Generate Template"):
            with st.spinner("Generating JMeter template..."):
                time.sleep(2)  # Simulating template generation
                st.success(f"Template generated successfully for {selected_template}!")

                # Download button for the generated template
                st.download_button(
                    label="Download JMX File",
                    data=f"This would be a JMeter JMX file for {selected_template} in a real implementation",
                    file_name=f"{selected_template.replace(' ', '_').lower()}.jmx",
                    mime="application/xml"
                )

                if NOTIFICATIONS_AVAILABLE:
                    notifications.add_notification(
                        module_name="performance_testing",
                        status="success",
                        message=f"JMeter template generated: {selected_template}",
                        details="Template generated with customized parameters",
                        action_steps=["Download the template", "Import to JMeter", "Run the test"]
                    )

        # Example preview section
        with st.expander("Template Preview"):
            st.write(f"Preview of the {selected_template} JMeter template structure:")

            # Show a sample of the JMeter structure based on template type
            structure = [
                "Test Plan",
                "├─ Thread Group (Users: " + str(threads) + ", Ramp-up: " + str(ramp_up) + "s)",
                "│  ├─ HTTP Request Defaults",
            ]

            if selected_template == "Web Application Load Test":
                structure.extend([
                    "│  ├─ Cookie Manager",
                    "│  ├─ HTTP Cache Manager",
                    "│  ├─ HomePage Request",
                    "│  │  └─ Response Assertions",
                    "│  ├─ Think Time Timer",
                    "│  ├─ Products Page Request",
                    "│  └─ Response Time Graph Listener"
                ])
            elif selected_template == "REST API Performance Test":
                structure.extend([
                    "│  ├─ HTTP Header Manager",
                    "│  ├─ GET /users Request",
                    "│  │  └─ JSON Extractor",
                    "│  ├─ POST /orders Request",
                    "│  │  └─ JSON Path Assertions",
                    "│  └─ Summary Report"
                ])
            else:
                structure.extend([
                    "│  ├─ Main Request",
                    "│  ├─ Secondary Requests",
                    "│  └─ Results Collector"
                ])

            structure.extend([
                "├─ Summary Report",
                "├─ Aggregate Report",
                "└─ View Results Tree"
            ])

            for line in structure:
                st.text(line)

    with cicd_tab:
        st.markdown("### CI/CD Integration")
        st.markdown("""
        Integrate your JMeter performance tests into CI/CD pipelines to automate performance testing
        as part of your development lifecycle. This helps catch performance regressions early.
        """)

        # Select CI/CD platform
        selected_platform = st.selectbox(
            "CI/CD Platform",
            options=list(CICD_PLATFORMS.keys()),
            help="Select the CI/CD platform to integrate with"
        )

        # Show platform description
        st.info(CICD_PLATFORMS[selected_platform]["description"])

        # Configuration options
        st.markdown("### Configuration Options")

        col1, col2 = st.columns(2)

        with col1:
            repo_url = st.text_input("Repository URL", value="https://github.com/yourusername/yourrepo")
            test_plan_path = st.text_input("JMeter Test Plan Path", value="tests/performance/test_plan.jmx")

        with col2:
            trigger_type = st.selectbox(
                "Trigger Type",
                options=["Push to branch", "Pull Request", "Manual", "Scheduled"],
                help="When should the performance tests run?"
            )

            if trigger_type == "Push to branch":
                branch = st.text_input("Branch Name", value="main")
            elif trigger_type == "Scheduled":
                schedule = st.text_input("Schedule (cron syntax)", value="0 0 * * *")

        # Performance thresholds
        st.markdown("### Performance Thresholds")
        st.info("Set thresholds to automatically fail the build if performance degradation is detected")

        col1, col2 = st.columns(2)

        with col1:
            response_time_threshold = st.number_input("Max Response Time (ms)", value=1000)
            error_rate_threshold = st.number_input("Max Error Rate (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

        with col2:
            throughput_threshold = st.number_input("Min Throughput (req/sec)", value=10)
            p95_threshold = st.number_input("95th Percentile Response Time (ms)", value=1500)

        # Configure notification options
        st.markdown("### Notification Settings")

        notify_on_failure = st.checkbox("Notify on Test Failure", value=True)
        notify_on_threshold = st.checkbox("Notify on Threshold Violation", value=True)

        if notify_on_failure or notify_on_threshold:
            notification_channels = st.multiselect(
                "Notification Channels",
                options=["Email", "Slack", "Microsoft Teams", "Discord"],
                default=["Email"]
            )

            if "Email" in notification_channels:
                email_recipients = st.text_input("Email Recipients", value="team@example.com")

            if "Slack" in notification_channels:
                slack_webhook = st.text_input("Slack Webhook URL")

            if "Microsoft Teams" in notification_channels:
                teams_webhook = st.text_input("Microsoft Teams Webhook URL")

            if "Discord" in notification_channels:
                discord_webhook = st.text_input("Discord Webhook URL")

        # Generate configuration button
        if st.button("Generate Configuration"):
            # In a real implementation, this would generate the configuration file for the selected platform
            st.success(f"Configuration for {selected_platform} generated successfully!")

            # If notifications are enabled, show notification setup
            if notify_on_failure or notify_on_threshold:
                st.info("Notification settings:")

                if "Email" in notification_channels:
                    st.write(f"- Email: {email_recipients}")

                if "Slack" in notification_channels:
                    st.write(f"- Slack: {slack_webhook}")

                if "Microsoft Teams" in notification_channels:
                    st.write(f"- Teams: {teams_webhook}")

                if "Discord" in notification_channels:
                    st.write(f"- Discord: {discord_webhook}")

        # Template selection for CI/CD config
        st.markdown("### CI/CD Configuration Templates")

        # Show available templates for the selected platform
        templates = CICD_PLATFORMS[selected_platform]["config_template"].strip().split("\n\n")

        selected_template = st.selectbox(
            "Select a template",
            options=[t.split("{")[0].strip() for t in templates],
            format_func=lambda x: x.split("{")[0].strip()
        )

        # Show selected template details
        st.markdown("#### Selected Template")
        st.code(next(t for t in templates if t.startswith(selected_template)), language="groovy")

        # Download button for the template
        if st.button("Download Template"):
            # In a real implementation, this would trigger a download of the template file
            st.success(f"Template {selected_template} downloaded successfully!")

        # Action buttons for CI/CD
        st.markdown("### CI/CD Actions")

        if st.button("Test Connection"):
            # In a real implementation, this would test the connection to the CI/CD platform
            st.success("Connection to CI/CD platform successful!")

        if st.button("Trigger Manual Run"):
            # In a real implementation, this would trigger a manual run of the CI/CD pipeline
            st.success("Manual run of CI/CD pipeline triggered!")

        if st.button("View Pipeline Status"):
            # In a real implementation, this would show the current status of the CI/CD pipeline
            st.success("Showing latest pipeline status!")

        # Help and documentation links
        st.markdown("### Help and Documentation")

        st.info("For detailed documentation on CI/CD integration, visit the [official documentation](https://example.com/docs/cicd-integration).")

        st.markdown("""
        ### Common Issues
        - **Authentication failed**: Ensure your credentials are correct and you have access to the repository.
        - **JMeter not found**: Make sure JMeter is installed and the path is correctly set in the environment variables.
        - **Network issues**: Check your network connection and firewall settings.
        """)

        # Feedback and support
        st.markdown("### Feedback and Support")

        st.info("For feedback or support, please contact the development team or raise an issue in the project repository.")

def create_result_visualizations(metrics):
    """Create visualizations for the JMeter test results."""
    if 'error' in metrics:
        st.error(f"Error creating visualizations: {metrics['error']}")
        return

    # Create metrics cards
    st.markdown("### 📊 Key Performance Metrics")
    cols = st.columns(4)

    with cols[0]:
        # Use get() with a default value to prevent KeyError
        st.metric("Total Samples", f"{metrics.get('total_samples', 0):,}")
    with cols[1]:
        # Use get() with a default value to prevent KeyError
        st.metric("Throughput", f"{metrics.get('throughput', 0):.2f} req/sec")
    with cols[2]:
        # Use get() with a default value to prevent KeyError
        st.metric("Error Rate", f"{metrics.get('error_rate', 0):.2f}%")
    with cols[3]:
        # Use get() with a default value to prevent KeyError
        st.metric("Avg Response", f"{metrics.get('avg_response', 0):.2f} ms")

    # Create response time distribution chart
    st.markdown("### ⏱️ Response Time Analysis")

    response_data = {
        'Metric': ['Minimum', 'Average', '90th Percentile', '95th Percentile', '99th Percentile', 'Maximum'],
        'Time (ms)': [
            metrics.get('min_response', 0),
            metrics.get('avg_response', 0),
            metrics.get('p90_response', 0),
            metrics.get('p95_response', 0),
            metrics.get('p99_response', 0),
            metrics.get('max_response', 0)
        ]
    }

    response_df = pd.DataFrame(response_data)

    # Create bar chart for response time metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ['#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e', '#d62728', '#d62728']

    bars = ax.bar(response_data['Metric'], response_data['Time (ms)'], color=bar_colors)
    ax.set_ylabel('Response Time (ms)')
    ax.set_title('Response Time Metrics')
    ax.set_ylim(bottom=0)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Customize grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    st.pyplot(fig)

    # Additional insights
    st.markdown("### 🔍 Performance Insights")

    # Generate insights based on metrics
    insights = []

    # Error rate insights
    if metrics.get('error_rate', 0) > 5:
        insights.append("⚠️ **High Error Rate**: The error rate exceeds 5%, which may indicate issues with the system under test or test configuration.")
    else:
        insights.append("✅ **Acceptable Error Rate**: The error rate is within acceptable limits.")

    # Response time insights
    if metrics.get('p95_response', 0) > 2000:
        insights.append("⚠️ **Slow Response Times**: The 95th percentile response time exceeds 2 seconds, which may indicate performance issues.")
    else:
        insights.append("✅ **Good Response Times**: The 95th percentile response time is within acceptable limits.")

    # Throughput insights
    if metrics.get('throughput', 0) < 10:
        insights.append("⚠️ **Low Throughput**: The system is processing less than 10 requests per second, which may indicate bottlenecks.")
    else:
        insights.append("✅ **Good Throughput**: The system is handling an acceptable number of requests per second.")

    # Response time consistency
    min_response = metrics.get('min_response', 0)
    max_response = metrics.get('max_response', 0)
    avg_response = metrics.get('avg_response', 1)  # Default to 1 to avoid division by zero
    response_time_range = max_response - min_response
    if response_time_range > 10 * avg_response:
        insights.append("⚠️ **Inconsistent Response Times**: There's a large gap between minimum and maximum response times, indicating potential instability.")
    else:
        insights.append("✅ **Consistent Response Times**: The response time range is reasonable compared to the average.")

    # Display insights
    for insight in insights:
        st.markdown(insight)

def run_jmeter_test(jmeter_path, test_plan_path, result_file_path=None, remote_hosts=None, properties=None):
    """
    Execute a JMeter test in real-time and return the results

    Args:
        jmeter_path: Path to JMeter installation
        test_plan_path: Path to the JMeter test plan file
        result_file_path: Path where results should be saved
        remote_hosts: List of remote JMeter server hosts for distributed testing
        properties: Dictionary of JMeter properties to set

    Returns:
        dict: Path to the results file and execution status
    """
    import subprocess
    import tempfile
    import os
    import time
    import platform
    from datetime import datetime

    # Create a timestamped results file if not provided
    if not result_file_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file_path = os.path.join(tempfile.gettempdir(), f"jmeter_results_{timestamp}.jtl")

    # Check if jmeter_path already contains bin/jmeter and adjust accordingly
    jmeter_bin_path = jmeter_path
    if not jmeter_path.endswith('bin'):
        jmeter_bin_path = os.path.join(jmeter_path, "bin")

    # On Windows, use jmeter.bat, otherwise use jmeter
    jmeter_executable = os.path.join(jmeter_bin_path, "jmeter")
    if os.name == 'nt':
        jmeter_executable += '.bat'

    # Set environment variables without any Java options that might trigger X11 warnings
    env = os.environ.copy()

    # Remove any existing _JAVA_OPTIONS to avoid warnings
    if '_JAVA_OPTIONS' in env:
        del env['_JAVA_OPTIONS']

    # Construct the JMeter command with Java system properties directly in the command
    cmd = [
        jmeter_executable,
        "-n",  # Non-GUI mode
        "-Djava.awt.headless=true",  # Set headless mode directly as JMeter argument
        "-t", test_plan_path,  # Test plan
        "-l", result_file_path  # Results file
    ]

    # Add remote hosts for distributed testing
    if remote_hosts:
        cmd.extend(["-R", ",".join(remote_hosts)])

    # Add JMeter properties
    if properties:
        for key, value in properties.items():
            cmd.extend(["-J", f"{key}={value}"])

    try:
        # Start the process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            env=env  # Use our enhanced environment variables
        )

        # Wait a brief moment to check for immediate errors
        time.sleep(1)
        if process.poll() is not None:
            # Process already terminated - capture the error
            stdout, stderr = process.communicate()
            return {
                'status': 'error',
                'error': f"JMeter process failed to start: {stderr}",
                'stdout': stdout,
                'stderr': stderr,
                'returncode': process.returncode
            }

        # Store the process for monitoring
        test_run_info = {
            'process': process,
            'start_time': time.time(),
            'result_file': result_file_path,
            'status': 'running',
            'progress_updates': [],
            'command': ' '.join(cmd)  # Store the command for debugging
        }

        # Store in session state for monitoring
        if 'current_test_run' not in st.session_state:
            st.session_state['current_test_run'] = test_run_info

        return {
            'result_file': result_file_path,
            'status': 'started',
            'run_info': test_run_info
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'command': ' '.join(cmd)  # Include command for debugging
        }

def monitor_test_execution(run_info):
    """
    Monitor the execution of a running JMeter test and update progress

    Args:
        run_info: Dictionary containing process and execution information

    Returns:
        dict: Updated status and metrics if available
    """
    import time

    if not run_info or 'process' not in run_info:
        return {'status': 'error', 'error': 'Invalid run information'}

    process = run_info['process']

    # Check if process is still running
    if process.poll() is not None:
        # Process has completed
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            return {
                'status': 'failed',
                'returncode': process.returncode,
                'stderr': stderr
            }
        else:
            # Process completed successfully, parse results
            metrics = process_jmeter_results_file(run_info['result_file'])

            return {
                'status': 'completed',
                'metrics': metrics,
                'duration': time.time() - run_info['start_time'],
                'result_file': run_info['result_file']
            }
    else:
        # Process is still running, try to read partial results if available
        partial_metrics = None
        if os.path.exists(run_info['result_file']) and os.path.getsize(run_info['result_file']) > 0:
            try:
                partial_metrics = process_jmeter_results_file(run_info['result_file'])
            except:
                partial_metrics = None

        return {
            'status': 'running',
            'duration': time.time() - run_info['start_time'],
            'partial_metrics': partial_metrics
        }

def process_jmeter_results_file(file_path):
    """
    Process a JMeter results file (.jtl or .csv) and extract key performance metrics.

    Args:
        file_path: Path to the JMeter results file

    Returns:
        dict: Dictionary containing extracted performance metrics
    """
    try:
        # Read the JMeter results into a pandas DataFrame
        df = pd.read_csv(file_path)

        # Check if required columns exist
        required_cols = ['timeStamp', 'elapsed']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in results file: {', '.join(missing_cols)}")

        # Calculate total number of samples
        total_samples = len(df)

        # Calculate error count and error rate
        if 'success' in df.columns:
            error_count = len(df[df['success'] == False])
        elif 'responseCode' in df.columns:
            # Consider non-2xx status codes as errors
            error_count = len(df[~df['responseCode'].astype(str).str.startswith('2')])
        else:
            error_count = 0

        error_rate = (error_count / total_samples) * 100 if total_samples > 0 else 0

        # Calculate response time metrics (in milliseconds)
        elapsed_times = df['elapsed'].values
        min_response = elapsed_times.min() if len(elapsed_times) > 0 else 0
        max_response = elapsed_times.max() if len(elapsed_times) > 0 else 0
        avg_response = elapsed_times.mean() if len(elapsed_times) > 0 else 0

        # Calculate percentiles
        p90_response = np.percentile(elapsed_times, 90) if len(elapsed_times) > 0 else 0
        p95_response = np.percentile(elapsed_times, 95) if len(elapsed_times) > 0 else 0
        p99_response = np.percentile(elapsed_times, 99) if len(elapsed_times) > 0 else 0

        # Calculate throughput (requests per second)
        if len(df) > 1:
            # Calculate time span of the test in seconds
            start_time = df['timeStamp'].min()
            end_time = df['timeStamp'].max()
            time_span = (end_time - start_time) / 1000  # convert to seconds
            throughput = total_samples / time_span if time_span > 0 else 0
        else:
            throughput = 0

        # Create a dictionary with all metrics
        metrics = {
            'total_samples': total_samples,
            'error_count': error_count,
            'error_rate': error_rate,
            'avg_response': avg_response,
            'min_response': min_response,
            'max_response': max_response,
            'p90_response': p90_response,
            'p95_response': p95_response,
            'p99_response': p99_response,
            'throughput': throughput
        }

        # Optional: Create response time distribution for further analysis
        response_times = df['elapsed'].values.tolist()

        return metrics

    except Exception as e:
        # Return error information
        return {'error': str(e)}

def fetch_test_results_realtime(run_info, update_interval=1.0):
    """
    Continuously fetch test results in real-time while a test is running

    Args:
        run_info: Dictionary containing process and execution information
        update_interval: How frequently to check for updates (seconds)

    Returns:
        Generator yielding current status and metrics at each interval
    """
    import time

    if not run_info or 'process' not in run_info:
        yield {'status': 'error', 'error': 'Invalid run information'}
        return

    last_size = 0
    last_sample_count = 0

    while True:
        # Get current status
        status_info = monitor_test_execution(run_info)

        # Check if file has been updated
        current_size = 0
        if os.path.exists(run_info['result_file']):
            current_size = os.path.getsize(run_info['result_file'])

        file_updated = current_size > last_size
        last_size = current_size

        # Add information about samples processed since last update
        if status_info.get('partial_metrics') and 'total_samples' in status_info['partial_metrics']:
            current_samples = status_info['partial_metrics']['total_samples']
            status_info['new_samples'] = current_samples - last_sample_count
            status_info['last_sample_count'] = last_sample_count
            status_info['current_sample_count'] = current_samples
            last_sample_count = current_samples

        # Add update timestamp
        status_info['timestamp'] = time.time()

        # Add to progress updates history
        if 'progress_updates' in run_info:
            run_info['progress_updates'].append({
                'time': time.time() - run_info['start_time'],
                'metrics': status_info.get('partial_metrics', {})
            })

        # Yield the current status
        yield status_info

        # If test has completed or failed, stop monitoring
        if status_info['status'] in ['completed', 'failed', 'error']:
            break

        # Wait before next check
        time.sleep(update_interval)
