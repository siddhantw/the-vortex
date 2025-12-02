"""
RoboCop Lint Checker Module

This module provides an interface for analyzing Robot Framework code in the repository
using the RoboCop static analysis tool. It highlights issues with actionable and insightful reports.
"""

import streamlit as st
import os
import subprocess
import tempfile
import json
import pandas as pd
from datetime import datetime
import sys
import re

# Import notifications module (created in main_ui.py)
try:
    import notifications
except ImportError:
    # Create a dummy notifications module if the real one isn't available
    class DummyNotifications:
        @staticmethod
        def add_notification(*args, **kwargs):
            pass

        @staticmethod
        def handle_execution_result(*args, **kwargs):
            pass

    notifications = DummyNotifications()

# Set module name for notifications and tracking
MODULE_NAME = "robocop_lint_checker"

def install_robocop():
    """
    Install the RoboCop package if not already installed.
    Returns True if installation is successful or package exists.
    """
    try:
        # Check if robocop is already installed
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "robotframework-robocop"],
            capture_output=True,
            text=True
        )

        if "Name: robotframework-robocop" in result.stdout:
            return True

        # Install robocop
        st.info("Installing RoboCop linter... This may take a moment.")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "robotframework-robocop"],
            capture_output=True,
            text=True,
            check=True
        )
        st.success("RoboCop linter installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to install RoboCop: {e}")
        return False


def run_robocop_analysis(target_path, config=None, ignore=None, include=None):
    """
    Run RoboCop analysis on Robot Framework files.

    Args:
        target_path: Path to analyze (can be a file or directory)
        config: Optional path to configuration file
        ignore: List of rules to ignore
        include: List of rules to include exclusively

    Returns:
        Tuple of (success, results, raw_output)
    """
    if not install_robocop():
        return False, [], "Failed to install RoboCop"

    # Update command structure to use 'check' command for this version of RoboCop
    cmd = [sys.executable, "-m", "robocop", "check"]

    # Add configuration file if provided
    if config:
        cmd.extend(["--configure", config])

    # Add rules to ignore
    if ignore:
        ignore_rules = ",".join(ignore)
        cmd.extend(["--exclude", ignore_rules])

    # Add rules to include
    if include:
        include_rules = ",".join(include)
        cmd.extend(["--include", include_rules])

    # Add target path
    cmd.append(target_path)

    # Debug message
    st.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse regular text output
        if result.stdout:
            findings = parse_robocop_output(result.stdout)
            return True, findings, result.stdout
        else:
            # No issues found
            return True, [], result.stderr

    except subprocess.CalledProcessError as e:
        if e.returncode == 1 and e.stdout:
            # RoboCop returns 1 when it finds issues, but that's not an error for us
            findings = parse_robocop_output(e.stdout)
            return True, findings, e.stdout
        else:
            # Real error occurred - provide detailed error information
            st.warning("RoboCop command failed. See error details below.")
            return False, [], f"Error running RoboCop: {e.stderr}"


def parse_robocop_output(output_text):
    """
    Parse RoboCop's standard text output format into structured data.

    Example line:
    /path/to/file.robot:10:5 [W] 0502 Too few steps in keyword (1/2)

    Returns:
        List of dictionaries with structured findings
    """
    results = []
    for line in output_text.splitlines():
        # Skip empty lines
        if not line.strip():
            continue

        # Try to match the standard robocop output format
        match = re.match(r'(.*?):(\d+):(\d+) \[([EWI])\] (\d+) (.*)', line)
        if match:
            source, line_num, col, severity_code, rule_id, message = match.groups()

            # Map severity code to full severity name
            severity_map = {
                'E': 'ERROR',
                'W': 'WARNING',
                'I': 'INFO'
            }
            severity = severity_map.get(severity_code, 'UNKNOWN')

            # Extract rule name - the part after the rule ID
            rule_id_full = rule_id.strip()
            rule_name = message

            results.append({
                'source': source,
                'line': int(line_num),
                'col': int(col),
                'severity': severity,
                'rule_id': rule_id_full,
                'rule_name': rule_name,
                'message': message
            })
        else:
            # If line doesn't match expected format, just skip it
            continue

    return results


def generate_summary(results):
    """Generate a summary of RoboCop findings."""
    if not results:
        return "No issues found. Your code looks good! 🎉"

    # Count issues by severity
    severity_counts = {"ERROR": 0, "WARNING": 0, "INFO": 0}

    for result in results:
        severity = result.get("severity", "").upper()
        if severity in severity_counts:
            severity_counts[severity] += 1

    total_issues = sum(severity_counts.values())

    summary = f"Found {total_issues} issues: "
    details = []

    for severity, count in severity_counts.items():
        if count > 0:
            details.append(f"{count} {severity.lower()}")

    summary += ", ".join(details)

    # Group issues by rule ID to identify common problems
    rule_counts = {}
    for result in results:
        rule_id = result.get("rule_id", "unknown")
        if rule_id not in rule_counts:
            rule_counts[rule_id] = {"count": 0, "severity": result.get("severity", ""), "description": result.get("rule_name", "")}
        rule_counts[rule_id]["count"] += 1

    # Find top 3 most common issues
    top_rules = sorted(rule_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:3]

    if top_rules:
        summary += "\n\nMost common issues:"
        for rule_id, data in top_rules:
            summary += f"\n- {rule_id}: {data['description']} ({data['count']} occurrences)"

    return summary


def get_severity_icon(severity):
    """Return an appropriate icon for the severity level."""
    severity = severity.upper()
    if severity == "ERROR":
        return "❌"
    elif severity == "WARNING":
        return "⚠️"
    else:  # INFO
        return "ℹ️"


def create_action_steps(results):
    """Create actionable steps based on the findings."""
    if not results:
        return ["No action needed - code looks good!"]

    # Group issues by rule ID
    rule_groups = {}
    for result in results:
        rule_id = result.get("rule_id", "unknown")
        if rule_id not in rule_groups:
            rule_groups[rule_id] = {
                "count": 0,
                "severity": result.get("severity", ""),
                "description": result.get("rule_name", ""),
                "examples": []
            }

        rule_groups[rule_id]["count"] += 1

        # Keep track of a few examples for each rule (limit to 3)
        if len(rule_groups[rule_id]["examples"]) < 3:
            rule_groups[rule_id]["examples"].append({
                "message": result.get("message", ""),
                "line": result.get("line", 0),
                "file_path": result.get("source", "")
            })

    # Sort rules by severity and count
    severity_rank = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    sorted_rules = sorted(
        rule_groups.items(),
        key=lambda x: (severity_rank.get(x[1]["severity"].upper(), 999), -x[1]["count"])
    )

    # Create action steps for the top issues (limit to 5 to avoid overwhelming)
    action_steps = []

    for rule_id, data in sorted_rules[:5]:
        severity = data["severity"].upper()
        count = data["count"]
        description = data["description"]

        # Create a general action step for this rule
        action = f"Fix {count} {severity.lower()}(s) related to {rule_id}: {description}"
        action_steps.append(action)

        # Add an example for context if available
        if data["examples"]:
            example = data["examples"][0]
            file_name = os.path.basename(example["file_path"])
            action_steps.append(
                f"Example: {example['message']} in {file_name} line {example['line']}"
            )

    # Add a general recommendation
    action_steps.append("Review the full report and prioritize fixing ERROR level issues first")

    return action_steps


def export_results(results, format_type="csv"):
    """
    Export the RoboCop results to the specified format.

    Args:
        results: The RoboCop results to export
        format_type: The format to export to ("csv", "json", "html")

    Returns:
        The path to the exported file
    """
    if not results:
        return None

    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert results to DataFrame for easier exporting
    df = pd.DataFrame(results)

    # Create exports directory if it doesn't exist
    export_dir = os.path.join(os.getcwd(), "robocop_results")
    os.makedirs(export_dir, exist_ok=True)

    # Export based on format type
    if format_type == "csv":
        export_path = os.path.join(export_dir, f"robocop_report_{timestamp}.csv")
        df.to_csv(export_path, index=False)
    elif format_type == "json":
        export_path = os.path.join(export_dir, f"robocop_report_{timestamp}.json")
        with open(export_path, "w") as f:
            json.dump(results, f, indent=2)
    elif format_type == "html":
        export_path = os.path.join(export_dir, f"robocop_report_{timestamp}.html")
        df.to_html(export_path, index=False)
    else:
        return None

    return export_path


def show_ui():
    """Show the UI for the RoboCop Lint Checker module."""
    st.title("RoboCop Lint Checker")

    st.write("""
    This tool analyzes Robot Framework test files for code quality issues using the RoboCop static analysis tool.
    It will identify issues and provide actionable insights to improve your test code.
    """)

    # Get the workspace root directory
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    # Set up the sidebar with options
    with st.sidebar:
        st.header("Analysis Options")

        # Select directory to analyze
        analysis_options = [
            "Entire workspace",
            "tests directory",
            "generated_tests directory",
            "Custom path"
        ]
        analysis_target = st.selectbox(
            "What would you like to analyze?",
            options=analysis_options
        )

        # Handle custom path option
        if analysis_target == "Custom path":
            custom_path = st.text_input(
                "Enter the path to analyze",
                placeholder="e.g., tests/api_tests"
            )
            if custom_path:
                if not custom_path.startswith("/"):
                    # Relative path - make it absolute
                    target_path = os.path.join(workspace_root, custom_path)
                else:
                    # Already an absolute path
                    target_path = custom_path
            else:
                target_path = workspace_root
        else:
            # Map the selection to actual paths
            if analysis_target == "Entire workspace":
                target_path = workspace_root
            elif analysis_target == "tests directory":
                target_path = os.path.join(workspace_root, "tests")
            else:  # generated_tests directory
                target_path = os.path.join(workspace_root, "generated_tests")

        # Advanced options
        st.subheader("Advanced Options")
        show_advanced = st.checkbox("Show advanced options")

        if show_advanced:
            # Rule configuration options
            exclude_rules = st.text_input(
                "Rules to exclude (comma-separated)",
                placeholder="e.g., 0508,0201"
            )
            include_rules = st.text_input(
                "Rules to include (comma-separated)",
                placeholder="e.g., 0901,0902"
            )

            severity_options = ["All", "Error only", "Error and warning"]
            severity_filter = st.radio("Severity filter", severity_options)

            # Convert to actual filter lists
            ignore_list = exclude_rules.split(",") if exclude_rules else None
            include_list = include_rules.split(",") if include_rules else None

            # Apply severity filter
            if severity_filter == "Error only":
                if not include_list:
                    include_list = ["E"]  # Only error rules
            elif severity_filter == "Error and warning":
                if not include_list:
                    include_list = ["E", "W"]  # Error and warning rules
        else:
            ignore_list = None
            include_list = None

        # Export options
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Export format",
            options=["CSV", "JSON", "HTML", "None"],
            index=3  # Default to None
        )

    # Main area - Run analysis button
    col1, col2 = st.columns([2, 1])

    with col1:
        run_button = st.button("🔍 Run Analysis", use_container_width=True)

    with col2:
        if st.button("📝 View Robot Documentation", use_container_width=True):
            st.markdown("[Robot Framework User Guide](https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html)")
            st.markdown("[RoboCop Documentation](https://robocop.readthedocs.io/)")

    # Run the analysis when the button is clicked
    if run_button:
        with st.spinner("Running RoboCop analysis..."):
            # Make sure the target path exists
            if not os.path.exists(target_path):
                st.error(f"Path does not exist: {target_path}")
            else:
                # Run the analysis
                success, results, raw_output = run_robocop_analysis(
                    target_path=target_path,
                    ignore=ignore_list,
                    include=include_list
                )

                if success:
                    # Generate summary and display it
                    summary = generate_summary(results)

                    # Display results as tabs
                    if results:
                        # Create a summary container with appropriate styling
                        summary_severity = "error" if any(r.get("severity", "").upper() == "ERROR" for r in results) else "warning"

                        if summary_severity == "error":
                            st.error(f"Analysis Summary: {summary}")
                        else:
                            st.warning(f"Analysis Summary: {summary}")

                        # Show results in a tabular format
                        df = pd.DataFrame(results)

                        # Add an icon column for better visualization
                        if not df.empty and "severity" in df.columns:
                            df["icon"] = df["severity"].apply(get_severity_icon)
                            # Reorder columns to put icon first
                            cols = df.columns.tolist()
                            cols.insert(0, cols.pop(cols.index("icon")))
                            df = df[cols]

                        # Display the results
                        st.dataframe(
                            df,
                            use_container_width=True,
                            column_config={
                                "icon": "Status",
                                "rule_id": "Rule ID",
                                "line": "Line",
                                "col": "Column",
                                "severity": "Severity",
                                "message": "Message",
                                "rule_name": "Rule Name",
                                "source": "File"
                            }
                        )

                        # Create tabs for different views of the data
                        tab1, tab2 = st.tabs(["Issues by Severity", "Issues by File"])

                        with tab1:
                            # Group by severity
                            severity_counts = df["severity"].value_counts()
                            st.bar_chart(severity_counts)

                            # Show breakdown by severity
                            for severity in ["ERROR", "WARNING", "INFO"]:
                                if severity.lower() in df["severity"].str.lower().values:
                                    with st.expander(f"{get_severity_icon(severity)} {severity.title()} Issues"):
                                        severity_df = df[df["severity"].str.lower() == severity.lower()]
                                        st.dataframe(
                                            severity_df,
                                            use_container_width=True,
                                            hide_index=True
                                        )

                        with tab2:
                            # Group by file
                            file_counts = df["source"].value_counts()

                            # Show a chart of issues per file
                            st.bar_chart(file_counts)

                            # Show issues grouped by file
                            for file_path in file_counts.index:
                                file_name = os.path.basename(file_path)
                                with st.expander(f"{file_name} ({file_counts[file_path]} issues)"):
                                    file_df = df[df["source"] == file_path]
                                    st.dataframe(
                                        file_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )

                        # Export the results if requested
                        if export_format != "None":
                            export_path = export_results(results, export_format.lower())
                            if export_path:
                                st.success(f"Results exported to: {export_path}")

                                # Create download button
                                with open(export_path, "rb") as f:
                                    st.download_button(
                                        label=f"Download {export_format} Report",
                                        data=f,
                                        file_name=os.path.basename(export_path),
                                        mime="text/csv" if export_format == "CSV" else "application/json" if export_format == "JSON" else "text/html"
                                    )

                        # Add advice section
                        st.subheader("📋 Recommendations")

                        # Generate action steps
                        action_steps = create_action_steps(results)

                        for step in action_steps:
                            st.markdown(f"- {step}")

                        # Add notification
                        try:
                            # Generate a notification
                            notifications.add_notification(
                                module_name=MODULE_NAME,
                                status="warning" if results else "success",
                                message=f"RoboCop analysis found {len(results)} issues in {os.path.basename(target_path)}",
                                details=summary,
                                action_steps=action_steps
                            )
                        except Exception as e:
                            st.error(f"Failed to create notification: {e}")
                    else:
                        # No issues found
                        st.success("No issues found! Your Robot Framework code looks good. 🎉")

                        # Update notification system
                        try:
                            notifications.add_notification(
                                module_name=MODULE_NAME,
                                status="success",
                                message=f"RoboCop analysis found no issues in {os.path.basename(target_path)}",
                                details="Your Robot Framework code follows best practices.",
                                action_steps=["Continue maintaining clean code standards"]
                            )
                        except Exception as e:
                            st.error(f"Failed to create notification: {e}")
                else:
                    st.error(f"Analysis failed: {raw_output}")

    # Help and documentation section
    with st.expander("ℹ️ About RoboCop", expanded=True):
        st.markdown("""
        ### What is RoboCop?
        
        RoboCop is a static code analysis tool for Robot Framework that helps identify code smells, 
        anti-patterns, and errors in your test files. It performs checks based on rules organized into 
        several categories.
        
        ### Common Rules
        
        - **0101**: Line is too long (exceeds character limit)
        - **0201**: Missing documentation for keyword
        - **0301**: Trailing whitespace detected
        - **0501**: Too few steps in keyword (at least 2 steps required)
        - **0502**: Too many steps in keyword (too complex)
        - **0701**: Invalid section name
        - **0802**: Test case does not have any keywords
        - **0901**: Keyword name does not follow case convention
        
        ### Severity Levels
        
        - **ERROR**: Indicates a serious problem that should be fixed immediately
        - **WARNING**: Indicates a potential problem that should be investigated
        - **INFO**: General information about code style or minor improvements
        
        [Full RoboCop documentation](https://robocop.readthedocs.io/)
        """)

    # Usage tips section
    with st.expander("💡 Usage Tips", expanded=True):
        st.markdown("""
        ### Effective Usage Tips
        
        1. **Run regularly**: Integrate linting into your development workflow to catch issues early
        2. **Focus on errors first**: Prioritize fixing ERROR level issues before moving to warnings
        3. **Use filtering**: For large codebases, focus on specific rule categories or directories
        4. **Export results**: Share results with your team by exporting to CSV or HTML
        5. **Customize rules**: Use the advanced options to ignore rules that don't apply to your project
        
        ### Common Fixes
        
        - **Line too long**: Split long lines into multiple lines
        - **Missing documentation**: Add descriptive docstrings to keywords and test cases
        - **Naming conventions**: Follow consistent naming patterns (snake_case or CamelCase)
        - **Trailing whitespace**: Remove extra spaces at the end of lines
        - **Empty sections**: Remove empty sections or add content
        """)


if __name__ == "__main__":
    show_ui()
