import streamlit as st
import os
import pandas as pd
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import sys
import time
import random
import bs4
from bs4 import BeautifulSoup

# Add the project root to path for imports
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import notifications module for action feedback
try:
    import notifications
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
    print("Notifications module not available. Notification features will be disabled.")

# Constants for healing strategies
FUZZY_MATCHING = "Fuzzy Matching"
COMPUTER_VISION = "Computer Vision"
DOM_ANALYSIS = "DOM Analysis"
HYBRID = "Hybrid (Multiple Techniques)"

# Mapping of healing strategies to descriptions
HEALING_STRATEGIES = {
    FUZZY_MATCHING: "Uses string similarity algorithms to find elements with similar attributes.",
    COMPUTER_VISION: "Uses image recognition to locate UI elements visually.",
    DOM_ANALYSIS: "Analyzes DOM structure to find elements with similar positions or hierarchies.",
    HYBRID: "Combines multiple strategies for better accuracy."
}

# Default configurations
DEFAULT_CONFIGS = {
    "similarity_threshold": 0.7,
    "max_healing_suggestions": 3,
    "include_visual_analysis": True,
    "save_healing_history": True
}

# Initialize session state variables
def init_session_state():
    if 'healing_history' not in st.session_state:
        st.session_state.healing_history = []
    if 'failed_tests' not in st.session_state:
        st.session_state.failed_tests = []
    if 'healing_configs' not in st.session_state:
        st.session_state.healing_configs = DEFAULT_CONFIGS.copy()
    if 'healing_progress' not in st.session_state:
        st.session_state.healing_progress = 0


def parse_robot_output_xml(xml_content):
    """Parse Robot Framework's output.xml to extract failed tests"""
    try:
        root = ET.fromstring(xml_content)
        failed_tests = []

        for test in root.findall(".//test"):
            status = test.find("status")
            if status is not None and status.get("status") == "FAIL":
                test_name = test.get("name")
                error_msg = status.text

                # Extract locator information from error messages
                locator_info = extract_locator_from_error(error_msg)

                failed_tests.append({
                    "name": test_name,
                    "error": error_msg,
                    "suite": test.getparent().get("name") if hasattr(test, "getparent") else "",
                    "locator": locator_info.get("locator", ""),
                    "locator_type": locator_info.get("type", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        return failed_tests
    except Exception as e:
        st.error(f"Error parsing XML: {str(e)}")
        return []


def parse_junit_xml(xml_content):
    """Parse JUnit XML test reports"""
    try:
        root = ET.fromstring(xml_content)
        failed_tests = []

        for test_case in root.findall(".//testcase"):
            failure = test_case.find("failure")
            if failure is not None:
                test_name = test_case.get("name")
                error_msg = failure.text or failure.get("message", "")

                # Extract locator information from error messages
                locator_info = extract_locator_from_error(error_msg)

                failed_tests.append({
                    "name": test_name,
                    "error": error_msg,
                    "suite": test_case.get("classname", ""),
                    "locator": locator_info.get("locator", ""),
                    "locator_type": locator_info.get("type", ""),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

        return failed_tests
    except Exception as e:
        st.error(f"Error parsing XML: {str(e)}")
        return []


def parse_robot_html_report(html_content):
    """Parse Robot Framework's HTML report to extract failed tests"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        failed_tests = []

        # Find all failed test cases in the HTML report
        # Robot Framework HTML reports typically mark failed tests with class="fail"
        failed_elements = soup.find_all(class_="fail")

        for element in failed_elements:
            # Look for the test name
            test_name_elem = element.find_previous(class_="name")
            if not test_name_elem:
                continue

            test_name = test_name_elem.get_text().strip()

            # Get the error message
            error_msg_elem = element.find(class_="message")
            error_msg = error_msg_elem.get_text().strip() if error_msg_elem else "Unknown error"

            # Extract locator information from the error message
            locator_info = extract_locator_from_error(error_msg)

            # Try to find the test suite name
            suite_elem = element.find_previous(class_="suite")
            suite_name = suite_elem.get_text().strip() if suite_elem else ""

            failed_tests.append({
                "name": test_name,
                "error": error_msg,
                "suite": suite_name,
                "locator": locator_info.get("locator", ""),
                "locator_type": locator_info.get("type", ""),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        return failed_tests
    except Exception as e:
        st.error(f"Error parsing HTML report: {str(e)}")
        return []


def extract_locator_from_error(error_msg):
    """Extract locator information from error messages"""
    # Common locator patterns in error messages
    patterns = [
        r"(?:Element|Locator|Selector)\s+['\"]([^'\"]+)['\"](?:\s+with\s+(?:type|strategy)\s+['\"]([^'\"]+)['\"])?",
        r"(?:Unable to find|No such element|Cannot locate)[^'\"]*['\"]([^'\"]+)['\"]",
        r"xpath=([^\s]+)",
        r"css=([^\s]+)",
        r"id=([^\s]+)",
        r"name=([^\s]+)",
        r"class=([^\s]+)",
        r"tag=([^\s]+)",
        r"link=([^\s]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg, re.IGNORECASE)
        if match:
            locator = match.group(1)
            # Try to determine the locator type
            locator_type = None

            if "xpath=" in locator or "xpath=" in error_msg:
                locator_type = "xpath"
            elif "css=" in locator or "css=" in error_msg:
                locator_type = "css"
            elif "id=" in locator or "id=" in error_msg:
                locator_type = "id"
            elif "//" in locator and "[" in locator and "]" in locator:
                locator_type = "xpath"
            elif locator.startswith("#") or locator.startswith("."):
                locator_type = "css"
            elif match.groups() > 1:
                locator_type = match.group(2)

            return {"locator": locator, "type": locator_type or "unknown"}

    return {"locator": "", "type": "unknown"}


def process_uploaded_file(uploaded_file):
    """Process an uploaded test report file"""
    content = uploaded_file.read()
    content_str = content.decode("utf-8")

    if uploaded_file.name.endswith(".xml"):
        # Try to determine if it's a Robot Framework or JUnit XML
        if "<robot" in content_str:
            failed_tests = parse_robot_output_xml(content_str)
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="self_healing_tests",
                    status="info" if failed_tests else "warning",
                    message=f"Processed Robot Framework XML report",
                    details=f"Found {len(failed_tests)} failed tests in {uploaded_file.name}"
                )
            return failed_tests
        else:
            failed_tests = parse_junit_xml(content_str)
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="self_healing_tests",
                    status="info" if failed_tests else "warning",
                    message=f"Processed JUnit XML report",
                    details=f"Found {len(failed_tests)} failed tests in {uploaded_file.name}"
                )
            return failed_tests
    elif uploaded_file.name.endswith(".json"):
        try:
            data = json.loads(content_str)
            # Process based on common JSON report formats
            # This is a simplified implementation; you'd need to adapt it to your specific JSON format
            failed_tests = []

            # Handle different JSON formats based on structure
            if "tests" in data:
                for test in data["tests"]:
                    if test.get("status") == "failed" or test.get("result") == "failed":
                        error_msg = test.get("message", "")
                        locator_info = extract_locator_from_error(error_msg)

                        failed_tests.append({
                            "name": test.get("name", "Unknown Test"),
                            "error": error_msg,
                            "suite": test.get("suite", ""),
                            "locator": locator_info.get("locator", ""),
                            "locator_type": locator_info.get("type", ""),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="self_healing_tests",
                    status="info" if failed_tests else "warning",
                    message=f"Processed JSON test report",
                    details=f"Found {len(failed_tests)} failed tests in {uploaded_file.name}"
                )
            return failed_tests
        except Exception as e:
            if NOTIFICATIONS_AVAILABLE:
                notifications.add_notification(
                    module_name="self_healing_tests",
                    status="error",
                    message="Error parsing JSON report",
                    details=f"Error while parsing {uploaded_file.name}: {str(e)}",
                    action_steps=["Check if the file is valid JSON", "Ensure the JSON structure contains test results"]
                )
            st.error(f"Error parsing JSON: {str(e)}")
            return []
    elif uploaded_file.name.endswith(".html"):
        # Try to parse as Robot Framework HTML report
        failed_tests = parse_robot_html_report(content_str)
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="self_healing_tests",
                status="info" if failed_tests else "warning",
                message=f"Processed Robot Framework HTML report",
                details=f"Found {len(failed_tests)} failed tests in {uploaded_file.name}"
            )
        return failed_tests
    else:
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="self_healing_tests",
                status="error",
                message="Unsupported file format",
                details=f"The file {uploaded_file.name} is not supported. Please upload XML, JSON, or HTML test reports.",
                action_steps=["Convert your test report to a supported format", "Upload a file with .xml, .json, or .html extension"]
            )
        st.error(f"Unsupported file format: {uploaded_file.name}")
        return []


def suggest_alternative_locators(failed_test):
    """Generate alternative locator suggestions based on the original locator"""
    original_locator = failed_test.get("locator", "")
    locator_type = failed_test.get("locator_type", "unknown")

    suggestions = []

    if not original_locator:
        return [
            {"locator": "//div[contains(@class, 'element-name')]", "type": "xpath", "confidence": 0.5, "strategy": FUZZY_MATCHING},
            {"locator": "#element-id", "type": "css", "confidence": 0.4, "strategy": FUZZY_MATCHING}
        ]

    # For XPath locators
    if locator_type == "xpath":
        # Make more flexible by using contains() for attributes
        if "=" in original_locator:
            flexible_locator = original_locator.replace('="', "=contains(., '").replace('"', "')]")
            suggestions.append({"locator": flexible_locator, "type": "xpath", "confidence": 0.8, "strategy": FUZZY_MATCHING})

        # Use parent-child relationship
        suggestions.append({"locator": f"{original_locator}/parent::*/child::*", "type": "xpath", "confidence": 0.6, "strategy": DOM_ANALYSIS})

        # Use text content
        if "text()" not in original_locator and "//" in original_locator:
            base_element = original_locator.split("[")[0] if "[" in original_locator else original_locator
            suggestions.append({"locator": f"{base_element}[contains(text(), 'element')]", "type": "xpath", "confidence": 0.5, "strategy": DOM_ANALYSIS})

    # For CSS selectors
    elif locator_type == "css":
        # Remove specific class from multiple classes
        if "." in original_locator and " " not in original_locator:
            parts = original_locator.split(".")
            if len(parts) > 2:  # Has multiple classes
                simplified = parts[0] + "." + parts[1]  # Keep only first class
                suggestions.append({"locator": simplified, "type": "css", "confidence": 0.7, "strategy": FUZZY_MATCHING})

        # Use parent-child relationship
        suggestions.append({"locator": f"{original_locator} > *", "type": "css", "confidence": 0.6, "strategy": DOM_ANALYSIS})

    # For ID selectors
    elif locator_type == "id":
        # Suggest using partial ID match with XPath
        id_value = original_locator.replace("id=", "")
        suggestions.append({"locator": f"//[contains(@id, '{id_value}')]", "type": "xpath", "confidence": 0.8, "strategy": FUZZY_MATCHING})

    # Add some general purpose suggestions
    suggestions.append({"locator": "Using image recognition to find the element", "type": "visual", "confidence": 0.7, "strategy": COMPUTER_VISION})
    suggestions.append({"locator": "Combined DOM structure analysis with visual cues", "type": "hybrid", "confidence": 0.85, "strategy": HYBRID})

    # Randomize confidence a bit to simulate real-world analysis
    for suggestion in suggestions:
        suggestion["confidence"] = min(0.95, suggestion["confidence"] + random.uniform(-0.1, 0.1))

    return suggestions


def fix_test_file(test_info, selected_suggestion):
    """Generate fixed test code based on the selected suggestion"""
    # In a real implementation, this would analyze and modify actual test files
    # This is a simplified simulation of that process

    test_name = test_info.get("name", "Unknown Test")
    original_locator = test_info.get("locator", "")
    new_locator = selected_suggestion.get("locator", "")
    strategy = selected_suggestion.get("strategy", FUZZY_MATCHING)

    # Generate sample code showing the fix
    if strategy == COMPUTER_VISION:
        fixed_code = f"""
# Test: {test_name}
# FIXED using {strategy}
# Original issue: Element not found with locator: {original_locator}

# Added visual element detection
from computer_vision_library import find_element_by_image

# Take screenshot and find element
element = find_element_by_image("element_reference.png", confidence=0.7)
element.click()
"""
    elif strategy == HYBRID:
        fixed_code = f"""
# Test: {test_name}
# FIXED using {strategy}
# Original issue: Element not found with locator: {original_locator}

# Added hybrid element detection
try:
    # First try original locator
    element = driver.find_element_by_xpath("{original_locator}")
except:
    try:
        # Then try fuzzy matching
        element = find_element_with_fuzzy_match("{new_locator}")
    except:
        # Finally try visual detection
        element = find_element_by_visual_cues("element_reference.png")

element.click()
"""
    else:
        # For standard DOM-based approaches
        if "xpath" in selected_suggestion.get("type", ""):
            method = "find_element_by_xpath"
        elif "css" in selected_suggestion.get("type", ""):
            method = "find_element_by_css_selector"
        else:
            method = f"find_element_by_{selected_suggestion.get('type', 'xpath')}"

        fixed_code = f"""
# Test: {test_name}
# FIXED using {strategy}
# Original locator: {original_locator}
# New locator: {new_locator}

# Updated element locator
element = driver.{method}("{new_locator}")
element.click()
"""

    return fixed_code


def heal_tests(failed_tests, healing_strategy):
    """Apply the selected healing strategy to the failed tests"""
    if not failed_tests:
        if NOTIFICATIONS_AVAILABLE:
            notifications.add_notification(
                module_name="self_healing_tests",
                status="warning",
                message="No tests to heal",
                details="No failed tests were found for healing"
            )
        return []

    # Set up progress tracking
    total_tests = len(failed_tests)
    st.session_state.healing_progress = 0
    progress_bar = st.progress(0)

    if NOTIFICATIONS_AVAILABLE:
        notifications.add_notification(
            module_name="self_healing_tests",
            status="info",
            message="Started test healing process",
            details=f"Analyzing {total_tests} failed tests using the {healing_strategy} strategy"
        )

    healing_results = []

    for i, test in enumerate(failed_tests):
        # Update progress
        progress = (i + 1) / total_tests
        st.session_state.healing_progress = progress
        progress_bar.progress(progress)

        # Simulate analysis time
        time.sleep(0.5)

        # Generate alternative locator suggestions
        suggestions = suggest_alternative_locators(test)

        # Filter suggestions based on selected strategy
        if healing_strategy != HYBRID:
            suggestions = [s for s in suggestions if s["strategy"] == healing_strategy]

        # Sort by confidence
        suggestions = sorted(suggestions, key=lambda x: x.get("confidence", 0), reverse=True)

        # Take the top suggestion
        best_suggestion = suggestions[0] if suggestions else None

        # Generate fixed code if we have a suggestion
        fixed_code = fix_test_file(test, best_suggestion) if best_suggestion else ""

        healing_results.append({
            "test_name": test["name"],
            "original_locator": test["locator"],
            "suggestions": suggestions,
            "best_suggestion": best_suggestion,
            "fixed_code": fixed_code,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    # Reset progress when done
    st.session_state.healing_progress = 1.0
    progress_bar.progress(1.0)
    time.sleep(0.5)
    st.session_state.healing_progress = 0

    # Notify about healing results
    if NOTIFICATIONS_AVAILABLE:
        success_count = sum(1 for r in healing_results if r.get("best_suggestion") is not None)

        if success_count > 0:
            notifications.add_notification(
                module_name="self_healing_tests",
                status="success",
                message="Tests healing completed",
                details=f"Successfully healed {success_count} out of {len(failed_tests)} tests using {healing_strategy}",
                action_steps=["Review the suggested fixes", "Apply the fixes to your test code"]
            )
        else:
            notifications.add_notification(
                module_name="self_healing_tests",
                status="warning",
                message="Tests healing completed with issues",
                details=f"Could not find good alternatives for any of the {len(failed_tests)} tests using {healing_strategy}",
                action_steps=["Try a different healing strategy", "Consider manual test fixes"]
            )

    # Save healing history
    if st.session_state.healing_configs["save_healing_history"]:
        st.session_state.healing_history.extend(healing_results)

    return healing_results


def show_ui():
    """Main UI function for self-healing tests module"""
    st.title("Self-Healing Tests")

    # Initialize session state
    init_session_state()

    st.markdown("""
    This tool helps automatically repair broken tests when UI elements change.
    Upload your failed test reports, select a healing strategy, and let the AI suggest fixes.
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        # File uploader for test reports
        uploaded_file = st.file_uploader(
            "Upload Failed Test Report",
            type=["xml", "json", "html"],
            help="Upload Robot Framework, JUnit XML, or custom JSON test reports"
        )

        # Process the uploaded file
        if uploaded_file is not None:
            with st.spinner("Processing test report..."):
                failed_tests = process_uploaded_file(uploaded_file)
                st.session_state.failed_tests = failed_tests

                if failed_tests:
                    st.success(f"Successfully identified {len(failed_tests)} failed tests")
                else:
                    st.warning("No failed tests found in the report or the report format is not recognized")

    with col2:
        # Settings and configuration
        with st.expander("Settings", expanded=False):
            st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.healing_configs["similarity_threshold"],
                step=0.05,
                help="Minimum similarity score for alternative locator suggestions",
                key="similarity_threshold"
            )

            st.number_input(
                "Max Suggestions",
                min_value=1,
                max_value=10,
                value=st.session_state.healing_configs["max_healing_suggestions"],
                help="Maximum number of alternative locator suggestions to provide",
                key="max_healing_suggestions"
            )

            st.checkbox(
                "Include Visual Analysis",
                value=st.session_state.healing_configs["include_visual_analysis"],
                help="Use image recognition techniques for element identification",
                key="include_visual_analysis"
            )

            st.checkbox(
                "Save Healing History",
                value=st.session_state.healing_configs["save_healing_history"],
                help="Keep a record of all healing operations",
                key="save_healing_history"
            )

            # Update configs when settings change
            for key in DEFAULT_CONFIGS:
                st.session_state.healing_configs[key] = st.session_state[key]

    # Display identified failed tests
    if st.session_state.failed_tests:
        st.markdown("### Failed Tests")

        # Convert to DataFrame for better display
        df_failed = pd.DataFrame(st.session_state.failed_tests)
        if not df_failed.empty:
            # Select relevant columns
            display_cols = ["name", "suite", "locator_type", "locator"]
            display_cols = [col for col in display_cols if col in df_failed.columns]

            st.dataframe(df_failed[display_cols], use_container_width=True)

            # Healing strategy selection
            st.markdown("### Healing Configuration")

            healing_strategy = st.selectbox(
                "Healing Strategy",
                options=[FUZZY_MATCHING, COMPUTER_VISION, DOM_ANALYSIS, HYBRID],
                help="Select the approach to use for finding alternative locators",
                index=0
            )

            st.markdown(f"**Strategy Description:** {HEALING_STRATEGIES[healing_strategy]}")

            # Heal tests button
            if st.button("Repair Tests", key="repair_btn"):
                with st.spinner("Analyzing and repairing tests..."):
                    healing_results = heal_tests(st.session_state.failed_tests, healing_strategy)

                    if healing_results:
                        st.success(f"Successfully repaired {len(healing_results)} tests")

                        # Display results
                        st.markdown("### Healing Results")

                        for i, result in enumerate(healing_results):
                            with st.expander(f"Test: {result['test_name']}", expanded=i == 0):
                                st.markdown(f"**Original Locator:** `{result['original_locator']}`")

                                if result['best_suggestion']:
                                    st.markdown(f"**Best Alternative:** `{result['best_suggestion']['locator']}`")
                                    st.markdown(f"**Confidence:** {result['best_suggestion']['confidence']:.2f}")
                                    st.markdown(f"**Strategy:** {result['best_suggestion']['strategy']}")

                                    st.markdown("**Fixed Code:**")
                                    st.code(result['fixed_code'], language="python")

                                    # Option to save the fix
                                    if st.button(f"Apply Fix for {result['test_name']}", key=f"apply_{i}"):
                                        st.success(f"Fix for '{result['test_name']}' has been applied")

                                        # In a real implementation, this would modify the actual test file
                                        # For this demo, we'll just show a success message
                                else:
                                    st.warning("Couldn't find a suitable alternative for this test")

                        # Option to download a report
                        report_json = json.dumps({
                            "healing_results": healing_results,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "strategy": healing_strategy
                        })

                        st.download_button(
                            label="Download Healing Report",
                            data=report_json,
                            file_name=f"healing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                        )
                    else:
                        st.error("Failed to repair tests. Please try a different healing strategy.")

    # Display healing history
    if st.session_state.healing_history:
        with st.expander("Healing History", expanded=False):
            # Create a DataFrame for better display
            history_data = []
            for entry in st.session_state.healing_history:
                history_data.append({
                    "Test Name": entry["test_name"],
                    "Original Locator": entry["original_locator"],
                    "Best Suggestion": entry.get("best_suggestion", {}).get("locator", "None"),
                    "Confidence": entry.get("best_suggestion", {}).get("confidence", 0),
                    "Strategy": entry.get("best_suggestion", {}).get("strategy", "Unknown"),
                    "Timestamp": entry["timestamp"]
                })

            if history_data:
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True)

                # Option to clear history
                if st.button("Clear History", key="clear_history"):
                    st.session_state.healing_history = []
                    st.rerun()
            else:
                st.info("No healing history available")

    # Additional information and resources
    with st.expander("About Self-Healing Tests", expanded=True):
        st.markdown("""
        ### What are Self-Healing Tests?

        Self-healing tests are automated tests that can automatically adapt to changes in the application's UI,
        preventing tests from failing due to minor UI modifications like ID changes, class renaming, or element restructuring.

        ### Benefits:

        1. **Reduced Maintenance:** Less time spent fixing broken tests
        2. **Higher Stability:** Tests continue working even when UI changes
        3. **Faster Feedback:** No need to wait for manual test fixes

        ### How It Works:

        1. **Locator Analysis:** Extracts failed locators from test reports
        2. **Alternative Generation:** Suggests different ways to locate the same elements
        3. **Test Repair:** Applies the most promising alternatives to fix the tests
        4. **Learning:** Records successful fixes to improve future suggestions

        ### Common Healing Strategies:

        - **Fuzzy Matching:** Uses string similarity to find elements with similar attributes
        - **DOM Analysis:** Examines the structure and hierarchy of elements
        - **Computer Vision:** Uses image recognition to locate elements visually
        - **Hybrid Approaches:** Combines multiple techniques for better accuracy
        """)


if __name__ == "__main__":
    show_ui()
