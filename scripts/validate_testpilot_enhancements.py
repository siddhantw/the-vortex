#!/usr/bin/env python3
"""
TestPilot Enhancement Validation Script

This script validates that all TestPilot enhancements are working correctly.
"""

import sys
import os

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "gen_ai", "use_cases"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "gen_ai"))
sys.path.insert(0, ROOT_DIR)

def validate_imports():
    """Validate that all required imports work"""
    print("=" * 60)
    print("VALIDATION: Checking Imports")
    print("=" * 60)

    try:
        import test_pilot
        print("✅ test_pilot module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import test_pilot: {e}")
        return False

    return True


def validate_classes():
    """Validate that all new classes are available"""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking Classes")
    print("=" * 60)

    import test_pilot

    checks = [
        ("TestStep", hasattr(test_pilot, "TestStep")),
        ("TestCase", hasattr(test_pilot, "TestCase")),
        ("JiraZephyrIntegration", hasattr(test_pilot, "JiraZephyrIntegration")),
        ("BrowserAutomationManager", hasattr(test_pilot, "BrowserAutomationManager")),
        ("RecordingParser", hasattr(test_pilot, "RecordingParser")),
        ("TestPilotEngine", hasattr(test_pilot, "TestPilotEngine")),
    ]

    all_passed = True
    for class_name, available in checks:
        status = "✅" if available else "❌"
        print(f"{status} {class_name}: {'Available' if available else 'Missing'}")
        if not available:
            all_passed = False

    return all_passed


def validate_robotmcp():
    """Validate RobotMCP integration status"""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking RobotMCP Integration")
    print("=" * 60)

    import test_pilot

    print(f"RobotMCP Available: {test_pilot.ROBOTMCP_AVAILABLE}")

    if test_pilot.ROBOTMCP_AVAILABLE:
        print("✅ RobotMCP is installed and available")
    else:
        print("⚠️  RobotMCP not available (optional)")
        print("   Install with: pip install robotmcp")

    return True  # Not required


def validate_browser_automation_features():
    """Validate BrowserAutomationManager features"""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking BrowserAutomationManager Features")
    print("=" * 60)

    import test_pilot

    if not hasattr(test_pilot, "BrowserAutomationManager"):
        print("❌ BrowserAutomationManager class not found")
        return False

    browser_mgr_class = test_pilot.BrowserAutomationManager

    required_methods = [
        "initialize_browser",
        "capture_network_logs",
        "capture_console_errors",
        "capture_dom_snapshot",
        "capture_screenshot",
        "capture_performance_metrics",
        "execute_step_smartly",
        "generate_ai_bug_report",
        "cleanup"
    ]

    all_passed = True
    for method_name in required_methods:
        has_method = hasattr(browser_mgr_class, method_name)
        status = "✅" if has_method else "❌"
        print(f"{status} {method_name}: {'Present' if has_method else 'Missing'}")
        if not has_method:
            all_passed = False

    return all_passed


def validate_engine_features():
    """Validate TestPilotEngine enhanced features"""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking TestPilotEngine Features")
    print("=" * 60)

    import test_pilot

    engine_class = test_pilot.TestPilotEngine

    required_methods = [
        "_scan_existing_keywords",
        "_scan_existing_locators",
        "analyze_and_generate_with_browser_automation",
        "_use_robotmcp_for_analysis",
        "analyze_steps_with_ai",
        "generate_robot_script"
    ]

    all_passed = True
    for method_name in required_methods:
        has_method = hasattr(engine_class, method_name)
        status = "✅" if has_method else "❌"
        print(f"{status} {method_name}: {'Present' if has_method else 'Missing'}")
        if not has_method:
            all_passed = False

    return all_passed


def validate_selenium():
    """Validate Selenium availability"""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking Selenium")
    print("=" * 60)

    try:
        from selenium import webdriver
        print("✅ Selenium is installed")

        try:
            from selenium.webdriver.chrome.options import Options
            print("✅ Chrome WebDriver support available")
        except ImportError:
            print("⚠️  Chrome WebDriver support not available")

        return True
    except ImportError:
        print("❌ Selenium not installed")
        print("   Install with: pip install selenium")
        return False


def validate_dependencies():
    """Validate other dependencies"""
    print("\n" + "=" * 60)
    print("VALIDATION: Checking Dependencies")
    print("=" * 60)

    dependencies = [
        ("beautifulsoup4", "bs4"),
        ("requests", "requests"),
        ("streamlit", "streamlit"),
    ]

    all_passed = True
    for package_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} not installed")
            print(f"   Install with: pip install {package_name}")
            all_passed = False

    return all_passed


def print_summary(results):
    """Print validation summary"""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "-" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("-" * 60)

    if failed == 0:
        print("\n🎉 All validations passed! TestPilot enhancements are ready to use.")
    else:
        print(f"\n⚠️  {failed} validation(s) failed. Please review the output above.")


def main():
    """Main validation runner"""
    print("\n" + "=" * 60)
    print("TestPilot Enhancement Validation")
    print("=" * 60)
    print("This script validates all TestPilot enhancements.")
    print("Date: November 13, 2025")
    print("=" * 60)

    results = {}

    # Run validations
    results["Imports"] = validate_imports()

    if results["Imports"]:
        results["Classes"] = validate_classes()
        results["BrowserAutomationManager Features"] = validate_browser_automation_features()
        results["TestPilotEngine Features"] = validate_engine_features()
        results["RobotMCP Integration"] = validate_robotmcp()
        results["Selenium"] = validate_selenium()
        results["Dependencies"] = validate_dependencies()
    else:
        print("\n❌ Import validation failed. Cannot proceed with other validations.")
        return 1

    # Print summary
    print_summary(results)

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

