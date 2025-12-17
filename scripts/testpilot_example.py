#!/usr/bin/env python3
"""
TestPilot Browser Automation - Quick Start Example

This script demonstrates how to use the enhanced TestPilot with browser automation
to generate Robot Framework test scripts with real locators and bug reports.
"""

import sys
import os
import asyncio

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "gen_ai", "use_cases"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "gen_ai"))
sys.path.insert(0, ROOT_DIR)

# Import TestPilot
from test_pilot import TestCase, TestStep, TestPilotEngine, BrowserAutomationManager

# Try to import Azure client (optional)
try:
    from azure_openai_client import AzureOpenAIClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️  Azure OpenAI not available - bug reports will be basic")


async def example_wordpress_plan_selection():
    """
    Example: WordPress Plan Selection Test

    This example demonstrates:
    1. Creating a test case with steps
    2. Using browser automation to execute steps
    3. Capturing real locators and network logs
    4. Generating AI-powered bug report
    5. Creating Robot Framework script
    """
    print("\n" + "=" * 60)
    print("Example: WordPress Plan Selection Test with Browser Automation")
    print("=" * 60)

    # Initialize Azure client (optional)
    azure_client = None
    if AZURE_AVAILABLE:
        try:
            azure_client = AzureOpenAIClient()
            if azure_client.is_configured():
                print("✅ Azure OpenAI configured - will generate AI bug report")
            else:
                print("⚠️  Azure OpenAI not configured - will use basic bug report")
                azure_client = None
        except Exception as e:
            print(f"⚠️  Could not initialize Azure client: {e}")
            azure_client = None

    # Create test case
    print("\n📝 Creating test case...")
    test_case = TestCase(
        id="EXAMPLE-001",
        title="WordPress Plan Selection",
        description="Test WordPress Cloud plan selection on Bluehost website",
        priority="High",
        tags=["ui", "wordpress", "bluehost", "example"],
        source="example"
    )

    # Add test steps
    test_case.steps = [
        TestStep(
            step_number=1,
            description="Navigate to https://www.bluehost.com"
        ),
        TestStep(
            step_number=2,
            description="Click on 'WordPress' menu item"
        ),
        TestStep(
            step_number=3,
            description="Select 'WordPress Cloud' from submenu"
        ),
        TestStep(
            step_number=4,
            description="Click 'Explore Plans' button"
        ),
        TestStep(
            step_number=5,
            description="Verify WordPress Cloud plans are displayed"
        )
    ]

    print(f"✅ Test case created: {test_case.title}")
    print(f"   Steps: {len(test_case.steps)}")

    # Initialize TestPilot engine
    print("\n🔧 Initializing TestPilot engine...")
    engine = TestPilotEngine(azure_client=azure_client)
    print("✅ Engine initialized")

    # Run browser automation
    print("\n🌐 Starting browser automation...")
    print("   Base URL: https://www.bluehost.com")
    print("   Headless: True")
    print("   This may take 30-60 seconds...")

    try:
        success, script_content, file_path, bug_report = await engine.analyze_and_generate_with_browser_automation(
            test_case=test_case,
            base_url="https://www.bluehost.com",
            headless=True,
            use_robotmcp=False
        )

        if success:
            print("\n" + "=" * 60)
            print("✅ SUCCESS! Script Generated")
            print("=" * 60)
            print(f"\n📁 Script saved to: {file_path}")

            # Show script preview
            print("\n📜 Script Preview (first 30 lines):")
            print("-" * 60)
            lines = script_content.split('\n')[:30]
            for line in lines:
                print(line)
            if len(script_content.split('\n')) > 30:
                print("...")
                print(f"(Total: {len(script_content.split('\n'))} lines)")
            print("-" * 60)

            # Show bug report summary
            print("\n🐛 Bug Report Summary:")
            print("-" * 60)
            bug_lines = bug_report.split('\n')[:20]
            for line in bug_lines:
                print(line)
            if len(bug_report.split('\n')) > 20:
                print("...")
                print(f"(Total: {len(bug_report.split('\n'))} lines)")
            print("-" * 60)

            # Show captured data
            if test_case.metadata.get('captured_locators'):
                print(f"\n📍 Captured {len(test_case.metadata['captured_locators'])} locators")
                print("   Sample locators:")
                for i, (name, value) in enumerate(list(test_case.metadata['captured_locators'].items())[:5]):
                    print(f"   - {name} = '{value}'")
                if len(test_case.metadata['captured_locators']) > 5:
                    print(f"   ... and {len(test_case.metadata['captured_locators']) - 5} more")

            if test_case.metadata.get('screenshots'):
                print(f"\n📸 Captured {len(test_case.metadata['screenshots'])} screenshots")
                print("   Screenshots saved to:")
                for screenshot in test_case.metadata['screenshots'][:3]:
                    print(f"   - {screenshot}")
                if len(test_case.metadata['screenshots']) > 3:
                    print(f"   ... and {len(test_case.metadata['screenshots']) - 3} more")

            if test_case.metadata.get('dom_snapshots'):
                print(f"\n🗂️  Captured {test_case.metadata['dom_snapshots']} DOM snapshots")

            print("\n" + "=" * 60)
            print("Next Steps:")
            print("=" * 60)
            print("1. Review the generated script")
            print("2. Check the bug report for issues")
            print("3. Update any NEED_TO_UPDATE locators")
            print("4. Run the test: robot " + file_path)

        else:
            print("\n❌ Script generation failed")
            print(f"   Error: {file_path}")
            print(f"\n   Bug report may still have useful information:")
            print(bug_report[:500])

    except Exception as e:
        print(f"\n❌ Error during browser automation: {e}")
        import traceback
        traceback.print_exc()


async def example_basic_navigation():
    """
    Example: Simple Navigation Test

    This is a simpler example for quick testing.
    """
    print("\n" + "=" * 60)
    print("Example: Simple Navigation Test")
    print("=" * 60)

    # Create simple test case
    test_case = TestCase(
        id="EXAMPLE-002",
        title="Simple Website Navigation",
        description="Navigate to website and verify page loads",
        source="example"
    )

    test_case.steps = [
        TestStep(1, "Navigate to https://www.google.com"),
        TestStep(2, "Verify Google search page is displayed"),
    ]

    print(f"✅ Test case created: {test_case.title}")

    # Initialize engine
    engine = TestPilotEngine()

    # Run browser automation
    print("\n🌐 Running browser automation...")

    try:
        success, script, path, report = await engine.analyze_and_generate_with_browser_automation(
            test_case=test_case,
            base_url="https://www.google.com",
            headless=True
        )

        if success:
            print(f"\n✅ Success! Script saved to: {path}")
            print(f"   Captured {len(test_case.metadata.get('captured_locators', {}))} locators")
        else:
            print(f"\n❌ Failed: {path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


def print_usage():
    """Print usage instructions"""
    print("\n" + "=" * 60)
    print("TestPilot Browser Automation - Quick Start")
    print("=" * 60)
    print("\nThis script demonstrates TestPilot's browser automation features.")
    print("\nAvailable Examples:")
    print("  1. WordPress Plan Selection (comprehensive)")
    print("  2. Simple Navigation (quick test)")
    print("\nUsage:")
    print("  python3 scripts/testpilot_example.py [example_number]")
    print("\nExamples:")
    print("  python3 scripts/testpilot_example.py 1   # Run WordPress example")
    print("  python3 scripts/testpilot_example.py 2   # Run simple navigation")
    print("  python3 scripts/testpilot_example.py     # Run both examples")
    print("\nNote: Requires Selenium and Chrome browser installed.")
    print("=" * 60)


async def main():
    """Main entry point"""
    import sys

    # Check if help requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        return 0

    # Determine which example to run
    example = sys.argv[1] if len(sys.argv) > 1 else "all"

    if example == "1":
        await example_wordpress_plan_selection()
    elif example == "2":
        await example_basic_navigation()
    else:
        # Run both examples
        print_usage()

        response = input("\nRun WordPress example? (y/n): ")
        if response.lower() == 'y':
            await example_wordpress_plan_selection()

        response = input("\nRun simple navigation example? (y/n): ")
        if response.lower() == 'y':
            await example_basic_navigation()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - TESTPILOT_ENHANCEMENTS.md")
    print("  - TESTPILOT_ENHANCEMENT_SUMMARY.md")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    # Run async main
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

