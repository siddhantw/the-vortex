#!/usr/bin/env python3
"""
TestPilot CLI - Command-line interface for TestPilot operations
Usage: python testpilot_cli.py [command] [options]

Examples:
    python testpilot_cli.py generate --url https://example.com --output test_example.robot
    python testpilot_cli.py metrics
    python testpilot_cli.py export --format json
    python testpilot_cli.py health
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from test_pilot import TestPilotEngine, TestPilotMetrics, TestCase, TestStep
    from gen_ai.azure_openai_client import AzureOpenAIClient
except ImportError as e:
    print(f"⚠️ Warning: Could not import TestPilot modules: {e}")
    print("Running in standalone mode with limited functionality.")
    TestPilotEngine = None
    TestPilotMetrics = None


class TestPilotCLI:
    """Command-line interface for TestPilot"""

    def __init__(self):
        self.engine = None
        self.metrics = None

        if TestPilotEngine:
            try:
                azure_client = AzureOpenAIClient() if 'AzureOpenAIClient' in globals() else None
                self.engine = TestPilotEngine(azure_client=azure_client)
                self.metrics = TestPilotMetrics()
                print("✅ TestPilot initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: TestPilot initialization failed: {e}")

    def generate_test(self, url: str, output: str, use_selenium: bool = True):
        """Generate test from URL"""
        if not self.engine:
            print("❌ Error: TestPilot engine not available")
            return False

        print(f"🔄 Generating test from {url}...")
        print(f"📝 Output will be saved to: {output}")

        try:
            # Create a test case from URL
            test_case = TestCase(
                title=f"Test for {url}",
                description=f"Automatically generated test for {url}",
                steps=[]
            )

            # Add URL navigation step
            test_case.steps.append(TestStep(
                step_number=1,
                description=f"Navigate to {url}",
                action="navigate"
            ))

            # Use browser automation if available
            if use_selenium:
                print("🌐 Using Selenium for dynamic content extraction...")
                result = self.engine.analyze_and_generate_with_browser_automation(
                    url=url,
                    test_case=test_case
                )
            else:
                print("📄 Generating test without browser automation...")
                result = self.engine.generate_robot_script(test_case)

            if result:
                print(f"✅ Test generated successfully!")
                print(f"📁 File saved: {output}")
                return True
            else:
                print("❌ Test generation failed")
                return False

        except Exception as e:
            print(f"❌ Error during test generation: {str(e)}")
            return False

    def run_test(self, test_file: str):
        """Run generated test"""
        if not os.path.exists(test_file):
            print(f"❌ Error: Test file not found: {test_file}")
            return False

        print(f"🚀 Running test: {test_file}")

        try:
            # Use robot command to run the test
            import subprocess
            result = subprocess.run(
                ['robot', test_file],
                capture_output=True,
                text=True
            )

            print(result.stdout)
            if result.returncode == 0:
                print("✅ Test passed!")
                return True
            else:
                print("❌ Test failed!")
                print(result.stderr)
                return False

        except FileNotFoundError:
            print("❌ Error: Robot Framework not installed")
            print("Install with: pip install robotframework")
            return False
        except Exception as e:
            print(f"❌ Error running test: {str(e)}")
            return False

    def show_metrics(self):
        """Display current metrics"""
        if not self.metrics:
            print("❌ Error: Metrics not available")
            return

        try:
            summary = self.metrics.get_summary()

            print("\n" + "="*60)
            print("📊 TestPilot Metrics Summary")
            print("="*60)
            print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Uptime: {summary['uptime_seconds']:.0f} seconds")
            print()

            print("🏥 Health Score:")
            health = summary['health_score']
            health_emoji = "🟢" if health > 80 else "🟡" if health > 60 else "🔴"
            print(f"  {health_emoji} {health:.1f}/100")
            print()

            print("📝 Test Generation:")
            gen = summary['metrics']['test_generation']
            print(f"  Total: {gen['total_generated']}")
            print(f"  Successful: {gen['successful']} ({gen['successful']/max(1,gen['total_generated'])*100:.1f}%)")
            print(f"  Failed: {gen['failed']}")
            print(f"  Flaky Detected: {gen['flaky_detected']}")
            print(f"  Avg Time: {gen['avg_generation_time']:.2f}s")
            print()

            print("🚀 Test Execution:")
            exe = summary['metrics']['test_execution']
            print(f"  Total: {exe['total_executed']}")
            print(f"  Passed: {exe['passed']} ({exe['passed']/max(1,exe['total_executed'])*100:.1f}%)")
            print(f"  Failed: {exe['failed']}")
            print(f"  Skipped: {exe['skipped']}")
            print(f"  Avg Time: {exe['avg_execution_time']:.2f}s")
            print()

            print("🔧 Self-Healing:")
            heal = summary['metrics']['self_healing']
            print(f"  Total Attempts: {heal['total_heals']}")
            print(f"  Successful: {heal['successful_heals']} ({heal['successful_heals']/max(1,heal['total_heals'])*100:.1f}%)")
            print(f"  Failed: {heal['failed_heals']}")
            print(f"  Avg Time: {heal['avg_heal_time']:.2f}s")
            print()

            print("🤖 API Usage:")
            api = summary['metrics']['api_usage']
            print(f"  Total Calls: {api['total_calls']}")
            print(f"  Successful: {api['successful_calls']} ({api['successful_calls']/max(1,api['total_calls'])*100:.1f}%)")
            print(f"  Total Tokens: {api['total_tokens']:,}")
            print(f"  Total Cost: ${api['total_cost']:.2f}")
            print()

            print("="*60)

        except Exception as e:
            print(f"❌ Error displaying metrics: {str(e)}")

    def export_report(self, format: str = 'json', output: str = None):
        """Export comprehensive report"""
        if not self.metrics:
            print("❌ Error: Metrics not available")
            return False

        if output is None:
            output = f"testpilot_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"

        print(f"📄 Exporting report in {format} format to {output}...")

        try:
            summary = self.metrics.get_summary()

            if format == 'json':
                import json
                with open(output, 'w') as f:
                    json.dump(summary, f, indent=2)

            elif format == 'txt':
                with open(output, 'w') as f:
                    f.write("TestPilot Report\n")
                    f.write("="*60 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Health Score: {summary['health_score']:.1f}/100\n\n")

                    f.write("Test Generation:\n")
                    gen = summary['metrics']['test_generation']
                    f.write(f"  Total: {gen['total_generated']}\n")
                    f.write(f"  Successful: {gen['successful']}\n")
                    f.write(f"  Failed: {gen['failed']}\n\n")

                    f.write("Test Execution:\n")
                    exe = summary['metrics']['test_execution']
                    f.write(f"  Total: {exe['total_executed']}\n")
                    f.write(f"  Passed: {exe['passed']}\n")
                    f.write(f"  Failed: {exe['failed']}\n\n")

            else:
                print(f"❌ Unsupported format: {format}")
                return False

            print(f"✅ Report exported successfully to {output}")
            return True

        except Exception as e:
            print(f"❌ Error exporting report: {str(e)}")
            return False

    def health_check(self):
        """Perform health check"""
        if not self.metrics:
            print("❌ Error: Metrics not available")
            return

        try:
            summary = self.metrics.get_summary()
            health = summary['health_score']

            print("\n" + "="*60)
            print("🏥 TestPilot Health Check")
            print("="*60)

            if health > 90:
                print("✅ Status: EXCELLENT")
                print("   All systems operating optimally")
            elif health > 80:
                print("✅ Status: GOOD")
                print("   Systems operating normally")
            elif health > 70:
                print("⚠️  Status: FAIR")
                print("   Some issues detected, review recommended")
            elif health > 60:
                print("⚠️  Status: POOR")
                print("   Multiple issues detected, action required")
            else:
                print("❌ Status: CRITICAL")
                print("   Severe issues detected, immediate action required")

            print(f"\nHealth Score: {health:.1f}/100")

            # Recommendations
            print("\n📋 Recommendations:")
            gen = summary['metrics']['test_generation']
            if gen['total_generated'] > 0:
                fail_rate = gen['failed'] / gen['total_generated']
                if fail_rate > 0.2:
                    print(f"  ⚠️  High test generation failure rate ({fail_rate*100:.1f}%)")
                    print("     → Review input quality and AI configuration")

            exe = summary['metrics']['test_execution']
            if exe['total_executed'] > 0:
                fail_rate = exe['failed'] / exe['total_executed']
                if fail_rate > 0.1:
                    print(f"  ⚠️  High test execution failure rate ({fail_rate*100:.1f}%)")
                    print("     → Review test stability and environment setup")

            heal = summary['metrics']['self_healing']
            if heal['total_heals'] > 0:
                success_rate = heal['successful_heals'] / heal['total_heals']
                if success_rate < 0.8:
                    print(f"  ⚠️  Low self-healing success rate ({success_rate*100:.1f}%)")
                    print("     → Review locator strategies and page stability")

            api = summary['metrics']['api_usage']
            if api['total_cost'] > 50:
                print(f"  💰 High API costs (${api['total_cost']:.2f})")
                print("     → Consider prompt optimization or caching")

            if health > 80:
                print("  ✅ No critical issues detected")

            print("="*60 + "\n")

        except Exception as e:
            print(f"❌ Error performing health check: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='TestPilot CLI - Command-line interface for test automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate test from URL:
    %(prog)s generate --url https://example.com --output test_example.robot
    
  Show metrics:
    %(prog)s metrics
    
  Export report:
    %(prog)s export --format json --output report.json
    
  Health check:
    %(prog)s health
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate test from URL')
    gen_parser.add_argument('--url', required=True, help='URL to test')
    gen_parser.add_argument('--output', default='generated_test.robot', help='Output file path')
    gen_parser.add_argument('--no-selenium', action='store_true', help='Disable Selenium automation')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run generated test')
    run_parser.add_argument('test_file', help='Path to test file')

    # Metrics command
    subparsers.add_parser('metrics', help='Show current metrics')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export report')
    export_parser.add_argument('--format', choices=['json', 'txt'], default='json', help='Export format')
    export_parser.add_argument('--output', help='Output file path')

    # Health command
    subparsers.add_parser('health', help='Perform health check')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = TestPilotCLI()

    # Execute command
    try:
        if args.command == 'generate':
            success = cli.generate_test(args.url, args.output, use_selenium=not args.no_selenium)
            sys.exit(0 if success else 1)

        elif args.command == 'run':
            success = cli.run_test(args.test_file)
            sys.exit(0 if success else 1)

        elif args.command == 'metrics':
            cli.show_metrics()
            sys.exit(0)

        elif args.command == 'export':
            success = cli.export_report(args.format, args.output)
            sys.exit(0 if success else 1)

        elif args.command == 'health':
            cli.health_check()
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

