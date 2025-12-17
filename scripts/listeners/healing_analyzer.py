"""
Healing Report Analyzer and Dashboard
Analyzes self-healing reports and provides insights
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class HealingAnalyzer:
    """Analyze self-healing reports and provide insights"""

    def __init__(self, heal_reports_dir: str = "heal_reports"):
        self.heal_reports_dir = Path(heal_reports_dir)
        self.reports = []
        self.flaky_tests = {}
        self.healing_stats = defaultdict(int)

    def load_reports(self, days: int = 30):
        """Load healing reports from the last N days"""
        if not self.heal_reports_dir.exists():
            print(f"No healing reports found at {self.heal_reports_dir}")
            return

        cutoff_date = datetime.now() - timedelta(days=days)

        for report_file in self.heal_reports_dir.glob("heal_report_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)

                # Check date
                timestamp_str = report.get("session_info", {}).get("timestamp", "")
                if timestamp_str:
                    try:
                        report_date = datetime.fromisoformat(timestamp_str)
                        if report_date >= cutoff_date:
                            self.reports.append(report)
                    except ValueError:
                        # If timestamp parsing fails, include the report anyway
                        self.reports.append(report)
            except Exception as e:
                print(f"Error loading report {report_file}: {e}")

        print(f"Loaded {len(self.reports)} healing reports from last {days} days")

        # Load flaky tests report
        flaky_file = self.heal_reports_dir / "flaky_tests.json"
        if flaky_file.exists():
            try:
                with open(flaky_file, 'r') as f:
                    self.flaky_tests = json.load(f)
            except Exception as e:
                print(f"Error loading flaky tests: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all reports"""
        stats = {
            "total_sessions": len(self.reports),
            "total_tests": 0,
            "total_failures": 0,
            "total_healed": 0,
            "total_flaky_detected": 0,
            "total_ai_suggestions": 0,
            "total_auto_healed": 0,
            "failure_patterns": defaultdict(int),
            "healing_success_rate": 0.0,
            "most_common_failures": [],
            "most_healed_tests": [],
        }

        test_healing_counts = defaultdict(int)

        for report in self.reports:
            session_stats = report.get("statistics", {})
            stats["total_tests"] += session_stats.get("total_tests", 0)
            stats["total_failures"] += session_stats.get("failed_tests", 0)
            stats["total_healed"] += session_stats.get("healed_tests", 0)
            stats["total_flaky_detected"] += session_stats.get("flaky_tests_detected", 0)
            stats["total_ai_suggestions"] += session_stats.get("ai_suggestions", 0)
            stats["total_auto_healed"] += session_stats.get("auto_healed", 0)

            # Aggregate failure patterns
            for pattern, count in report.get("failure_patterns", {}).items():
                stats["failure_patterns"][pattern] += count

            # Track healed tests
            for test_name in report.get("healed_tests", {}).keys():
                test_healing_counts[test_name] += 1

        # Calculate healing success rate
        if stats["total_failures"] > 0:
            stats["healing_success_rate"] = (
                stats["total_healed"] / stats["total_failures"] * 100
            )

        # Get most common failure types
        stats["most_common_failures"] = sorted(
            stats["failure_patterns"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Get most healed tests
        stats["most_healed_tests"] = sorted(
            test_healing_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return dict(stats)

    def get_flaky_tests_summary(self) -> List[Dict[str, Any]]:
        """Get summary of flaky tests"""
        flaky_list = []

        for test in self.flaky_tests.get("flaky_tests", []):
            flaky_list.append({
                "test_name": test.get("test_name"),
                "failure_count": test.get("failure_count", 0),
                "last_failure": test.get("failures", [{}])[-1].get("timestamp", "N/A") if test.get("failures") else "N/A",
                "failure_types": list(set([
                    f.get("failure_type", "UNKNOWN")
                    for f in test.get("failures", [])
                ]))
            })

        return sorted(flaky_list, key=lambda x: x["failure_count"], reverse=True)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Robot Framework self-healing reports"
    )

    parser.add_argument(
        "--reports-dir",
        default="heal_reports",
        help="Directory containing healing reports"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to analyze (default: 30)"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export summary to JSON file"
    )

    args = parser.parse_args()

    analyzer = HealingAnalyzer(args.reports_dir)
    analyzer.load_reports(days=args.days)

    if args.export:
        analyzer.export_summary()


if __name__ == "__main__":
    main()

