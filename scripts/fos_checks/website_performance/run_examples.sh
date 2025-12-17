#!/bin/bash

# GTMetrix-Style Performance Testing Examples
# This script demonstrates various ways to use the performance testing suite

echo "🚀 GTMetrix-Style Performance Testing Suite - Example Usage"
echo "============================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node.js and Lighthouse are available
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found. Please install Node.js and Lighthouse:"
    echo "   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "   nvm install node"
    echo "   npm install -g lighthouse"
fi

if ! command -v lighthouse &> /dev/null; then
    echo "⚠️  Lighthouse not found. Please install Lighthouse:"
    echo "   npm install -g lighthouse"
fi

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "📁 Current directory: $SCRIPT_DIR"
echo ""

# Function to run example with error handling
run_example() {
    local example_name="$1"
    local command="$2"
    
    echo "📊 Running Example: $example_name"
    echo "Command: $command"
    echo "----------------------------------------"
    
    if eval "$command"; then
        echo "✅ $example_name completed successfully!"
    else
        echo "❌ $example_name failed!"
        return 1
    fi
    
    echo ""
}

# Example 1: Test single URL (quick test)
echo "Example 1: Single URL Performance Test"
echo "======================================"
run_example "Single URL Test" \
    "python3 gtmetrix_style_performance_checker.py --urls https://www.google.com --devices desktop --workers 1"

# Example 2: Test multiple URLs from command line
echo "Example 2: Multiple URLs from Command Line"
echo "=========================================="
run_example "Multiple URLs Test" \
    "python3 gtmetrix_style_performance_checker.py --urls https://www.google.com https://www.github.com --devices desktop mobile --workers 2"

# Example 3: Test URLs from CSV file
echo "Example 3: URLs from CSV File"
echo "=============================="
run_example "CSV File Test" \
    "python3 gtmetrix_style_performance_checker.py --file sample_urls.csv --devices desktop --workers 2"

# Example 4: Test URLs from JSON file
echo "Example 4: URLs from JSON File"
echo "==============================="
run_example "JSON File Test" \
    "python3 gtmetrix_style_performance_checker.py --file sample_urls.json --devices desktop mobile --workers 1"

# Example 5: Test URLs from TXT file
echo "Example 5: URLs from TXT File"
echo "=============================="
run_example "TXT File Test" \
    "python3 gtmetrix_style_performance_checker.py --file sample_urls.txt --devices desktop --workers 1"

# Example 6: Generate dashboard (no server)
echo "Example 6: Generate Enhanced Dashboard"
echo "======================================"
run_example "Dashboard Generation" \
    "python3 enhanced_gtmetrix_dashboard.py --generate-only"

# Example 7: Start dashboard server (background process)
echo "Example 7: Start Dashboard Server"
echo "=================================="
echo "📊 Starting dashboard server in background..."
echo "🌐 Dashboard will be available at: http://localhost:8001"
echo "⏹️  Press Ctrl+C to stop the server when done viewing"
echo ""

# Start dashboard server
python3 enhanced_gtmetrix_dashboard.py --port 8001 &
DASHBOARD_PID=$!

echo "🆔 Dashboard server PID: $DASHBOARD_PID"
echo "📊 Dashboard server started successfully!"
echo ""
echo "✨ You can now:"
echo "   1. Open http://localhost:8001 in your browser"
echo "   2. View historical performance data"
echo "   3. Analyze trends and insights"
echo "   4. Access detailed reports"
echo ""
echo "⏹️  To stop the dashboard server, run:"
echo "   kill $DASHBOARD_PID"
echo ""

# Function to show available reports
show_reports() {
    echo "📋 Available Performance Reports:"
    echo "================================"
    
    if [ -d "performance_reports" ]; then
        find performance_reports -name "performance_report.html" -type f | head -5 | while read -r report; do
            echo "📄 $report"
        done
        
        echo ""
        echo "🌐 To view a report, open it in your browser:"
        echo "   open performance_reports/performance_report_*/performance_report.html"
    else
        echo "⚠️  No performance reports found. Run some tests first!"
    fi
}

# Show available reports
show_reports

echo ""
echo "🎉 Example demonstrations completed!"
echo ""
echo "📚 Next Steps:"
echo "============="
echo "1. 🔍 View the generated HTML reports in your browser"
echo "2. 📊 Access the dashboard at http://localhost:8001"
echo "3. 📈 Analyze performance trends and recommendations"
echo "4. 🔧 Customize the tests for your specific websites"
echo "5. 📋 Read the README for advanced usage options"
echo ""
echo "💡 Tips:"
echo "======="
echo "• Use --workers to control parallel testing"
echo "• Test both desktop and mobile for complete analysis"
echo "• Set up CrUX API key for real user metrics"
echo "• Schedule regular tests for continuous monitoring"
echo ""
echo "🆘 For help and troubleshooting:"
echo "• Check the README file"
echo "• Review the logs for error details"
echo "• Ensure all dependencies are installed"
echo ""
echo "Happy performance testing! 🚀"
