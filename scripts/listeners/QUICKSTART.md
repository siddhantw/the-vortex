# 🚀 Quick Start Guide: Azure OpenAI Self-Healing for Robot Framework

## 5-Minute Setup

### Step 1: Verify Installation (1 minute)

```bash
# Check if robotframework-heal is installed
pip list | grep robotframework-heal

# If not installed, run:
pip install -r requirements.txt
```

### Step 2: Configure Azure OpenAI (2 minutes)

```bash
# Set environment variables (replace with your actual values)
export AZURE_OPENAI_ENDPOINT="https://mh-open-ai-east-us2.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key-here"
export AZURE_OPENAI_DEPLOYMENT="dev-chat-ai-gpt4.1-mini"
export AZURE_OPENAI_API_VERSION="2024-10-21"

# Or use the template
cp scripts/gen_ai/azure_openai_env_template.sh .env
# Edit .env with your credentials
source .env
```

### Step 3: Run Your First Self-Healing Test (2 minutes)

```bash
# Run with self-healing detection
python scripts/listeners/run_with_healing.py tests/

# Or run a specific test file
python scripts/listeners/run_with_healing.py tests/keywords/health_check_monitor.robot
```

That's it! 🎉

## What Just Happened?

Your tests ran with these enhancements:

✅ **AI Analysis**: Failed tests analyzed by Azure OpenAI  
✅ **Smart Suggestions**: Healing recommendations generated  
✅ **Flaky Detection**: Unreliable tests automatically identified  
✅ **Detailed Reports**: Comprehensive insights saved to `heal_reports/`

## View Results

### 1. Check Healing Reports

```bash
# View latest healing report
cat heal_reports/heal_report_*.json | jq .

# View flaky tests
cat heal_reports/flaky_tests.json | jq .
```

### 2. Analyze Results

```bash
# Get summary of healing sessions
python scripts/listeners/healing_analyzer.py

# Export detailed analysis
python scripts/listeners/healing_analyzer.py --export
```

### 3. View in Dashboard

```bash
# Start Streamlit dashboard
streamlit run scripts/gen_ai/main_ui.py

# Navigate to: Robot Framework Dashboard Analytics
# Scroll to: Self-Healing Insights section
```

## Common Use Cases

### Use Case 1: Daily Smoke Tests

```bash
# Run smoke tests with healing
python scripts/listeners/run_with_healing.py tests/ \
  --include smoke \
  --flaky-threshold 2
```

### Use Case 2: Regression Suite

```bash
# Full regression with detailed tracking
python scripts/listeners/run_with_healing.py tests/ \
  --output-dir regression_results \
  --flaky-threshold 3
```

### Use Case 3: CI/CD Integration

```yaml
# Add to your CI/CD pipeline (Jenkins/GitLab/GitHub Actions)
test:
  script:
    - export AZURE_OPENAI_ENDPOINT=$AZURE_ENDPOINT
    - export AZURE_OPENAI_API_KEY=$AZURE_KEY
    - python scripts/listeners/run_with_healing.py tests/
  artifacts:
    paths:
      - results/
      - heal_reports/
```

### Use Case 4: Flaky Test Hunt

```bash
# Run tests multiple times to detect flakiness
for i in {1..5}; do
  python scripts/listeners/run_with_healing.py tests/ --flaky-threshold 2
done

# Analyze accumulated flaky tests
python scripts/listeners/healing_analyzer.py
```

## Understanding the Output

### Console Output

```
==========================================
🤖 Robot Framework with Azure AI Self-Healing
==========================================
Test Path: tests/
Output Dir: results
Auto-Heal: False
Flaky Threshold: 3
==========================================
✅ Azure OpenAI configured

🚀 Starting test execution...
```

During execution, you'll see:
- `❌ Test failed: Login Test` - Test failure detected
- `🤖 AI Suggestion for Login Test:` - AI analysis complete
- `🔄 Flaky test detected: Checkout Test` - Flaky test identified

### Healing Report Structure

```json
{
  "statistics": {
    "total_tests": 50,          // Total tests run
    "failed_tests": 5,          // Tests that failed
    "healed_tests": 3,          // Tests with healing suggestions
    "flaky_tests_detected": 2,  // Unreliable tests found
    "ai_suggestions": 5         // AI recommendations generated
  },
  "failure_patterns": {
    "LOCATOR_ISSUE": 3,         // Element not found
    "TIMEOUT": 1,               // Wait timeouts
    "ASSERTION_FAILURE": 1      // Expected vs actual mismatches
  }
}
```

## Next Steps

### 1. Review AI Suggestions

```bash
# Find your latest report
ls -lt heal_reports/heal_report_*.json | head -1

# View suggestions
cat heal_reports/heal_report_*.json | jq '.healed_tests'
```

### 2. Fix Flaky Tests

```bash
# Check flaky tests
cat heal_reports/flaky_tests.json | jq '.flaky_tests[] | {test: .test_name, failures: .failure_count}'

# Use AI suggestions to fix them
```

### 3. Enable Auto-Healing (Advanced)

⚠️ **Warning**: Only after reviewing suggestions manually

```bash
# Enable automatic healing
python scripts/listeners/run_with_healing.py tests/ --auto-heal
```

### 4. Integrate with Dashboard

View trends and insights in the Streamlit dashboard:

```bash
streamlit run scripts/gen_ai/main_ui.py
```

Navigate to **Robot Framework Dashboard Analytics** → **Self-Healing Insights**

## Troubleshooting

### Problem: No AI suggestions generated

```bash
# Test Azure OpenAI connection
python scripts/gen_ai/quick_azure_test.py

# Check environment variables
echo $AZURE_OPENAI_ENDPOINT
echo $AZURE_OPENAI_API_KEY
```

### Problem: Tests pass but no reports

Self-healing reports are only generated when tests fail. Intentionally fail a test to see healing in action:

```robot
*** Test Cases ***
Test For Healing Demo
    [Documentation]    This will fail to demonstrate healing
    Click Element    id=non-existent-element
```

### Problem: Flaky tests not detected

Lower the threshold:

```bash
python scripts/listeners/run_with_healing.py tests/ --flaky-threshold 2
```

## Useful Commands

```bash
# View all healing reports
ls -lh heal_reports/

# Count flaky tests
cat heal_reports/flaky_tests.json | jq '.flaky_tests | length'

# Show failure patterns
cat heal_reports/heal_report_*.json | jq '.failure_patterns'

# Export healing summary
python scripts/listeners/healing_analyzer.py --export

# Clean old reports (keep last 30 days)
find heal_reports/ -name "heal_report_*.json" -mtime +30 -delete
```

## Getting Help

1. **Check the comprehensive README**: `scripts/listeners/README.md`
2. **Review healing reports**: Detailed insights in `heal_reports/`
3. **Test Azure connection**: `python scripts/gen_ai/quick_azure_test.py`
4. **View dashboard**: `streamlit run scripts/gen_ai/main_ui.py`

## Pro Tips

💡 **Start Simple**: Run with detection only first, review suggestions  
💡 **Track Trends**: Run regularly to build healing history  
💡 **Review Weekly**: Check flaky tests report weekly  
💡 **Share Insights**: Use dashboard to share with team  
💡 **Iterate**: Apply AI suggestions, measure improvements  

---

**Ready to make your tests self-heal!** 🔧🤖✨

For detailed documentation, see: `scripts/listeners/README.md`

