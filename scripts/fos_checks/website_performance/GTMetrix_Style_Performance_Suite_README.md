# GTMetrix-Style Performance Testing Suite

A comprehensive website performance testing suite that provides detailed analysis similar to GTMetrix, including Core Web Vitals, performance grades, waterfall charts, and historical tracking.

## Features

### Performance Checker (`gtmetrix_style_performance_checker.py`)
- **Multi-URL Analysis**: Test multiple websites from files or command line
- **Device Mode Testing**: Desktop and mobile performance analysis
- **Core Web Vitals**: LCP, FCP, CLS, INP measurement with accurate grading
- **Lighthouse Integration**: Full Lighthouse performance audits
- **Waterfall Charts**: Network request visualization
- **Screenshots**: Visual page captures
- **Performance Grades**: A-F grading system similar to GTMetrix
- **Detailed Recommendations**: Actionable performance improvement suggestions
- **CrUX Integration**: Chrome User Experience Report data (with API key)
- **Historical Data Storage**: Automatic archiving of all test results

### Enhanced Dashboard (`enhanced_gtmetrix_dashboard.py`)
- **Interactive Web Interface**: Modern, responsive dashboard
- **Historical Trends**: Time series analysis of performance metrics
- **Core Web Vitals Grading**: Visual grade tracking over time
- **Performance Insights**: AI-driven recommendations and alerts
- **Comparative Analysis**: Side-by-side performance comparisons
- **Speed Visualization**: Multiple chart types for data analysis
- **Report History**: Access to all historical performance reports
- **Real-time Updates**: Live dashboard with latest performance data

## Installation

1. **Prerequisites**:
   ```bash
   # Install Node.js and Lighthouse
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
   nvm install node
   npm install -g lighthouse
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

2. **Required Python Packages**:
   ```bash
   pip install pandas matplotlib seaborn numpy requests selenium beautifulsoup4 jinja2
   ```

3. **Chrome/Chromium Setup**:
   - Install Google Chrome or Chromium browser
   - Ensure ChromeDriver is in PATH or install via selenium manager

## Usage

### Performance Checker

#### Command Line Options
```bash
python gtmetrix_style_performance_checker.py [OPTIONS]
```

**Options:**
- `--urls URL1 URL2 ...`: List of URLs to analyze
- `--file FILE`: File containing URLs (CSV, JSON, or TXT format)
- `--output DIR`: Output directory for reports (default: ./performance_reports)
- `--devices DEVICE1 DEVICE2`: Device modes to test (desktop, mobile)
- `--workers N`: Number of parallel workers (default: 3)
- `--crux-api-key KEY`: Chrome User Experience Report API key

#### Examples

**Test single URL:**
```bash
python gtmetrix_style_performance_checker.py --urls https://example.com
```

**Test multiple URLs:**
```bash
python gtmetrix_style_performance_checker.py --urls https://site1.com https://site2.com --devices desktop mobile
```

**Test from file:**
```bash
python gtmetrix_style_performance_checker.py --file urls.csv --output ./my_reports
```

**Desktop only with custom output:**
```bash
python gtmetrix_style_performance_checker.py --file urls.txt --devices desktop --output ./desktop_reports
```

### URL File Formats

#### CSV Format
```csv
url,name
https://example.com,Example Website
https://test.com,Test Site
```

#### JSON Format
```json
[
  {"url": "https://example.com", "name": "Example Website"},
  {"url": "https://test.com", "name": "Test Site"}
]
```

#### TXT Format
```
https://example.com
https://test.com
# This is a comment
https://another-site.com
```

### Enhanced Dashboard

#### Start Dashboard Server
```bash
python enhanced_gtmetrix_dashboard.py
```

#### Generate Dashboard Only
```bash
python enhanced_gtmetrix_dashboard.py --generate-only
```

#### Custom Port and Directory
```bash
python enhanced_gtmetrix_dashboard.py --port 8080 --base-dir /path/to/reports
```

## Report Features

### Performance Report Structure
```
performance_report_YYYYMMDD_HHMMSS/
├── performance_report.html          # Main HTML report
├── screenshots/                     # Page screenshots
│   ├── screenshot_hash_desktop.png
│   └── screenshot_hash_mobile.png
├── waterfalls/                      # Network waterfall charts
│   └── waterfall_hash.png
├── charts/                          # Performance comparison charts
│   ├── performance_comparison.png
│   └── core_web_vitals_comparison.png
└── data/                           # Raw performance data
    └── metrics.csv
```

### Dashboard Features
- **Performance Trends**: Historical performance tracking
- **Core Web Vitals Grading**: A-F grade system with pass/fail rates
- **Speed Visualization**: Multiple chart types for analysis
- **Performance Matrix**: Heatmap showing site performance
- **Business Insights**: Automated recommendations and alerts
- **Report History**: Access to all historical reports
- **Responsive Design**: Works on desktop and mobile devices

## Performance Grading

### Core Web Vitals Thresholds
- **LCP (Largest Contentful Paint)**:
  - Good: ≤ 2.5s
  - Needs Improvement: ≤ 4.0s
  - Poor: > 4.0s

- **FCP (First Contentful Paint)**:
  - Good: ≤ 1.8s
  - Needs Improvement: ≤ 3.0s
  - Poor: > 3.0s

- **CLS (Cumulative Layout Shift)**:
  - Good: ≤ 0.1
  - Needs Improvement: ≤ 0.25
  - Poor: > 0.25

- **INP (Interaction to Next Paint)**:
  - Good: ≤ 200ms
  - Needs Improvement: ≤ 500ms
  - Poor: > 500ms

### Overall Performance Grades
- **A**: 90-100 points (Excellent)
- **B**: 80-89 points (Good)
- **C**: 70-79 points (Fair)
- **D**: 60-69 points (Poor)
- **F**: 0-59 points (Fail)

## Advanced Features

### Chrome User Experience Report (CrUX) Integration
Set up CrUX API key for real user metrics:
```bash
export CRUX_API_KEY="your_crux_api_key"
# or
python gtmetrix_style_performance_checker.py --crux-api-key "your_key"
```

### Custom Performance Thresholds
Edit the `performance_thresholds` dictionary in the checker to customize grading:
```python
self.performance_thresholds = {
    'lcp': {'good': 2.5, 'needs_improvement': 4.0},
    'fcp': {'good': 1.8, 'needs_improvement': 3.0},
    # ... customize as needed
}
```

### Historical Data Integration
All test results are automatically stored in the historical database:
- Performance metrics over time
- Grade progression tracking
- Comparative analysis across sites
- Trend identification and alerts

## API Integration

### Google PageSpeed Insights
The tool integrates with Google PageSpeed Insights API for additional data:
- Set `PAGESPEED_API_KEY` environment variable
- Fallback to encoded key for basic usage

### Lighthouse CLI
Direct integration with Lighthouse CLI for accurate performance measurement:
- Multiple runs for statistical accuracy
- Background process isolation
- Asset saving and archiving

## Troubleshooting

### Common Issues

1. **Lighthouse Command Not Found**:
   ```bash
   npm install -g lighthouse
   which lighthouse  # Verify installation
   ```

2. **Chrome/ChromeDriver Issues**:
   ```bash
   # Update Chrome
   # Install latest ChromeDriver
   pip install --upgrade selenium
   ```

3. **Permission Errors**:
   ```bash
   chmod +x gtmetrix_style_performance_checker.py
   chmod +x enhanced_gtmetrix_dashboard.py
   ```

4. **Memory Issues with Large Sites**:
   - Reduce number of parallel workers
   - Increase system memory limits
   - Use `--workers 1` for memory-constrained environments

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Best Practices
1. **Parallel Processing**: Use optimal worker count based on system resources
2. **Resource Management**: Automatic cleanup of temporary files
3. **Memory Efficiency**: Streaming data processing for large datasets
4. **Network Optimization**: Request throttling and retry mechanisms

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 1GB per 100 performance reports
- **Network**: Stable internet connection for accurate measurements

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the examples and documentation

## Changelog

### Version 1.0.0
- Initial release with GTMetrix-style performance checking
- Enhanced dashboard with historical tracking
- Core Web Vitals grading system
- Waterfall chart generation
- Screenshot capture
- Multi-device testing support
- Comprehensive reporting system
