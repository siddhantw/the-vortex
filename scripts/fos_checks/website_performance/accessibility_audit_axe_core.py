import json
import os
import time
import tempfile
from datetime import datetime
from urllib.parse import urlparse

from axe_selenium_python import Axe
from selenium import webdriver

# Define brands
BRANDS = [
    {"name": "Bluehost", "url": "https://www.bluehost.com/"},
    {"name": "Domain", "url": "https://www.domain.com/"},
    {"name": "HostGator", "url": "https://www.hostgator.com/"},
    {"name": "Network Solutions", "url": "https://www.networksolutions.com/"},
    {"name": "Register", "url": "https://www.register.com/"},
    {"name": "Web", "url": "https://www.web.com/"}
]

# Create directories for reports
os.makedirs("accessibility_reports/json", exist_ok=True)


def run_axe_audit(brand):
    """Run accessibility audit for a single brand"""
    url = brand["url"]
    name = brand["name"]

    print(f"Starting accessibility audit for {name} ({url})...")

    # Set up headless Chrome
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

    # Use a unique temporary directory for user data to avoid conflicts in parallel runs
    with tempfile.TemporaryDirectory(prefix=f"chrome_user_data_{os.getpid()}_") as user_data_dir:
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")
        driver = webdriver.Chrome(options=chrome_options)
        start_time = time.time()

        try:
            driver.get(url)
            # Wait for page to load
            time.sleep(5)

            axe = Axe(driver)
            axe.inject()

            results = axe.run()
            results["brand"] = name
            results["url"] = url
            results["timestamp"] = datetime.now().isoformat()
            results["load_time"] = time.time() - start_time

            # Save JSON report
            domain_name = urlparse(url).netloc.replace("www.", "")
            report_file = f"accessibility_reports/json/{domain_name}_accessibility_report.json"
            with open(report_file, "w") as f:
                json.dump(results, f, indent=4)

            print(f"✓ Completed accessibility audit for {name}")
            return results

        except Exception as e:
            print(f"✗ Error running accessibility audit for {name}: {str(e)}")
            return {
                "brand": name,
                "url": url,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            driver.quit()


def extract_metrics(results):
    """Extract key metrics from the audit results"""
    if "error" in results:
        return {
            "brand": results["brand"],
            "url": results["url"],
            "status": "Error",
            "error": results["error"],
            "timestamp": results["timestamp"],
            "violations_count": 0,
            "violations_by_impact": {},
            "passes_count": 0,
            "incomplete_count": 0,
            "inapplicable_count": 0,
            "top_violations": []
        }

    violations = results.get("violations", [])
    passes = results.get("passes", [])
    incomplete = results.get("incomplete", [])
    inapplicable = results.get("inapplicable", [])

    # Count violations by impact
    impact_counts = {"critical": 0, "serious": 0, "moderate": 0, "minor": 0}
    for violation in violations:
        impact = violation.get("impact", "unknown")
        if impact in impact_counts:
            impact_counts[impact] += 1

    # Get top violations (most nodes affected)
    top_violations = sorted(violations, key=lambda v: len(v.get("nodes", [])), reverse=True)[:5]
    top_violations_data = []

    for v in top_violations:
        top_violations_data.append({
            "id": v.get("id", "unknown"),
            "impact": v.get("impact", "unknown"),
            "description": v.get("description", "No description"),
            "help": v.get("help", "No help text"),
            "help_url": v.get("helpUrl", "#"),
            "nodes_count": len(v.get("nodes", []))
        })

    return {
        "brand": results["brand"],
        "url": results["url"],
        "status": "Success",
        "timestamp": results["timestamp"],
        "load_time": results.get("load_time", 0),
        "violations_count": len(violations),
        "violations_by_impact": impact_counts,
        "passes_count": len(passes),
        "incomplete_count": len(incomplete),
        "inapplicable_count": len(inapplicable),
        "top_violations": top_violations_data
    }


def generate_html_report(all_metrics):
    """Generate HTML report with the results"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_json = json.dumps(all_metrics)  # Serialize metrics data for JavaScript

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accessibility Audit Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .card-header {{ background-color: #f8f9fa; }}
        .table th {{ position: sticky; top: 0; background-color: #f8f9fa; z-index: 1; }}
        .metric-card {{ text-align: center; padding: 15px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
        .metric-label {{ font-size: 14px; color: #6c757d; }}
        .critical {{ color: #dc3545; }}
        .serious {{ color: #fd7e14; }}
        .moderate {{ color: #ffc107; }}
        .minor {{ color: #6c757d; }}
        .table-responsive {{ max-height: 500px; overflow-y: auto; }}
        .brand-logo {{ height: 30px; margin-right: 10px; }}
        .filters {{ margin-bottom: 20px; }}
        #search {{ width: 100%; }}
        .violation-desc {{ max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .violation-desc:hover {{ overflow: visible; white-space: normal; max-width: none; }}
        .notes {{
            margin-bottom: 24px;
            padding: 12px 18px;
            background-color: #e9f5e9;
            border-left: 4px solid #0d6efd;
            border-radius: 4px;
            font-size: 1rem;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row mb-4">
            <div class="col">
                <h1><i class="bi bi-check2-circle text-primary"></i> Accessibility Audit Report</h1>
                <p class="text-muted">Generated on {timestamp}</p>
            </div>
        </div>

        <!-- Notes Section -->
        <div class="notes">
            <p>
                <strong>Note:</strong> This report summarizes accessibility issues detected using <a href="https://www.deque.com/axe/core-documentation/" target="_blank">axe-core</a> automated testing.
                <ul>
                    <li><strong>Status:</strong> "Critical" status is shown if any critical or serious violations are present, or if total violations exceed 10. "Healthy" and "Caution" reflect lower violation counts and no critical/serious issues.</li>
                    <li><strong>Impacts:</strong> <span style="color:#dc3545;">Critical</span> and <span style="color:#fd7e14;">Serious</span> issues have the highest impact on accessibility and should be prioritized for remediation.</li>
                    <li><strong>Top Violation:</strong> The most widespread issue (by affected nodes) is highlighted for each brand.</li>
                    <li>Automated tools may not catch all accessibility issues. Manual testing is recommended for comprehensive coverage.</li>
                    <li>For more information on each violation, click the "More info" link in the detailed view.</li>
                    <li>For full details, see the downloadable JSON report for each brand.</li>
                    <li>For more information on accessibility best practices, visit the <a href="https://www.w3.org/WAI/WCAG21/quickref/" target="_blank">W3C WCAG Quick Reference</a>.</li>
                </ul>
            </p>
        </div>

        <!-- Summary Stats -->
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="bi bi-bar-chart-fill"></i> Violations Summary by Brand</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="violationsChart" height="250"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="bi bi-pie-chart-fill"></i> Violations by Impact</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="impactChart" height="250"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Filters -->
        <div class="card mb-4">
            <div class="card-header">
                <h5><i class="bi bi-funnel-fill"></i> Filters</h5>
            </div>
            <div class="card-body">
                <div class="row filters">
                    <div class="col-md-3 mb-2">
                        <label for="search" class="form-label">Search:</label>
                        <input type="text" id="search" class="form-control" placeholder="Search brands, violations...">
                    </div>
                    <div class="col-md-3 mb-2">
                        <label for="impactFilter" class="form-label">Filter by Impact:</label>
                        <select id="impactFilter" class="form-select">
                            <option value="all">All Impacts</option>
                            <option value="critical">Critical</option>
                            <option value="serious">Serious</option>
                            <option value="moderate">Moderate</option>
                            <option value="minor">Minor</option>
                        </select>
                    </div>
                    <div class="col-md-3 mb-2">
                        <label for="sortBy" class="form-label">Sort by:</label>
                        <select id="sortBy" class="form-select">
                            <option value="violations">Violations Count</option>
                            <option value="critical">Critical Issues</option>
                            <option value="brand">Brand Name</option>
                        </select>
                    </div>
                    <div class="col-md-3 mb-2">
                        <label for="sortOrder" class="form-label">Order:</label>
                        <select id="sortOrder" class="form-select">
                            <option value="desc">Descending</option>
                            <option value="asc">Ascending</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Table -->
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5><i class="bi bi-table"></i> Detailed Results</h5>
                <span id="recordCount" class="badge bg-primary"></span>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table id="resultsTable" class="table table-striped table-hover">
                        <thead>
                            <tr>
                                <th>Brand</th>
                                <th>Status</th>
                                <th>Violations</th>
                                <th>Critical</th>
                                <th>Serious</th>
                                <th>Moderate</th>
                                <th>Minor</th>
                                <th>Passes</th>
                                <th>Top Violation</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Data rows will be inserted here by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Detailed View Modal -->
        <div class="modal fade" id="detailModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Brand Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body" id="modalContent">
                        <!-- Modal content will be inserted here by JavaScript -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const metricsData = {metrics_json};

        // Store filtered/sorted data for table rendering
        let filteredData = [...metricsData];

        // Charts initialization
        function initCharts() {{
            const brandNames = metricsData.map((m) => m.brand);
            const violationCounts = metricsData.map((m) => m.violations_count);

            new Chart(document.getElementById('violationsChart'), {{
                type: 'bar',
                data: {{
                    labels: brandNames,
                    datasets: [{{
                        label: 'Total Violations',
                        data: violationCounts,
                        backgroundColor: '#0d6efd',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Number of Violations'
                            }}
                        }}
                    }}
                }}
            }});

            // Defensive: ensure all impact keys exist and are numbers
            function safeImpactSum(key) {{
                return metricsData.reduce((sum, m) => sum + (m.violations_by_impact && typeof m.violations_by_impact[key] === 'number' ? m.violations_by_impact[key] : 0), 0);
            }}

            const impactData = {{
                critical: safeImpactSum('critical'),
                serious: safeImpactSum('serious'),
                moderate: safeImpactSum('moderate'),
                minor: safeImpactSum('minor')
            }};

            new Chart(document.getElementById('impactChart'), {{
                type: 'pie',
                data: {{
                    labels: ['Critical', 'Serious', 'Moderate', 'Minor'],
                    datasets: [{{
                        data: [impactData.critical, impactData.serious, impactData.moderate, impactData.minor],
                        backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#6c757d']
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'right'
                        }}
                    }}
                }}
            }});
        }}

        // Helper to get Bootstrap badge class for impact
        function getImpactBadge(impact) {{
            if (impact === "critical") return "badge bg-danger";
            if (impact === "serious") return "badge bg-warning text-dark";
            if (impact === "moderate") return "badge bg-info text-dark";
            if (impact === "minor") return "badge bg-secondary";
            return "badge bg-light text-dark";
        }}

        // Helper to render all violations for modal
        function renderViolationsList(violations) {{
            if (!violations || violations.length === 0) {{
                return `<div class="alert alert-success mb-2">No accessibility violations found!</div>`;
            }}
            return `
                <div class="table-responsive">
                <table class="table table-bordered table-sm align-middle">
                    <thead>
                        <tr>
                            <th>Impact</th>
                            <th>Description</th>
                            <th>Nodes Affected</th>
                            <th>Fix Guidance</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${{violations.map(v => `
                            <tr>
                                <td><span class="${{getImpactBadge(v.impact)}}">${{v.impact || "unknown"}}</span></td>
                                <td>
                                    <div><strong>${{v.id}}</strong></div>
                                    <div>${{v.description}}</div>
                                    <div><a href="${{v.help_url || v.helpUrl || '#'}}" target="_blank">More info</a></div>
                                </td>
                                <td>${{v.nodes_count || (v.nodes ? v.nodes.length : 0)}}</td>
                                <td>${{v.help || "No fix guidance available."}}</td>
                            </tr>
                        `).join('')}}
                    </tbody>
                </table>
                </div>
            `;
        }}

        function getStatusAndClass(item) {{
            // Defensive: handle missing keys
            const impacts = item.violations_by_impact || {{}};
            if ((impacts.critical || 0) > 0 || (impacts.serious || 0) > 0) {{
                return {{ status: 'Critical', statusClass: 'text-danger' }};
            }} else {{
                const totalViolations = item.violations_count || 0;
                if (totalViolations <= 5) {{
                    return {{ status: 'Healthy', statusClass: 'text-success' }};
                }} else if (totalViolations <= 10) {{
                    return {{ status: 'Caution', statusClass: 'text-warning' }};
                }} else {{
                    return {{ status: 'Critical', statusClass: 'text-danger' }};
                }}
            }}
        }}

        function populateTable(data) {{
            const tbody = document.querySelector('#resultsTable tbody');
            tbody.innerHTML = '';

            data.forEach((item) => {{
                const tr = document.createElement('tr');
                const {{ status, statusClass }} = getStatusAndClass(item);

                let topViolation = "None";
                if (item.top_violations && item.top_violations.length > 0) {{
                    const top = item.top_violations[0];
                    topViolation = `<span class="violation-desc">${{top.description}}</span>`;
                }}

                const impacts = item.violations_by_impact || {{}};

                tr.innerHTML = `
                    <td>${{item.brand}}</td>
                    <td class="${{statusClass}}">${{status}}</td>
                    <td>${{item.violations_count || 0}}</td>
                    <td>${{impacts.critical || 0}}</td>
                    <td>${{impacts.serious || 0}}</td>
                    <td>${{impacts.moderate || 0}}</td>
                    <td>${{impacts.minor || 0}}</td>
                    <td>${{item.passes_count || 0}}</td>
                    <td>${{topViolation}}</td>
                    <td><button class="btn btn-primary btn-sm">View</button></td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function applyFiltersAndSort() {{
            let data = [...metricsData];

            const selectedImpact = document.getElementById('impactFilter').value;
            if (selectedImpact !== 'all') {{
                data = data.filter(item => ((item.violations_by_impact && item.violations_by_impact[selectedImpact]) || 0) > 0);
            }}

            const searchValue = document.getElementById('search').value.toLowerCase();
            if (searchValue) {{
                data = data.filter(item => {{
                    const brand = (item.brand || '').toLowerCase();
                    const topViolation = (item.top_violations && item.top_violations[0]?.description || '').toLowerCase();
                    return brand.includes(searchValue) || topViolation.includes(searchValue);
                }});
            }}

            const sortBy = document.getElementById('sortBy').value;
            const sortOrder = document.getElementById('sortOrder').value;
            data.sort((a, b) => {{
                let aValue, bValue;
                if (sortBy === 'violations') {{
                    aValue = a.violations_count || 0;
                    bValue = b.violations_count || 0;
                }} else if (sortBy === 'critical') {{
                    aValue = (a.violations_by_impact && a.violations_by_impact.critical) || 0;
                    bValue = (b.violations_by_impact && b.violations_by_impact.critical) || 0;
                }} else if (sortBy === 'brand') {{
                    aValue = (a.brand || '').toLowerCase();
                    bValue = (b.brand || '').toLowerCase();
                    if (aValue < bValue) return sortOrder === 'asc' ? -1 : 1;
                    if (aValue > bValue) return sortOrder === 'asc' ? 1 : -1;
                    return 0;
                }}
                if (typeof aValue === 'number' && typeof bValue === 'number') {{
                    return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
                }}
                return 0;
            }});

            filteredData = data;
            populateTable(filteredData);
            document.getElementById('recordCount').innerText = filteredData.length + ' Records Found';
        }}

        document.getElementById('search').addEventListener('input', applyFiltersAndSort);
        document.getElementById('impactFilter').addEventListener('change', applyFiltersAndSort);
        document.getElementById('sortBy').addEventListener('change', applyFiltersAndSort);
        document.getElementById('sortOrder').addEventListener('change', applyFiltersAndSort);

        document.addEventListener('click', function(e) {{
            if (e.target && e.target.matches('#resultsTable button')) {{
                const row = e.target.closest('tr');
                const idx = Array.from(row.parentNode.children).indexOf(row);
                const item = filteredData[idx];
                const {{ status }} = getStatusAndClass(item);

                // Find the full violations list for this brand (from metricsData)
                const fullBrandData = metricsData.find(m => m.brand === item.brand);
                // If not found, fallback to item.top_violations
                let violations = [];
                if (fullBrandData && fullBrandData.url) {{
                    // Try to load the JSON report for this brand for full violation details
                    // But since we only have summary in metricsData, use top_violations as fallback
                    violations = (fullBrandData.violations || item.top_violations || []);
                }}
                // If not available, fallback to top_violations
                if (!violations.length && item.top_violations) {{
                    violations = item.top_violations;
                }}

                // If we have a full violations array in the metricsData, use it, else try to fetch from JSON
                // But for this static HTML, we only have summary, so use top_violations and stats
                // Show all available violations from the summary
                let modalStats = `
                    <div class="mb-2">
                        <span class="me-3"><strong>Status:</strong> ${{status}}</span>
                        <span class="me-3"><strong>Total Violations:</strong> ${{item.violations_count || 0}}</span>
                        <span class="me-3"><strong>Critical:</strong> ${{(item.violations_by_impact && item.violations_by_impact.critical) || 0}}</span>
                        <span class="me-3"><strong>Serious:</strong> ${{(item.violations_by_impact && item.violations_by_impact.serious) || 0}}</span>
                        <span class="me-3"><strong>Moderate:</strong> ${{(item.violations_by_impact && item.violations_by_impact.moderate) || 0}}</span>
                        <span class="me-3"><strong>Minor:</strong> ${{(item.violations_by_impact && item.violations_by_impact.minor) || 0}}</span>
                        <span class="me-3"><strong>Passes:</strong> ${{item.passes_count || 0}}</span>
                    </div>
                `;

                // If available, show all top violations (up to 5, as per summary)
                let violationsHtml = renderViolationsList(item.top_violations);

                // Modal content
                const modalContent = `
                    <h5>${{item.brand}} - Accessibility Violations</h5>
                    ${{modalStats}}
                    <div class="mb-2">
                        <strong>Showing up to 5 most widespread violations (by affected nodes):</strong>
                    </div>
                    ${{violationsHtml}}
                    <div class="mt-2 text-muted" style="font-size:0.95em;">
                        For full details, see the downloadable JSON report for this brand.
                    </div>
                `;
                document.getElementById('modalContent').innerHTML = modalContent;
                const modal = new bootstrap.Modal(document.getElementById('detailModal'));
                modal.show();
            }}
        }});

        document.addEventListener('DOMContentLoaded', function() {{
            initCharts();
            applyFiltersAndSort();
            document.getElementById('recordCount').classList.add('badge', 'bg-primary');
            document.getElementById('recordCount').classList.remove('bg-secondary');            
        }});
    </script>
</body>
</html>
"""
    # Save the HTML report
    with open("accessibility_reports/report.html", "w") as f:
        f.write(html_content)
    print("✓ HTML report generated successfully.")

    # Open the report in the default browser
    try:
        import webbrowser
        report_path = os.path.abspath("accessibility_reports/report.html")
        webbrowser.open('file://' + report_path)
    except Exception as e:
        print(f"Could not open report automatically: {str(e)}")


def main():
    """Main function to run the accessibility audit process"""
    print("Starting accessibility audits for all brands...")

    # Run audits for all brands
    all_results = []
    for brand in BRANDS:
        results = run_axe_audit(brand)
        all_results.append(results)

    # Extract metrics from results
    all_metrics = [extract_metrics(results) for results in all_results]

    # Generate an HTML Report with the Accessibility Metrics
    generate_html_report(all_metrics)

    print("\n✓ Accessibility audit completed for all brands.")
    print(f"Report saved to: {os.path.abspath('accessibility_reports/report.html')}")


if __name__ == "__main__":
    main()
